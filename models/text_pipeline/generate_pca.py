import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(PIPELINE_DIR, "..", ".."))

sys.path.append(PIPELINE_DIR)
from train import TextEmotionModel, TextEmotionDataset, build_training_data, CLASS_NAMES

def main():
    metadata_path = os.path.join(PIPELINE_DIR, "metadata.csv")
    if not os.path.exists(metadata_path):
        sys.exit(1)
        
    df = pd.read_csv(metadata_path)
    
    _, _, test_df = build_training_data(df)
    
    encoder = LabelEncoder()
    encoder.fit(CLASS_NAMES)
    
    saved_models_dir = os.path.join(PIPELINE_DIR, "saved_models")
    if os.path.exists(os.path.join(saved_models_dir, "tokenizer_config.json")):
        tokenizer = AutoTokenizer.from_pretrained(saved_models_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
    test_dataset = TextEmotionDataset(test_df, encoder, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextEmotionModel(num_classes=len(CLASS_NAMES)).to(device)
    
    best_model_path = os.path.join(saved_models_dir, "best_model.pth")
    state_dict = torch.load(best_model_path, map_location=device)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    
    embeddings = []
    def hook_fn(module, input, output):
        cls_output = output.last_hidden_state[:, 0, :]
        embeddings.append(cls_output.detach().cpu())
        
    hook = model.bert.register_forward_hook(hook_fn)
    
    all_emotions = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]
            _ = model(input_ids, attention_mask)
            for l in labels:
                all_emotions.append(CLASS_NAMES[l.item()])
                
    hook.remove()
    
    embeddings = torch.cat(embeddings, dim=0).numpy()
    
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    out_df = pd.DataFrame(embeddings_2d, columns=["PCA_1", "PCA_2"])
    out_df["Emotion"] = all_emotions
    
    plots_dir = os.path.join(PROJECT_ROOT, "results", "text_pipeline", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    csv_path = os.path.join(plots_dir, "text_pca.csv")
    out_df.to_csv(csv_path, index=False)
    
    plt.figure(figsize=(8, 6))
    colors = {
        "anger": "#ef5350",
        "disgust": "#66bb6a",
        "fear": "#ab47bc",
        "happiness": "#ffee58",
        "neutral": "#90a4ae",
        "sadness": "#42a5f5",
        "surprise": "#ffa726",
    }
    
    for emotion in CLASS_NAMES:
        emotion_df = out_df[out_df["Emotion"] == emotion]
        plt.scatter(
            emotion_df["PCA_1"],
            emotion_df["PCA_2"],
            label=emotion,
            color=colors.get(emotion, "#000000"),
            alpha=0.7,
            edgecolors="none"
        )
        
    plt.title("Text Contextual Modelling Embeddings PCA")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    png_path = os.path.join(plots_dir, "text_pca.png")
    plt.savefig(png_path)
    plt.close()

if __name__ == "__main__":
    main()
