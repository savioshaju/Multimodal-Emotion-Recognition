# Plot PCA for Text Pipeline

import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Resolve paths
pipeline_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(pipeline_dir, "..", ".."))
metadata_path = os.path.join(pipeline_dir, "metadata.csv")
test_split_path = os.path.join(pipeline_dir, "test_split.csv")
model_path = os.path.join(pipeline_dir, "saved_models", "best_model.pth")
results_dir = os.path.join(project_root, "results", "text_pipeline", "plots")
os.makedirs(results_dir, exist_ok=True)

# Add pipeline directory to path to import helpers
sys.path.append(pipeline_dir)
from train import TextEmotionModel, CLASS_NAMES, MODEL_NAME, MAX_LEN, build_training_data

# Ensure test_split.csv exists
if not os.path.exists(test_split_path):
    print(f"test_split.csv not found at {test_split_path}.")
    print("Recreating split CSVs using the same metadata and logic as train.py...")
    if not os.path.exists(metadata_path):
        print(f"metadata.csv not found at {metadata_path}. Running preprocess.py first...")
        from preprocess import main as preprocess_main
        preprocess_main()
    
    df = pd.read_csv(metadata_path)
    build_training_data(df)
    print("Successfully recreated train_split.csv, val_split.csv, and test_split.csv.")

# Load test split for PCA
test_df = pd.read_csv(test_split_path)
labels = test_df["emotion"].tolist()
texts = test_df["text"].fillna("").tolist()

# Determine number of classes
num_classes = len(CLASS_NAMES)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextEmotionModel(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Initialize Tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

embeddings = []
print("Extracting embeddings for PCA...")
with torch.no_grad():
    for txt in texts:
        encoded = tokenizer(
            txt,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        
        # Get CLS token embedding from BERT (before classifier)
        outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
        embeddings.append(cls_emb)

embeddings = np.stack(embeddings)

# PCA
pca = PCA(n_components=2)
pcs = pca.fit_transform(embeddings)

# Save CSV
csv_path = os.path.join(results_dir, "text_pca.csv")
pd.DataFrame({"pc1": pcs[:, 0], "pc2": pcs[:, 1], "label": labels}).to_csv(csv_path, index=False)

# Plot
plt.figure(figsize=(8, 6))
for emotion in sorted(set(labels)):
    idxs = [i for i, l in enumerate(labels) if l == emotion]
    plt.scatter(pcs[idxs, 0], pcs[idxs, 1], label=emotion, alpha=0.7)
plt.title("Text Embeddings PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
png_path = os.path.join(results_dir, "text_pca.png")
plt.savefig(png_path)
plt.close()

print(f"Saved PCA plot to {png_path}")
print(f"Saved CSV to {csv_path}")
