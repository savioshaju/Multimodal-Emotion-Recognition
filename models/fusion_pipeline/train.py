import os
import json
import random
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- Paths & Setup ---
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(PIPELINE_DIR, "..", ".."))
METADATA_PATH = os.path.join(PIPELINE_DIR, "metadata.csv")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "fusion_pipeline")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
RESULTS_OUT_DIR = os.path.join(RESULTS_DIR, "results")
MODELS_DIR = os.path.join(PIPELINE_DIR, "saved_models")

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_OUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# --- Config ---
MODEL_SPEECH_NAME = "microsoft/wavlm-base"
MODEL_TEXT_NAME = "distilbert-base-uncased"
BASE_TRAIN_SPEAKER = "oaf"
TARGET_SPEAKER = "yaf"
ADAPT_RATIO = 0.05
SR = 16000
DURATION = 3
MAX_AUDIO_LEN = SR * DURATION
MAX_TEXT_LEN = 32
BATCH_SIZE = 8
EPOCHS = 30
PATIENCE = 5
LR = 2e-5
WEIGHT_DECAY = 1e-4

EMOTIONS = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
NUM_CLASSES = len(EMOTIONS)
EMOTION_TO_IDX = {emo: i for i, emo in enumerate(EMOTIONS)}
IDX_TO_EMOTION = {i: emo for i, emo in enumerate(EMOTIONS)}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- Data Augmentation (Speech) ---
def augment_audio(y, sr):
    aug_type = random.choice(["noise", "volume", "shift", "pitch", "stretch", "none"])
    if aug_type == "noise":
        noise_amp = 0.005 * np.random.uniform() * np.amax(y)
        y = y + noise_amp * np.random.normal(size=y.shape[0])
    elif aug_type == "volume":
        y = y * np.random.uniform(0.8, 1.2)
    elif aug_type == "shift":
        shift_range = int(np.random.uniform(0.1, 0.3) * sr)
        y = np.roll(y, shift_range)
    elif aug_type == "pitch":
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=np.random.uniform(-2, 2))
    elif aug_type == "stretch":
        y = librosa.effects.time_stretch(y, rate=np.random.uniform(0.8, 1.2))
    return y

# --- Dataset ---
class FusionDataset(Dataset):
    def __init__(self, df, tokenizer, is_train=False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = row["file_path"]
        text = row["text"]
        emotion = row["emotion"]
        label = EMOTION_TO_IDX[emotion]

        # Speech Preprocessing
        try:
            y, _ = librosa.load(file_path, sr=SR)
            y, _ = librosa.effects.trim(y, top_db=30)
            if self.is_train and random.random() < 0.5:
                y = augment_audio(y, SR)
            
            if len(y) > MAX_AUDIO_LEN:
                y = y[:MAX_AUDIO_LEN]
            else:
                y = np.pad(y, (0, MAX_AUDIO_LEN - len(y)), "constant")
            
            y = librosa.util.normalize(y)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            y = np.zeros(MAX_AUDIO_LEN)
            
        # Text Preprocessing
        text_inputs = self.tokenizer(
            text,
            max_length=MAX_TEXT_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_values": torch.tensor(y, dtype=torch.float32),
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

# --- Model ---
class MultimodalFusionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.speech_encoder = AutoModel.from_pretrained(MODEL_SPEECH_NAME)
        self.text_encoder = AutoModel.from_pretrained(MODEL_TEXT_NAME)
        
        speech_hidden = self.speech_encoder.config.hidden_size
        text_hidden = self.text_encoder.config.hidden_size
        
        self.speech_proj = nn.Sequential(
            nn.Linear(speech_hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(text_hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_values, input_ids, attention_mask):
        # Speech branch
        speech_outputs = self.speech_encoder(input_values)
        speech_features = speech_outputs.last_hidden_state.mean(dim=1) # Mean pooling
        speech_proj = self.speech_proj(speech_features)
        
        # Text branch
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :] # CLS token
        text_proj = self.text_proj(text_features)
        
        # Fusion
        fused = torch.cat([speech_proj, text_proj], dim=1)
        logits = self.classifier(fused)
        
        return logits

# --- Training Loop ---
def train():
    seed_everything(42)
    
    if not os.path.exists(METADATA_PATH):
        print(f"Error: {METADATA_PATH} not found. Run preprocess.py first.")
        return

    df = pd.read_csv(METADATA_PATH)
    
    SEED = 42
    df["speaker_id"] = df["speaker_id"].str.lower()
    df["emotion"] = df["emotion"].str.lower()
    
    base_train_df = df[df["speaker_id"] == BASE_TRAIN_SPEAKER].reset_index(drop=True)
    target_df = df[df["speaker_id"] == TARGET_SPEAKER].reset_index(drop=True)
    
    adapt_parts = []
    remaining_parts = []
    
    for emotion in sorted(target_df["emotion"].unique()):
        emotion_df = target_df[target_df["emotion"] == emotion].reset_index(drop=True)
        adapt_df, remain_df = train_test_split(
            emotion_df,
            train_size=ADAPT_RATIO,
            random_state=SEED,
            shuffle=True
        )
        adapt_parts.append(adapt_df)
        remaining_parts.append(remain_df)
        
    adapt_df = pd.concat(adapt_parts).reset_index(drop=True)
    remaining_target_df = pd.concat(remaining_parts).reset_index(drop=True)
    
    val_df, test_df = train_test_split(
        remaining_target_df,
        test_size=0.70,
        random_state=SEED,
        stratify=remaining_target_df["emotion"]
    )
    
    train_df = pd.concat([base_train_df, adapt_df], axis=0)
    train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    train_df.to_csv(os.path.join(PIPELINE_DIR, "train_split.csv"), index=False)
    val_df.to_csv(os.path.join(PIPELINE_DIR, "val_split.csv"), index=False)
    test_df.to_csv(os.path.join(PIPELINE_DIR, "test_split.csv"), index=False)
    
    print(f"Train split: {len(train_df)}")
    print("Train Class Distribution:")
    print(train_df["emotion"].value_counts())
    print(f"\nVal split: {len(val_df)}")
    print("Val Class Distribution:")
    print(val_df["emotion"].value_counts())
    print(f"\nTest split: {len(test_df)}")
    print("Test Class Distribution:")
    print(test_df["emotion"].value_counts())

    tokenizer = AutoTokenizer.from_pretrained(MODEL_TEXT_NAME)
    tokenizer.save_pretrained(MODELS_DIR)
    
    train_dataset = FusionDataset(train_df, tokenizer, is_train=True)
    val_dataset = FusionDataset(val_df, tokenizer, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = MultimodalFusionModel(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    best_val_f1 = 0
    patience_counter = 0
    
    history = {"epoch": [], "train_loss": [], "val_acc": [], "val_uar": [], "val_f1": []}
    
    print(f"Training started on {DEVICE}...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch in train_loader:
            input_values = batch["input_values"].to(DEVICE)
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(input_values, input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_values = batch["input_values"].to(DEVICE)
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                
                logits = model(input_values, input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1)
                
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                
        val_acc = accuracy_score(val_labels, val_preds)
        val_uar = recall_score(val_labels, val_preds, average="macro")
        val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["val_uar"].append(val_uar)
        history["val_f1"].append(val_f1)
        
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val UAR: {val_uar:.4f} | Val F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "best_model.pth"))
            print("  --> Saved new best model!")
        else:
            patience_counter += 1
            
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break
            
    # Save config
    model_config = {
        "model_speech": MODEL_SPEECH_NAME,
        "model_text": MODEL_TEXT_NAME,
        "num_classes": NUM_CLASSES,
        "emotions": EMOTIONS,
        "sr": SR,
        "duration": DURATION,
        "max_text_len": MAX_TEXT_LEN
    }
    with open(os.path.join(MODELS_DIR, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=4)
        
    # Save training curves
    plt.figure(figsize=(10, 5))
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
    plt.plot(history["epoch"], history["val_f1"], label="Val F1 (Macro)")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.title("Training Curve")
    plt.savefig(os.path.join(PLOTS_DIR, "training_curve.png"))
    plt.close()
    
    pd.DataFrame(history).to_csv(os.path.join(METRICS_DIR, "training_metrics.csv"), index=False)
    
    # Eval on Test Set
    print("\n--- Evaluating on Test Set ---")
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "best_model.pth"), map_location=DEVICE))
    model.eval()
    
    test_dataset = FusionDataset(test_df, tokenizer, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_values = batch["input_values"].to(DEVICE)
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            
            logits = model(input_values, input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            
    test_acc = accuracy_score(test_labels, test_preds)
    test_uar = recall_score(test_labels, test_preds, average="macro")
    test_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)
    
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print(f"Final Test UAR: {test_uar:.4f}")
    print(f"Final Test Macro F1: {test_f1:.4f}")
    
    clf_report = classification_report(test_labels, test_preds, target_names=EMOTIONS, digits=4)
    print("\nClassification Report:")
    print(clf_report)
    
    with open(os.path.join(RESULTS_OUT_DIR, "classification_report.txt"), "w") as f:
        f.write(clf_report)
        
    clf_dict = classification_report(test_labels, test_preds, target_names=EMOTIONS, output_dict=True)
    pd.DataFrame(clf_dict).transpose().to_csv(os.path.join(RESULTS_OUT_DIR, "classification_report.csv"))
    
    cm = confusion_matrix(test_labels, test_preds)
    pd.DataFrame(cm, index=EMOTIONS, columns=EMOTIONS).to_csv(os.path.join(RESULTS_OUT_DIR, "confusion_matrix.csv"))
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"))
    plt.close()
    
    fusion_metrics = {
        "dataset": "TESS",
        "speech_model_name": MODEL_SPEECH_NAME,
        "text_model_name": MODEL_TEXT_NAME,
        "architecture": "WavLM + DistilBERT + Concat + MLP",
        "best_epoch": int(np.argmax(history["val_f1"]) + 1) if len(history["val_f1"]) > 0 else 0,
        "best_validation_macro_f1": float(best_val_f1),
        "test_accuracy": float(test_acc),
        "test_uar": float(test_uar),
        "test_macro_f1": float(test_f1),
        "classes": EMOTIONS,
        "split_strategy": "Speaker-aware split",
        "base_train_speaker": BASE_TRAIN_SPEAKER,
        "target_speaker": TARGET_SPEAKER,
        "adapt_ratio": ADAPT_RATIO,
        "sr": SR,
        "duration": DURATION,
        "max_text_len": MAX_TEXT_LEN,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "seed": 42,
        "note": "Fusion model metrics"
    }
    with open(os.path.join(METRICS_DIR, "fusion_metrics.json"), "w") as f:
        json.dump(fusion_metrics, f, indent=4)
        
    summary = {
        "test_accuracy": test_acc,
        "test_uar": test_uar,
        "test_macro_f1": test_f1,
        "speech_model_name": MODEL_SPEECH_NAME,
        "text_model_name": MODEL_TEXT_NAME
    }
    pd.DataFrame([summary]).to_csv(os.path.join(RESULTS_OUT_DIR, "summary.csv"), index=False)
    
if __name__ == "__main__":
    train()
