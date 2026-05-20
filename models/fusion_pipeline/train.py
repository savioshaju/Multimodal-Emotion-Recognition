import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    recall_score,
)
import matplotlib.pyplot as plt
# Paths

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

# Configuration

MODEL_TEXT_NAME = "distilbert-base-uncased"

BASE_TRAIN_SPEAKER = "oaf"
TARGET_SPEAKER = "yaf"
ADAPT_RATIO = 0.05

MAX_TEXT_LEN = 32
BATCH_SIZE = 16
EPOCHS = 30
PATIENCE = 5
LR = 2e-5
WEIGHT_DECAY = 1e-4
SEED = 42

EMOTIONS = [
    "anger",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise",
]

NUM_CLASSES = len(EMOTIONS)
EMOTION_TO_IDX = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}
IDX_TO_EMOTION = {idx: emotion for emotion, idx in EMOTION_TO_IDX.items()}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seeds

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Dataset

class FusionDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        feature_path = row["feature_path"]
        text = str(row["text"])
        emotion = row["emotion"]

        speech_features = np.load(feature_path).astype(np.float32)

        text_inputs = self.tokenizer(
            text,
            max_length=MAX_TEXT_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        label = EMOTION_TO_IDX[emotion]

        return {
            "speech_features": torch.tensor(speech_features, dtype=torch.float32),
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }

# Speech branch

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        scores = self.attention(lstm_output).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(lstm_output * weights.unsqueeze(-1), dim=1)
        return context

class SpeechCNNBiLSTMAttention(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )

        self.lstm = nn.LSTM(
            input_size=128 * 16,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        self.attention = AttentionPooling(hidden_dim=256)

        self.projection = nn.Sequential(
            nn.Linear(256, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        x = self.cnn(x)

        
        x = x.permute(0, 3, 1, 2)

        batch_size, time_steps, channels, freq_bins = x.shape
        x = x.reshape(batch_size, time_steps, channels * freq_bins)

        lstm_output, _ = self.lstm(x)
        context = self.attention(lstm_output)

        speech_embedding = self.projection(context)

        return speech_embedding

# Model

class LightweightFusionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.speech_branch = SpeechCNNBiLSTMAttention(embedding_dim=256)

        self.text_encoder = AutoModel.from_pretrained(MODEL_TEXT_NAME)
        text_hidden = self.text_encoder.config.hidden_size

        self.text_projection = nn.Sequential(
            nn.Linear(text_hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, speech_features, input_ids, attention_mask):
        speech_embedding = self.speech_branch(speech_features)

        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        cls_embedding = text_outputs.last_hidden_state[:, 0, :]
        text_embedding = self.text_projection(cls_embedding)

        fused = torch.cat([speech_embedding, text_embedding], dim=1)

        logits = self.classifier(fused)

        return logits

    def embed(self, speech_features, input_ids, attention_mask):
        """Return fused embedding before classifier."""
        speech_emb = self.speech_branch(speech_features)
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_emb = text_outputs.last_hidden_state[:, 0, :]
        text_emb = self.text_projection(cls_emb)
        fused = torch.cat([speech_emb, text_emb], dim=1)
        return fused

# Splits

def create_speaker_aware_splits(df):
    df["speaker_id"] = df["speaker_id"].str.lower()
    df["emotion"] = df["emotion"].str.lower()

    base_train_df = df[df["speaker_id"] == BASE_TRAIN_SPEAKER].reset_index(drop=True)
    target_df = df[df["speaker_id"] == TARGET_SPEAKER].reset_index(drop=True)

    if len(base_train_df) == 0:
        raise ValueError(f"No samples found for base train speaker: {BASE_TRAIN_SPEAKER}")

    if len(target_df) == 0:
        raise ValueError(f"No samples found for target speaker: {TARGET_SPEAKER}")

    adapt_parts = []
    remaining_parts = []

    for emotion in sorted(target_df["emotion"].unique()):
        emotion_df = target_df[target_df["emotion"] == emotion].reset_index(drop=True)

        adapt_df, remain_df = train_test_split(
            emotion_df,
            train_size=ADAPT_RATIO,
            random_state=SEED,
            shuffle=True,
        )

        adapt_parts.append(adapt_df)
        remaining_parts.append(remain_df)

    adapt_df = pd.concat(adapt_parts).reset_index(drop=True)
    remaining_target_df = pd.concat(remaining_parts).reset_index(drop=True)

    val_df, test_df = train_test_split(
        remaining_target_df,
        test_size=0.70,
        random_state=SEED,
        stratify=remaining_target_df["emotion"],
    )

    train_df = pd.concat([base_train_df, adapt_df], axis=0)
    train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_df.to_csv(os.path.join(PIPELINE_DIR, "train_split.csv"), index=False)
    val_df.to_csv(os.path.join(PIPELINE_DIR, "val_split.csv"), index=False)
    test_df.to_csv(os.path.join(PIPELINE_DIR, "test_split.csv"), index=False)

    return train_df, val_df, test_df

def run_validation(model, data_loader):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            speech_features = batch["speech_features"].to(DEVICE)
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits = model(speech_features, input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    uar = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return acc, uar, macro_f1, all_labels, all_preds

def save_confusion_matrix_plot(cm):
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm)

    ax.set_xticks(np.arange(len(EMOTIONS)))
    ax.set_yticks(np.arange(len(EMOTIONS)))
    ax.set_xticklabels(EMOTIONS, rotation=45, ha="right")
    ax.set_yticklabels(EMOTIONS)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Fusion Confusion Matrix")

    for i in range(len(EMOTIONS)):
        for j in range(len(EMOTIONS)):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"))
    plt.close()

def save_training_curve(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
    plt.plot(history["epoch"], history["val_f1"], label="Validation Macro F1")
    plt.plot(history["epoch"], history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Fusion Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "training_curve.png"))
    plt.close()

# Main

def train():
    seed_everything(SEED)

    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            f"{METADATA_PATH} not found. Run preprocess.py first."
        )

    df = pd.read_csv(METADATA_PATH)

    required_columns = [
        "feature_path",
        "text",
        "emotion",
        "speaker_id",
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"metadata.csv is missing required column: {col}")

    print("\nCreating speaker-aware train/val/test splits...")
    train_df, val_df, test_df = create_speaker_aware_splits(df)

    print(f"\nTrain samples: {len(train_df)}")
    print(train_df["emotion"].value_counts().sort_index())

    print(f"\nValidation samples: {len(val_df)}")
    print(val_df["emotion"].value_counts().sort_index())

    print(f"\nTest samples: {len(test_df)}")
    print(test_df["emotion"].value_counts().sort_index())

    tokenizer = AutoTokenizer.from_pretrained(MODEL_TEXT_NAME)
    tokenizer.save_pretrained(MODELS_DIR)

    train_dataset = FusionDataset(train_df, tokenizer)
    val_dataset = FusionDataset(val_df, tokenizer)
    test_dataset = FusionDataset(test_df, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = LightweightFusionModel(NUM_CLASSES).to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "val_acc": [],
        "val_uar": [],
        "val_f1": [],
    }

    print(f"\nTraining lightweight fusion model on {DEVICE}...\n")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            speech_features = batch["speech_features"].to(DEVICE)
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()

            logits = model(speech_features, input_ids, attention_mask)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        val_acc, val_uar, val_f1, _, _ = run_validation(model, val_loader)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["val_uar"].append(val_uar)
        history["val_f1"].append(val_f1)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val UAR: {val_uar:.4f} | "
            f"Val Macro F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            patience_counter = 0

            torch.save(
                model.state_dict(),
                os.path.join(MODELS_DIR, "best_model.pth"),
            )

            print("  --> Saved new best model.")
        else:
            patience_counter += 1
            print(f"  --> No improvement. Patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    pd.DataFrame(history).to_csv(
        os.path.join(METRICS_DIR, "training_metrics.csv"),
        index=False,
    )

    save_training_curve(history)

    model_config = {
        "dataset": "TESS",
        "architecture": "CNN + BiLSTM + Attention speech branch + DistilBERT text branch + Concatenation + MLP",
        "speech_branch": "Mel Spectrogram + Delta + MFCC -> CNN -> BiLSTM -> Attention",
        "text_branch": MODEL_TEXT_NAME,
        "fusion_method": "Concatenation + MLP classifier",
        "num_classes": NUM_CLASSES,
        "classes": EMOTIONS,
        "base_train_speaker": BASE_TRAIN_SPEAKER,
        "target_speaker": TARGET_SPEAKER,
        "adapt_ratio": ADAPT_RATIO,
        "max_text_len": MAX_TEXT_LEN,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
    }

    with open(os.path.join(MODELS_DIR, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=4)

    print("\nEvaluating best model on test set...")

    model.load_state_dict(
        torch.load(
            os.path.join(MODELS_DIR, "best_model.pth"),
            map_location=DEVICE,
        )
    )

    test_acc, test_uar, test_f1, test_labels, test_preds = run_validation(
        model,
        test_loader,
    )

    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    print(f"Final Test UAR: {test_uar:.4f}")
    print(f"Final Test Macro F1: {test_f1:.4f}")

    report_text = classification_report(
        test_labels,
        test_preds,
        target_names=EMOTIONS,
        digits=4,
        zero_division=0,
    )

    print("\nClassification Report:")
    print(report_text)

    with open(os.path.join(RESULTS_OUT_DIR, "classification_report.txt"), "w") as f:
        f.write(report_text)

    report_dict = classification_report(
        test_labels,
        test_preds,
        target_names=EMOTIONS,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report_dict).transpose().to_csv(
        os.path.join(RESULTS_OUT_DIR, "classification_report.csv")
    )

    cm = confusion_matrix(test_labels, test_preds)

    pd.DataFrame(
        cm,
        index=EMOTIONS,
        columns=EMOTIONS,
    ).to_csv(os.path.join(RESULTS_OUT_DIR, "confusion_matrix.csv"))

    save_confusion_matrix_plot(cm)

    fusion_metrics = {
        "dataset": "TESS",
        "architecture": "CNN + BiLSTM + Attention speech branch + DistilBERT text branch + Concatenation + MLP",
        "speech_branch": "Mel Spectrogram + Delta + MFCC -> CNN -> BiLSTM -> Attention",
        "text_branch": MODEL_TEXT_NAME,
        "fusion_method": "Concatenation",
        "best_epoch": best_epoch,
        "best_validation_macro_f1": float(best_val_f1),
        "test_accuracy": float(test_acc),
        "test_uar": float(test_uar),
        "test_macro_f1": float(test_f1),
        "classes": EMOTIONS,
        "split_strategy": "speaker_aware",
        "base_train_speaker": BASE_TRAIN_SPEAKER,
        "target_speaker": TARGET_SPEAKER,
        "adapt_ratio": ADAPT_RATIO,
        "max_text_len": MAX_TEXT_LEN,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "seed": SEED,
        "note": "Lightweight fusion model using CNN-BiLSTM-Attention speech features and DistilBERT text representation.",
    }

    with open(os.path.join(METRICS_DIR, "fusion_metrics.json"), "w") as f:
        json.dump(fusion_metrics, f, indent=4)

    summary = {
        "test_accuracy": test_acc,
        "test_uar": test_uar,
        "test_macro_f1": test_f1,
        "speech_model_name": "CNN + BiLSTM + Attention",
        "text_model_name": MODEL_TEXT_NAME,
        "fusion_model_name": "CNN-BiLSTM-Attention + DistilBERT",
    }

    pd.DataFrame([summary]).to_csv(
        os.path.join(RESULTS_OUT_DIR, "summary.csv"),
        index=False,
    )

    print("\nSaved outputs:")
    print(f"- Best model: {os.path.join(MODELS_DIR, 'best_model.pth')}")
    print(f"- Model config: {os.path.join(MODELS_DIR, 'model_config.json')}")
    print(f"- Training metrics: {os.path.join(METRICS_DIR, 'training_metrics.csv')}")
    print(f"- Fusion metrics: {os.path.join(METRICS_DIR, 'fusion_metrics.json')}")
    print(f"- Summary: {os.path.join(RESULTS_OUT_DIR, 'summary.csv')}")
    print(f"- Classification report: {os.path.join(RESULTS_OUT_DIR, 'classification_report.csv')}")
    print(f"- Confusion matrix: {os.path.join(RESULTS_OUT_DIR, 'confusion_matrix.csv')}")
    print(f"- Plots: {PLOTS_DIR}")

if __name__ == "__main__":
    train()