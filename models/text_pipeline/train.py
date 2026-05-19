import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    recall_score,
    f1_score,
    accuracy_score
)

from transformers import AutoTokenizer, AutoModel

# Resolve project paths

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(PIPELINE_DIR, "..", ".."))

METADATA_PATH = os.path.join(PIPELINE_DIR, "metadata.csv")
MODELS_DIR    = os.path.join(PIPELINE_DIR, "saved_models")

OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "results", "text_pipeline")
METRICS_DIR = os.path.join(OUTPUT_ROOT, "metrics")
PLOTS_DIR   = os.path.join(OUTPUT_ROOT, "plots")
RESULTS_DIR = os.path.join(OUTPUT_ROOT, "results")

for d in [MODELS_DIR, METRICS_DIR, PLOTS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# Model configuration

BATCH_SIZE = 16
EPOCHS = 20
LR = 2e-5
PATIENCE = 5
SEED = 42
MAX_LEN = 32

BASE_TRAIN_SPEAKER = "oaf"
TARGET_SPEAKER = "yaf"
ADAPT_RATIO = 0.05

MODEL_NAME = "distilbert-base-uncased"

CLASS_NAMES = [
    "anger",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise"
]

# Set random seeds

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Dataset definition

class TextEmotionDataset(Dataset):
    def __init__(self, df, encoder, tokenizer, max_len=MAX_LEN):
        self.df = df.reset_index(drop=True)
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        text = str(row["text"])
        emotion = row["emotion"]

        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        label = self.encoder.transform([emotion])[0]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(label, dtype=torch.long)
        }

# Model architecture

class TextEmotionModel(nn.Module):
    def __init__(self, num_classes, model_name=MODEL_NAME):
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Use CLS token representation
        cls_output = outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(cls_output)
        return logits

# Build train, validation, and test splits

def build_training_data(df):
    df["speaker_id"] = df["speaker_id"].str.lower()
    df["emotion"] = df["emotion"].str.lower()
    df["text"] = df["text"].astype(str).str.lower().str.strip()

    base_train_df = df[df["speaker_id"] == BASE_TRAIN_SPEAKER].reset_index(drop=True)
    target_df = df[df["speaker_id"] == TARGET_SPEAKER].reset_index(drop=True)

    if len(base_train_df) == 0:
        raise ValueError(f"No samples found for BASE_TRAIN_SPEAKER = {BASE_TRAIN_SPEAKER}")

    if len(target_df) == 0:
        raise ValueError(f"No samples found for TARGET_SPEAKER = {TARGET_SPEAKER}")

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

    print("\nSplit Summary")
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

    print("\nTrain class distribution:")
    print(train_df["emotion"].value_counts().sort_index())

    print("\nValidation class distribution:")
    print(val_df["emotion"].value_counts().sort_index())

    print("\nTest class distribution:")
    print(test_df["emotion"].value_counts().sort_index())

    return train_df, val_df, test_df

# Evaluation helpers

def evaluate(model, loader, device):
    model.eval()

    predictions = []
    actuals = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, dim=1)

            predictions.extend(predicted.cpu().numpy())
            actuals.extend(labels.cpu().numpy())

    accuracy = accuracy_score(actuals, predictions) * 100
    uar = recall_score(actuals, predictions, average="macro", zero_division=0) * 100
    f1 = f1_score(actuals, predictions, average="macro", zero_division=0) * 100

    return accuracy, uar, f1, actuals, predictions

# Plotting functions

def save_curves(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_f1"], label="Validation Macro F1")
    plt.title("Text Pipeline Training Loss and Validation Macro F1")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "training_curve.png"))
    plt.close()

# Main training loop

def main():
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            "metadata.csv not found. Run text_pipeline/preprocess.py first."
        )

    df = pd.read_csv(METADATA_PATH)

    required_columns = ["text", "emotion", "speaker_id"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column in metadata.csv: {col}")

    encoder = LabelEncoder()
    encoder.fit(CLASS_NAMES)

    train_df, val_df, test_df = build_training_data(df)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = TextEmotionDataset(train_df, encoder, tokenizer)
    val_dataset = TextEmotionDataset(val_df, encoder, tokenizer)
    test_dataset = TextEmotionDataset(test_df, encoder, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing Device: {device}")

    model = TextEmotionModel(num_classes=len(CLASS_NAMES)).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best_val_f1 = 0.0
    best_epoch = 0
    counter = 0

    history = {
        "train_loss": [],
        "val_accuracy": [],
        "val_uar": [],
        "val_f1": []
    }

    print("\nStarting Transformer-based Text Emotion Recognition Training...\n")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        val_acc, val_uar, val_f1, _, _ = evaluate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_accuracy"].append(val_acc)
        history["val_uar"].append(val_uar)
        history["val_f1"].append(val_f1)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Accuracy: {val_acc:.2f}% | "
            f"Val UAR: {val_uar:.2f}% | "
            f"Val Macro F1: {val_f1:.2f}%"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            counter = 0

            torch.save(
                model.state_dict(),
                os.path.join(MODELS_DIR, "best_model.pth")
            )

            config_data = {
                "model_name": MODEL_NAME,
                "classes": CLASS_NAMES,
                "max_len": MAX_LEN
            }

            with open(os.path.join(MODELS_DIR, "model_config.json"), "w") as f:
                json.dump(config_data, f, indent=4)

            tokenizer.save_pretrained(MODELS_DIR)

        else:
            counter += 1

            if counter >= PATIENCE:
                print("\nEarly stopping triggered")
                break

    print(
        f"\nTraining Complete. "
        f"Best Epoch: {best_epoch}, "
        f"Best Validation Macro F1: {best_val_f1:.2f}%"
    )

    model.load_state_dict(
        torch.load(
            os.path.join(MODELS_DIR, "best_model.pth"),
            map_location=device,
            weights_only=True
        )
    )

    test_acc, test_uar, test_f1, test_actuals, test_predictions = evaluate(
        model,
        test_loader,
        device
    )

    print("\nFinal Test Results")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test UAR: {test_uar:.2f}%")
    print(f"Test Macro F1: {test_f1:.2f}%")

    report_txt = classification_report(
        test_actuals,
        test_predictions,
        target_names=CLASS_NAMES,
        zero_division=0
    )

    print("\nClassification Report:\n", report_txt)

    report_dict = classification_report(
        test_actuals,
        test_predictions,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report_dict).transpose().to_csv(
        os.path.join(RESULTS_DIR, "classification_report.csv")
    )

    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
        f.write(report_txt)

    cm = confusion_matrix(test_actuals, test_predictions)
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_df.to_csv(os.path.join(RESULTS_DIR, "confusion_matrix.csv"))

    summary_df = pd.DataFrame([{
        "test_accuracy": test_acc,
        "test_uar": test_uar,
        "test_macro_f1": test_f1,
        "model_name": MODEL_NAME
    }])

    summary_df.to_csv(
        os.path.join(RESULTS_DIR, "summary.csv"),
        index=False
    )

    text_metrics = {
        "dataset": "TESS (Toronto Emotional Speech Set)",
        "model_name": MODEL_NAME,
        "architecture": "DistilBERT Transformer Encoder + CLS Pooling + Classification Head",
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_val_f1,
        "test_accuracy": test_acc,
        "test_uar": test_uar,
        "test_macro_f1": test_f1,
        "classes": CLASS_NAMES,
        "split_strategy": "speaker_aware",
        "base_train_speaker": BASE_TRAIN_SPEAKER,
        "target_speaker": TARGET_SPEAKER,
        "adapt_ratio": ADAPT_RATIO,
        "max_len": MAX_LEN,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "seed": SEED,
        "note": (
            "TESS text content is extracted from filenames/transcripts and is limited. "
            "Text-only emotion performance may not generalize like models trained on full natural-language emotion datasets."
        )
    }

    with open(os.path.join(METRICS_DIR, "text_metrics.json"), "w") as f:
        json.dump(text_metrics, f, indent=4)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )
    plt.title("Text Pipeline Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"))
    plt.close()

    metrics_df = pd.DataFrame({
        "Epoch": list(range(1, len(history["train_loss"]) + 1)),
        "Train_Loss": history["train_loss"],
        "Val_Accuracy": history["val_accuracy"],
        "Val_UAR": history["val_uar"],
        "Val_Macro_F1": history["val_f1"]
    })

    metrics_df.to_csv(
        os.path.join(METRICS_DIR, "training_metrics.csv"),
        index=False
    )

    save_curves(history)

    print("\nAll Text Pipeline Outputs Saved Successfully")

if __name__ == "__main__":
    main()