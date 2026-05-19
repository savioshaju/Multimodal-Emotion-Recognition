import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
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

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "fusion_pipeline_MELD")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
RESULTS_OUT_DIR = os.path.join(RESULTS_DIR, "results")
MODELS_DIR = os.path.join(PIPELINE_DIR, "saved_models")

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_OUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# Model configuration
MODEL_TEXT_NAME = "distilbert-base-uncased"

MAX_TEXT_LEN = 64
BATCH_SIZE = 8
EPOCHS = 12
PATIENCE = 3
SEED = 42

SPEECH_LR = 1e-4
TEXT_LR = 1e-5
HEAD_LR = 2e-5
WEIGHT_DECAY = 1e-4

FREEZE_TEXT_ENCODER_EPOCHS = 1
USE_CLASS_WEIGHTED_LOSS = True

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


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FusionMELDDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        feature_path = row["feature_path"]
        text = str(row["text"])
        emotion = str(row["emotion"]).lower().strip()

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
            dropout=0.4,
        )

        self.attention = AttentionPooling(hidden_dim=256)

        self.projection = nn.Sequential(
            nn.Linear(256, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
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


class MELDFusionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.speech_branch = SpeechCNNBiLSTMAttention(embedding_dim=256)

        self.text_encoder = AutoModel.from_pretrained(MODEL_TEXT_NAME)
        text_hidden = self.text_encoder.config.hidden_size

        self.text_projection = nn.Sequential(
            nn.Linear(text_hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),

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


def freeze_text_encoder(model):
    for param in model.text_encoder.parameters():
        param.requires_grad = False


def unfreeze_text_encoder(model):
    for param in model.text_encoder.parameters():
        param.requires_grad = True


def load_meld_splits():
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            f"{METADATA_PATH} not found. Run preprocess.py first."
        )

    df = pd.read_csv(METADATA_PATH)

    required_columns = [
        "feature_path",
        "text",
        "emotion",
        "split",
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"metadata.csv is missing required column: {col}")

    df["emotion"] = df["emotion"].astype(str).str.lower().str.strip()
    df["split"] = df["split"].astype(str).str.lower().str.strip()

    df = df[df["emotion"].isin(EMOTIONS)].reset_index(drop=True)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "dev"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    if len(train_df) == 0:
        raise ValueError("No train samples found in metadata.csv")

    if len(val_df) == 0:
        raise ValueError("No dev/validation samples found in metadata.csv")

    if len(test_df) == 0:
        raise ValueError("No test samples found in metadata.csv")

    train_df.to_csv(os.path.join(PIPELINE_DIR, "train_split.csv"), index=False)
    val_df.to_csv(os.path.join(PIPELINE_DIR, "val_split.csv"), index=False)
    test_df.to_csv(os.path.join(PIPELINE_DIR, "test_split.csv"), index=False)

    return train_df, val_df, test_df


def compute_class_weights(train_df):
    labels = train_df["emotion"].map(EMOTION_TO_IDX).values
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)

    class_weights = 1.0 / np.maximum(class_counts, 1)
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
    ).to(DEVICE)

    return class_weights, class_counts


def build_optimizer(model):
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.speech_branch.parameters(),
                "lr": SPEECH_LR,
            },
            {
                "params": model.text_encoder.parameters(),
                "lr": TEXT_LR,
            },
            {
                "params": model.text_projection.parameters(),
                "lr": HEAD_LR,
            },
            {
                "params": model.classifier.parameters(),
                "lr": HEAD_LR,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    return optimizer


def build_frozen_optimizer(model):
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.speech_branch.parameters(),
                "lr": SPEECH_LR,
            },
            {
                "params": model.text_projection.parameters(),
                "lr": HEAD_LR,
            },
            {
                "params": model.classifier.parameters(),
                "lr": HEAD_LR,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    return optimizer


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
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return acc, uar, macro_f1, weighted_f1, all_labels, all_preds


def save_confusion_matrix_plot(cm):
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm)

    ax.set_xticks(np.arange(len(EMOTIONS)))
    ax.set_yticks(np.arange(len(EMOTIONS)))
    ax.set_xticklabels(EMOTIONS, rotation=45, ha="right")
    ax.set_yticklabels(EMOTIONS)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("MELD Fusion Confusion Matrix")

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
    plt.title("MELD Fusion Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "training_curve.png"))
    plt.close()


def train():
    seed_everything(SEED)

    train_df, val_df, test_df = load_meld_splits()

    print("\nUsing official MELD train/dev/test splits.")

    print(f"\nTrain samples: {len(train_df)}")
    print(train_df["emotion"].value_counts().sort_index())

    print(f"\nValidation samples: {len(val_df)}")
    print(val_df["emotion"].value_counts().sort_index())

    print(f"\nTest samples: {len(test_df)}")
    print(test_df["emotion"].value_counts().sort_index())

    tokenizer = AutoTokenizer.from_pretrained(MODEL_TEXT_NAME)
    tokenizer.save_pretrained(MODELS_DIR)

    train_dataset = FusionMELDDataset(train_df, tokenizer)
    val_dataset = FusionMELDDataset(val_df, tokenizer)
    test_dataset = FusionMELDDataset(test_df, tokenizer)

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

    model = MELDFusionModel(NUM_CLASSES).to(DEVICE)

    class_weights, class_counts = compute_class_weights(train_df)

    print("\nUsing class-weighted loss for MELD class imbalance.")
    print("Train class counts and loss weights:")

    for emotion, count, weight in zip(EMOTIONS, class_counts, class_weights.detach().cpu().numpy()):
        print(f"{emotion:10s}: count={count:5d} | weight={weight:.4f}")

    if USE_CLASS_WEIGHTED_LOSS:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=0.03,
        )
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.03)

    if FREEZE_TEXT_ENCODER_EPOCHS > 0:
        freeze_text_encoder(model)
        optimizer = build_frozen_optimizer(model)
        print(f"\nText encoder frozen for first {FREEZE_TEXT_ENCODER_EPOCHS} epoch(s).")
    else:
        optimizer = build_optimizer(model)

    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "val_acc": [],
        "val_uar": [],
        "val_f1": [],
        "val_weighted_f1": [],
    }

    print(f"\nTraining MELD fusion model on {DEVICE}...\n")

    for epoch in range(EPOCHS):
        if epoch == FREEZE_TEXT_ENCODER_EPOCHS:
            unfreeze_text_encoder(model)
            optimizer = build_optimizer(model)
            print("\nText encoder unfrozen.")

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        val_acc, val_uar, val_f1, val_weighted_f1, _, _ = run_validation(
            model,
            val_loader,
        )

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["val_uar"].append(val_uar)
        history["val_f1"].append(val_f1)
        history["val_weighted_f1"].append(val_weighted_f1)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}% | "
            f"Val UAR: {val_uar * 100:.2f}% | "
            f"Val Macro F1: {val_f1 * 100:.2f}% | "
            f"Val Weighted F1: {val_weighted_f1 * 100:.2f}%"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            patience_counter = 0

            torch.save(
                model.state_dict(),
                os.path.join(MODELS_DIR, "best_model.pth"),
            )

            print("Best model saved.")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    pd.DataFrame(history).to_csv(
        os.path.join(METRICS_DIR, "training_metrics.csv"),
        index=False,
    )

    save_training_curve(history)

    model_config = {
        "dataset": "MELD",
        "architecture": "CNN + BiLSTM + Attention speech branch + DistilBERT text branch + Concatenation + MLP",
        "speech_branch": "Mel Spectrogram + Delta + MFCC -> CNN -> BiLSTM -> Attention",
        "text_branch": MODEL_TEXT_NAME,
        "fusion_method": "Concatenation + MLP classifier",
        "num_classes": NUM_CLASSES,
        "classes": EMOTIONS,
        "split_strategy": "official MELD train/dev/test split",
        "max_text_len": MAX_TEXT_LEN,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "speech_learning_rate": SPEECH_LR,
        "text_learning_rate": TEXT_LR,
        "head_learning_rate": HEAD_LR,
        "weight_decay": WEIGHT_DECAY,
        "class_weighted_loss": USE_CLASS_WEIGHTED_LOSS,
        "weighted_sampler": False,
        "freeze_text_encoder_epochs": FREEZE_TEXT_ENCODER_EPOCHS,
        "seed": SEED,
    }

    with open(os.path.join(MODELS_DIR, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=4)

    print("\nEvaluating best model on MELD test set...")

    best_model_path = os.path.join(MODELS_DIR, "best_model.pth")

    if not os.path.exists(best_model_path):
        raise FileNotFoundError("No best_model.pth was saved during training.")

    model.load_state_dict(
        torch.load(
            best_model_path,
            map_location=DEVICE,
        )
    )

    test_acc, test_uar, test_f1, test_weighted_f1, test_labels, test_preds = run_validation(
        model,
        test_loader,
    )

    print(f"\nFinal Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Final Test UAR: {test_uar * 100:.2f}%")
    print(f"Final Test Macro F1: {test_f1 * 100:.2f}%")
    print(f"Final Test Weighted F1: {test_weighted_f1 * 100:.2f}%")

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
        "dataset": "MELD",
        "architecture": "CNN + BiLSTM + Attention speech branch + DistilBERT text branch + Concatenation + MLP",
        "speech_branch": "Mel Spectrogram + Delta + MFCC -> CNN -> BiLSTM -> Attention",
        "text_branch": MODEL_TEXT_NAME,
        "fusion_method": "Concatenation",
        "best_epoch": best_epoch,
        "best_validation_macro_f1": float(best_val_f1),
        "test_accuracy": float(test_acc),
        "test_uar": float(test_uar),
        "test_macro_f1": float(test_f1),
        "test_weighted_f1": float(test_weighted_f1),
        "classes": EMOTIONS,
        "split_strategy": "official MELD train/dev/test split",
        "max_text_len": MAX_TEXT_LEN,
        "batch_size": BATCH_SIZE,
        "speech_learning_rate": SPEECH_LR,
        "text_learning_rate": TEXT_LR,
        "head_learning_rate": HEAD_LR,
        "class_weighted_loss": USE_CLASS_WEIGHTED_LOSS,
        "weighted_sampler": False,
        "freeze_text_encoder_epochs": FREEZE_TEXT_ENCODER_EPOCHS,
        "seed": SEED,
        "note": "MELD fusion model using class-weighted loss instead of weighted sampling.",
    }

    with open(os.path.join(METRICS_DIR, "fusion_metrics.json"), "w") as f:
        json.dump(fusion_metrics, f, indent=4)

    summary = {
        "dataset": "MELD",
        "test_accuracy": test_acc,
        "test_uar": test_uar,
        "test_macro_f1": test_f1,
        "test_weighted_f1": test_weighted_f1,
        "speech_model_name": "CNN + BiLSTM + Attention",
        "text_model_name": MODEL_TEXT_NAME,
        "fusion_model_name": "CNN-BiLSTM-Attention + DistilBERT",
        "imbalance_strategy": "class_weighted_loss",
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