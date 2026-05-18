import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# =========================
# PATH CONFIGURATION
# =========================

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(PIPELINE_DIR, "..", ".."))

METADATA_PATH = os.path.join(PIPELINE_DIR, "metadata.csv")
MODELS_DIR = os.path.join(PIPELINE_DIR, "saved_models")

OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "results", "speech_pipeline")
METRICS_DIR = os.path.join(OUTPUT_ROOT, "metrics")
PLOTS_DIR = os.path.join(OUTPUT_ROOT, "plots")
RESULTS_DIR = os.path.join(OUTPUT_ROOT, "results")

for d in [MODELS_DIR, METRICS_DIR, PLOTS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# =========================
# CONFIGURATION
# =========================

SEED = 42
BATCH_SIZE = 16
EPOCHS = 40
LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 7

BASE_TRAIN_SPEAKER = "oaf"
TARGET_SPEAKER = "yaf"
ADAPT_RATIO = 0.05

USE_WEIGHTED_SAMPLER = True
USE_CLASS_WEIGHTS = False

CLASS_NAMES = [
    "anger",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise"
]

MODEL_NAME = "CNN_BiLSTM_Attention_MelDeltaMFCC"

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================
# DATASET
# =========================

class SpeechFeatureDataset(Dataset):
    def __init__(self, df, label_encoder, augment=False):
        self.df = df.reset_index(drop=True)
        self.label_encoder = label_encoder
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def augment_feature(self, x):
        """
        Light feature-space augmentation.
        x shape: (3, 128, time)
        """
        x = x.copy()

        # Add small Gaussian noise
        if random.random() < 0.40:
            noise = np.random.normal(0, 0.015, size=x.shape).astype(np.float32)
            x = x + noise

        # Random gain
        if random.random() < 0.40:
            gain = np.random.uniform(0.90, 1.10)
            x = x * gain

        # Time masking
        if random.random() < 0.35:
            time_len = x.shape[-1]
            mask_size = random.randint(5, 18)
            start = random.randint(0, max(0, time_len - mask_size))
            x[:, :, start:start + mask_size] = 0

        # Frequency masking
        if random.random() < 0.35:
            freq_len = x.shape[1]
            mask_size = random.randint(4, 14)
            start = random.randint(0, max(0, freq_len - mask_size))
            x[:, start:start + mask_size, :] = 0

        return x.astype(np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        feature_path = row["feature_path"]

        if not os.path.isabs(feature_path):
            feature_path = os.path.join(PIPELINE_DIR, feature_path)

        x = np.load(feature_path).astype(np.float32)

        if self.augment:
            x = self.augment_feature(x)

        y = self.label_encoder.transform([row["emotion"]])[0]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# =========================
# MODEL
# =========================

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv_block(x)
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        x shape: (batch, time, hidden_dim)
        """
        scores = self.attention(x)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * x, dim=1)
        return context


class CNNBiLSTMAttentionSER(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.cnn = nn.Sequential(
            ResBlock(3, 32, stride=1),
            nn.MaxPool2d(kernel_size=2),

            ResBlock(32, 64, stride=1),
            nn.MaxPool2d(kernel_size=2),

            ResBlock(64, 128, stride=1),
            nn.MaxPool2d(kernel_size=2),

            ResBlock(128, 256, stride=1),
            nn.MaxPool2d(kernel_size=2)
        )

        self.dropout_cnn = nn.Dropout(0.30)

        # Input feature shape from preprocess.py is normally:
        # (3, 128, 188)
        # After four MaxPool2d(2): frequency 128 -> 8, time 188 -> 11
        # CNN output channels = 256
        # LSTM input size = 256 * 8 = 2048
        self.lstm_input_size = 256 * 8

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.30
        )

        self.attention = AttentionPooling(hidden_dim=512)

        self.classifier = nn.Sequential(
            nn.LayerNorm(512),
            nn.Dropout(0.40),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """
        x shape: (batch, 3, 128, time)
        """
        x = self.cnn(x)
        x = self.dropout_cnn(x)

        # x shape: (batch, channels, freq, time)
        batch, channels, freq, time = x.shape

        # Convert to sequence for LSTM:
        # (batch, time, channels * freq)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(batch, time, channels * freq)

        lstm_out, _ = self.lstm(x)

        pooled = self.attention(lstm_out)

        logits = self.classifier(pooled)

        return logits


# =========================
# SPLIT STRATEGY
# =========================

def build_training_data(df):
    df = df.copy()

    df["speaker_id"] = df["speaker_id"].astype(str).str.lower()
    df["emotion"] = df["emotion"].astype(str).str.lower()

    required_columns = ["feature_path", "emotion", "speaker_id"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(
                f"metadata.csv is missing required column: {col}. "
                "Run the new feature-based preprocess.py first."
            )

    base_train_df = df[df["speaker_id"] == BASE_TRAIN_SPEAKER].reset_index(drop=True)
    target_df = df[df["speaker_id"] == TARGET_SPEAKER].reset_index(drop=True)

    if len(base_train_df) == 0:
        raise ValueError(f"No samples found for base train speaker: {BASE_TRAIN_SPEAKER}")

    if len(target_df) == 0:
        raise ValueError(f"No samples found for target speaker: {TARGET_SPEAKER}")

    adapt_parts = []
    remaining_parts = []

    for emotion in CLASS_NAMES:
        emotion_df = target_df[target_df["emotion"] == emotion].reset_index(drop=True)

        if len(emotion_df) == 0:
            raise ValueError(f"No target-speaker samples found for emotion: {emotion}")

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
    print(train_df["emotion"].value_counts().reindex(CLASS_NAMES, fill_value=0))

    print("\nValidation class distribution:")
    print(val_df["emotion"].value_counts().reindex(CLASS_NAMES, fill_value=0))

    print("\nTest class distribution:")
    print(test_df["emotion"].value_counts().reindex(CLASS_NAMES, fill_value=0))

    return train_df, val_df, test_df


# =========================
# BALANCING
# =========================

def create_weighted_sampler(train_df, label_encoder):
    labels = label_encoder.transform(train_df["emotion"])
    class_counts = np.bincount(labels, minlength=len(label_encoder.classes_))

    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[labels]

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler


def create_class_weights(train_df, label_encoder, device):
    labels = label_encoder.transform(train_df["emotion"])
    class_counts = np.bincount(labels, minlength=len(label_encoder.classes_))

    total = class_counts.sum()
    weights = total / (len(class_counts) * np.maximum(class_counts, 1))

    weights = torch.tensor(weights, dtype=torch.float32).to(device)

    return weights


# =========================
# TRAIN / EVALUATE
# =========================

def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    predictions = []
    actuals = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            predictions.extend(preds.cpu().numpy())
            actuals.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(len(loader), 1)

    accuracy = accuracy_score(actuals, predictions) * 100
    uar = recall_score(actuals, predictions, average="macro", zero_division=0) * 100
    macro_f1 = f1_score(actuals, predictions, average="macro", zero_division=0) * 100

    return avg_loss, accuracy, uar, macro_f1, actuals, predictions


def save_training_curve(history):
    plt.figure(figsize=(9, 6))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.plot(history["val_macro_f1"], label="Validation Macro F1")
    plt.title("Speech Training Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "training_curve.png"))
    plt.close()


def save_confusion_matrix(cm):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"))
    plt.close()


# =========================
# MAIN
# =========================

def main():
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            "metadata.csv not found. Run preprocess.py first."
        )

    df = pd.read_csv(METADATA_PATH)

    if "feature_path" not in df.columns:
        raise ValueError(
            "metadata.csv does not contain feature_path. "
            "You are using the old metadata-only preprocess.py. "
            "Run the new CNN/BiLSTM feature-based preprocess.py first."
        )

    label_encoder = LabelEncoder()
    label_encoder.fit(CLASS_NAMES)

    train_df, val_df, test_df = build_training_data(df)

    train_dataset = SpeechFeatureDataset(train_df, label_encoder, augment=True)
    val_dataset = SpeechFeatureDataset(val_df, label_encoder, augment=False)
    test_dataset = SpeechFeatureDataset(test_df, label_encoder, augment=False)

    if USE_WEIGHTED_SAMPLER:
        train_sampler = create_weighted_sampler(train_df, label_encoder)
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=train_sampler,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nUsing Device: {device}")
    print("\nStarting CNN + BiLSTM + Attention Speech Emotion Training...\n")

    model = CNNBiLSTMAttentionSER(num_classes=len(CLASS_NAMES)).to(device)

    if USE_CLASS_WEIGHTS:
        class_weights = create_class_weights(train_df, label_encoder, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.03)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.03)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_uar": [],
        "val_macro_f1": []
    }

    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0.0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS}"
        )

        for features, labels in progress_bar:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(features)
            loss = criterion(outputs, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()

            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(len(train_loader), 1)

        val_loss, val_acc, val_uar, val_f1, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step(val_f1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_uar"].append(val_uar)
        history["val_macro_f1"].append(val_f1)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}% | "
            f"Val UAR: {val_uar:.2f}% | "
            f"Val Macro F1: {val_f1:.2f}%"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            patience_counter = 0

            torch.save(
                model.state_dict(),
                os.path.join(MODELS_DIR, "best_model.pth")
            )

            model_config = {
                "dataset": "TESS (Toronto Emotional Speech Set)",
                "model_name": MODEL_NAME,
                "architecture": "Mel Spectrogram + Delta + MFCC + Residual CNN + BiLSTM + Attention + Classification Head",
                "classes": CLASS_NAMES,
                "input_feature": "3-channel feature tensor",
                "feature_channels": ["mel_spectrogram_db", "delta_mel", "mfcc_expanded"],
                "base_train_speaker": BASE_TRAIN_SPEAKER,
                "target_speaker": TARGET_SPEAKER,
                "adapt_ratio": ADAPT_RATIO,
                "batch_size": BATCH_SIZE,
                "learning_rate": LR,
                "seed": SEED
            }

            with open(os.path.join(MODELS_DIR, "model_config.json"), "w") as f:
                json.dump(model_config, f, indent=4)

            print("Best model saved.")

        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")

            if patience_counter >= PATIENCE:
                print("\nEarly stopping triggered.")
                break

    print(
        f"\nTraining Complete. "
        f"Best Epoch: {best_epoch}, "
        f"Best Validation Macro F1: {best_val_f1:.2f}%"
    )

    best_model_path = os.path.join(MODELS_DIR, "best_model.pth")

    model.load_state_dict(
        torch.load(
            best_model_path,
            map_location=device,
            weights_only=True
        )
    )

    test_loss, test_acc, test_uar, test_f1, test_actuals, test_predictions = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    print("\nFinal Test Results")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test UAR: {test_uar:.2f}%")
    print(f"Test Macro F1: {test_f1:.2f}%")

    report_txt = classification_report(
        test_actuals,
        test_predictions,
        target_names=CLASS_NAMES,
        zero_division=0
    )

    print("\nClassification Report:\n")
    print(report_txt)

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
        "model_name": MODEL_NAME,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_val_f1,
        "split_strategy": "speaker_aware_adaptation",
        "base_train_speaker": BASE_TRAIN_SPEAKER,
        "target_speaker": TARGET_SPEAKER,
        "adapt_ratio": ADAPT_RATIO
    }])

    summary_df.to_csv(
        os.path.join(RESULTS_DIR, "summary.csv"),
        index=False
    )

    speech_metrics = {
        "dataset": "TESS (Toronto Emotional Speech Set)",
        "model_name": MODEL_NAME,
        "architecture": "Mel Spectrogram + Delta + MFCC + Residual CNN + BiLSTM + Attention + Classification Head",
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_val_f1,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_uar": test_uar,
        "test_macro_f1": test_f1,
        "classes": CLASS_NAMES,
        "split_strategy": "speaker_aware_adaptation",
        "base_train_speaker": BASE_TRAIN_SPEAKER,
        "target_speaker": TARGET_SPEAKER,
        "adapt_ratio": ADAPT_RATIO,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
        "note": (
            "This is a feature-based non-transformer speech emotion model. "
            "It uses Mel spectrogram, delta, and MFCC features with a CNN, BiLSTM, "
            "and attention classifier. The evaluation uses speaker-aware adaptation, "
            "not strict zero-shot unseen-speaker testing."
        )
    }

    with open(os.path.join(METRICS_DIR, "speech_metrics.json"), "w") as f:
        json.dump(speech_metrics, f, indent=4)

    metrics_df = pd.DataFrame({
        "Epoch": list(range(1, len(history["train_loss"]) + 1)),
        "Train_Loss": history["train_loss"],
        "Val_Loss": history["val_loss"],
        "Val_Accuracy": history["val_accuracy"],
        "Val_UAR": history["val_uar"],
        "Val_Macro_F1": history["val_macro_f1"]
    })

    metrics_df.to_csv(
        os.path.join(METRICS_DIR, "training_metrics.csv"),
        index=False
    )

    save_confusion_matrix(cm)
    save_training_curve(history)

    print("\nAll Outputs Saved Successfully")
    print(f"Saved model -> {best_model_path}")
    print(f"Results folder -> {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()