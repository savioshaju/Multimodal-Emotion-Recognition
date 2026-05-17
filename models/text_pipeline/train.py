import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    recall_score,
    f1_score
)


# =========================
# PATH CONFIGURATION
# =========================

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(PIPELINE_DIR, "..", ".."))

PROCESSED_DIR = os.path.join(PIPELINE_DIR, "processed_data")
SAVED_MODELS_DIR = os.path.join(PIPELINE_DIR, "saved_models")

OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "results", "text_pipeline")
METRICS_DIR = os.path.join(OUTPUT_ROOT, "metrics")
PLOTS_DIR = os.path.join(OUTPUT_ROOT, "plots")

TRAIN_PATH = os.path.join(PROCESSED_DIR, "train.csv")
VAL_PATH = os.path.join(PROCESSED_DIR, "val.csv")
TEST_PATH = os.path.join(PROCESSED_DIR, "test.csv")

MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "text_emotion_model.pth")
CONFIG_PATH = os.path.join(SAVED_MODELS_DIR, "model_config.json")

CLASSIFICATION_REPORT_PATH = os.path.join(METRICS_DIR, "classification_report.txt")
CLASSIFICATION_REPORT_CSV_PATH = os.path.join(METRICS_DIR, "classification_report.csv")
CONFUSION_MATRIX_PATH = os.path.join(METRICS_DIR, "confusion_matrix.csv")
TEXT_METRICS_PATH = os.path.join(METRICS_DIR, "text_metrics.json")
TRAINING_METRICS_PATH = os.path.join(METRICS_DIR, "training_metrics.csv")

TRAINING_CURVE_PATH = os.path.join(PLOTS_DIR, "training_curve.png")
CONFUSION_MATRIX_PLOT_PATH = os.path.join(PLOTS_DIR, "confusion_matrix.png")
CLASS_DISTRIBUTION_PLOT_PATH = os.path.join(PLOTS_DIR, "class_distribution.png")

for d in [SAVED_MODELS_DIR, METRICS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)


# =========================
# TRAINING CONFIGURATION
# =========================

MODEL_NAME = "roberta-base"

MAX_LENGTH = 96
BATCH_SIZE = 32
EPOCHS = 8

LR = 1.2e-5
WEIGHT_DECAY = 0.01
PATIENCE = 2
SEED = 42

WARMUP_RATIO = 0.06
LABEL_SMOOTHING = 0.0
DROPOUT_RATE = 0.30
GRAD_CLIP = 1.0

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


# =========================
# REPRODUCIBILITY
# =========================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


set_seed(SEED)


# =========================
# DATASET CLASS
# =========================

class EncodedTextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "label": self.labels[idx]
        }


# =========================
# MODEL CLASS
# =========================

class TextEmotionModel(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(256, num_classes)
        )

    def mean_pooling(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled = self.mean_pooling(
            outputs.last_hidden_state,
            attention_mask
        )

        logits = self.classifier(pooled)
        return logits


# =========================
# HELPER FUNCTIONS
# =========================

def validate_files():
    for path in [TRAIN_PATH, VAL_PATH, TEST_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Run preprocess.py first.")


def clean_dataframe(df, name):
    required_columns = {"text", "emotion"}

    if not required_columns.issubset(df.columns):
        raise ValueError(f"{name} must contain 'text' and 'emotion' columns.")

    df = df[df["emotion"].isin(CLASS_NAMES)].copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 2].reset_index(drop=True)

    return df


def check_dialogue_leakage(train_df, val_df, test_df):
    if "dialogue_id" not in train_df.columns:
        print("\nWarning: dialogue_id column missing. Cannot verify dialogue-level leakage.")
        return

    train_ids = set(train_df["dialogue_id"].unique())
    val_ids = set(val_df["dialogue_id"].unique())
    test_ids = set(test_df["dialogue_id"].unique())

    if (
        train_ids.intersection(val_ids)
        or train_ids.intersection(test_ids)
        or val_ids.intersection(test_ids)
    ):
        raise ValueError("Dialogue-level leakage detected. Re-run preprocess.py.")

    print("\nLeakage check passed: no dialogue_id overlap.")


def print_distribution(name, df):
    print(f"\n{name} class distribution:")
    print(df["emotion"].value_counts().reindex(CLASS_NAMES, fill_value=0))


def tokenize_dataframe(df, tokenizer):
    texts = df["text"].tolist()

    encodings = tokenizer(
        texts,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    return encodings


def create_class_weights(train_df, device):
    if not USE_CLASS_WEIGHTS:
        weights = np.ones(len(CLASS_NAMES), dtype=np.float32)

        class_weights = torch.tensor(
            weights,
            dtype=torch.float32
        ).to(device)

        print("\nClass weights disabled because WeightedRandomSampler is enabled:")
        for cls, weight in zip(CLASS_NAMES, weights):
            print(f"{cls:10s} | Weight: {weight:.4f}")

        return class_weights

    counts = train_df["emotion"].value_counts().reindex(CLASS_NAMES, fill_value=1)
    total = counts.sum()
    weights = total / (len(CLASS_NAMES) * counts.values)

    class_weights = torch.tensor(
        weights,
        dtype=torch.float32
    ).to(device)

    print("\nClass weights:")
    for cls, count, weight in zip(CLASS_NAMES, counts.values, weights):
        print(f"{cls:10s} | Count: {int(count):6d} | Weight: {weight:.4f}")

    return class_weights


def create_weighted_sampler(train_df, label_encoder):
    labels = label_encoder.transform(train_df["emotion"])

    class_counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    class_counts = np.maximum(class_counts, 1)

    class_sample_weights = 1.0 / class_counts
    sample_weights = class_sample_weights[labels]

    sample_weights = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    print("\nWeightedRandomSampler enabled.")
    print("Sampler class counts:")
    for cls, count in zip(CLASS_NAMES, class_counts):
        print(f"{cls:10s} | Count: {int(count):6d}")

    return sampler


def evaluate(model, dataloader, device, criterion):
    model.eval()

    total_loss = 0.0
    actuals = []
    predictions = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)

            actuals.extend(labels.cpu().numpy())
            predictions.extend(preds.cpu().numpy())

    avg_loss = total_loss / max(1, len(dataloader))

    accuracy = accuracy_score(actuals, predictions) * 100

    uar = recall_score(
        actuals,
        predictions,
        average="macro",
        zero_division=0
    ) * 100

    macro_f1 = f1_score(
        actuals,
        predictions,
        average="macro",
        zero_division=0
    ) * 100

    weighted_f1 = f1_score(
        actuals,
        predictions,
        average="weighted",
        zero_division=0
    ) * 100

    return avg_loss, accuracy, uar, macro_f1, weighted_f1, actuals, predictions


def plot_training(history):
    plt.figure(figsize=(8, 5))

    plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
    plt.plot(history["epoch"], history["val_loss"], label="Validation Loss")
    plt.plot(history["epoch"], history["val_macro_f1"], label="Validation Macro F1")

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Text Model Training Curve")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(TRAINING_CURVE_PATH)
    plt.close()


def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(9, 7))

    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.xticks(range(len(classes)), classes, rotation=45)
    plt.yticks(range(len(classes)), classes)

    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PLOT_PATH)
    plt.close()


def plot_class_distribution(train_df, val_df, test_df):
    distribution_df = pd.DataFrame({
        "train": train_df["emotion"].value_counts().reindex(CLASS_NAMES, fill_value=0),
        "validation": val_df["emotion"].value_counts().reindex(CLASS_NAMES, fill_value=0),
        "test": test_df["emotion"].value_counts().reindex(CLASS_NAMES, fill_value=0)
    })

    distribution_df.plot(kind="bar", figsize=(10, 6))

    plt.title("Text Dataset Class Distribution")
    plt.xlabel("Emotion")
    plt.ylabel("Sample Count")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(CLASS_DISTRIBUTION_PLOT_PATH)
    plt.close()


# =========================
# MAIN FUNCTION
# =========================

def main():
    validate_files()

    train_df = clean_dataframe(pd.read_csv(TRAIN_PATH), "train.csv")
    val_df = clean_dataframe(pd.read_csv(VAL_PATH), "val.csv")
    test_df = clean_dataframe(pd.read_csv(TEST_PATH), "test.csv")

    check_dialogue_leakage(train_df, val_df, test_df)

    label_encoder = LabelEncoder()
    label_encoder.fit(CLASS_NAMES)

    train_labels = torch.tensor(
        label_encoder.transform(train_df["emotion"]),
        dtype=torch.long
    )

    val_labels = torch.tensor(
        label_encoder.transform(val_df["emotion"]),
        dtype=torch.long
    )

    test_labels = torch.tensor(
        label_encoder.transform(test_df["emotion"]),
        dtype=torch.long
    )

    print("\nText emotion classes:")
    print(list(label_encoder.classes_))

    print_distribution("Train", train_df)
    print_distribution("Validation", val_df)
    print_distribution("Test", test_df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("\nTokenizing data once...")
    train_encodings = tokenize_dataframe(train_df, tokenizer)
    val_encodings = tokenize_dataframe(val_df, tokenizer)
    test_encodings = tokenize_dataframe(test_df, tokenizer)

    train_dataset = EncodedTextDataset(train_encodings, train_labels)
    val_dataset = EncodedTextDataset(val_encodings, val_labels)
    test_dataset = EncodedTextDataset(test_encodings, test_labels)

    num_workers = 2 if os.name == "nt" else 4
    pin_memory = torch.cuda.is_available()

    if USE_WEIGHTED_SAMPLER:
        train_sampler = create_weighted_sampler(train_df, label_encoder)

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=train_sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    model = TextEmotionModel(
        MODEL_NAME,
        len(CLASS_NAMES)
    ).to(device)

    class_weights = create_class_weights(train_df, device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=LABEL_SMOOTHING
    )

    optimizer = AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=torch.cuda.is_available()
    )

    config = {
        "dataset": "DailyDialog",
        "model_name": MODEL_NAME,
        "architecture": "RoBERTa Transformer Encoder + Mean Pooling + Classification Head",
        "classes": CLASS_NAMES,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "patience": PATIENCE,
        "seed": SEED,
        "warmup_ratio": WARMUP_RATIO,
        "label_smoothing": LABEL_SMOOTHING,
        "dropout_rate": DROPOUT_RATE,
        "weighted_sampler": USE_WEIGHTED_SAMPLER,
        "class_weights_enabled": USE_CLASS_WEIGHTS,
        "mixed_precision": torch.cuda.is_available(),
        "split_requirement": "dialogue_id_group_split_no_leakage",
        "output_structure": {
            "metrics_dir": METRICS_DIR,
            "plots_dir": PLOTS_DIR,
            "classification_report_txt": CLASSIFICATION_REPORT_PATH,
            "classification_report_csv": CLASSIFICATION_REPORT_CSV_PATH,
            "confusion_matrix_csv": CONFUSION_MATRIX_PATH,
            "text_metrics_json": TEXT_METRICS_PATH,
            "training_metrics_csv": TRAINING_METRICS_PATH,
            "training_curve_png": TRAINING_CURVE_PATH,
            "confusion_matrix_png": CONFUSION_MATRIX_PLOT_PATH,
            "class_distribution_png": CLASS_DISTRIBUTION_PLOT_PATH
        }
    }

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_uar": [],
        "val_macro_f1": [],
        "val_weighted_f1": []
    }

    print("\nStarting optimized text emotion training with sampler-only balancing...\n")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS}"
        )

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                "cuda",
                enabled=torch.cuda.is_available()
            ):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP
            )

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()

            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}"
            })

        train_loss = running_loss / max(1, len(train_loader))

        val_loss, val_acc, val_uar, val_f1, val_weighted_f1, _, _ = evaluate(
            model,
            val_loader,
            device,
            criterion
        )

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_uar"].append(val_uar)
        history["val_macro_f1"].append(val_f1)
        history["val_weighted_f1"].append(val_weighted_f1)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}% | "
            f"Val UAR: {val_uar:.2f}% | "
            f"Val Macro F1: {val_f1:.2f}% | "
            f"Val Weighted F1: {val_weighted_f1:.2f}%"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            patience_counter = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "label_classes": CLASS_NAMES,
                "config": config,
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_val_f1
            }, MODEL_PATH)

            print("Best model saved.")

        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")

            if patience_counter >= PATIENCE:
                print("\nEarly stopping triggered.")
                break

    print("\nTraining complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation macro F1: {best_val_f1:.2f}%")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    test_loss, test_acc, test_uar, test_f1, test_weighted_f1, test_actuals, test_predictions = evaluate(
        model,
        test_loader,
        device,
        criterion
    )

    print("\nFinal test results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test UAR: {test_uar:.2f}%")
    print(f"Test Macro F1: {test_f1:.2f}%")
    print(f"Test Weighted F1: {test_weighted_f1:.2f}%")

    report = classification_report(
        test_actuals,
        test_predictions,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        zero_division=0
    )

    print("\nClassification Report:\n")
    print(report)

    with open(CLASSIFICATION_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    report_dict = classification_report(
        test_actuals,
        test_predictions,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report_dict).transpose().to_csv(
        CLASSIFICATION_REPORT_CSV_PATH
    )

    cm = confusion_matrix(
        test_actuals,
        test_predictions,
        labels=list(range(len(CLASS_NAMES)))
    )

    cm_df = pd.DataFrame(
        cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES
    )

    cm_df.to_csv(CONFUSION_MATRIX_PATH)

    metrics = {
        "dataset": "DailyDialog",
        "model_name": MODEL_NAME,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_val_f1,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_uar": test_uar,
        "test_macro_f1": test_f1,
        "test_weighted_f1": test_weighted_f1,
        "classes": CLASS_NAMES,
        "train_samples": int(len(train_df)),
        "val_samples": int(len(val_df)),
        "test_samples": int(len(test_df)),
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "label_smoothing": LABEL_SMOOTHING,
        "dropout_rate": DROPOUT_RATE,
        "weighted_sampler": USE_WEIGHTED_SAMPLER,
        "class_weights_enabled": USE_CLASS_WEIGHTS,
        "mixed_precision": torch.cuda.is_available(),
        "leakage_check": "dialogue_id checked"
    }

    with open(TEXT_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    pd.DataFrame(history).to_csv(
        TRAINING_METRICS_PATH,
        index=False
    )

    plot_training(history)
    plot_confusion_matrix(cm, CLASS_NAMES)
    plot_class_distribution(train_df, val_df, test_df)

    print("\nSaved outputs:")
    print(MODEL_PATH)
    print(CONFIG_PATH)
    print(CLASSIFICATION_REPORT_PATH)
    print(CLASSIFICATION_REPORT_CSV_PATH)
    print(CONFUSION_MATRIX_PATH)
    print(TEXT_METRICS_PATH)
    print(TRAINING_METRICS_PATH)
    print(TRAINING_CURVE_PATH)
    print(CONFUSION_MATRIX_PLOT_PATH)
    print(CLASS_DISTRIBUTION_PLOT_PATH)

    print("\nAll outputs saved successfully.")


if __name__ == "__main__":
    main()