import os
import json
import random
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, recall_score, f1_score, accuracy_score
from transformers import AutoFeatureExtractor, AutoModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METADATA_PATH = os.path.join(BASE_DIR, "metadata.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
METRICS_DIR = os.path.join(BASE_DIR, "metrics")

for d in [PLOTS_DIR, RESULTS_DIR, MODELS_DIR, METRICS_DIR]:
    os.makedirs(d, exist_ok=True)

SR = 16000
DURATION = 3
MAX_LEN = SR * DURATION
BATCH_SIZE = 16
EPOCHS = 30
LR = 2e-5
PATIENCE = 5
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

BASE_TRAIN_SPEAKER = "oaf"
TARGET_SPEAKER = "yaf"
ADAPT_RATIO = 0.14

CLASS_NAMES = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]

MODEL_NAME = "microsoft/wavlm-base"

def preprocess_audio(path):
    audio, _ = librosa.load(path, sr=SR)
    audio, _ = librosa.effects.trim(audio, top_db=30)
    if len(audio) > MAX_LEN:
        audio = audio[:MAX_LEN]
    else:
        audio = np.pad(audio, (0, MAX_LEN - len(audio)))
    audio = librosa.util.normalize(audio)
    return audio.astype(np.float32)

def augment_audio(audio):
    aug = audio.copy()
    if random.random() < 0.50:
        noise = np.random.randn(len(aug)).astype(np.float32)
        aug += np.random.uniform(0.002, 0.008) * noise
    if random.random() < 0.50:
        aug *= np.random.uniform(0.75, 1.25)
    if random.random() < 0.40:
        aug = np.roll(aug, np.random.randint(-1600, 1600))
    if random.random() < 0.35:
        aug = librosa.effects.pitch_shift(y=aug, sr=SR, n_steps=np.random.uniform(-1.5, 1.5))
    if random.random() < 0.35:
        aug = librosa.effects.time_stretch(y=aug, rate=np.random.uniform(0.92, 1.08))
    if len(aug) > MAX_LEN:
        aug = aug[:MAX_LEN]
    else:
        aug = np.pad(aug, (0, MAX_LEN - len(aug)))
    aug = librosa.util.normalize(aug)
    return aug.astype(np.float32)

class TransformerSERDataset(Dataset):
    def __init__(self, df, encoder, feature_extractor, augment=False):
        self.df = df.reset_index(drop=True)
        self.encoder = encoder
        self.augment = augment
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio = preprocess_audio(row["file_path"])
        if self.augment:
            audio = augment_audio(audio)
        
        inputs = self.feature_extractor(
            audio, 
            sampling_rate=SR, 
            return_tensors="pt",
            padding="max_length",
            max_length=MAX_LEN,
            truncation=True
        )
        
        x = inputs["input_values"].squeeze(0)
        y = self.encoder.transform([row["emotion"]])[0]
        return x, torch.tensor(y, dtype=torch.long)

class TransformerSERModel(nn.Module):
    def __init__(self, num_classes, model_name=MODEL_NAME):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        
        hidden_size = self.transformer.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        outputs = self.transformer(x)
        hidden_states = outputs.last_hidden_state
        pooled_output = torch.mean(hidden_states, dim=1)
        logits = self.classifier(pooled_output)
        return logits

def build_training_data(df):
    df["speaker_id"] = df["speaker_id"].str.lower()
    df["emotion"] = df["emotion"].str.lower()

    base_train_df = df[df["speaker_id"] == BASE_TRAIN_SPEAKER].reset_index(drop=True)
    target_df = df[df["speaker_id"] == TARGET_SPEAKER].reset_index(drop=True)

    adapt_parts = []
    remaining_parts = []

    for emotion in sorted(target_df["emotion"].unique()):
        emotion_df = target_df[target_df["emotion"] == emotion].reset_index(drop=True)
        adapt_df, remain_df = train_test_split(emotion_df, train_size=ADAPT_RATIO, random_state=SEED, shuffle=True)
        adapt_parts.append(adapt_df)
        remaining_parts.append(remain_df)

    adapt_df = pd.concat(adapt_parts).reset_index(drop=True)
    remaining_target_df = pd.concat(remaining_parts).reset_index(drop=True)

    val_df, test_df = train_test_split(
        remaining_target_df, test_size=0.70, random_state=SEED, stratify=remaining_target_df["emotion"]
    )

    train_df = pd.concat([base_train_df, adapt_df], axis=0)
    train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_df.to_csv(os.path.join(BASE_DIR, "train_split.csv"), index=False)
    val_df.to_csv(os.path.join(BASE_DIR, "val_split.csv"), index=False)
    test_df.to_csv(os.path.join(BASE_DIR, "test_split.csv"), index=False)

    return train_df, val_df, test_df

def evaluate(model, loader, device):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
            actuals.extend(labels.cpu().numpy())
            
    accuracy = accuracy_score(actuals, predictions) * 100
    uar = recall_score(actuals, predictions, average="macro", zero_division=0) * 100
    f1 = f1_score(actuals, predictions, average="macro", zero_division=0) * 100
    return accuracy, uar, f1, actuals, predictions

def save_curves(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_f1"], label="Validation Macro F1")
    plt.title("Training Loss and Validation Macro F1")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "training_curve.png"))
    plt.close()

def main():
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError("metadata.csv not found. Run preprocess.py first.")

    df = pd.read_csv(METADATA_PATH)
    encoder = LabelEncoder()
    encoder.fit(CLASS_NAMES)

    train_df, val_df, test_df = build_training_data(df)

    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    
    train_dataset = TransformerSERDataset(train_df, encoder, feature_extractor, augment=True)
    val_dataset = TransformerSERDataset(val_df, encoder, feature_extractor, augment=False)
    test_dataset = TransformerSERDataset(test_df, encoder, feature_extractor, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing Device: {device}")

    model = TransformerSERModel(num_classes=len(encoder.classes_)).to(device)

    class_weights = torch.tensor([1.1, 1.0, 1.0, 1.15, 1.0, 1.0, 1.05], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.03)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best_val_f1 = 0.0
    best_epoch = 0
    counter = 0

    history = {"train_loss": [], "val_accuracy": [], "val_uar": [], "val_f1": []}

    print("\nStarting Transformer-based Speech Emotion Recognition Training...\n")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
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

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Accuracy: {val_acc:.2f}% | Val UAR: {val_uar:.2f}% | Val Macro F1: {val_f1:.2f}%")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            counter = 0
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "best_model.pth"))
            
            config_data = {
                "model_name": MODEL_NAME,
                "classes": CLASS_NAMES,
                "sr": SR,
                "duration": DURATION
            }
            with open(os.path.join(MODELS_DIR, "model_config.json"), "w") as f:
                json.dump(config_data, f)
        else:
            counter += 1
            if counter >= PATIENCE:
                print("\nEarly stopping triggered")
                break

    print(f"\nTraining Complete. Best Epoch: {best_epoch}, Best Validation Macro F1: {best_val_f1:.2f}%")

    # Reload the best checkpoint for final evaluation
    model.load_state_dict(
        torch.load(
            os.path.join(MODELS_DIR, "best_model.pth"),
            map_location=device,
            weights_only=True
        )
    )
    test_acc, test_uar, test_f1, test_actuals, test_predictions = evaluate(model, test_loader, device)

    print("\nFinal Test Results")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test UAR: {test_uar:.2f}%")
    print(f"Test Macro F1: {test_f1:.2f}%")

    report = classification_report(test_actuals, test_predictions, target_names=CLASS_NAMES, zero_division=0)
    print("\nClassification Report:\n", report)

    # Save classification report as CSV and plain text
    report_dict = classification_report(
        test_actuals, test_predictions,
        target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    pd.DataFrame(report_dict).transpose().to_csv(
        os.path.join(RESULTS_DIR, "classification_report.csv")
    )

    report_txt = classification_report(
        test_actuals, test_predictions,
        target_names=CLASS_NAMES, zero_division=0
    )
    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
        f.write(report_txt)

    # Save confusion matrix
    cm = confusion_matrix(test_actuals, test_predictions)
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_df.to_csv(os.path.join(RESULTS_DIR, "confusion_matrix.csv"))

    # Save summary CSV for GUI metrics panel
    summary_df = pd.DataFrame([{
        "test_accuracy": test_acc,
        "test_uar": test_uar,
        "test_macro_f1": test_f1,
        "model_name": MODEL_NAME
    }])
    summary_df.to_csv(os.path.join(RESULTS_DIR, "summary.csv"), index=False)

    # Save comprehensive metrics JSON for submission
    import json
    speech_metrics = {
        "dataset": "TESS (Toronto Emotional Speech Set)",
        "model_name": MODEL_NAME,
        "architecture": "WavLM Transformer Encoder + Mean Pooling + Classification Head",
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
        "sr": SR,
        "duration": DURATION,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "seed": SEED,
        "note": (
            "Results are specific to the TESS controlled dataset. "
            "Performance on spontaneous real-world speech is expected to be lower."
        )
    }
    os.makedirs(METRICS_DIR, exist_ok=True)
    with open(os.path.join(METRICS_DIR, "speech_metrics.json"), "w") as f:
        json.dump(speech_metrics, f, indent=4)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Confusion Matrix")
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
    metrics_df.to_csv(os.path.join(METRICS_DIR, "training_metrics.csv"), index=False)
    save_curves(history)
    print("\nAll Outputs Saved Successfully")

if __name__ == "__main__":
    main()