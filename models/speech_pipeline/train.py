import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

import torchaudio.transforms as T

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    recall_score,
    f1_score,
    accuracy_score
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PLOTS_DIR = os.path.join(BASE_DIR, "plots")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
METRICS_DIR = os.path.join(BASE_DIR, "metrics")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 40
LR = 0.0005

class SERDataset(Dataset):

    def __init__(self, df, encoder, augment=False):

        self.df = df.reset_index(drop=True)

        self.encoder = encoder

        self.augment = augment

        self.freq_mask = T.FrequencyMasking(
            freq_mask_param=20
        )

        self.time_mask = T.TimeMasking(
            time_mask_param=40
        )

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        x = np.load(row["feature_path"])

        x = torch.tensor(
            x,
            dtype=torch.float32
        )

        if self.augment:

            x = self.freq_mask(x)

            x = self.time_mask(x)

        y = self.encoder.transform(
            [row["emotion"]]
        )[0]

        return x, torch.tensor(
            y,
            dtype=torch.long
        )

class ResBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.bn1 = nn.BatchNorm2d(
            out_channels
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(
            out_channels
        )

        self.shortcut = nn.Sequential()

        if in_channels != out_channels:

            self.shortcut = nn.Sequential(

                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1
                ),

                nn.BatchNorm2d(
                    out_channels
                )
            )

    def forward(self, x):

        identity = self.shortcut(x)

        out = torch.relu(
            self.bn1(
                self.conv1(x)
            )
        )

        out = self.bn2(
            self.conv2(out)
        )

        out += identity

        out = torch.relu(out)

        out = nn.MaxPool2d(2)(out)

        return out

class UnifiedSERModel(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.cnn = nn.Sequential(

            ResBlock(3, 32),
            nn.Dropout2d(0.3),

            ResBlock(32, 64),
            nn.Dropout2d(0.3),

            ResBlock(64, 128),
            nn.Dropout2d(0.3)
        )

        with torch.no_grad():

            dummy = torch.zeros(
                1,
                3,
                128,
                188
            )

            out = self.cnn(dummy)

            _, c, f, t = out.shape

            lstm_input_size = c * f

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.attention = nn.Sequential(

            nn.Linear(256, 128),

            nn.Tanh(),

            nn.Linear(128, 1)
        )

        self.classifier = nn.Sequential(

            nn.Linear(256, 128),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(
                128,
                num_classes
            )
        )

    def forward(self, x):

        x = self.cnn(x)

        b, c, f, t = x.shape

        x = x.permute(
            0,
            3,
            1,
            2
        )

        x = x.reshape(
            b,
            t,
            c * f
        )

        lstm_out, _ = self.lstm(x)

        attention_weights = torch.softmax(

            self.attention(lstm_out),

            dim=1
        )

        context = torch.sum(

            attention_weights * lstm_out,

            dim=1
        )

        output = self.classifier(context)

        return output

def main():

    metadata_path = os.path.join(
        BASE_DIR,
        "metadata.csv"
    )

    if not os.path.exists(metadata_path):

        print(
            "metadata.csv not found"
        )

        return

    df = pd.read_csv(
        metadata_path
    )

    encoder = LabelEncoder()

    encoder.fit(
        df["emotion"]
    )

    print("\nEmotion Classes:")

    print(
        encoder.classes_
    )

    gss = GroupShuffleSplit(

        n_splits=1,

        test_size=0.5,

        random_state=42
    )

    train_idx, test_idx = next(

        gss.split(

            df,

            groups=df["speaker_id"]
        )
    )

    train_df = df.iloc[
        train_idx
    ]

    test_df = df.iloc[
        test_idx
    ]

    print("\nTraining Samples:")

    print(len(train_df))

    print("\nTesting Samples:")

    print(len(test_df))

    train_dataset = SERDataset(

        train_df,

        encoder,

        augment=True
    )

    test_dataset = SERDataset(

        test_df,

        encoder,

        augment=False
    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True
    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE
    )

    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"
    )

    print("\nUsing Device:")

    print(device)

    model = UnifiedSERModel(

        num_classes=len(
            encoder.classes_
        )

    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(

        model.parameters(),

        lr=LR,

        weight_decay=1e-2
    )

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(

        optimizer,

        T_0=10,

        T_mult=2
    )

    best_uar = 0.0

    history = {

        "loss": [],

        "uar": [],

        "f1": [],

        "accuracy": []
    }

    print("\nStarting Training...\n")

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0

        for inputs, labels in tqdm(train_loader):

            inputs = inputs.to(device)

            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                2.0
            )

            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        model.eval()

        predictions = []

        actuals = []

        with torch.no_grad():

            for inputs, labels in test_loader:

                inputs = inputs.to(device)

                labels = labels.to(device)

                outputs = model(inputs)

                _, predicted = torch.max(
                    outputs,
                    1
                )

                predictions.extend(
                    predicted.cpu().numpy()
                )

                actuals.extend(
                    labels.cpu().numpy()
                )

        accuracy = accuracy_score(
            actuals,
            predictions
        ) * 100

        uar = recall_score(

            actuals,

            predictions,

            average="macro"
        ) * 100

        f1 = f1_score(

            actuals,

            predictions,

            average="macro"
        ) * 100

        epoch_loss = (
            running_loss / len(train_loader)
        )

        history["loss"].append(
            epoch_loss
        )

        history["uar"].append(
            uar
        )

        history["f1"].append(
            f1
        )

        history["accuracy"].append(
            accuracy
        )

        print(

            f"Epoch {epoch+1}/{EPOCHS} | "

            f"Loss: {epoch_loss:.4f} | "

            f"Accuracy: {accuracy:.2f}% | "

            f"UAR: {uar:.2f}% | "

            f"F1: {f1:.2f}%"
        )

        with open(

            os.path.join(
                LOGS_DIR,
                "training_log.txt"
            ),

            "a"
        ) as f:

            f.write(

                f"Epoch {epoch+1}/{EPOCHS} | "

                f"Loss: {epoch_loss:.4f} | "

                f"Accuracy: {accuracy:.2f}% | "

                f"UAR: {uar:.2f}% | "

                f"F1: {f1:.2f}%\n"
            )

        if uar > best_uar:

            best_uar = uar

            best_predictions = predictions

            best_actuals = actuals

            torch.save(

                model.state_dict(),

                os.path.join(

                    MODELS_DIR,

                    "best_model.pth"
                )
            )

    print("\nTraining Complete")

    print(

        f"\nBest Speaker Independent UAR: {best_uar:.2f}%"
    )

    report = classification_report(

        best_actuals,

        best_predictions,

        target_names=encoder.classes_,

        output_dict=True
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    report_df.to_csv(

        os.path.join(

            RESULTS_DIR,

            "classification_report.csv"
        )
    )

    cm = confusion_matrix(
        best_actuals,
        best_predictions
    )

    plt.figure(figsize=(10, 8))

    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        cmap="Blues",

        xticklabels=encoder.classes_,

        yticklabels=encoder.classes_
    )

    plt.title(
        "Confusion Matrix"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            PLOTS_DIR,

            "confusion_matrix.png"
        )
    )

    plt.close()

    plt.figure(figsize=(8, 5))

    plt.plot(
        history["loss"]
    )

    plt.title(
        "Training Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.grid()

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            PLOTS_DIR,

            "loss_curve.png"
        )
    )

    plt.close()

    plt.figure(figsize=(8, 5))

    plt.plot(
        history["uar"]
    )

    plt.title(
        "UAR Curve"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "UAR"
    )

    plt.grid()

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            PLOTS_DIR,

            "uar_curve.png"
        )
    )

    plt.close()

    plt.figure(figsize=(8, 5))

    plt.plot(
        history["f1"]
    )

    plt.title(
        "F1 Curve"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "F1"
    )

    plt.grid()

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            PLOTS_DIR,

            "f1_curve.png"
        )
    )

    plt.close()

    metrics_df = pd.DataFrame({

        "Epoch": list(
            range(1, EPOCHS + 1)
        ),

        "Loss": history["loss"],

        "Accuracy": history["accuracy"],

        "UAR": history["uar"],

        "F1": history["f1"]
    })

    metrics_df.to_csv(

        os.path.join(

            METRICS_DIR,

            "training_metrics.csv"
        ),

        index=False
    )

    print("\nAll outputs saved successfully.")

if __name__ == "__main__":

    main()