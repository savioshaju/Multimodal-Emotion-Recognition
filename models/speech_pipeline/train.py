import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import librosa

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# --------------------------
# 1. Dynamic Unified Dataset
# --------------------------
class UnifiedAudioDataset(Dataset):
    def __init__(self, df, label_encoder, augment=False, max_len=16000*3):
        self.df = df
        self.augment = augment
        self.max_len = max_len
        self.label_encoder = label_encoder
        
        # Audio feature extractors
        self.mel_transform = T.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=256, n_mels=128)
        self.amplitude_to_db = T.AmplitudeToDB()
        self.compute_deltas = T.ComputeDeltas()
        
        # SpecAugment
        self.freq_mask = T.FrequencyMasking(freq_mask_param=30)
        self.time_mask = T.TimeMasking(time_mask_param=60)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = str(row['file_path'])
        
        # Load pre-processed 16kHz audio using librosa to avoid torchcodec dependency
        audio_np, _ = librosa.load(file_path, sr=16000)
        waveform = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0) # (1, Time)
        
        # Ensure fixed length via padding/truncation
        if waveform.shape[1] > self.max_len:
            waveform = waveform[:, :self.max_len]
        else:
            pad_amount = self.max_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

        # Waveform Augmentations
        if self.augment:
            # 1. Random Volume Scaling (fast, no memory explosion)
            if torch.rand(1).item() > 0.5:
                scale = torch.empty(1).uniform_(0.8, 1.2).item()
                waveform = waveform * scale
                
            # 2. Additive Background Noise
            if torch.rand(1).item() > 0.5:
                noise = torch.randn_like(waveform) * 0.005
                waveform = waveform + noise

        # Feature Extraction
        mel = self.mel_transform(waveform)
        mel_db = self.amplitude_to_db(mel)
        
        # Calculate Deltas
        delta1 = self.compute_deltas(mel_db)
        delta2 = self.compute_deltas(delta1)
        
        # Stack channels: (3, 128, Time)
        x = torch.cat([mel_db, delta1, delta2], dim=0)

        # Instance Normalization (Crucial for Speaker Independence)
        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True)
        x = (x - mean) / (std + 1e-8)

        # SpecAugment
        if self.augment:
            x = self.freq_mask(x)
            x = self.time_mask(x)

        # Encode Label
        y = self.label_encoder.transform([row['emotion']])[0]
        
        return x, torch.tensor(y, dtype=torch.long)

# --------------------------
# 2. ResNet-CRNN Architecture
# --------------------------
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Sequential()

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return nn.MaxPool2d(2)(torch.relu(out))

class UnifiedSERModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            ResBlock(3, 32), nn.Dropout2d(0.3),
            ResBlock(32, 64), nn.Dropout2d(0.3),
            ResBlock(64, 128), nn.Dropout2d(0.3)
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 128, 188) # 16000*3 // 256
            out = self.cnn(dummy)
            lstm_input_size = out.shape[1] * out.shape[2]

        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=128, num_layers=2, 
                            batch_first=True, bidirectional=True, dropout=0.3)

        self.attention = nn.Sequential(nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1))
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)
        lstm_out, _ = self.lstm(x)
        
        att_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(att_weights * lstm_out, dim=1)
        return self.classifier(context)

# --------------------------
# 3. Main Training Logic
# --------------------------
def main():
    if not os.path.exists("metadata.csv"):
        print("Error: metadata.csv not found. Please run preprocess.py first.")
        return

    df = pd.read_csv("metadata.csv")
    print(f"Loaded {len(df)} total samples from datasets: {df['dataset'].unique()}")

    encoder = LabelEncoder()
    df['emotion_encoded'] = encoder.fit_transform(df['emotion'])
    
    # Robust Speaker-Independent Split (GroupShuffleSplit ensures no speaker overlaps between train/test)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df['speaker_id']))
    
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    print(f"Training Speakers: {train_df['speaker_id'].nunique()}")
    print(f"Testing Speakers: {test_df['speaker_id'].nunique()} (Zero Overlap Guaranteed)")

    train_dataset = UnifiedAudioDataset(train_df, encoder, augment=True)
    test_dataset = UnifiedAudioDataset(test_df, encoder, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UnifiedSERModel(num_classes=len(encoder.classes_)).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    EPOCHS = 40
    best_uar = 0.0

    os.makedirs("plots", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)

    print("\nStarting Unified Training Pipeline...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        model.eval()
        predictions, actuals = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())
                actuals.extend(labels.cpu().numpy())

        uar = 100 * recall_score(actuals, predictions, average='macro')
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | Test UAR: {uar:.1f}%")

        if uar > best_uar:
            best_uar = uar
            best_predictions = predictions
            best_actuals = actuals
            torch.save(model.state_dict(), "saved_models/best_unified_cnn.pth")

    print(f"\nFinal Best Speaker-Independent UAR: {best_uar:.2f}%")
    
    # Save Report
    report = classification_report(best_actuals, best_predictions, target_names=encoder.classes_, output_dict=True)
    pd.DataFrame(report).transpose().to_csv("results/unified_cnn_report.csv")

if __name__ == "__main__":
    main()