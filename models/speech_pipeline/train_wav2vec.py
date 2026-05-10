import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2Model

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# --------------------------
# 1. Dataset for Raw Audio
# --------------------------
class RawAudioDataset(Dataset):
    def __init__(self, df, label_encoder, processor, max_length=16000*3):
        self.df = df
        self.label_encoder = label_encoder
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = str(row['file_path'])
        
        # Audio is already 16kHz and trimmed by preprocess.py
        audio, _ = librosa.load(file_path, sr=16000)
        
        # Process using HuggingFace processor
        processed = self.processor(
            audio, 
            sampling_rate=16000, 
            max_length=self.max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        
        y = self.label_encoder.transform([row['emotion']])[0]
        return processed.input_values.squeeze(0), torch.tensor(y, dtype=torch.long)

# --------------------------
# 2. Wav2Vec2 SER Architecture
# --------------------------
class Wav2Vec2SER(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.wav2vec2.feature_extractor._freeze_parameters()
        
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        outputs = self.wav2vec2(x)
        pooled_output = torch.mean(outputs.last_hidden_state, dim=1)
        return self.classifier(pooled_output)

# --------------------------
# 3. Main Training Script
# --------------------------
def main():
    if not os.path.exists("metadata.csv"):
        print("Error: metadata.csv not found. Please run preprocess.py first.")
        return

    df = pd.read_csv("metadata.csv")
    print(f"Loaded {len(df)} total samples from datasets: {df['dataset'].unique()}")

    encoder = LabelEncoder()
    df['emotion_encoded'] = encoder.fit_transform(df['emotion'])

    # Robust Speaker-Independent Split across all datasets
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df['speaker_id']))
    
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    print(f"\nTraining Speakers: {train_df['speaker_id'].nunique()}")
    print(f"Testing Speakers: {test_df['speaker_id'].nunique()} (Zero Overlap Guaranteed)")

    print("\nLoading Wav2Vec2 Processor...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

    train_dataset = RawAudioDataset(train_df, encoder, processor)
    test_dataset = RawAudioDataset(test_df, encoder, processor)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=8, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    model = Wav2Vec2SER(num_classes=len(encoder.classes_)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()

    EPOCHS = 15
    best_uar = 0.0

    os.makedirs("plots", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)

    print("\nStarting Unified Wav2Vec2 Fine-Tuning...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

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
        print(f"Epoch {epoch+1} Loss: {running_loss/len(train_loader):.4f} | Test UAR: {uar:.1f}%")

        if uar > best_uar:
            best_uar = uar
            best_predictions = predictions
            best_actuals = actuals
            torch.save(model.state_dict(), "saved_models/best_unified_wav2vec2.pth")

    print(f"\nFinal Best Speaker-Independent UAR: {best_uar:.2f}%")
    
    report = classification_report(best_actuals, best_predictions, target_names=encoder.classes_, output_dict=True)
    pd.DataFrame(report).transpose().to_csv("results/unified_wav2vec2_report.csv")

if __name__ == "__main__":
    main()
