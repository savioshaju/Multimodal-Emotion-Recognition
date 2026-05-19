import os
import sys
import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import customtkinter as ctk
import torch
import torch.nn as nn
import librosa
import numpy as np
import pandas as pd
from PIL import Image

from transformers import AutoModel, AutoTokenizer

try:
    import sounddevice as sd
except Exception:
    sd = None

# UI configuration

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

EMOTION_COLORS = {
    "anger": "#ef5350",
    "disgust": "#66bb6a",
    "fear": "#ab47bc",
    "happiness": "#ffee58",
    "neutral": "#90a4ae",
    "sadness": "#42a5f5",
    "surprise": "#ffa726",
    "uncertain": "#bdbdbd",
}

EMOTION_EMOJIS = {
    "anger": "😡",
    "disgust": "🤢",
    "fear": "😨",
    "happiness": "😄",
    "neutral": "😐",
    "sadness": "😢",
    "surprise": "😲",
    "uncertain": "❓",
}

DEFAULT_CLASS_NAMES = [
    "anger",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise",
]

DEFAULT_SR = 16000
DEFAULT_DURATION = 3
DEFAULT_MAX_TEXT_LEN = 32
DEFAULT_TEXT_MODEL = "distilbert-base-uncased"

N_MELS = 128
N_MFCC = 40

# Extract acoustic features

def normalize_feature(x):
    return (x - x.mean()) / (x.std() + 1e-8)

def preprocess_audio_array(audio, sr=DEFAULT_SR, max_audio_len=DEFAULT_SR * DEFAULT_DURATION):
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim > 1:
        audio = audio.flatten()

    audio, _ = librosa.effects.trim(audio, top_db=30)

    if len(audio) > max_audio_len:
        audio = audio[:max_audio_len]
    else:
        audio = np.pad(audio, (0, max_audio_len - len(audio)), mode="constant")

    audio = librosa.util.normalize(audio)

    return audio.astype(np.float32)

def extract_audio_features(audio, sr=DEFAULT_SR):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=1024,
        hop_length=256,
        n_mels=N_MELS,
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    delta = librosa.feature.delta(mel_db)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=1024,
        hop_length=256,
    )

    mfcc = librosa.util.fix_length(
        mfcc,
        size=mel_db.shape[1],
        axis=1,
    )

    mfcc = np.repeat(mfcc, 4, axis=0)[:N_MELS]

    mel_db = normalize_feature(mel_db)
    delta = normalize_feature(delta)
    mfcc = normalize_feature(mfcc)

    features = np.stack([mel_db, delta, mfcc], axis=0)

    return features.astype(np.float32)

# Fusion model architecture

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

class LightweightFusionModel(nn.Module):
    def __init__(self, num_classes, model_text_name):
        super().__init__()

        self.speech_branch = SpeechCNNBiLSTMAttention(embedding_dim=256)

        self.text_encoder = AutoModel.from_pretrained(model_text_name)
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



class FusionEmotionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multimodal Fusion Emotion Recognition")
        self.root.geometry("1250x850")
        self.root.minsize(1150, 760)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.abspath(os.path.join(self.base_dir, "..", ".."))

        print("RUNNING APP FROM:", self.base_dir)

        self.output_root = os.path.join(self.project_root, "results", "fusion_pipeline")
        self.models_dir = os.path.join(self.base_dir, "saved_models")

        self.model_path = os.path.join(self.models_dir, "best_model.pth")
        self.config_path = os.path.join(self.models_dir, "model_config.json")

        self.classes = DEFAULT_CLASS_NAMES
        self.sr = DEFAULT_SR
        self.duration = DEFAULT_DURATION
        self.max_audio_len = self.sr * self.duration
        self.max_text_len = DEFAULT_MAX_TEXT_LEN
        self.text_model_name = DEFAULT_TEXT_MODEL

        self.recording = False
        self.audio_data = []
        self.stream = None
        self.current_audio = None
        self.current_audio_path = None

        self.load_config()

        self.tokenizer = None
        self.model = None

        self.build_ui()
        self.load_model()


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        audio_path_arg = sys.argv[1]
        text_arg = " ".join(sys.argv[2:])
        predict_cli(audio_path_arg, text_arg)
    else:
        root = ctk.CTk()
        app = FusionEmotionApp(root)
        root.mainloop()