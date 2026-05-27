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
        self.root.title("Multimodal Fusion Emotion Recognition (MELD)")
        self.root.geometry("1250x850")
        self.root.minsize(1150, 760)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.abspath(os.path.join(self.base_dir, "..", ".."))

        print("RUNNING APP FROM:", self.base_dir)

        self.output_root = os.path.join(self.project_root, "results", "fusion_pipeline_MELD")
        self.models_dir = os.path.join(self.base_dir, "saved_models")

        self.model_path = os.path.join(self.models_dir, "best_model.pth")
        self.config_path = os.path.join(self.models_dir, "model_config.json")

        self.report_txt = os.path.join(self.output_root, "results", "classification_report.txt")
        self.cm_csv = os.path.join(self.output_root, "results", "confusion_matrix.csv")
        self.metrics_json = os.path.join(self.output_root, "metrics", "fusion_metrics.json")
        self.plot_cm = os.path.join(self.output_root, "plots", "meld_confusion_matrix.png")
        self.plot_curve = os.path.join(self.output_root, "plots", "training_curve.png")

        self.classes = DEFAULT_CLASS_NAMES
        self.sr = DEFAULT_SR
        self.duration = DEFAULT_DURATION
        self.max_audio_len = self.sr * self.duration
        self.max_text_len = DEFAULT_MAX_TEXT_LEN
        self.text_model_name = DEFAULT_TEXT_MODEL

        self.current_audio = None
        self.current_audio_path = None

        self.tokenizer = None
        self.model = None

        self.load_config()
        self.build_ui()
        self.load_model()

    def load_config(self):
        if not os.path.exists(self.config_path):
            print("Config file not found. Using default config.")
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            self.classes = config.get("classes", DEFAULT_CLASS_NAMES)
            self.max_text_len = int(config.get("max_text_len", DEFAULT_MAX_TEXT_LEN))

            text_branch = config.get("text_branch", DEFAULT_TEXT_MODEL)

            if isinstance(text_branch, str) and "distilbert" in text_branch.lower():
                self.text_model_name = text_branch
            else:
                self.text_model_name = DEFAULT_TEXT_MODEL

            print("Fusion config loaded successfully.")
            print("Classes:", self.classes)
            print("Max text length:", self.max_text_len)
            print("Text model:", self.text_model_name)

        except Exception as e:
            print("Failed to load config. Using default config.")
            print(e)

    def build_ui(self):
        self.header_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        self.header_frame.pack(fill="x", pady=(25, 15), padx=35)

        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text="Multimodal Fusion Emotion Recognition (MELD)",
            font=ctk.CTkFont(size=30, weight="bold")
        )
        self.title_label.pack(side="left")

        self.status_badge = ctk.CTkFrame(
            self.header_frame,
            corner_radius=15,
            fg_color="#1b5e20"
        )
        self.status_badge.pack(side="right", pady=5)

        self.status_label = ctk.CTkLabel(
            self.status_badge,
            text="Loading...",
            text_color="#69f0ae",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.status_label.pack(padx=15, pady=5)

        self.main_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        self.main_frame.pack(fill="both", expand=True, padx=35, pady=(0, 25))

        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(0, weight=1)

        self.input_card = ctk.CTkFrame(
            self.main_frame,
            corner_radius=20,
            fg_color="#212121"
        )
        self.input_card.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        self.input_title = ctk.CTkLabel(
            self.input_card,
            text="Fusion Input",
            font=ctk.CTkFont(size=22, weight="bold")
        )
        self.input_title.pack(pady=(30, 15))

        self.audio_label = ctk.CTkLabel(
            self.input_card,
            text="No audio selected",
            font=ctk.CTkFont(size=14),
            text_color="gray75"
        )
        self.audio_label.pack(padx=30, pady=(0, 10), fill="x")

        self.upload_audio_btn = ctk.CTkButton(
            self.input_card,
            text="Select Audio File",
            height=45,
            corner_radius=10,
            command=self.upload_audio
        )
        self.upload_audio_btn.pack(padx=30, pady=8, fill="x")

        self.text_label = ctk.CTkLabel(
            self.input_card,
            text="Input Text / Transcript",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.text_label.pack(padx=30, pady=(25, 8), anchor="w")

        self.textbox = ctk.CTkTextbox(
            self.input_card,
            height=120,
            corner_radius=10,
            font=ctk.CTkFont(size=14)
        )
        self.textbox.pack(padx=30, pady=(0, 15), fill="x")

        self.predict_btn = ctk.CTkButton(
            self.input_card,
            text="Predict Fusion Emotion",
            height=50,
            corner_radius=10,
            font=ctk.CTkFont(size=16, weight="bold"),
            command=self.predict_emotion
        )
        self.predict_btn.pack(padx=30, pady=10, fill="x")

        self.clear_btn = ctk.CTkButton(
            self.input_card,
            text="Clear",
            height=40,
            corner_radius=10,
            fg_color="#424242",
            hover_color="#616161",
            command=self.clear_input
        )
        self.clear_btn.pack(padx=30, pady=8, fill="x")

        self.reports_title = ctk.CTkLabel(
            self.input_card,
            text="Reports & Metrics",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.reports_title.pack(pady=(25, 10))

        self.reports_grid = ctk.CTkFrame(self.input_card, fg_color="transparent")
        self.reports_grid.pack(padx=30, pady=(0, 20), fill="x")

        ctk.CTkButton(
            self.reports_grid,
            text="Classification Report",
            height=35,
            command=self.show_classification_report
        ).grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        ctk.CTkButton(
            self.reports_grid,
            text="Confusion Matrix",
            height=35,
            command=self.show_confusion_matrix
        ).grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        ctk.CTkButton(
            self.reports_grid,
            text="Metrics Summary",
            height=35,
            command=self.show_metrics_summary
        ).grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        ctk.CTkButton(
            self.reports_grid,
            text="View Plots",
            height=35,
            command=self.show_plots
        ).grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        self.reports_grid.columnconfigure(0, weight=1)
        self.reports_grid.columnconfigure(1, weight=1)

        self.result_card = ctk.CTkFrame(
            self.main_frame,
            corner_radius=20,
            fg_color="#263238"
        )
        self.result_card.grid(row=0, column=1, sticky="nsew", padx=(12, 0))

        self.result_emoji = ctk.CTkLabel(
            self.result_card,
            text="🎧",
            font=ctk.CTkFont(size=120)
        )
        self.result_emoji.pack(pady=(60, 20), expand=True)

        self.result_text = ctk.CTkLabel(
            self.result_card,
            text="Ready for Fusion Input",
            font=ctk.CTkFont(size=36, weight="bold")
        )
        self.result_text.pack(pady=(0, 10))

        self.confidence_text = ctk.CTkLabel(
            self.result_card,
            text="Confidence: --",
            font=ctk.CTkFont(size=18),
            text_color="gray70",
            justify="center"
        )
        self.confidence_text.pack(pady=(0, 20))

        self.progress = ctk.CTkProgressBar(
            self.result_card,
            width=400,
            height=15,
            progress_color="#00e676",
            fg_color="#37474f"
        )
        self.progress.pack(pady=(0, 60))
        self.progress.set(0)

    def set_status(self, text, badge_color="#1b5e20", text_color="#69f0ae"):
        self.status_label.configure(text=text, text_color=text_color)
        self.status_badge.configure(fg_color=badge_color)

    def load_model(self):
        if not os.path.exists(self.model_path):
            messagebox.showerror(
                "Model Missing",
                f"Model file not found:\n{self.model_path}\n\nRun train.py first."
            )
            self.set_status("Model Missing", badge_color="#b71c1c", text_color="#ff5252")
            self.predict_btn.configure(state="disabled")
            return

        try:
            if os.path.exists(os.path.join(self.models_dir, "tokenizer_config.json")):
                self.tokenizer = AutoTokenizer.from_pretrained(self.models_dir)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)

            self.model = LightweightFusionModel(
                num_classes=len(self.classes),
                model_text_name=self.text_model_name
            )

            checkpoint = torch.load(
                self.model_path,
                map_location=self.device
            )

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)

            self.model.to(self.device)
            self.model.eval()

            self.set_status(f"Model Loaded: {self.device}")
            self.predict_btn.configure(state="normal")

            print("Fusion model loaded successfully.")
            print("Device:", self.device)

        except Exception as e:
            messagebox.showerror(
                "Model Error",
                f"Failed to load fusion model:\n{e}"
            )
            self.set_status("Model Error", badge_color="#b71c1c", text_color="#ff5252")
            self.predict_btn.configure(state="disabled")

    def upload_audio(self):
        path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[
                ("Audio Files", "*.wav *.mp3 *.flac *.ogg"),
                ("WAV Files", "*.wav"),
                ("All Files", "*.*")
            ]
        )

        if not path:
            return

        try:
            audio, sr = librosa.load(path, sr=self.sr)
            audio = preprocess_audio_array(audio, sr=self.sr, max_audio_len=self.max_audio_len)

            self.current_audio = audio
            self.current_audio_path = path

            self.audio_label.configure(text=f"Selected: {os.path.basename(path)}")
            self.set_status("Audio Loaded")

        except Exception as e:
            messagebox.showerror("Audio Error", f"Failed to load audio:\n{e}")

    def predict_emotion(self):
        if self.model is None or self.tokenizer is None:
            messagebox.showerror("Model Error", "Model is not loaded.")
            return

        if self.current_audio is None:
            messagebox.showwarning("Input Error", "Please upload an audio file first.")
            return

        text = self.textbox.get("1.0", "end-1c").strip()

        if not text:
            messagebox.showwarning("Input Error", "Please enter transcript/text.")
            return

        try:
            self.set_status("Processing...", badge_color="#f57f17", text_color="#ffee58")
            self.root.update()

            features = extract_audio_features(self.current_audio, sr=self.sr)
            speech_features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

            encoded = self.tokenizer(
                text,
                max_length=self.max_text_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            with torch.no_grad():
                logits = self.model(speech_features, input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)

            top_probs, top_indices = torch.topk(probs, k=3, dim=1)

            top3_results = []

            for i in range(3):
                emotion = self.classes[top_indices[0][i].item()]
                score = top_probs[0][i].item() * 100
                top3_results.append((emotion, score))

            prediction = top3_results[0][0]
            confidence = top3_results[0][1]

            second_confidence = top3_results[1][1]
            confidence_gap = confidence - second_confidence

            is_low_confidence = confidence < 60 or confidence_gap < 10

            self.update_ui(prediction, confidence, top3_results, is_low_confidence)

        except Exception as e:
            messagebox.showerror("Prediction Error", f"Failed to predict:\n{e}")
            self.set_status("Prediction Error", badge_color="#b71c1c", text_color="#ff5252")

    def update_ui(self, emotion, confidence, top3_results=None, is_low_confidence=False):
        color = EMOTION_COLORS.get(emotion, "#00e676")
        emoji = EMOTION_EMOJIS.get(emotion, "🎧")

        self.result_emoji.configure(text=emoji)
        self.result_text.configure(text=emotion.upper(), text_color=color)

        top_text = " | ".join(
            [f"{emo}: {score:.2f}%" for emo, score in (top3_results or [])]
        )

        if is_low_confidence:
            self.confidence_text.configure(
                text=f"Confidence: {confidence:.2f}%\nStatus: Low confidence\nTop predictions: {top_text}"
            )
        else:
            self.confidence_text.configure(
                text=f"Confidence: {confidence:.2f}%\nTop predictions: {top_text}"
            )

        self.progress.configure(progress_color=color)
        self.progress.set(confidence / 100)

        self.set_status("Prediction Complete")

    def clear_input(self):
        self.textbox.delete("1.0", "end")
        self.current_audio = None
        self.current_audio_path = None
        self.audio_label.configure(text="No audio selected")

        self.result_emoji.configure(text="🎧")
        self.result_text.configure(text="Ready for Fusion Input", text_color="white")
        self.confidence_text.configure(text="Confidence: --")
        self.progress.configure(progress_color="#00e676")
        self.progress.set(0)

        self.set_status("Model Loaded")

    def safe_value_to_string(self, value):
        if value is None:
            return ""

        if isinstance(value, (list, tuple)):
            return ", ".join(map(str, value))

        if isinstance(value, dict):
            return json.dumps(value, indent=2)

        try:
            if pd.isna(value):
                return ""
        except Exception:
            pass

        try:
            float_val = float(value)

            if float_val.is_integer():
                return str(int(float_val))

            return f"{float_val:.4f}"

        except Exception:
            return str(value)

    def create_table_popup(self, title, df, table_type):
        top = ctk.CTkToplevel(self.root)
        top.title(title)
        top.geometry("950x620")
        top.transient(self.root)

        title_label = ctk.CTkLabel(
            top,
            text=title,
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(pady=(20, 10))

        table_frame = ctk.CTkFrame(top, corner_radius=10)
        table_frame.pack(fill="both", expand=True, padx=30, pady=(10, 20))

        style = ttk.Style(top)
        style.theme_use("default")
        style.configure(
            "Treeview",
            background="#2b2b2b",
            foreground="white",
            rowheight=35,
            fieldbackground="#2b2b2b",
            font=("Helvetica", 12)
        )
        style.map("Treeview", background=[("selected", "#1f538d")])
        style.configure(
            "Treeview.Heading",
            background="#1f538d",
            foreground="white",
            font=("Helvetica", 13, "bold")
        )

        y_scroll = ttk.Scrollbar(table_frame, orient="vertical")
        y_scroll.pack(side="right", fill="y")

        x_scroll = ttk.Scrollbar(table_frame, orient="horizontal")
        x_scroll.pack(side="bottom", fill="x")

        tree = ttk.Treeview(
            table_frame,
            style="Treeview",
            yscrollcommand=y_scroll.set,
            xscrollcommand=x_scroll.set
        )
        tree.pack(side="left", fill="both", expand=True)

        y_scroll.config(command=tree.yview)
        x_scroll.config(command=tree.xview)

        columns = list(df.columns)
        tree["columns"] = columns
        tree["show"] = "headings"

        for col in columns:
            tree.heading(col, text=str(col))

            if table_type == "summary":
                tree.column(col, anchor="w" if col == columns[0] else "center", width=320)
            elif table_type == "classification":
                tree.column(col, anchor="w" if col == columns[0] else "center", width=140)
            else:
                tree.column(col, anchor="center", width=120)

        for _, row in df.iterrows():
            formatted_row = [self.safe_value_to_string(val) for val in row]
            tree.insert("", "end", values=formatted_row)

        close_btn = ctk.CTkButton(
            top,
            text="Close",
            command=top.destroy,
            width=150,
            height=40
        )
        close_btn.pack(pady=(0, 20))

    def show_classification_report(self):
        if not os.path.exists(self.report_txt):
            messagebox.showinfo(
                "Not Found",
                f"Classification Report not found at:\n{self.report_txt}"
            )
            return

        try:
            with open(self.report_txt, "r", encoding="utf-8") as f:
                lines = f.readlines()

            data = []
            columns = ["Class", "Precision", "Recall", "F1-Score", "Support"]

            for line in lines:
                parts = line.split()

                if not parts:
                    continue

                if parts[0] == "precision":
                    continue

                if len(parts) >= 5 and parts[0] not in ["macro", "weighted", "accuracy"]:
                    data.append(parts[:5])

                elif len(parts) >= 6 and parts[1] == "avg":
                    data.append([
                        f"{parts[0]} {parts[1]}",
                        parts[2],
                        parts[3],
                        parts[4],
                        parts[5]
                    ])

                elif len(parts) >= 3 and parts[0] == "accuracy":
                    data.append([
                        "accuracy",
                        "",
                        "",
                        parts[1],
                        parts[2]
                    ])

            if not data:
                raise ValueError("Could not parse classification report.")

            df = pd.DataFrame(data, columns=columns)
            self.create_table_popup("Classification Report", df, "classification")

        except Exception as e:
            messagebox.showerror(
                "Parse Error",
                f"Failed to load classification report:\n{e}"
            )

    def show_confusion_matrix(self):
        if not os.path.exists(self.cm_csv):
            messagebox.showinfo(
                "Not Found",
                f"Confusion Matrix not found at:\n{self.cm_csv}"
            )
            return

        try:
            df = pd.read_csv(self.cm_csv)
            df = df.fillna("")
            self.create_table_popup("Confusion Matrix", df, "confusion")

        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Failed to load confusion matrix:\n{e}"
            )

    def show_metrics_summary(self):
        if not os.path.exists(self.metrics_json):
            messagebox.showinfo(
                "Not Found",
                f"Fusion metrics not found at:\n{self.metrics_json}"
            )
            return

        try:
            with open(self.metrics_json, "r", encoding="utf-8") as f:
                data = json.load(f)

            rows = []

            for key, value in data.items():
                metric_name = key.replace("_", " ").title()
                metric_value = self.safe_value_to_string(value)
                rows.append([metric_name, metric_value])

            df = pd.DataFrame(rows, columns=["Metric", "Value"])
            self.create_table_popup("Fusion Metrics Summary", df, "summary")

        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Failed to load fusion metrics:\n{e}"
            )

    def show_plots(self):
        if not os.path.exists(self.plot_cm) and not os.path.exists(self.plot_curve):
            messagebox.showinfo(
                "Not Found",
                "No plots found in the fusion_pipeline plots directory."
            )
            return

        top = ctk.CTkToplevel(self.root)
        top.title("Fusion Plots")
        top.geometry("1050x820")
        top.transient(self.root)

        title_label = ctk.CTkLabel(
            top,
            text="Fusion Dataset & Evaluation Plots",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(pady=(20, 10))

        scroll_frame = ctk.CTkScrollableFrame(top, fg_color="transparent")
        scroll_frame.pack(fill="both", expand=True, padx=20, pady=10)

        def add_image(frame, title, path):
            lbl_title = ctk.CTkLabel(
                frame,
                text=title,
                font=ctk.CTkFont(size=18, weight="bold")
            )
            lbl_title.pack(pady=(30, 10))

            try:
                img = Image.open(path)
                w, h = img.size
                ratio = 900 / w if w > 900 else 1
                new_size = (int(w * ratio), int(h * ratio))

                ctk_img = ctk.CTkImage(
                    light_image=img,
                    dark_image=img,
                    size=new_size
                )

                img_lbl = ctk.CTkLabel(frame, text="", image=ctk_img)
                img_lbl.image = ctk_img
                img_lbl.pack(pady=(0, 20))

            except Exception as e:
                err_lbl = ctk.CTkLabel(
                    frame,
                    text=f"Error loading image: {e}",
                    text_color="red"
                )
                err_lbl.pack()

        if os.path.exists(self.plot_curve):
            add_image(scroll_frame, "Training Curve", self.plot_curve)

        if os.path.exists(self.plot_cm):
            add_image(scroll_frame, "Confusion Matrix", self.plot_cm)

        close_btn = ctk.CTkButton(
            top,
            text="Close",
            command=top.destroy,
            width=150,
            height=40
        )
        close_btn.pack(pady=(20, 20))

def predict_cli(audio_path, text):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, "..", ".."))
    models_dir = os.path.join(base_dir, "saved_models")

    model_path = os.path.join(models_dir, "best_model.pth")
    config_path = os.path.join(models_dir, "model_config.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes = DEFAULT_CLASS_NAMES
    max_text_len = DEFAULT_MAX_TEXT_LEN
    text_model_name = DEFAULT_TEXT_MODEL

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        classes = config.get("classes", DEFAULT_CLASS_NAMES)
        max_text_len = int(config.get("max_text_len", DEFAULT_MAX_TEXT_LEN))

        text_branch = config.get("text_branch", DEFAULT_TEXT_MODEL)
        if isinstance(text_branch, str) and "distilbert" in text_branch.lower():
            text_model_name = text_branch

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    if os.path.exists(os.path.join(models_dir, "tokenizer_config.json")):
        tokenizer = AutoTokenizer.from_pretrained(models_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(text_model_name)

    model = LightweightFusionModel(
        num_classes=len(classes),
        model_text_name=text_model_name
    )

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    audio, _ = librosa.load(audio_path, sr=DEFAULT_SR)
    audio = preprocess_audio_array(audio, sr=DEFAULT_SR, max_audio_len=DEFAULT_SR * DEFAULT_DURATION)
    features = extract_audio_features(audio, sr=DEFAULT_SR)

    speech_features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

    encoded = tokenizer(
        text,
        max_length=max_text_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(speech_features, input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)

    top_probs, top_indices = torch.topk(probs, k=3, dim=1)

    print("\nFusion Prediction")
    print("-----------------")

    for i in range(3):
        emotion = classes[top_indices[0][i].item()]
        score = top_probs[0][i].item() * 100
        print(f"{i + 1}. {emotion}: {score:.2f}%")

    prediction = classes[top_indices[0][0].item()]
    confidence = top_probs[0][0].item() * 100

    print(f"\nFinal Prediction: {prediction.upper()}")
    print(f"Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        audio_path_arg = sys.argv[1]
        text_arg = " ".join(sys.argv[2:])
        predict_cli(audio_path_arg, text_arg)
    else:
        root = ctk.CTk()
        app = FusionEmotionApp(root)
        root.mainloop()