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


# =========================
# UI CONFIG
# =========================

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


# =========================
# AUDIO FEATURE PREPROCESSING
# =========================

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


# =========================
# FUSION MODEL
# =========================

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

        # x: [batch, channels, freq, time]
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


# =========================
# GUI APP
# =========================

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

    # =========================
    # CONFIG + MODEL LOADING
    # =========================

    def load_config(self):
        if not os.path.exists(self.config_path):
            return

        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)

            self.classes = (
                config.get("classes")
                or config.get("emotions")
                or DEFAULT_CLASS_NAMES
            )

            self.sr = config.get("sr", DEFAULT_SR)
            self.duration = config.get("duration", DEFAULT_DURATION)
            self.max_audio_len = self.sr * self.duration
            self.max_text_len = config.get("max_text_len", DEFAULT_MAX_TEXT_LEN)

            self.text_model_name = (
                config.get("model_text")
                or config.get("text_model_name")
                or config.get("model_text_name")
                or config.get("text_branch")
                or DEFAULT_TEXT_MODEL
            )

            if self.text_model_name != DEFAULT_TEXT_MODEL and "distilbert" not in self.text_model_name.lower():
                self.text_model_name = DEFAULT_TEXT_MODEL

        except Exception as e:
            messagebox.showwarning("Config Warning", f"Could not load config:\n{e}")

    def load_model(self):
        if not os.path.exists(self.model_path):
            self.set_status("Model Missing", "#b71c1c", "#ff5252")
            messagebox.showerror(
                "Model Missing",
                f"best_model.pth not found:\n{self.model_path}\n\nRun train.py first."
            )
            return

        try:
            self.set_status("Loading Model...", "#f57f17", "#ffee58")
            self.root.update()

            self.tokenizer = AutoTokenizer.from_pretrained(self.models_dir)

            self.model = LightweightFusionModel(
                num_classes=len(self.classes),
                model_text_name=self.text_model_name,
            ).to(self.device)

            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()

            self.set_status("Model Loaded", "#1b5e20", "#69f0ae")

        except Exception as e:
            self.set_status("Model Error", "#b71c1c", "#ff5252")
            messagebox.showerror(
                "Model Error",
                f"{e}\n\nThis usually means best_model.pth was trained with a different architecture.\n"
                f"Use the checkpoint trained with CNN + BiLSTM + Attention + DistilBERT."
            )

    # =========================
    # UI BUILD
    # =========================

    def build_ui(self):
        self.header_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        self.header_frame.pack(fill="x", pady=(20, 10), padx=35)

        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text="Multimodal Fusion Emotion Recognition",
            font=ctk.CTkFont(family="Helvetica", size=30, weight="bold"),
        )
        self.title_label.pack(side="left")

        self.status_badge = ctk.CTkFrame(
            self.header_frame,
            corner_radius=15,
            fg_color="#263238",
        )
        self.status_badge.pack(side="right", pady=5)

        self.status_label = ctk.CTkLabel(
            self.status_badge,
            text="Starting...",
            text_color="#ffffff",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.status_label.pack(padx=15, pady=5)

        self.body_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        self.body_frame.pack(fill="both", expand=True, padx=35, pady=(0, 30))

        self.body_frame.columnconfigure(0, weight=2)
        self.body_frame.columnconfigure(1, weight=3)
        self.body_frame.rowconfigure(0, weight=1)

        self.controls_card = ctk.CTkScrollableFrame(
            self.body_frame,
            corner_radius=20,
            fg_color="#212121",
        )
        self.controls_card.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        self.controls_title = ctk.CTkLabel(
            self.controls_card,
            text="Speech + Transcript Input",
            font=ctk.CTkFont(size=21, weight="bold"),
        )
        self.controls_title.pack(pady=(18, 10))

        self.upload_btn = ctk.CTkButton(
            self.controls_card,
            text="Upload Audio File",
            height=42,
            corner_radius=10,
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self.upload_audio,
        )
        self.upload_btn.pack(padx=28, pady=(5, 7), fill="x")

        self.record_btn = ctk.CTkButton(
            self.controls_card,
            text="Hold To Record",
            height=42,
            corner_radius=10,
            fg_color="#d32f2f",
            hover_color="#b71c1c",
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        self.record_btn.pack(padx=28, pady=7, fill="x")

        if sd is not None:
            self.record_btn.bind("<ButtonPress-1>", self.start_recording)
            self.record_btn.bind("<ButtonRelease-1>", self.stop_recording)
        else:
            self.record_btn.configure(state="disabled", text="sounddevice Missing")

        self.text_label = ctk.CTkLabel(
            self.controls_card,
            text="Transcript Extracted From Filename",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        self.text_label.pack(anchor="w", padx=30, pady=(10, 5))

        self.text_box = ctk.CTkTextbox(
            self.controls_card,
            height=65,
            corner_radius=10,
            font=ctk.CTkFont(size=14),
        )
        self.text_box.pack(padx=28, pady=(0, 8), fill="x")
        self.text_box.insert("1.0", "Upload a TESS audio file to auto-fill transcript")
        self.text_box.configure(state="disabled")

        self.predict_btn = ctk.CTkButton(
            self.controls_card,
            text="Predict Emotion",
            height=45,
            corner_radius=10,
            fg_color="#1565c0",
            hover_color="#0d47a1",
            font=ctk.CTkFont(size=16, weight="bold"),
            command=self.predict_from_current_input,
        )
        self.predict_btn.pack(padx=28, pady=(6, 10), fill="x")

        self.audio_file_label = ctk.CTkLabel(
            self.controls_card,
            text="No audio selected",
            text_color="gray70",
            font=ctk.CTkFont(size=12),
        )
        self.audio_file_label.pack(padx=30, pady=(0, 6))

        self.reports_title = ctk.CTkLabel(
            self.controls_card,
            text="Reports & Metrics",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        self.reports_title.pack(pady=(5, 5))

        self.reports_grid = ctk.CTkFrame(self.controls_card, fg_color="transparent")
        self.reports_grid.pack(padx=28, pady=(0, 10), fill="x")

        ctk.CTkButton(
            self.reports_grid,
            text="Classification Report",
            height=34,
            command=self.show_classification_report,
        ).grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        ctk.CTkButton(
            self.reports_grid,
            text="Confusion Matrix",
            height=34,
            command=self.show_confusion_matrix,
        ).grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        ctk.CTkButton(
            self.reports_grid,
            text="Metrics Summary",
            height=34,
            command=self.show_metrics_summary,
        ).grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        ctk.CTkButton(
            self.reports_grid,
            text="Training Metrics",
            height=34,
            command=self.show_training_metrics,
        ).grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        ctk.CTkButton(
            self.reports_grid,
            text="Fusion JSON",
            height=34,
            command=self.show_fusion_metrics_json,
        ).grid(row=2, column=0, padx=5, pady=5, sticky="ew")

        ctk.CTkButton(
            self.reports_grid,
            text="View Plots",
            height=34,
            command=self.show_plots,
        ).grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        self.reports_grid.columnconfigure(0, weight=1)
        self.reports_grid.columnconfigure(1, weight=1)

        self.result_card = ctk.CTkFrame(
            self.body_frame,
            corner_radius=20,
            fg_color="#263238",
        )
        self.result_card.grid(row=0, column=1, sticky="nsew", padx=(12, 0))

        self.result_emoji = ctk.CTkLabel(
            self.result_card,
            text="🔀",
            font=ctk.CTkFont(size=115),
        )
        self.result_emoji.pack(pady=(55, 18), expand=True)

        self.result_text = ctk.CTkLabel(
            self.result_card,
            text="Ready for Fusion Input",
            font=ctk.CTkFont(size=36, weight="bold"),
        )
        self.result_text.pack(pady=(0, 10))

        self.confidence_text = ctk.CTkLabel(
            self.result_card,
            text="Confidence: --",
            font=ctk.CTkFont(size=18),
            text_color="gray70",
        )
        self.confidence_text.pack(pady=(0, 20))

        self.progress = ctk.CTkProgressBar(
            self.result_card,
            width=430,
            height=15,
            progress_color="#00e676",
            fg_color="#37474f",
        )
        self.progress.pack(pady=(0, 25))
        self.progress.set(0)

        self.all_probs_label = ctk.CTkLabel(
            self.result_card,
            text="",
            justify="left",
            font=ctk.CTkFont(size=14),
            text_color="gray85",
        )
        self.all_probs_label.pack(pady=(0, 45))

    def set_status(self, text, badge_color="#1b5e20", text_color="#69f0ae"):
        self.status_label.configure(text=text, text_color=text_color)
        self.status_badge.configure(fg_color=badge_color)

    # =========================
    # TEXTBOX HELPER
    # =========================

    def set_transcript_text(self, text):
        self.text_box.configure(state="normal")
        self.text_box.delete("1.0", "end")
        self.text_box.insert("1.0", text)
        self.text_box.configure(state="disabled")

    # =========================
    # AUDIO INPUT
    # =========================

    def upload_audio(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.ogg *.m4a")]
        )

        if not filepath:
            return

        try:
            self.set_status("Loading Audio...", "#f57f17", "#ffee58")
            self.root.update()

            audio_np, _ = librosa.load(filepath, sr=self.sr, mono=True)
            self.current_audio = audio_np
            self.current_audio_path = filepath

            filename = os.path.basename(filepath)
            self.audio_file_label.configure(text=f"Selected: {filename}")

            extracted_text = self.extract_text_from_filename(filename)

            if extracted_text:
                self.set_transcript_text(extracted_text)
            else:
                self.set_transcript_text("unknown")

            self.set_status("Audio Ready", "#1b5e20", "#69f0ae")

        except Exception as e:
            self.set_status("Audio Error", "#b71c1c", "#ff5252")
            messagebox.showerror("Audio Error", str(e))

    def extract_text_from_filename(self, filename):
        try:
            name = os.path.splitext(filename)[0]
            parts = name.split("_")

            if len(parts) >= 3:
                text_parts = parts[1:-1]
                text = " ".join(text_parts).replace("-", " ").replace("_", " ")
                text = text.lower().strip()

                if text:
                    return text

            return None

        except Exception:
            return None

    def start_recording(self, event):
        if sd is None:
            return

        self.recording = True
        self.audio_data = []

        self.set_status("Recording...", "#b71c1c", "#ff5252")
        self.record_btn.configure(text="Recording...", fg_color="#b71c1c")

        def callback(indata, frames, time, status):
            if self.recording:
                self.audio_data.append(indata.copy())

        try:
            self.stream = sd.InputStream(
                samplerate=self.sr,
                channels=1,
                dtype="float32",
                callback=callback,
            )
            self.stream.start()

        except Exception as e:
            self.recording = False
            self.set_status("Recording Error", "#b71c1c", "#ff5252")
            self.record_btn.configure(text="Hold To Record", fg_color="#d32f2f")
            messagebox.showerror("Recording Error", str(e))

    def stop_recording(self, event):
        if sd is None:
            return

        self.recording = False

        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
                self.stream = None
        except Exception:
            pass

        self.record_btn.configure(text="Hold To Record", fg_color="#d32f2f")

        if len(self.audio_data) == 0:
            self.set_status("No Audio", "#b71c1c", "#ff5252")
            return

        audio_np = np.concatenate(self.audio_data, axis=0).flatten()
        self.current_audio = audio_np
        self.current_audio_path = None
        self.audio_file_label.configure(text="Recorded audio ready")

        self.set_transcript_text("unknown")
        self.set_status("Audio Ready", "#1b5e20", "#69f0ae")

    # =========================
    # PREDICTION
    # =========================

    def predict_from_current_input(self):
        if self.model is None or self.tokenizer is None:
            messagebox.showerror("Model Error", "Model is not loaded.")
            return

        if self.current_audio is None:
            messagebox.showwarning("Missing Audio", "Upload or record audio first.")
            return

        text = self.text_box.get("1.0", "end").strip()

        if not text:
            messagebox.showwarning("Missing Text", "Transcript is empty.")
            return

        self.predict(self.current_audio, text)

    def predict(self, audio_np, text):
        try:
            self.set_status("Predicting...", "#f57f17", "#ffee58")
            self.root.update()

            audio_np = preprocess_audio_array(
                audio_np,
                sr=self.sr,
                max_audio_len=self.max_audio_len,
            )

            speech_features = extract_audio_features(audio_np, sr=self.sr)
            speech_features = torch.tensor(
                speech_features,
                dtype=torch.float32,
            ).unsqueeze(0).to(self.device)

            text_inputs = self.tokenizer(
                text,
                max_length=self.max_text_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            input_ids = text_inputs["input_ids"].to(self.device)
            attention_mask = text_inputs["attention_mask"].to(self.device)

            with torch.no_grad():
                logits = self.model(speech_features, input_ids, attention_mask)
                probabilities = torch.softmax(logits, dim=1).squeeze(0)

                pred_idx = torch.argmax(probabilities).item()
                pred_emotion = self.classes[pred_idx]
                confidence = probabilities[pred_idx].item() * 100

            display_emotion = pred_emotion if confidence >= 55 else "uncertain"
            self.update_ui(display_emotion, confidence, probabilities)

        except Exception as e:
            self.set_status("Prediction Error", "#b71c1c", "#ff5252")
            messagebox.showerror("Prediction Error", str(e))

    def update_ui(self, emotion, confidence, probabilities):
        color = EMOTION_COLORS.get(emotion, "#00e676")
        emoji = EMOTION_EMOJIS.get(emotion, "🔀")

        self.result_emoji.configure(text=emoji)
        self.result_text.configure(text=emotion.upper(), text_color=color)
        self.confidence_text.configure(text=f"Confidence: {confidence:.2f}%")
        self.progress.configure(progress_color=color)
        self.progress.set(confidence / 100)

        lines = []

        for i, cls in enumerate(self.classes):
            lines.append(f"{cls:<10}: {probabilities[i].item() * 100:.2f}%")

        self.all_probs_label.configure(text="\n".join(lines))
        self.set_status("Prediction Complete", "#1b5e20", "#69f0ae")

    # =========================
    # REPORT VIEWERS
    # =========================

    def format_cell_value(self, value):
        if pd.isna(value):
            return ""

        try:
            float_val = float(value)

            if float_val.is_integer():
                return str(int(float_val))

            return f"{float_val:.4f}"

        except Exception:
            return str(value)

    def show_csv_table(self, title, csv_relative_path, table_type):
        filepath = os.path.join(self.output_root, csv_relative_path)

        if not os.path.exists(filepath):
            messagebox.showinfo("Not Found", f"{title} not found at:\n{filepath}")
            return

        try:
            df = pd.read_csv(filepath).fillna("")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load {csv_relative_path}:\n{e}")
            return

        top = ctk.CTkToplevel(self.root)
        top.title(title)
        top.geometry("950x620")
        top.transient(self.root)

        title_label = ctk.CTkLabel(
            top,
            text=title,
            font=ctk.CTkFont(size=24, weight="bold"),
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
            font=("Helvetica", 12),
        )
        style.map("Treeview", background=[("selected", "#1f538d")])
        style.configure(
            "Treeview.Heading",
            background="#1f538d",
            foreground="white",
            font=("Helvetica", 13, "bold"),
        )

        y_scroll = ttk.Scrollbar(table_frame, orient="vertical")
        y_scroll.pack(side="right", fill="y")

        x_scroll = ttk.Scrollbar(table_frame, orient="horizontal")
        x_scroll.pack(side="bottom", fill="x")

        tree = ttk.Treeview(
            table_frame,
            yscrollcommand=y_scroll.set,
            xscrollcommand=x_scroll.set,
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
                tree.column(col, anchor="center", width=180)
            elif table_type == "classification":
                tree.column(col, anchor="center", width=140)
            else:
                tree.column(col, anchor="center", width=120)

        for _, row in df.iterrows():
            formatted_row = [self.format_cell_value(val) for val in row]
            tree.insert("", "end", values=formatted_row)

        close_btn = ctk.CTkButton(
            top,
            text="Close",
            command=top.destroy,
            width=150,
            height=40,
        )
        close_btn.pack(pady=(0, 20))

    def show_classification_report(self):
        self.show_csv_table(
            "Fusion Classification Report",
            os.path.join("results", "classification_report.csv"),
            "classification",
        )

    def show_confusion_matrix(self):
        self.show_csv_table(
            "Fusion Confusion Matrix",
            os.path.join("results", "confusion_matrix.csv"),
            "confusion",
        )

    def show_metrics_summary(self):
        self.show_csv_table(
            "Fusion Metrics Summary",
            os.path.join("results", "summary.csv"),
            "summary",
        )

    def show_training_metrics(self):
        self.show_csv_table(
            "Fusion Training Metrics",
            os.path.join("metrics", "training_metrics.csv"),
            "summary",
        )

    def show_fusion_metrics_json(self):
        filepath = os.path.join(self.output_root, "metrics", "fusion_metrics.json")

        if not os.path.exists(filepath):
            messagebox.showinfo("Not Found", f"fusion_metrics.json not found at:\n{filepath}")
            return

        try:
            with open(filepath, "r") as f:
                data = json.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load fusion_metrics.json:\n{e}")
            return

        top = ctk.CTkToplevel(self.root)
        top.title("Fusion Metrics JSON")
        top.geometry("900x680")
        top.transient(self.root)

        title_label = ctk.CTkLabel(
            top,
            text="Fusion Metrics JSON",
            font=ctk.CTkFont(size=24, weight="bold"),
        )
        title_label.pack(pady=(20, 10))

        text_frame = ctk.CTkFrame(top, corner_radius=10)
        text_frame.pack(fill="both", expand=True, padx=25, pady=(10, 20))

        textbox = ctk.CTkTextbox(
            text_frame,
            font=ctk.CTkFont(family="Consolas", size=13),
            wrap="none",
        )
        textbox.pack(fill="both", expand=True, padx=12, pady=12)

        pretty_json = json.dumps(data, indent=4)
        textbox.insert("1.0", pretty_json)
        textbox.configure(state="disabled")

        close_btn = ctk.CTkButton(
            top,
            text="Close",
            command=top.destroy,
            width=150,
            height=40,
        )
        close_btn.pack(pady=(0, 20))

    def show_plots(self):
        plot_curve = os.path.join(self.output_root, "plots", "training_curve.png")
        plot_cm = os.path.join(self.output_root, "plots", "confusion_matrix.png")
        plot_cm_test = os.path.join(self.output_root, "plots", "confusion_matrix_test.png")

        if not os.path.exists(plot_curve) and not os.path.exists(plot_cm) and not os.path.exists(plot_cm_test):
            messagebox.showinfo("Not Found", "No plots found in the plots/ directory.")
            return

        top = ctk.CTkToplevel(self.root)
        top.title("Fusion Plots")
        top.geometry("1000x800")
        top.transient(self.root)

        title_label = ctk.CTkLabel(
            top,
            text="Fusion Training & Evaluation Plots",
            font=ctk.CTkFont(size=24, weight="bold"),
        )
        title_label.pack(pady=(20, 10))

        scroll_frame = ctk.CTkScrollableFrame(top, fg_color="transparent")
        scroll_frame.pack(fill="both", expand=True, padx=20, pady=10)

        def add_image(frame, title, path):
            if not os.path.exists(path):
                return

            lbl_title = ctk.CTkLabel(
                frame,
                text=title,
                font=ctk.CTkFont(size=18, weight="bold"),
            )
            lbl_title.pack(pady=(30, 10))

            try:
                img = Image.open(path)
                w, h = img.size
                ratio = 850 / w if w > 850 else 1
                new_size = (int(w * ratio), int(h * ratio))

                ctk_img = ctk.CTkImage(
                    light_image=img,
                    dark_image=img,
                    size=new_size,
                )

                img_lbl = ctk.CTkLabel(frame, text="", image=ctk_img)
                img_lbl.image = ctk_img
                img_lbl.pack(pady=(0, 20))

            except Exception as e:
                err_lbl = ctk.CTkLabel(
                    frame,
                    text=f"Error loading image: {e}",
                    text_color="red",
                )
                err_lbl.pack()

        add_image(scroll_frame, "Training Curve", plot_curve)
        add_image(scroll_frame, "Confusion Matrix", plot_cm)
        add_image(scroll_frame, "Confusion Matrix Test", plot_cm_test)

        close_btn = ctk.CTkButton(
            top,
            text="Close",
            command=top.destroy,
            width=150,
            height=40,
        )
        close_btn.pack(pady=(20, 20))


# =========================
# CLI PREDICTION
# =========================

def predict_cli(audio_path, text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "saved_models")
    model_path = os.path.join(models_dir, "best_model.pth")
    config_path = os.path.join(models_dir, "model_config.json")

    if not os.path.exists(model_path):
        print(f"Model file not found:\n{model_path}")
        return

    if not os.path.exists(audio_path):
        print(f"Audio file not found:\n{audio_path}")
        return

    classes = DEFAULT_CLASS_NAMES
    sr = DEFAULT_SR
    duration = DEFAULT_DURATION
    max_text_len = DEFAULT_MAX_TEXT_LEN
    text_model_name = DEFAULT_TEXT_MODEL

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)

        classes = config.get("classes") or config.get("emotions") or DEFAULT_CLASS_NAMES
        sr = config.get("sr", DEFAULT_SR)
        duration = config.get("duration", DEFAULT_DURATION)
        max_text_len = config.get("max_text_len", DEFAULT_MAX_TEXT_LEN)

        text_model_name = (
            config.get("model_text")
            or config.get("text_model_name")
            or config.get("model_text_name")
            or config.get("text_branch")
            or DEFAULT_TEXT_MODEL
        )

        if text_model_name != DEFAULT_TEXT_MODEL and "distilbert" not in text_model_name.lower():
            text_model_name = DEFAULT_TEXT_MODEL

    max_audio_len = sr * duration

    tokenizer = AutoTokenizer.from_pretrained(models_dir)

    model = LightweightFusionModel(
        num_classes=len(classes),
        model_text_name=text_model_name,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    audio_np, _ = librosa.load(audio_path, sr=sr, mono=True)

    audio_np = preprocess_audio_array(
        audio_np,
        sr=sr,
        max_audio_len=max_audio_len,
    )

    speech_features = extract_audio_features(audio_np, sr=sr)
    speech_features = torch.tensor(
        speech_features,
        dtype=torch.float32,
    ).unsqueeze(0).to(device)

    text_inputs = tokenizer(
        text,
        max_length=max_text_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = text_inputs["input_ids"].to(device)
    attention_mask = text_inputs["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(speech_features, input_ids, attention_mask)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

    pred_idx = torch.argmax(probabilities).item()
    pred_emotion = classes[pred_idx]
    confidence = probabilities[pred_idx].item() * 100

    print(f"\nPredicted Emotion: {pred_emotion.upper()}")
    print(f"Confidence Score: {confidence:.2f}%")
    print("\nConfidence for all classes:")

    for i, cls in enumerate(classes):
        print(f"{cls}: {probabilities[i].item() * 100:.2f}%")


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        audio_path_arg = sys.argv[1]
        text_arg = " ".join(sys.argv[2:])
        predict_cli(audio_path_arg, text_arg)
    else:
        root = ctk.CTk()
        app = FusionEmotionApp(root)
        root.mainloop()