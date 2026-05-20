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
    "uncertain": "#bdbdbd"
}

EMOTION_EMOJIS = {
    "anger": "😡",
    "disgust": "🤢",
    "fear": "😨",
    "happiness": "😄",
    "neutral": "😐",
    "sadness": "😢",
    "surprise": "😲",
    "uncertain": "❓"
}

# Resolve project paths

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(PIPELINE_DIR, "..", ".."))

MODEL_PATH = os.path.join(PIPELINE_DIR, "saved_models", "best_model.pth")
CONFIG_PATH = os.path.join(PIPELINE_DIR, "saved_models", "model_config.json")

OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "results", "speech_pipeline")

# Configure audio settings

SR = 16000
DURATION = 3
MAX_LEN = SR * DURATION
N_MELS = 128
N_MFCC = 40

CLASS_NAMES = [
    "anger",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise"
]

# Extract acoustic features

def preprocess_audio_array(audio, fs=SR, max_len=MAX_LEN):
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim > 1:
        audio = audio.flatten()

    audio, _ = librosa.effects.trim(audio, top_db=30)

    if len(audio) > max_len:
        audio = audio[:max_len]
    else:
        audio = np.pad(audio, (0, max_len - len(audio)))

    audio = librosa.util.normalize(audio)

    return audio.astype(np.float32)

def normalize_feature(feature):
    return (feature - feature.mean()) / (feature.std() + 1e-8)

def extract_features_from_audio(audio):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_fft=1024,
        hop_length=256,
        n_mels=N_MELS
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    delta = librosa.feature.delta(
        mel_db
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SR,
        n_mfcc=N_MFCC,
        n_fft=1024,
        hop_length=256
    )

    mfcc = librosa.util.fix_length(
        mfcc,
        size=mel_db.shape[1],
        axis=1
    )

    mfcc = np.repeat(
        mfcc,
        4,
        axis=0
    )[:N_MELS]

    mel_db = normalize_feature(mel_db)
    delta = normalize_feature(delta)
    mfcc = normalize_feature(mfcc)

    features = np.stack(
        [mel_db, delta, mfcc],
        axis=0
    )

    return features.astype(np.float32)

# Model architecture

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
        x = self.cnn(x)
        x = self.dropout_cnn(x)

        batch, channels, freq, time = x.shape

        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(batch, time, channels * freq)

        lstm_out, _ = self.lstm(x)

        pooled = self.attention(lstm_out)

        logits = self.classifier(pooled)

        return logits

# User interface setup

class SERApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Speech Emotion Recognition")
        self.root.geometry("1000x650")
        self.root.minsize(900, 500)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = CLASS_NAMES

        self.fs = SR
        self.max_len = MAX_LEN

        self.load_config()

        self.model = CNNBiLSTMAttentionSER(
            num_classes=len(self.classes)
        ).to(self.device)

        self.build_ui()
        self.load_model()

    def load_config(self):
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r") as f:
                    config = json.load(f)

                self.classes = config.get("classes", CLASS_NAMES)

            except Exception as e:
                print(f"Config load warning: {e}")

    def build_ui(self):
        self.header_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        self.header_frame.pack(fill="x", pady=(30, 20), padx=40)

        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text="Speech Emotion Recognition",
            font=ctk.CTkFont(family="Helvetica", size=32, weight="bold")
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
            text="Model Ready",
            text_color="#69f0ae",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.status_label.pack(padx=15, pady=5)

        self.body_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        self.body_frame.pack(fill="both", expand=True, padx=40, pady=(0, 40))

        self.body_frame.columnconfigure(0, weight=1)
        self.body_frame.columnconfigure(1, weight=2)
        self.body_frame.rowconfigure(0, weight=1)

        self.controls_card = ctk.CTkFrame(
            self.body_frame,
            corner_radius=20,
            fg_color="#212121"
        )
        self.controls_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        self.controls_title = ctk.CTkLabel(
            self.controls_card,
            text="Input Audio",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.controls_title.pack(pady=(40, 30))

        self.upload_btn = ctk.CTkButton(
            self.controls_card,
            text="Upload File",
            height=50,
            corner_radius=10,
            font=ctk.CTkFont(size=16, weight="bold"),
            command=self.upload_audio
        )
        self.upload_btn.pack(padx=30, pady=15, fill="x")


        self.reports_title = ctk.CTkLabel(
            self.controls_card,
            text="Reports & Metrics",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.reports_title.pack(pady=(20, 10))

        self.reports_grid = ctk.CTkFrame(
            self.controls_card,
            fg_color="transparent"
        )
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
            self.body_frame,
            corner_radius=20,
            fg_color="#263238"
        )
        self.result_card.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        self.result_emoji = ctk.CTkLabel(
            self.result_card,
            text="🎙️",
            font=ctk.CTkFont(size=120)
        )
        self.result_emoji.pack(pady=(60, 20), expand=True)

        self.result_text = ctk.CTkLabel(
            self.result_card,
            text="Ready for Input",
            font=ctk.CTkFont(size=38, weight="bold")
        )
        self.result_text.pack(pady=(0, 10))

        self.confidence_text = ctk.CTkLabel(
            self.result_card,
            text="Confidence: --",
            font=ctk.CTkFont(size=18),
            text_color="gray70"
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
        if not os.path.exists(MODEL_PATH):
            messagebox.showerror("Error", f"best_model.pth not found:\n{MODEL_PATH}")
            self.set_status("Model Missing", badge_color="#b71c1c", text_color="#ff5252")
            return

        try:
            state_dict = torch.load(MODEL_PATH, map_location=self.device)

            if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]

            self.model.load_state_dict(state_dict)
            self.model.eval()

            self.set_status("Model Loaded")

        except Exception as e:
            messagebox.showerror("Model Error", str(e))
            self.set_status("Model Error", badge_color="#b71c1c", text_color="#ff5252")

    def upload_audio(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.ogg *.m4a")]
        )

        if not filepath:
            return

        try:
            self.set_status("Processing...", badge_color="#f57f17", text_color="#ffee58")
            self.root.update()

            audio_np, _ = librosa.load(filepath, sr=self.fs, mono=True)
            self.predict(audio_np)

        except Exception as e:
            messagebox.showerror("Audio Error", str(e))
            self.set_status("Audio Error", badge_color="#b71c1c", text_color="#ff5252")


    def predict(self, audio_np):
        try:
            audio_np = preprocess_audio_array(
                audio_np,
                fs=self.fs,
                max_len=self.max_len
            )

            features = extract_features_from_audio(audio_np)
            features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(features)
                probabilities = torch.softmax(output, dim=1).squeeze()

                pred_idx = torch.argmax(probabilities).item()
                pred_emotion = self.classes[pred_idx]
                confidence = probabilities[pred_idx].item() * 100

                if confidence < 55:
                    display_emotion = "uncertain"
                else:
                    display_emotion = pred_emotion

                self.update_ui(display_emotion, confidence)

        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
            self.set_status("Prediction Error", badge_color="#b71c1c", text_color="#ff5252")

    def update_ui(self, emotion, confidence):
        color = EMOTION_COLORS.get(emotion, "#00e676")
        emoji = EMOTION_EMOJIS.get(emotion, "🎙️")

        self.result_emoji.configure(text=emoji)
        self.result_text.configure(text=emotion.upper(), text_color=color)
        self.confidence_text.configure(text=f"Confidence: {confidence:.2f}%")
        self.progress.configure(progress_color=color)
        self.progress.set(confidence / 100)

        self.set_status("Prediction Complete")

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

    def show_csv_table(self, title, csv_path, table_type):
        filepath = os.path.join(OUTPUT_ROOT, csv_path)

        if not os.path.exists(filepath):
            messagebox.showinfo("Not Found", f"{title} not found at:\n{filepath}")
            return

        try:
            df = pd.read_csv(filepath).fillna("")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load {csv_path}:\n{e}")
            return

        top = ctk.CTkToplevel(self.root)
        top.title(title)
        top.geometry("900x600")
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
                tree.column(col, anchor="w" if col == columns[0] else "center", width=250)
            elif table_type == "classification":
                tree.column(col, anchor="w" if col == columns[0] else "center", width=140)
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
            height=40
        )
        close_btn.pack(pady=(0, 20))

    def show_classification_report(self):
        self.show_csv_table(
            "Classification Report",
            "results/classification_report.csv",
            "classification"
        )

    def show_confusion_matrix(self):
        self.show_csv_table(
            "Confusion Matrix",
            "results/confusion_matrix.csv",
            "confusion"
        )

    def show_metrics_summary(self):
        self.show_csv_table(
            "Metrics Summary",
            "results/summary.csv",
            "summary"
        )

    def show_plots(self):
        plot_cm = os.path.join(OUTPUT_ROOT, "plots", "confusion_matrix.png")
        plot_curve = os.path.join(OUTPUT_ROOT, "plots", "training_curve.png")

        if not os.path.exists(plot_cm) and not os.path.exists(plot_curve):
            messagebox.showinfo("Not Found", "No plots found in the plots/ directory.")
            return

        top = ctk.CTkToplevel(self.root)
        top.title("View Plots")
        top.geometry("1000x800")
        top.transient(self.root)

        title_label = ctk.CTkLabel(
            top,
            text="Training & Evaluation Plots",
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
                ratio = 850 / w if w > 850 else 1
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

        if os.path.exists(plot_curve):
            add_image(scroll_frame, "Training Curve", plot_curve)

        if os.path.exists(plot_cm):
            add_image(scroll_frame, "Confusion Matrix", plot_cm)

        close_btn = ctk.CTkButton(
            top,
            text="Close",
            command=top.destroy,
            width=150,
            height=40
        )
        close_btn.pack(pady=(20, 20))

# Cli prediction

def predict_cli(audio_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes = CLASS_NAMES

    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                config = json.load(f)
            classes = config.get("classes", CLASS_NAMES)
        except Exception as e:
            print(f"Warning: Failed to load config file: {e}")

    if not os.path.exists(MODEL_PATH):
        print("Model file not found. Please train first.")
        return

    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        return

    model = CNNBiLSTMAttentionSER(
        num_classes=len(classes)
    ).to(device)

    state_dict = torch.load(MODEL_PATH, map_location=device)

    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    model.load_state_dict(state_dict)
    model.eval()

    try:
        audio_np, _ = librosa.load(audio_path, sr=SR, mono=True)
        audio_np = preprocess_audio_array(audio_np, SR, MAX_LEN)

        features = extract_features_from_audio(audio_np)
        features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(features)
            probabilities = torch.softmax(output, dim=1).squeeze()

            pred_idx = torch.argmax(probabilities).item()
            pred_emotion = classes[pred_idx]
            confidence = probabilities[pred_idx].item() * 100

        print(f"\nPredicted Emotion: {pred_emotion.upper()}")
        print(f"Confidence Score: {confidence:.2f}%")
        print("\nConfidence for all classes:")

        for i, cls in enumerate(classes):
            print(f"{cls}: {probabilities[i].item() * 100:.2f}%")

    except Exception as e:
        print(f"Error processing audio: {e}")

# Entry point

if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict_cli(sys.argv[1])
    else:
        root = ctk.CTk()
        app = SERApp(root)
        root.mainloop()