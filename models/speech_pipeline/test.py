import os
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import torch
import librosa
import numpy as np

try:
    import sounddevice as sd
except:
    sd = None

from train import UnifiedSERModel

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

SR = 16000
DURATION = 3
MAX_LEN = SR * DURATION
N_MELS = 128

CLASS_NAMES = [
    "anger",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise"
]

def preprocess_audio_array(audio):
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim > 1:
        audio = audio.flatten()

    audio, _ = librosa.effects.trim(audio, top_db=30)

    if len(audio) > MAX_LEN:
        audio = audio[:MAX_LEN]
    else:
        audio = np.pad(audio, (0, MAX_LEN - len(audio)))

    audio = librosa.util.normalize(audio)

    return audio.astype(np.float32)

def normalize_feature(x):
    return (x - x.mean()) / (x.std() + 1e-8)

def extract_features(audio):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_fft=1024,
        hop_length=256,
        n_mels=N_MELS
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)
    delta1 = librosa.feature.delta(mel_db)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SR,
        n_mfcc=40,
        n_fft=1024,
        hop_length=256
    )

    mfcc = librosa.util.fix_length(mfcc, size=mel_db.shape[1], axis=1)
    mfcc = np.repeat(mfcc, 4, axis=0)[:128]

    mel_db = normalize_feature(mel_db)
    delta1 = normalize_feature(delta1)
    mfcc = normalize_feature(mfcc)

    stacked = np.stack([mel_db, delta1, mfcc], axis=0)

    return stacked.astype(np.float32)

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
        self.recording = False
        self.audio_data = []
        self.stream = None

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(self.base_dir, "saved_models", "best_model.pth")

        self.model = UnifiedSERModel(num_classes=len(self.classes)).to(self.device)

        self.build_ui()
        self.load_model()

    def build_ui(self):
        # Header
        self.header_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        self.header_frame.pack(fill="x", pady=(30, 20), padx=40)

        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text="Speech Emotion Recognition",
            font=ctk.CTkFont(family="Helvetica", size=32, weight="bold")
        )
        self.title_label.pack(side="left")

        self.status_badge = ctk.CTkFrame(self.header_frame, corner_radius=15, fg_color="#1b5e20")
        self.status_badge.pack(side="right", pady=5)

        self.status_label = ctk.CTkLabel(
            self.status_badge,
            text="Model Ready",
            text_color="#69f0ae",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.status_label.pack(padx=15, pady=5)

        # Main Body
        self.body_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        self.body_frame.pack(fill="both", expand=True, padx=40, pady=(0, 40))

        self.body_frame.columnconfigure(0, weight=1)
        self.body_frame.columnconfigure(1, weight=2)
        self.body_frame.rowconfigure(0, weight=1)

        # Left Column - Controls
        self.controls_card = ctk.CTkFrame(self.body_frame, corner_radius=20, fg_color="#212121")
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

        self.record_btn = ctk.CTkButton(
            self.controls_card,
            text="Hold To Record",
            height=50,
            corner_radius=10,
            fg_color="#d32f2f",
            hover_color="#b71c1c",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.record_btn.pack(padx=30, pady=15, fill="x")

        if sd is not None:
            self.record_btn.bind("<ButtonPress-1>", self.start_recording)
            self.record_btn.bind("<ButtonRelease-1>", self.stop_recording)
        else:
            self.record_btn.configure(state="disabled", text="sounddevice Missing")

        # Right Column - Results
        self.result_card = ctk.CTkFrame(self.body_frame, corner_radius=20, fg_color="#263238")
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
        if not os.path.exists(self.model_path):
            messagebox.showerror("Error", f"best_model.pth not found:\n{self.model_path}")
            self.set_status("Model Missing", badge_color="#b71c1c", text_color="#ff5252")
            return

        try:
            state_dict = torch.load(self.model_path, map_location=self.device)
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

    def start_recording(self, event):
        if sd is None:
            return

        self.recording = True
        self.audio_data = []

        self.set_status("Recording...", badge_color="#b71c1c", text_color="#ff5252")
        self.record_btn.configure(text="Recording...", fg_color="#b71c1c")

        def callback(indata, frames, time, status):
            if self.recording:
                self.audio_data.append(indata.copy())

        try:
            self.stream = sd.InputStream(
                samplerate=self.fs,
                channels=1,
                dtype="float32",
                callback=callback
            )
            self.stream.start()

        except Exception as e:
            self.recording = False
            messagebox.showerror("Recording Error", str(e))
            self.set_status("Recording Error", badge_color="#b71c1c", text_color="#ff5252")
            self.record_btn.configure(text="Hold To Record", fg_color="#d32f2f")

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
            self.set_status("No Audio", badge_color="#b71c1c", text_color="#ff5252")
            return

        self.set_status("Processing...", badge_color="#f57f17", text_color="#ffee58")

        audio_np = np.concatenate(self.audio_data, axis=0).flatten()
        self.predict(audio_np)

    def predict(self, audio_np):
        try:
            audio_np = preprocess_audio_array(audio_np)
            features = extract_features(audio_np)

            if features.shape != (3, 128, 188):
                raise ValueError(f"Invalid feature shape: {features.shape}")

            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(x)
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


if __name__ == "__main__":
    root = ctk.CTk()
    app = SERApp(root)
    root.mainloop()