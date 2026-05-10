import os
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk

import torch
import torch.nn as nn
import torchaudio.transforms as T

import librosa
import numpy as np

try:
    import sounddevice as sd
except:
    sd = None

from train import SERModel

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

EMOTION_COLORS = {
    "anger": "#ff5252",
    "disgust": "#8bc34a",
    "fear": "#ab47bc",
    "happiness": "#ffd54f",
    "neutral": "#90a4ae",
    "sadness": "#42a5f5",
    "surprise": "#ff9800"
}

EMOTION_EMOJIS = {
    "anger": "😡",
    "disgust": "🤢",
    "fear": "😨",
    "happiness": "😄",
    "neutral": "😐",
    "sadness": "😢",
    "surprise": "😲"
}

class SERApp:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "Multimodal Emotion Recognition"
        )

        self.root.geometry("1200x750")

        self.root.minsize(1000, 650)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.classes = [
            "anger",
            "disgust",
            "fear",
            "happiness",
            "neutral",
            "sadness",
            "surprise"
        ]

        self.fs = 16000

        self.max_len = 16000 * 3

        self.recording = False

        self.audio_data = []

        self.stream = None

        self.base_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

        self.model_path = os.path.join(
            self.base_dir,
            "saved_models",
            "best_model.pth"
        )

        self.model = SERModel(
            num_classes=len(self.classes)
        ).to(self.device)

        self.mel_transform = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            hop_length=256,
            n_mels=128
        )

        self.amplitude_to_db = T.AmplitudeToDB()

        self.compute_deltas = T.ComputeDeltas()

        self.build_ui()

        self.load_model()

    def build_ui(self):

        self.sidebar = ctk.CTkFrame(
            self.root,
            width=260,
            corner_radius=0
        )

        self.sidebar.pack(
            side="left",
            fill="y"
        )

        self.logo = ctk.CTkLabel(

            self.sidebar,

            text="Emotion AI",

            font=ctk.CTkFont(
                size=28,
                weight="bold"
            )
        )

        self.logo.pack(
            pady=(40, 20)
        )

        self.upload_btn = ctk.CTkButton(

            self.sidebar,

            text="Upload Audio",

            height=50,

            font=ctk.CTkFont(
                size=16,
                weight="bold"
            ),

            command=self.upload_audio
        )

        self.upload_btn.pack(
            padx=20,
            pady=15,
            fill="x"
        )

        self.record_btn = ctk.CTkButton(

            self.sidebar,

            text="Hold To Record",

            height=50,

            fg_color="#d32f2f",

            hover_color="#b71c1c",

            font=ctk.CTkFont(
                size=16,
                weight="bold"
            )
        )

        self.record_btn.pack(
            padx=20,
            pady=15,
            fill="x"
        )

        if sd is not None:

            self.record_btn.bind(
                "<ButtonPress-1>",
                self.start_recording
            )

            self.record_btn.bind(
                "<ButtonRelease-1>",
                self.stop_recording
            )

        else:

            self.record_btn.configure(
                state="disabled",
                text="sounddevice Missing"
            )

        self.status_label = ctk.CTkLabel(

            self.sidebar,

            text="Model Ready",

            text_color="#00e676",

            font=ctk.CTkFont(
                size=14
            )
        )

        self.status_label.pack(
            pady=20
        )

        self.main_frame = ctk.CTkFrame(
            self.root,
            corner_radius=0
        )

        self.main_frame.pack(
            side="right",
            fill="both",
            expand=True
        )

        self.header = ctk.CTkLabel(

            self.main_frame,

            text="Speech Emotion Recognition",

            font=ctk.CTkFont(
                size=34,
                weight="bold"
            )
        )

        self.header.pack(
            pady=(40, 20)
        )

        self.result_card = ctk.CTkFrame(

            self.main_frame,

            width=700,

            height=260,

            corner_radius=25
        )

        self.result_card.pack(
            pady=30
        )

        self.result_emoji = ctk.CTkLabel(

            self.result_card,

            text="🎤",

            font=ctk.CTkFont(
                size=90
            )
        )

        self.result_emoji.pack(
            pady=(30, 10)
        )

        self.result_text = ctk.CTkLabel(

            self.result_card,

            text="Waiting For Prediction",

            font=ctk.CTkFont(
                size=28,
                weight="bold"
            )
        )

        self.result_text.pack()

        self.confidence_text = ctk.CTkLabel(

            self.result_card,

            text="Confidence: --",

            font=ctk.CTkFont(
                size=18
            )
        )

        self.confidence_text.pack(
            pady=10
        )

        self.progress = ctk.CTkProgressBar(

            self.main_frame,

            width=500,

            height=20,

            progress_color="#00e676"
        )

        self.progress.pack(
            pady=20
        )

        self.progress.set(0)

        self.bottom_frame = ctk.CTkFrame(

            self.main_frame,

            corner_radius=20
        )

        self.bottom_frame.pack(
            fill="x",
            padx=40,
            pady=20
        )

        self.info_text = ctk.CTkTextbox(

            self.bottom_frame,

            height=150,

            font=ctk.CTkFont(
                size=15
            )
        )

        self.info_text.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=20
        )

        self.info_text.insert(

            "end",

            "System Initialized...\n"
        )

        self.info_text.configure(
            state="disabled"
        )

    def log(self, text):

        self.info_text.configure(
            state="normal"
        )

        self.info_text.insert(
            "end",
            f"{text}\n"
        )

        self.info_text.see("end")

        self.info_text.configure(
            state="disabled"
        )

    def load_model(self):

        if not os.path.exists(self.model_path):

            messagebox.showerror(
                "Error",
                "Model not found."
            )

            return

        try:

            self.model.load_state_dict(

                torch.load(
                    self.model_path,
                    map_location=self.device
                )
            )

            self.model.eval()

            self.log(
                "Model Loaded Successfully"
            )

        except Exception as e:

            messagebox.showerror(
                "Model Error",
                str(e)
            )

    def upload_audio(self):

        filepath = filedialog.askopenfilename(

            filetypes=[
                ("Audio Files", "*.wav *.mp3")
            ]
        )

        if not filepath:

            return

        try:

            self.status_label.configure(
                text="Processing Audio...",
                text_color="yellow"
            )

            self.root.update()

            audio_np, _ = librosa.load(
                filepath,
                sr=self.fs
            )

            audio_np, _ = librosa.effects.trim(
                audio_np,
                top_db=30
            )

            self.predict(audio_np)

        except Exception as e:

            messagebox.showerror(
                "Audio Error",
                str(e)
            )

    def start_recording(self, event):

        self.recording = True

        self.audio_data = []

        self.status_label.configure(
            text="Recording...",
            text_color="#ff5252"
        )

        self.record_btn.configure(
            text="Recording..."
        )

        def callback(indata, frames, time, status):

            if self.recording:

                self.audio_data.append(
                    indata.copy()
                )

        self.stream = sd.InputStream(

            samplerate=self.fs,

            channels=1,

            callback=callback
        )

        self.stream.start()

    def stop_recording(self, event):

        self.recording = False

        self.stream.stop()

        self.stream.close()

        self.record_btn.configure(
            text="Hold To Record"
        )

        self.status_label.configure(
            text="Processing...",
            text_color="yellow"
        )

        if len(self.audio_data) > 0:

            audio_np = np.concatenate(
                self.audio_data,
                axis=0
            ).flatten()

            self.predict(audio_np)

    def predict(self, audio_np):

        waveform = torch.tensor(
            audio_np,
            dtype=torch.float32
        ).unsqueeze(0)

        if waveform.shape[1] > self.max_len:

            waveform = waveform[:, :self.max_len]

        else:

            pad_amount = self.max_len - waveform.shape[1]

            waveform = torch.nn.functional.pad(
                waveform,
                (0, pad_amount)
            )

        with torch.no_grad():

            mel = self.mel_transform(
                waveform
            )

            mel_db = self.amplitude_to_db(
                mel
            )

            delta1 = self.compute_deltas(
                mel_db
            )

            delta2 = self.compute_deltas(
                delta1
            )

            x = torch.cat(
                [mel_db, delta1, delta2],
                dim=0
            )

            mean = x.mean(
                dim=(1, 2),
                keepdim=True
            )

            std = x.std(
                dim=(1, 2),
                keepdim=True
            )

            x = (
                x - mean
            ) / (std + 1e-8)

            x = x.unsqueeze(0).to(
                self.device
            )

            output = self.model(x)

            probabilities = torch.softmax(
                output,
                dim=1
            ).squeeze()

            pred_idx = torch.argmax(
                probabilities
            ).item()

            pred_emotion = self.classes[
                pred_idx
            ]

            confidence = (
                probabilities[pred_idx].item()
                * 100
            )

            self.update_ui(
                pred_emotion,
                confidence
            )

    def update_ui(self, emotion, confidence):

        color = EMOTION_COLORS.get(
            emotion,
            "#00e676"
        )

        emoji = EMOTION_EMOJIS.get(
            emotion,
            "🎤"
        )

        self.result_card.configure(
            fg_color=color
        )

        self.result_emoji.configure(
            text=emoji
        )

        self.result_text.configure(

            text=emotion.upper()
        )

        self.confidence_text.configure(

            text=f"Confidence: {confidence:.2f}%"
        )

        self.progress.set(
            confidence / 100
        )

        self.status_label.configure(
            text="Prediction Complete",
            text_color="#00e676"
        )

        self.log(
            f"Prediction: {emotion} | Confidence: {confidence:.2f}%"
        )

if __name__ == "__main__":

    root = ctk.CTk()

    app = SERApp(root)

    root.mainloop()