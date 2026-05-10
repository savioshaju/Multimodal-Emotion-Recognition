import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import font as tkfont
import torch
import torchaudio.transforms as T
import librosa
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("WARNING: 'sounddevice' module not found. Please install it using: pip install sounddevice")
    sd = None

# Import the model architecture from the local train.py
from train import UnifiedSERModel

class SERApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Speech Emotion Recognition")
        self.root.geometry("450x450")
        self.root.configure(bg="#1E1E1E")
        self.font = tkfont.Font(family="Helvetica", size=12)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Alphabetical classes based on LabelEncoder from training
        self.classes = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
        self.model = UnifiedSERModel(num_classes=len(self.classes)).to(self.device)
        
        # Robust pathing (works whether run from root or inside speech_pipeline/)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(base_dir, "saved_models", "best_unified_cnn.pth")
        
        self.setup_ui()
        self.load_model()

        # Audio feature extractors (Identical to training)
        self.mel_transform = T.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=256, n_mels=128)
        self.amplitude_to_db = T.AmplitudeToDB()
        self.compute_deltas = T.ComputeDeltas()
        self.max_len = 16000 * 3 # 3 seconds
        
        self.fs = 16000
        self.recording = False
        self.audio_data = None
        self.stream = None

    def setup_ui(self):
        lbl = tk.Label(self.root, text="Speech Emotion Recognition", font=("Helvetica", 16, "bold"), bg="#1E1E1E", fg="#FFFFFF")
        lbl.pack(pady=20)

        self.btn_upload = tk.Button(self.root, text="📂 Upload Audio File", font=self.font, bg="#4CAF50", fg="white", command=self.upload_audio, width=25)
        self.btn_upload.pack(pady=10)

        self.btn_record = tk.Button(self.root, text="🎙️ Hold to Record Live", font=self.font, bg="#f44336", fg="white", width=25)
        self.btn_record.pack(pady=10)
        
        if sd is not None:
            self.btn_record.bind("<ButtonPress-1>", self.start_recording)
            self.btn_record.bind("<ButtonRelease-1>", self.stop_recording)
        else:
            self.btn_record.config(state=tk.DISABLED, text="Install 'sounddevice' for Live Audio")

        self.status_lbl = tk.Label(self.root, text="Ready", font=("Helvetica", 10, "italic"), bg="#1E1E1E", fg="#AAAAAA")
        self.status_lbl.pack(pady=10)

        self.result_frame = tk.Frame(self.root, bg="#2D2D2D", bd=2, relief=tk.GROOVE)
        self.result_frame.pack(pady=20, fill=tk.X, padx=20)
        
        self.result_lbl = tk.Label(self.result_frame, text="Prediction: None", font=("Helvetica", 18, "bold"), bg="#2D2D2D", fg="#00E676")
        self.result_lbl.pack(pady=20)

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.model.eval()
                self.status_lbl.config(text="Model Loaded Successfully!")
            except Exception as e:
                self.status_lbl.config(text="Error loading weights.", fg="red")
                messagebox.showerror("Weight Error", f"Could not load weights. Architecture mismatch?\n{e}")
        else:
            self.status_lbl.config(text=f"Model not found.", fg="red")
            messagebox.showwarning("Model Missing", f"Could not find model weights at:\n{self.model_path}\nPlease run train.py first.")

    def upload_audio(self):
        filepath = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3 *.ogg")])
        if not filepath:
            return
        
        self.status_lbl.config(text=f"Processing...", fg="yellow")
        self.root.update()
        
        try:
            audio_np, _ = librosa.load(filepath, sr=self.fs)
            audio_np, _ = librosa.effects.trim(audio_np, top_db=30)
            self.predict(audio_np)
        except Exception as e:
            messagebox.showerror("Audio Error", str(e))
            self.status_lbl.config(text="Error processing audio.", fg="red")

    def start_recording(self, event):
        if self.btn_record['state'] == tk.DISABLED: return
        self.recording = True
        self.btn_record.config(bg="#d32f2f", text="🎙️ Recording... (Release to stop)")
        self.status_lbl.config(text="Listening...", fg="red")
        
        self.audio_data = []
        def callback(indata, frames, time, status):
            if self.recording:
                self.audio_data.append(indata.copy())

        self.stream = sd.InputStream(samplerate=self.fs, channels=1, callback=callback)
        self.stream.start()

    def stop_recording(self, event):
        if self.btn_record['state'] == tk.DISABLED or not self.recording: return
        self.recording = False
        
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            
        self.btn_record.config(bg="#f44336", text="🎙️ Hold to Record Live")
        self.status_lbl.config(text="Processing live audio...", fg="yellow")
        self.root.update()
        
        if len(self.audio_data) > 0:
            audio_np = np.concatenate(self.audio_data, axis=0).flatten()
            audio_np, _ = librosa.effects.trim(audio_np, top_db=30)
            
            if len(audio_np) < self.fs * 0.5: # Less than 0.5s of audio
                messagebox.showwarning("Warning", "Audio too short or silent.")
                self.status_lbl.config(text="Ready", fg="#AAAAAA")
                return
                
            self.predict(audio_np)
        else:
            self.status_lbl.config(text="Ready", fg="#AAAAAA")

    def predict(self, audio_np):
        waveform = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
        
        # Pad or truncate to max length
        if waveform.shape[1] > self.max_len:
            waveform = waveform[:, :self.max_len]
        else:
            pad_amount = self.max_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
            
        with torch.no_grad():
            # Feature extraction exactly as in train.py
            mel = self.mel_transform(waveform)
            mel_db = self.amplitude_to_db(mel)
            delta1 = self.compute_deltas(mel_db)
            delta2 = self.compute_deltas(delta1)
            
            x = torch.cat([mel_db, delta1, delta2], dim=0)
            
            # Instance Normalization
            mean = x.mean(dim=(1, 2), keepdim=True)
            std = x.std(dim=(1, 2), keepdim=True)
            x = (x - mean) / (std + 1e-8)
            
            # Add batch dimension and move to device
            x = x.unsqueeze(0).to(self.device)
            
            # Predict
            output = self.model(x)
            probabilities = torch.softmax(output, dim=1).squeeze()
            pred_idx = torch.argmax(probabilities).item()
            pred_emotion = self.classes[pred_idx]
            confidence = probabilities[pred_idx].item() * 100
            
            # Update UI
            emoji_map = {'anger':'😡', 'disgust':'🤢', 'fear':'😨', 'happiness':'😄', 'neutral':'😐', 'sadness':'😢', 'surprise':'😲'}
            emoji = emoji_map.get(pred_emotion, '')
            
            self.result_lbl.config(text=f"{emoji} {pred_emotion.capitalize()}\nConf: {confidence:.1f}%")
            self.status_lbl.config(text="Ready", fg="#AAAAAA")

if __name__ == "__main__":
    root = tk.Tk()
    app = SERApp(root)
    root.mainloop()
