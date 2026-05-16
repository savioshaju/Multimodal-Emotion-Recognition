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
from transformers import AutoFeatureExtractor, AutoModel

try:
    import sounddevice as sd
except:
    sd = None

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

CLASS_NAMES = [
    "anger",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise"
]

class TransformerSERModel(nn.Module):
    def __init__(self, num_classes, model_name="microsoft/wavlm-base"):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        
        hidden_size = self.transformer.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        outputs = self.transformer(x)
        hidden_states = outputs.last_hidden_state
        pooled_output = torch.mean(hidden_states, dim=1)
        logits = self.classifier(pooled_output)
        return logits

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
        self.model_name = "microsoft/wavlm-base"
        
        self.recording = False
        self.audio_data = []
        self.stream = None

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(self.base_dir, "saved_models", "best_model.pth")
        self.config_path = os.path.join(self.base_dir, "saved_models", "model_config.json")

        self.load_config()

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
        self.model = TransformerSERModel(num_classes=len(self.classes), model_name=self.model_name).to(self.device)

        self.build_ui()
        self.load_model()

    def load_config(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    config = json.load(f)
                self.classes = config.get("classes", CLASS_NAMES)
                self.fs = config.get("sr", SR)
                self.max_len = self.fs * config.get("duration", DURATION)
                self.model_name = config.get("model_name", "microsoft/wavlm-base")
            except Exception as e:
                print(f"Error loading config: {e}")

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

        # Reports section
        self.reports_title = ctk.CTkLabel(
            self.controls_card,
            text="Reports & Metrics",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.reports_title.pack(pady=(20, 10))
        
        self.reports_grid = ctk.CTkFrame(self.controls_card, fg_color="transparent")
        self.reports_grid.pack(padx=30, pady=(0, 20), fill="x")
        
        ctk.CTkButton(self.reports_grid, text="Classification Report", height=35, command=self.show_classification_report).grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        ctk.CTkButton(self.reports_grid, text="Confusion Matrix", height=35, command=self.show_confusion_matrix).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ctk.CTkButton(self.reports_grid, text="Metrics Summary", height=35, command=self.show_metrics_summary).grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        ctk.CTkButton(self.reports_grid, text="View Plots", height=35, command=self.show_plots).grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        
        self.reports_grid.columnconfigure(0, weight=1)
        self.reports_grid.columnconfigure(1, weight=1)

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
            audio_np = preprocess_audio_array(audio_np, self.fs, self.max_len)
            
            inputs = self.feature_extractor(
                audio_np, 
                sampling_rate=self.fs, 
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_len,
                truncation=True
            )
            
            x = inputs["input_values"].to(self.device)

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

    # --- Reports Viewing ---

    def format_cell_value(self, value):
        if pd.isna(value):
            return ""
        try:
            float_val = float(value)
            if float_val.is_integer():
                return str(int(float_val))
            return f"{float_val:.4f}"
        except (ValueError, TypeError):
            return str(value)

    def show_csv_table(self, title, csv_path, table_type):
        filepath = os.path.join(self.base_dir, csv_path)
        if not os.path.exists(filepath):
            messagebox.showinfo("Not Found", f"{title} not found at {filepath}")
            return
            
        try:
            df = pd.read_csv(filepath)
            # Fill NaNs for better rendering
            df = df.fillna("")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load {csv_path}:\n{e}")
            return

        top = ctk.CTkToplevel(self.root)
        top.title(title)
        top.geometry("900x600")
        top.transient(self.root)
        
        title_label = ctk.CTkLabel(top, text=title, font=ctk.CTkFont(size=24, weight="bold"))
        title_label.pack(pady=(20, 10))

        # Table frame
        table_frame = ctk.CTkFrame(top, corner_radius=10)
        table_frame.pack(fill="both", expand=True, padx=30, pady=(10, 20))
        
        # Style
        style = ttk.Style(top)
        style.theme_use("default")
        style.configure("Treeview", 
                        background="#2b2b2b", 
                        foreground="white", 
                        rowheight=35, 
                        fieldbackground="#2b2b2b",
                        font=("Helvetica", 12))
        style.map('Treeview', background=[('selected', '#1f538d')])
        style.configure("Treeview.Heading", 
                        background="#1f538d", 
                        foreground="white", 
                        font=("Helvetica", 13, "bold"))
        
        # Scrollbars
        y_scroll = ttk.Scrollbar(table_frame, orient="vertical")
        y_scroll.pack(side="right", fill="y")
        
        x_scroll = ttk.Scrollbar(table_frame, orient="horizontal")
        x_scroll.pack(side="bottom", fill="x")

        tree = ttk.Treeview(table_frame, style="Treeview", yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        tree.pack(side="left", fill="both", expand=True)
        
        y_scroll.config(command=tree.yview)
        x_scroll.config(command=tree.xview)
        
        # Columns setup
        columns = list(df.columns)
        tree["columns"] = columns
        tree["show"] = "headings"
        
        for col in columns:
            tree.heading(col, text=str(col))
            # Adjust column width and alignment based on table type
            if table_type == "summary":
                tree.column(col, anchor="w" if col == columns[0] else "center", width=250)
            elif table_type == "classification":
                tree.column(col, anchor="w" if col == columns[0] else "center", width=140)
            else: # confusion matrix
                tree.column(col, anchor="center", width=120)
                
        # Rows setup
        for _, row in df.iterrows():
            formatted_row = [self.format_cell_value(val) for val in row]
            tree.insert("", "end", values=formatted_row)
            
        # Close button
        close_btn = ctk.CTkButton(top, text="Close", command=top.destroy, width=150, height=40)
        close_btn.pack(pady=(0, 20))

    def show_classification_report(self):
        self.show_csv_table("Classification Report", "results/classification_report.csv", "classification")

    def show_confusion_matrix(self):
        self.show_csv_table("Confusion Matrix", "results/confusion_matrix.csv", "confusion")

    def show_metrics_summary(self):
        filepath = os.path.join(self.base_dir, "results", "summary.csv")

        if not os.path.exists(filepath):
            messagebox.showinfo("Not Found", f"Metrics Summary not found at:\n{filepath}")
            return

        try:
            df = pd.read_csv(filepath).fillna("")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load summary.csv:\n{e}")
            return

        top = ctk.CTkToplevel(self.root)
        top.title("Metrics Summary")
        top.geometry("850x600")
        top.transient(self.root)
        top.grab_set()

        title_label = ctk.CTkLabel(
            top,
            text="Metrics Summary",
            font=ctk.CTkFont(size=26, weight="bold")
        )
        title_label.pack(pady=(22, 12))

        main_frame = ctk.CTkFrame(top, corner_radius=18, fg_color="#1f1f1f")
        main_frame.pack(fill="both", expand=True, padx=25, pady=(0, 20))

        cards_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        cards_frame.pack(fill="x", padx=20, pady=(20, 10))

        def clean_metric_name(name):
            return str(name).replace("_", " ").replace("-", " ").title()

        def clean_metric_value(value):
            try:
                value = float(value)
                if value <= 1:
                    return f"{value * 100:.2f}%"
                return f"{value:.2f}"
            except Exception:
                return str(value)

        def find_metric_value(possible_names):
            lower_columns = {str(col).lower().strip(): col for col in df.columns}

            for name in possible_names:
                key = name.lower().strip()
                if key in lower_columns:
                    val = df[lower_columns[key]].iloc[0]
                    return clean_metric_value(val)

            if df.shape[1] >= 2:
                metric_col = df.columns[0]
                value_col = df.columns[1]

                for _, row in df.iterrows():
                    metric_name = str(row[metric_col]).lower().strip()
                    for name in possible_names:
                        if name.lower().strip() in metric_name:
                            return clean_metric_value(row[value_col])

            return "--"

        summary_items = [
            ("Accuracy", find_metric_value(["accuracy", "test_accuracy"])),
            ("Macro F1", find_metric_value(["macro_f1", "macro f1", "test_macro_f1"])),
            ("Weighted F1", find_metric_value(["weighted_f1", "weighted f1", "test_weighted_f1"])),
            ("Samples", find_metric_value(["total_samples", "samples", "test_samples"]))
        ]

        for i, (metric, value) in enumerate(summary_items):
            card = ctk.CTkFrame(cards_frame, corner_radius=14, fg_color="#263238")
            card.grid(row=0, column=i, padx=8, pady=5, sticky="nsew")

            value_label = ctk.CTkLabel(
                card,
                text=value,
                font=ctk.CTkFont(size=22, weight="bold"),
                text_color="#64b5f6"
            )
            value_label.pack(pady=(16, 4), padx=12)

            metric_label = ctk.CTkLabel(
                card,
                text=metric,
                font=ctk.CTkFont(size=13),
                text_color="gray75"
            )
            metric_label.pack(pady=(0, 14), padx=12)

            cards_frame.columnconfigure(i, weight=1)

        table_title = ctk.CTkLabel(
            main_frame,
            text="Detailed Summary",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        table_title.pack(anchor="w", padx=25, pady=(12, 8))

        table_container = ctk.CTkFrame(main_frame, corner_radius=12, fg_color="#151515")
        table_container.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        style = ttk.Style(top)
        style.theme_use("default")

        style.configure(
            "Summary.Treeview",
            background="#151515",
            foreground="#ffffff",
            fieldbackground="#151515",
            rowheight=34,
            borderwidth=1,
            relief="solid",
            font=("Consolas", 12)
        )

        style.configure(
            "Summary.Treeview.Heading",
            background="#1565c0",
            foreground="#ffffff",
            relief="solid",
            font=("Helvetica", 12, "bold")
        )

        style.map(
            "Summary.Treeview",
            background=[("selected", "#0d47a1")],
            foreground=[("selected", "#ffffff")]
        )

        table_frame = tk.Frame(table_container, bg="#151515")
        table_frame.pack(fill="both", expand=True, padx=12, pady=12)

        y_scroll = ttk.Scrollbar(table_frame, orient="vertical")
        x_scroll = ttk.Scrollbar(table_frame, orient="horizontal")

        tree = ttk.Treeview(
            table_frame,
            style="Summary.Treeview",
            yscrollcommand=y_scroll.set,
            xscrollcommand=x_scroll.set
        )

        y_scroll.config(command=tree.yview)
        x_scroll.config(command=tree.xview)

        tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")

        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        columns = list(df.columns)
        tree["columns"] = columns
        tree["show"] = "headings"

        for col in columns:
            col_name = str(col)
            lower_col = col_name.lower()

            if any(key in lower_col for key in ["metric", "class", "label", "emotion", "name"]):
                anchor = "w"
                width = 260
            else:
                anchor = "center"
                width = 160

            tree.heading(
                col,
                text=clean_metric_name(col_name),
                anchor="center"
            )
            tree.column(
                col,
                anchor=anchor,
                width=width,
                minwidth=120,
                stretch=True
            )

        for _, row in df.iterrows():
            values = []
            for col in columns:
                value = row[col]
                try:
                    if pd.isna(value):
                        value = ""
                except Exception:
                    pass

                try:
                    num = float(value)
                    if num.is_integer():
                        value = str(int(num))
                    else:
                        value = f"{num:.4f}"
                except Exception:
                    value = str(value)

                values.append(value)

            tree.insert("", "end", values=values)

        close_btn = ctk.CTkButton(
            top,
            text="Close",
            width=140,
            height=38,
            corner_radius=10,
            fg_color="#424242",
            hover_color="#616161",
            command=top.destroy
        )
        close_btn.pack(pady=(0, 20))

    def show_plots(self):
        plot_cm = os.path.join(self.base_dir, "plots/confusion_matrix.png")
        plot_curve = os.path.join(self.base_dir, "plots/training_curve.png")
        
        if not os.path.exists(plot_cm) and not os.path.exists(plot_curve):
            messagebox.showinfo("Not Found", "No plots found in the plots/ directory.")
            return
            
        top = ctk.CTkToplevel(self.root)
        top.title("View Plots")
        top.geometry("1000x800")
        top.transient(self.root)
        
        title_label = ctk.CTkLabel(top, text="Training & Evaluation Plots", font=ctk.CTkFont(size=24, weight="bold"))
        title_label.pack(pady=(20, 10))
        
        scroll_frame = ctk.CTkScrollableFrame(top, fg_color="transparent")
        scroll_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        def add_image(frame, title, path):
            lbl_title = ctk.CTkLabel(frame, text=title, font=ctk.CTkFont(size=18, weight="bold"))
            lbl_title.pack(pady=(30, 10))
            
            try:
                img = Image.open(path)
                # Keep aspect ratio, max width 850
                w, h = img.size
                ratio = 850 / w if w > 850 else 1
                new_size = (int(w * ratio), int(h * ratio))
                
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=new_size)
                img_lbl = ctk.CTkLabel(frame, text="", image=ctk_img)
                img_lbl.pack(pady=(0, 20))
            except Exception as e:
                err_lbl = ctk.CTkLabel(frame, text=f"Error loading image: {e}", text_color="red")
                err_lbl.pack()
                
        if os.path.exists(plot_curve):
            add_image(scroll_frame, "Training Curve", plot_curve)
            
        if os.path.exists(plot_cm):
            add_image(scroll_frame, "Confusion Matrix", plot_cm)
            
        close_btn = ctk.CTkButton(top, text="Close", command=top.destroy, width=150, height=40)
        close_btn.pack(pady=(20, 20))


def predict_cli(audio_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "saved_models", "best_model.pth")
    config_path = os.path.join(base_dir, "saved_models", "model_config.json")
    
    classes = CLASS_NAMES
    fs = SR
    duration = DURATION
    model_name = "microsoft/wavlm-base"
    
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            classes = config.get("classes", CLASS_NAMES)
            fs = config.get("sr", SR)
            duration = config.get("duration", DURATION)
            model_name = config.get("model_name", "microsoft/wavlm-base")
        except Exception as e:
            print(f"Warning: Failed to load config file: {e}")
            
    max_len = fs * duration
        
    if not os.path.exists(model_path):
        print("Model file not found. Please train first.")
        return
        
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        return
        
    model = TransformerSERModel(num_classes=len(classes), model_name=model_name).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    
    try:
        audio_np, _ = librosa.load(audio_path, sr=fs, mono=True)
        audio_np = preprocess_audio_array(audio_np, fs, max_len)
        
        inputs = feature_extractor(
            audio_np, 
            sampling_rate=fs, 
            return_tensors="pt",
            padding="max_length",
            max_length=max_len,
            truncation=True
        )
        
        x = inputs["input_values"].to(device)

        with torch.no_grad():
            output = model(x)
            probabilities = torch.softmax(output, dim=1).squeeze()
            
            pred_idx = torch.argmax(probabilities).item()
            pred_emotion = classes[pred_idx]
            confidence = probabilities[pred_idx].item() * 100

        print(f"\nPredicted Emotion: {pred_emotion.upper()}")
        print(f"Confidence Score: {confidence:.2f}%")
        print("\nConfidence for all classes:")
        for i, cls in enumerate(classes):
            print(f"{cls}: {probabilities[i].item()*100:.2f}%")
            
    except Exception as e:
        print(f"Error processing audio: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict_cli(sys.argv[1])
    else:
        root = ctk.CTk()
        app = SERApp(root)
        root.mainloop()