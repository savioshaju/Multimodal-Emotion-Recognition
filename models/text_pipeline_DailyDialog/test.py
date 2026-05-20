import os
import re
import json
import pandas as pd
import tkinter as tk
from tkinter import messagebox, ttk
import customtkinter as ctk
from PIL import Image
import warnings
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings("ignore")

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Paths

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(PIPELINE_DIR, "..", ".."))

OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "results", "text_pipeline_DailyDialog")

MODEL_PATH = os.path.join(PIPELINE_DIR, "saved_models", "text_emotion_model.pth")
CONFIG_PATH = os.path.join(PIPELINE_DIR, "saved_models", "model_config.json")

REPORT_TXT = os.path.join(OUTPUT_ROOT, "metrics", "classification_report.txt")
CM_CSV = os.path.join(OUTPUT_ROOT, "metrics", "confusion_matrix.csv")
METRICS_JSON = os.path.join(OUTPUT_ROOT, "metrics", "text_metrics.json")

PLOT_CM = os.path.join(OUTPUT_ROOT, "plots", "confusion_matrix.png")
PLOT_CURVE = os.path.join(OUTPUT_ROOT, "plots", "training_curve.png")

# Configuration

EMOTION_COLORS = {
    "anger": "#ef5350",
    "disgust": "#66bb6a",
    "fear": "#ab47bc",
    "happiness": "#ffee58",
    "neutral": "#90a4ae",
    "sadness": "#42a5f5",
    "surprise": "#ffa726"
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

# Preprocessing

def clean_text(text):
    text = str(text).strip()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

# Model

class TextEmotionModel(nn.Module):
    def __init__(self, model_name, num_classes, dropout_rate=0.30):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def mean_pooling(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled_output = self.mean_pooling(
            outputs.last_hidden_state,
            attention_mask
        )

        logits = self.classifier(pooled_output)
        return logits

# GUI

class TextEmotionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Text Emotion Recognition")
        self.root.geometry("1100x700")
        self.root.minsize(1000, 600)

        self.model = None
        self.tokenizer = None
        self.config = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.build_ui()
        self.load_model()

    def build_ui(self):
        self.header_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        self.header_frame.pack(fill="x", pady=(30, 20), padx=40)

        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text="Text Emotion Recognition",
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
            text="Loading...",
            text_color="#69f0ae",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.status_label.pack(padx=15, pady=5)

        self.body_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        self.body_frame.pack(fill="both", expand=True, padx=40, pady=(0, 20))

        self.body_frame.columnconfigure(0, weight=1)
        self.body_frame.columnconfigure(1, weight=1)
        self.body_frame.rowconfigure(0, weight=1)

        self.controls_card = ctk.CTkFrame(
            self.body_frame,
            corner_radius=20,
            fg_color="#212121"
        )
        self.controls_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        self.controls_title = ctk.CTkLabel(
            self.controls_card,
            text="Input Text",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.controls_title.pack(pady=(30, 10))

        self.input_textbox = ctk.CTkTextbox(
            self.controls_card,
            height=120,
            corner_radius=10,
            font=ctk.CTkFont(size=14)
        )
        self.input_textbox.pack(padx=30, pady=(0, 20), fill="x")

        self.predict_btn = ctk.CTkButton(
            self.controls_card,
            text="Predict Emotion",
            height=50,
            corner_radius=10,
            font=ctk.CTkFont(size=16, weight="bold"),
            command=self.predict_emotion
        )
        self.predict_btn.pack(padx=30, pady=10, fill="x")

        self.btn_row = ctk.CTkFrame(self.controls_card, fg_color="transparent")
        self.btn_row.pack(padx=30, pady=10, fill="x")

        self.clear_btn = ctk.CTkButton(
            self.btn_row,
            text="Clear Input",
            height=40,
            corner_radius=10,
            fg_color="#424242",
            hover_color="#616161",
            command=self.clear_input
        )
        self.clear_btn.pack(fill="x")

        self.reports_title = ctk.CTkLabel(
            self.controls_card,
            text="Reports & Metrics",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.reports_title.pack(pady=(20, 10))

        self.reports_grid = ctk.CTkFrame(self.controls_card, fg_color="transparent")
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
            text="💬",
            font=ctk.CTkFont(size=120)
        )
        self.result_emoji.pack(pady=(50, 20), expand=True)

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
            width=350,
            height=15,
            progress_color="#00e676",
            fg_color="#37474f"
        )
        self.progress.pack(pady=(0, 50))
        self.progress.set(0)

    def set_status(self, text, badge_color="#1b5e20", text_color="#69f0ae"):
        self.status_label.configure(text=text, text_color=text_color)
        self.status_badge.configure(fg_color=badge_color)

    def load_model(self):
        if not os.path.exists(MODEL_PATH):
            messagebox.showerror(
                "Error",
                f"Model file not found:\n{MODEL_PATH}\n\nRun train.py first."
            )
            self.set_status("Model Missing", badge_color="#b71c1c", text_color="#ff5252")
            self.predict_btn.configure(state="disabled")
            return

        if not os.path.exists(CONFIG_PATH):
            messagebox.showerror(
                "Error",
                f"Config file not found:\n{CONFIG_PATH}\n\nRun train.py first."
            )
            self.set_status("Config Missing", badge_color="#b71c1c", text_color="#ff5252")
            self.predict_btn.configure(state="disabled")
            return

        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                self.config = json.load(f)

            self.model_name = self.config.get("model_name", "roberta-base")
            self.class_names = self.config.get(
                "classes",
                ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
            )
            self.max_length = int(self.config.get("max_length", 96))
            dropout_rate = float(self.config.get("dropout_rate", 0.30))

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            self.model = TextEmotionModel(
                self.model_name,
                len(self.class_names),
                dropout_rate=dropout_rate
            )

            checkpoint = torch.load(
                MODEL_PATH,
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

        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load model:\n{e}")
            self.set_status("Model Error", badge_color="#b71c1c", text_color="#ff5252")
            self.predict_btn.configure(state="disabled")

    def predict_emotion(self):
        if self.model is None:
            return

        user_input = self.input_textbox.get("1.0", "end-1c").strip()

        if not user_input:
            messagebox.showwarning("Input Error", "Please enter some text to predict.")
            return

        cleaned = clean_text(user_input)

        if not cleaned:
            messagebox.showwarning("Input Error", "Input is empty after cleaning.")
            return

        try:
            self.set_status("Processing...", badge_color="#f57f17", text_color="#ffee58")
            self.root.update()

            encoded = self.tokenizer(
                cleaned,
                add_special_tokens=True,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids, attention_mask)
                probs = torch.nn.functional.softmax(logits, dim=1)

            top_probs, top_indices = torch.topk(probs, k=3, dim=1)

            top3_results = []

            for i in range(3):
                emotion = self.class_names[top_indices[0][i].item()]
                score = top_probs[0][i].item() * 100
                top3_results.append((emotion, score))

            prediction = top3_results[0][0]
            confidence = top3_results[0][1]

            second_confidence = top3_results[1][1]
            confidence_gap = confidence - second_confidence

            is_low_confidence = confidence < 60 or confidence_gap < 10

            self.update_ui(
                prediction,
                confidence,
                top3_results,
                is_low_confidence
            )

        except Exception as e:
            messagebox.showerror("Prediction Error", f"Failed to predict:\n{e}")
            self.set_status("Prediction Error", badge_color="#b71c1c", text_color="#ff5252")

    def update_ui(self, emotion, confidence, top3_results=None, is_low_confidence=False):
        color = EMOTION_COLORS.get(emotion, "#00e676")
        emoji = EMOTION_EMOJIS.get(emotion, "💬")

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
        self.input_textbox.delete("1.0", "end")
        self.result_emoji.configure(text="💬")
        self.result_text.configure(text="Ready for Input", text_color="white")
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
                tree.column(col, anchor="w" if col == columns[0] else "center", width=300)
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
        if not os.path.exists(REPORT_TXT):
            messagebox.showinfo(
                "Not Found",
                f"Classification Report not found at:\n{REPORT_TXT}"
            )
            return

        try:
            with open(REPORT_TXT, "r", encoding="utf-8") as f:
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
                raise ValueError("Could not parse the classification report.")

            df = pd.DataFrame(data, columns=columns)
            self.create_table_popup("Classification Report", df, "classification")

        except Exception as e:
            messagebox.showerror(
                "Parse Error",
                f"Failed to load classification report:\n{e}"
            )

    def show_confusion_matrix(self):
        if not os.path.exists(CM_CSV):
            messagebox.showinfo(
                "Not Found",
                f"Confusion Matrix not found at:\n{CM_CSV}"
            )
            return

        try:
            df = pd.read_csv(CM_CSV)
            df = df.fillna("")
            self.create_table_popup("Confusion Matrix", df, "confusion")

        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Failed to load {CM_CSV}:\n{e}"
            )

    def show_metrics_summary(self):
        if not os.path.exists(METRICS_JSON):
            messagebox.showinfo(
                "Not Found",
                f"Metrics Summary not found at:\n{METRICS_JSON}"
            )
            return

        try:
            with open(METRICS_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)

            rows = []

            for key, value in data.items():
                metric_name = key.replace("_", " ").title()
                metric_value = self.safe_value_to_string(value)
                rows.append([metric_name, metric_value])

            df = pd.DataFrame(rows, columns=["Metric", "Value"])

            self.create_table_popup("Metrics Summary", df, "summary")

        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Failed to load {METRICS_JSON}:\n{e}"
            )

    def show_plots(self):
        if not os.path.exists(PLOT_CM) and not os.path.exists(PLOT_CURVE):
            messagebox.showinfo(
                "Not Found",
                "No plots found in the plots/ directory."
            )
            return

        top = ctk.CTkToplevel(self.root)
        top.title("View Plots")
        top.geometry("1000x800")
        top.transient(self.root)

        title_label = ctk.CTkLabel(
            top,
            text="Text Dataset & Evaluation Plots",
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

        if os.path.exists(PLOT_CURVE):
            add_image(scroll_frame, "Training Curve", PLOT_CURVE)

        if os.path.exists(PLOT_CM):
            add_image(scroll_frame, "Confusion Matrix", PLOT_CM)

        close_btn = ctk.CTkButton(
            top,
            text="Close",
            command=top.destroy,
            width=150,
            height=40
        )
        close_btn.pack(pady=(20, 20))


if __name__ == "__main__":
    root = ctk.CTk()
    app = TextEmotionApp(root)
    root.mainloop()