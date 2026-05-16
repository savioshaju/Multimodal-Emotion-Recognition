import os
import re
import json
import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer, AutoModel


# =========================
# PATH CONFIGURATION
# =========================

MODEL_PATH = "saved_models/text_emotion_model.pth"
CONFIG_PATH = "saved_models/model_config.json"
OUTPUT_PATH = "metrics/batch_text_predictions.csv"


# =========================
# TEXT CLEANING
# =========================

def clean_text(text):
    text = str(text).strip()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


# =========================
# MODEL CLASS
# =========================

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

        pooled = self.mean_pooling(
            outputs.last_hidden_state,
            attention_mask
        )

        logits = self.classifier(pooled)
        return logits


# =========================
# LOAD MODEL
# =========================

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    model_name = config.get("model_name", "roberta-base")
    class_names = config.get(
        "classes",
        ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
    )
    max_length = int(config.get("max_length", 96))
    dropout_rate = float(config.get("dropout_rate", 0.30))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = TextEmotionModel(
        model_name=model_name,
        num_classes=len(class_names),
        dropout_rate=dropout_rate
    )

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model, tokenizer, class_names, max_length, device


# =========================
# PREDICT SINGLE TEXT
# =========================

def predict_text(text, model, tokenizer, class_names, max_length, device):
    cleaned = clean_text(text)

    encoded = tokenizer(
        cleaned,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.nn.functional.softmax(logits, dim=1)

    top_probs, top_indices = torch.topk(probs, k=3, dim=1)

    top3 = []
    for i in range(3):
        emotion = class_names[top_indices[0][i].item()]
        score = top_probs[0][i].item() * 100
        top3.append((emotion, score))

    prediction = top3[0][0]
    confidence = top3[0][1]
    second_confidence = top3[1][1]
    confidence_gap = confidence - second_confidence

    if confidence < 50 or confidence_gap < 8:
        final_label = "uncertain"
    else:
        final_label = prediction

    return {
        "text": text,
        "cleaned_text": cleaned,
        "final_label": final_label,
        "top1": top3[0][0],
        "top1_confidence": round(top3[0][1], 2),
        "top2": top3[1][0],
        "top2_confidence": round(top3[1][1], 2),
        "top3": top3[2][0],
        "top3_confidence": round(top3[2][1], 2),
        "confidence_gap": round(confidence_gap, 2)
    }


# =========================
# MAIN
# =========================

def main():
    print("\nLoading text emotion model...")
    model, tokenizer, class_names, max_length, device = load_model()

    print(f"Model loaded successfully on: {device}")
    print("\nEnter multiple sentences.")
    print("Type one sentence per line.")
    print("Type DONE and press Enter to run prediction.\n")

    sentences = []

    while True:
        line = input("> ").strip()

        if line.upper() == "DONE":
            break

        if line:
            sentences.append(line)

    if not sentences:
        print("\nNo sentences entered.")
        return

    results = []

    print("\nPredictions:\n")

    for idx, sentence in enumerate(sentences, start=1):
        result = predict_text(
            sentence,
            model,
            tokenizer,
            class_names,
            max_length,
            device
        )

        results.append(result)

        print(f"{idx}. Text: {result['text']}")
        print(f"   Final Label: {result['final_label']}")
        print(
            f"   Top-3: "
            f"{result['top1']} ({result['top1_confidence']}%) | "
            f"{result['top2']} ({result['top2_confidence']}%) | "
            f"{result['top3']} ({result['top3_confidence']}%)"
        )
        print(f"   Confidence Gap: {result['confidence_gap']}%\n")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved predictions to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()