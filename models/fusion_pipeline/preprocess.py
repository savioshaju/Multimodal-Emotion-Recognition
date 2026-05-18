import os
import glob
import shutil
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

# =========================
# PATH CONFIGURATION
# =========================

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(PIPELINE_DIR, "..", ".."))

DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset")
METADATA_PATH = os.path.join(PIPELINE_DIR, "metadata.csv")
PROCESSED_DIR = os.path.join(PIPELINE_DIR, "processed_data")

# =========================
# AUDIO CONFIGURATION
# =========================

SR = 16000
DURATION = 3
MAX_LEN = SR * DURATION
N_MELS = 128
N_MFCC = 40

# =========================
# EMOTION MAPPING
# =========================

EMOTION_MAP = {
    "angry": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happiness",
    "neutral": "neutral",
    "sad": "sadness",
    "pleasant_surprise": "surprise",
    "pleasant_surprised": "surprise",
    "pleasant": "surprise",
    "surprise": "surprise",
    "ps": "surprise",
}

CLASS_NAMES = [
    "anger",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise",
]


# =========================
# UTILITY FUNCTIONS
# =========================

def reset_outputs():
    if os.path.exists(METADATA_PATH):
        os.remove(METADATA_PATH)

    if os.path.exists(PROCESSED_DIR):
        shutil.rmtree(PROCESSED_DIR)

    os.makedirs(PROCESSED_DIR, exist_ok=True)


def normalize_feature(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def get_emotion_from_folder(folder_name):
    folder_lower = folder_name.lower()
    parts = folder_lower.split("_")

    if len(parts) < 2:
        return None, None, None

    speaker_id = parts[0]
    raw_emotion = "_".join(parts[1:])
    emotion = EMOTION_MAP.get(raw_emotion)

    return speaker_id, raw_emotion, emotion


def extract_text_from_filename(filename):
    """
    TESS filename examples:
    OAF_back_angry.wav
    YAF_ditch_ps.wav

    Extracted text:
    back
    ditch
    """

    name = os.path.splitext(filename)[0]
    parts = name.split("_")

    if len(parts) < 3:
        return "unknown"

    text_parts = parts[1:-1]

    if len(text_parts) == 0:
        return "unknown"

    text = " ".join(text_parts)
    text = text.replace("_", " ").replace("-", " ")
    text = text.lower().strip()

    if text == "":
        return "unknown"

    return text


def preprocess_audio(file_path):
    audio, _ = librosa.load(file_path, sr=SR)
    audio, _ = librosa.effects.trim(audio, top_db=30)

    if len(audio) > MAX_LEN:
        audio = audio[:MAX_LEN]
    else:
        audio = np.pad(audio, (0, MAX_LEN - len(audio)), mode="constant")

    audio = librosa.util.normalize(audio)

    return audio.astype(np.float32)


def extract_audio_features(audio):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_fft=1024,
        hop_length=256,
        n_mels=N_MELS,
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    delta = librosa.feature.delta(mel_db)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SR,
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
# MAIN PREPROCESSING
# =========================

def main():
    print(f"Scanning dataset at: {DATASET_PATH}")

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_PATH}\n"
            "Place the TESS dataset inside the project-root dataset/ folder."
        )

    reset_outputs()

    wav_files = glob.glob(
        os.path.join(DATASET_PATH, "**", "*.wav"),
        recursive=True,
    )

    if len(wav_files) == 0:
        raise FileNotFoundError(f"No .wav files found inside: {DATASET_PATH}")

    print(f"\nFound {len(wav_files)} wav files\n")

    metadata = []
    skipped = []

    for file_path in tqdm(wav_files, desc="Preprocessing fusion data"):
        original_file = os.path.basename(file_path)
        original_folder = os.path.basename(os.path.dirname(file_path))

        speaker_id, raw_emotion, emotion = get_emotion_from_folder(original_folder)

        if speaker_id not in ["oaf", "yaf"]:
            skipped.append((file_path, "Invalid speaker"))
            continue

        if emotion is None:
            skipped.append((file_path, f"Unknown emotion: {raw_emotion}"))
            continue

        try:
            text = extract_text_from_filename(original_file)

            audio = preprocess_audio(file_path)
            features = extract_audio_features(audio)

            save_name = f"{speaker_id}_{emotion}_{os.path.splitext(original_file)[0]}.npy"
            feature_path = os.path.join(PROCESSED_DIR, save_name)

            np.save(feature_path, features)

            metadata.append({
                "file_path": file_path,
                "feature_path": feature_path,
                "text": text,
                "emotion": emotion,
                "speaker_id": speaker_id,
                "raw_emotion": raw_emotion,
                "original_folder": original_folder,
                "original_file": original_file,
                "dataset": "TESS",
            })

        except Exception as e:
            skipped.append((file_path, str(e)))

    df = pd.DataFrame(metadata)

    if len(df) == 0:
        raise ValueError("No valid samples were created. Check the TESS folder structure.")

    df.to_csv(METADATA_PATH, index=False)

    print("\nFusion preprocessing complete.")
    print(f"Total valid samples: {len(df)}")

    print("\nClass counts:")
    print(df["emotion"].value_counts().sort_index())

    print("\nSpeaker counts:")
    print(df["speaker_id"].value_counts().sort_index())

    print("\nSample rows:")
    print(df.head())

    if skipped:
        print(f"\nSkipped {len(skipped)} files/folders.")
        print("First 10 skipped items:")
        for item in skipped[:10]:
            print(item)

    print(f"\nMetadata saved to: {METADATA_PATH}")
    print(f"Processed features saved to: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()