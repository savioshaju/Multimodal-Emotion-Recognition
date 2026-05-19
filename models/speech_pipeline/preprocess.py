import os
import shutil
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# paths

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(PIPELINE_DIR, "..", ".."))

DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset")
PROCESSED_DIR = os.path.join(PIPELINE_DIR, "processed_data")
METADATA_PATH = os.path.join(PIPELINE_DIR, "metadata.csv")

# Configure audio settings

SR = 16000
DURATION = 3
MAX_LEN = SR * DURATION
N_MELS = 128
N_MFCC = 40

# Map raw labels to standard emotion names

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

# Utility functions

def reset_output():
    """Clear old processed files."""
    if os.path.exists(PROCESSED_DIR):
        shutil.rmtree(PROCESSED_DIR)

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    if os.path.exists(METADATA_PATH):
        os.remove(METADATA_PATH)

def get_emotion_from_folder(folder):
    """Extract speaker_id, raw_emotion, and emotion from folder name."""
    folder_lower = folder.lower().strip()
    parts = folder_lower.split("_")

    if len(parts) < 2:
        return None, None, None

    speaker_id = parts[0]
    raw_emotion = "_".join(parts[1:])
    emotion = EMOTION_MAP.get(raw_emotion, None)

    return speaker_id, raw_emotion, emotion

def preprocess_audio(file_path):
    """Load, trim, pad, and normalize audio."""
    audio, _ = librosa.load(file_path, sr=SR)

    audio, _ = librosa.effects.trim(audio, top_db=30)

    if len(audio) > MAX_LEN:
        audio = audio[:MAX_LEN]
    else:
        audio = np.pad(audio, (0, MAX_LEN - len(audio)))

    audio = librosa.util.normalize(audio)

    return audio.astype(np.float32)

def normalize_feature(feature):
    """Apply z-score normalization."""
    return (feature - feature.mean()) / (feature.std() + 1e-8)

def extract_features(audio):
    """Extract 3-channel acoustic features."""
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

    stacked_features = np.stack(
        [mel_db, delta, mfcc],
        axis=0
    )

    return stacked_features.astype(np.float32)

def make_safe_filename(speaker_id, emotion, original_file):
    """Create a clean filename for processed features."""
    base_name = os.path.splitext(original_file)[0]
    base_name = base_name.replace(" ", "_").replace("-", "_")
    return f"{speaker_id}_{emotion}_{base_name}.npy"

# Main preprocessing logic

def main():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_PATH}\n"
            "Place the TESS dataset inside the project-root 'dataset/' folder."
        )

    reset_output()

    metadata = []
    all_files = []
    skipped = []

    for folder in os.listdir(DATASET_PATH):
        folder_path = os.path.join(DATASET_PATH, folder)

        if not os.path.isdir(folder_path):
            continue

        for file in os.listdir(folder_path):
            if file.lower().endswith(".wav"):
                all_files.append((folder, file))

    print(f"\nFound {len(all_files)} wav files\n")

    if len(all_files) == 0:
        raise ValueError(
            f"No .wav files found inside {DATASET_PATH}. "
            "Check your TESS folder structure."
        )

    for folder, file in tqdm(all_files, desc="Extracting speech features"):
        folder_path = os.path.join(DATASET_PATH, folder)
        file_path = os.path.join(folder_path, file)

        try:
            speaker_id, raw_emotion, emotion = get_emotion_from_folder(folder)

            if speaker_id is None or raw_emotion is None:
                skipped.append((file_path, "Invalid folder name"))
                continue

            if emotion is None:
                skipped.append((file_path, f"Unknown emotion: {raw_emotion}"))
                continue

            audio = preprocess_audio(file_path)
            features = extract_features(audio)

            feature_file = make_safe_filename(speaker_id, emotion, file)
            feature_path = os.path.join(PROCESSED_DIR, feature_file)

            np.save(feature_path, features)

            metadata.append({
                "file_path": file_path,
                "feature_path": feature_path,
                "emotion": emotion,
                "speaker_id": speaker_id,
                "raw_emotion": raw_emotion,
                "original_folder": folder,
                "original_file": file,
                "dataset": "TESS",
                "sample_rate": SR,
                "duration": DURATION,
                "feature_type": "mel_delta_mfcc",
                "feature_shape": str(features.shape),
            })

        except Exception as e:
            skipped.append((file_path, str(e)))

    df = pd.DataFrame(metadata)

    if len(df) == 0:
        raise ValueError(
            "No valid samples were processed. "
            "Check dataset folder names and .wav files."
        )

    df.to_csv(METADATA_PATH, index=False)

    print("\nPreprocessing Complete\n")
    print(f"Total valid samples: {len(df)}")

    print("\nClass counts:")
    print(df["emotion"].value_counts().reindex(CLASS_NAMES, fill_value=0))

    print("\nSpeaker IDs:")
    print(df["speaker_id"].value_counts())

    print("\nFeature shape examples:")
    print(df[["feature_path", "feature_shape"]].head())

    if skipped:
        print("\nSkipped files/folders:")
        for item in skipped[:20]:
            print(item)

        if len(skipped) > 20:
            print(f"... and {len(skipped) - 20} more skipped items")

    print(f"\nMetadata Saved -> {METADATA_PATH}")
    print(f"Processed Features Saved -> {PROCESSED_DIR}")

if __name__ == "__main__":
    main()