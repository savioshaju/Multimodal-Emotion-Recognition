import os
import shutil
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm


PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(PIPELINE_DIR, "..", ".."))

DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset1")
METADATA_PATH = os.path.join(PIPELINE_DIR, "metadata.csv")
PROCESSED_DIR = os.path.join(PIPELINE_DIR, "processed_data")

TRAIN_CSV = os.path.join(DATASET_PATH, "train_sent_emo.csv")
DEV_CSV = os.path.join(DATASET_PATH, "dev_sent_emo.csv")
TEST_CSV = os.path.join(DATASET_PATH, "test_sent_emo.csv")

TRAIN_VIDEO_DIR = os.path.join(DATASET_PATH, "train", "train_splits")
DEV_VIDEO_DIR = os.path.join(DATASET_PATH, "dev", "dev_splits_complete")
TEST_VIDEO_DIR = os.path.join(DATASET_PATH, "test", "output_repeated_splits_test")

SR = 16000
DURATION = 3
MAX_LEN = SR * DURATION
N_MELS = 128
N_MFCC = 40

LABEL_MAP = {
    "anger": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "joy": "happiness",
    "neutral": "neutral",
    "sadness": "sadness",
    "surprise": "surprise",
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


def reset_outputs():
    if os.path.exists(METADATA_PATH):
        os.remove(METADATA_PATH)

    if os.path.exists(PROCESSED_DIR):
        shutil.rmtree(PROCESSED_DIR)

    os.makedirs(PROCESSED_DIR, exist_ok=True)


def normalize_feature(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = " ".join(text.split())

    return text.strip()


def find_column(df, possible_names):
    lower_map = {col.lower(): col for col in df.columns}

    for name in possible_names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    return None


def get_video_path(split_name, dialogue_id, utterance_id):
    file_name = f"dia{int(dialogue_id)}_utt{int(utterance_id)}.mp4"

    if split_name == "train":
        return os.path.join(TRAIN_VIDEO_DIR, file_name)

    if split_name == "dev":
        return os.path.join(DEV_VIDEO_DIR, file_name)

    if split_name == "test":
        return os.path.join(TEST_VIDEO_DIR, file_name)

    raise ValueError(f"Unknown split name: {split_name}")


def preprocess_audio(video_path):
    audio, _ = librosa.load(video_path, sr=SR, mono=True)

    if len(audio) == 0:
        raise ValueError("Empty audio loaded")

    audio, _ = librosa.effects.trim(audio, top_db=30)

    if len(audio) == 0:
        raise ValueError("Audio became empty after trimming")

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


def load_split(csv_path, split_name):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    utterance_col = find_column(df, ["Utterance", "utterance", "text"])
    emotion_col = find_column(df, ["Emotion", "emotion"])
    dialogue_col = find_column(df, ["Dialogue_ID", "dialogue_id", "DialogueId"])
    utterance_id_col = find_column(df, ["Utterance_ID", "utterance_id", "UtteranceId"])

    required = {
        "Utterance": utterance_col,
        "Emotion": emotion_col,
        "Dialogue_ID": dialogue_col,
        "Utterance_ID": utterance_id_col,
    }

    missing = [name for name, col in required.items() if col is None]

    if missing:
        raise ValueError(
            f"{csv_path} is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name}"):
        raw_emotion = str(row[emotion_col]).lower().strip()
        emotion = LABEL_MAP.get(raw_emotion)

        if emotion is None:
            continue

        dialogue_id = row[dialogue_col]
        utterance_id = row[utterance_id_col]
        text = clean_text(row[utterance_col])

        video_path = get_video_path(split_name, dialogue_id, utterance_id)

        if not os.path.exists(video_path):
            rows.append({
                "missing_video_path": video_path,
                "split": split_name,
                "dialogue_id": dialogue_id,
                "utterance_id": utterance_id,
                "raw_emotion": raw_emotion,
                "reason": "video_not_found",
            })
            continue

        try:
            audio = preprocess_audio(video_path)
            features = extract_audio_features(audio)

            save_name = f"{split_name}_dia{int(dialogue_id)}_utt{int(utterance_id)}.npy"
            feature_path = os.path.join(PROCESSED_DIR, save_name)

            np.save(feature_path, features)

            rows.append({
                "file_path": video_path,
                "feature_path": feature_path,
                "text": text,
                "emotion": emotion,
                "raw_emotion": raw_emotion,
                "dialogue_id": int(dialogue_id),
                "utterance_id": int(utterance_id),
                "split": split_name,
                "dataset": "MELD",
            })

        except Exception as e:
            rows.append({
                "missing_video_path": video_path,
                "split": split_name,
                "dialogue_id": dialogue_id,
                "utterance_id": utterance_id,
                "raw_emotion": raw_emotion,
                "reason": str(e),
            })

    return rows


def main():
    print(f"Using MELD dataset path: {DATASET_PATH}")

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset path not found: {DATASET_PATH}")

    reset_outputs()

    all_rows = []

    all_rows.extend(load_split(TRAIN_CSV, "train"))
    all_rows.extend(load_split(DEV_CSV, "dev"))
    all_rows.extend(load_split(TEST_CSV, "test"))

    valid_rows = [row for row in all_rows if "feature_path" in row]
    skipped_rows = [row for row in all_rows if "feature_path" not in row]

    df = pd.DataFrame(valid_rows)
    skipped_df = pd.DataFrame(skipped_rows)

    if len(df) == 0:
        raise ValueError("No valid MELD samples were processed. Check video paths and CSV columns.")

    df.to_csv(METADATA_PATH, index=False)

    if len(skipped_df) > 0:
        skipped_path = os.path.join(PIPELINE_DIR, "skipped_samples.csv")
        skipped_df.to_csv(skipped_path, index=False)

    print("\nMELD fusion preprocessing complete.")
    print(f"Total valid samples: {len(df)}")
    print(f"Total skipped samples: {len(skipped_df)}")

    print("\nSplit counts:")
    print(df["split"].value_counts().sort_index())

    print("\nClass counts:")
    print(df["emotion"].value_counts().sort_index())

    print("\nClass counts by split:")
    print(pd.crosstab(df["split"], df["emotion"]))

    print("\nSample rows:")
    print(df.head())

    print(f"\nMetadata saved to: {METADATA_PATH}")
    print(f"Processed features saved to: {PROCESSED_DIR}")

    if len(skipped_df) > 0:
        print(f"Skipped samples saved to: {os.path.join(PIPELINE_DIR, 'skipped_samples.csv')}")


if __name__ == "__main__":
    main()