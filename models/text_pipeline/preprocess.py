import os
import re
import json
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


# =========================
# PATH CONFIGURATION
# =========================

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(PIPELINE_DIR, "data")
PROCESSED_DIR = os.path.join(PIPELINE_DIR, "processed_data")

TEXT_PATH = os.path.join(DATASET_DIR, "dialogues_text.txt")
EMOTION_PATH = os.path.join(DATASET_DIR, "dialogues_emotion.txt")

METADATA_PATH = os.path.join(PROCESSED_DIR, "metadata.csv")
TRAIN_OUT = os.path.join(PROCESSED_DIR, "train.csv")
VAL_OUT = os.path.join(PROCESSED_DIR, "val.csv")
TEST_OUT = os.path.join(PROCESSED_DIR, "test.csv")
SPLIT_INFO_PATH = os.path.join(PROCESSED_DIR, "split_info.json")

os.makedirs(PROCESSED_DIR, exist_ok=True)


# =========================
# CONFIGURATION
# =========================

SEED = 42

EMOTION_MAP = {
    0: "neutral",
    1: "anger",
    2: "disgust",
    3: "fear",
    4: "happiness",
    5: "sadness",
    6: "surprise"
}

CLASS_NAMES = [
    "anger",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise"
]


# =========================
# TEXT CLEANING
# =========================

def clean_text(text):
    text = str(text).strip()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


# =========================
# FILE LOADING
# =========================

def load_lines(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    return [line.strip() for line in lines if line.strip()]


# =========================
# GROUP SPLIT
# =========================

def group_split(df):
    """
    Splits by dialogue_id to avoid data leakage.
    Same dialogue will never appear in more than one split.
    """

    splitter_1 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=SEED
    )

    train_idx, temp_idx = next(
        splitter_1.split(
            df,
            groups=df["dialogue_id"]
        )
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    splitter_2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=SEED
    )

    val_idx, test_idx = next(
        splitter_2.split(
            temp_df,
            groups=temp_df["dialogue_id"]
        )
    )

    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    return train_df, val_df, test_df


def check_dialogue_leakage(train_df, val_df, test_df):
    train_ids = set(train_df["dialogue_id"].unique())
    val_ids = set(val_df["dialogue_id"].unique())
    test_ids = set(test_df["dialogue_id"].unique())

    train_val_overlap = train_ids.intersection(val_ids)
    train_test_overlap = train_ids.intersection(test_ids)
    val_test_overlap = val_ids.intersection(test_ids)

    if train_val_overlap or train_test_overlap or val_test_overlap:
        raise ValueError("Dialogue-level leakage detected.")

    print("\nLeakage check passed: no dialogue_id overlap between train, validation, and test.")


def print_distribution(name, df):
    print(f"\n{name} samples: {len(df)}")
    print(df["emotion"].value_counts().reindex(CLASS_NAMES, fill_value=0))


# =========================
# MAIN
# =========================

def main():
    print("\nLoading DailyDialog files...")

    text_lines = load_lines(TEXT_PATH)
    emotion_lines = load_lines(EMOTION_PATH)

    print(f"Text dialogue lines: {len(text_lines)}")
    print(f"Emotion dialogue lines: {len(emotion_lines)}")

    if len(text_lines) != len(emotion_lines):
        raise ValueError(
            f"Mismatch: text lines={len(text_lines)}, emotion lines={len(emotion_lines)}"
        )

    rows = []
    skipped_dialogues = 0
    skipped_utterances = 0

    for dialogue_id, (text_line, emotion_line) in enumerate(zip(text_lines, emotion_lines)):
        utterances = [
            u.strip()
            for u in text_line.split("__eou__")
            if u.strip()
        ]

        emotion_ids = [
            e.strip()
            for e in emotion_line.split()
            if e.strip()
        ]

        if len(utterances) != len(emotion_ids):
            skipped_dialogues += 1
            continue

        for utterance_id, (utterance, emotion_id) in enumerate(zip(utterances, emotion_ids)):
            if not emotion_id.isdigit():
                skipped_utterances += 1
                continue

            emotion_id = int(emotion_id)

            if emotion_id not in EMOTION_MAP:
                skipped_utterances += 1
                continue

            text = clean_text(utterance)
            emotion = EMOTION_MAP[emotion_id]

            if len(text) <= 2:
                skipped_utterances += 1
                continue

            rows.append({
                "dialogue_id": dialogue_id,
                "utterance_id": utterance_id,
                "text": text,
                "emotion": emotion,
                "emotion_id": emotion_id
            })

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No valid samples created. Check DailyDialog file format.")

    df = df[df["emotion"].isin(CLASS_NAMES)].reset_index(drop=True)

    before_duplicates = len(df)
    df = df.drop_duplicates(subset=["dialogue_id", "utterance_id", "text", "emotion"]).reset_index(drop=True)
    after_duplicates = len(df)

    df.to_csv(METADATA_PATH, index=False)

    train_df, val_df, test_df = group_split(df)
    check_dialogue_leakage(train_df, val_df, test_df)

    train_df.to_csv(TRAIN_OUT, index=False)
    val_df.to_csv(VAL_OUT, index=False)
    test_df.to_csv(TEST_OUT, index=False)

    print("\nPreprocessing completed.")
    print(f"Total valid samples: {len(df)}")
    print(f"Skipped dialogues due to mismatch: {skipped_dialogues}")
    print(f"Skipped utterances: {skipped_utterances}")
    print(f"Duplicates removed: {before_duplicates - after_duplicates}")

    print_distribution("Full dataset", df)
    print_distribution("Train", train_df)
    print_distribution("Validation", val_df)
    print_distribution("Test", test_df)

    split_info = {
        "dataset": "DailyDialog",
        "split_type": "dialogue_id_group_split",
        "seed": SEED,
        "classes": CLASS_NAMES,
        "total_samples": int(len(df)),
        "train_samples": int(len(train_df)),
        "val_samples": int(len(val_df)),
        "test_samples": int(len(test_df)),
        "train_dialogues": int(train_df["dialogue_id"].nunique()),
        "val_dialogues": int(val_df["dialogue_id"].nunique()),
        "test_dialogues": int(test_df["dialogue_id"].nunique()),
        "leakage_check": "passed"
    }

    with open(SPLIT_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=4)

    print("\nSaved files:")
    print(METADATA_PATH)
    print(TRAIN_OUT)
    print(VAL_OUT)
    print(TEST_OUT)
    print(SPLIT_INFO_PATH)


if __name__ == "__main__":
    main()