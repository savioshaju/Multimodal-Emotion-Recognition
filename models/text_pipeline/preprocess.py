import os
import pandas as pd
from tqdm import tqdm

# =========================
# PATH CONFIGURATION
# =========================

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(PIPELINE_DIR, "..", ".."))

DATASET_PATH  = os.path.join(PROJECT_ROOT, "dataset")
METADATA_PATH = os.path.join(PIPELINE_DIR, "metadata.csv")

EMOTION_MAP = {
    "angry":              "anger",
    "disgust":            "disgust",
    "fear":               "fear",
    "happy":              "happiness",
    "neutral":            "neutral",
    "sad":                "sadness",
    "pleasant_surprise":  "surprise",
    "pleasant_surprised": "surprise",
    "pleasant":           "surprise",
    "surprise":           "surprise",
    "ps":                 "surprise",
}

CLASS_NAMES = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]


def get_emotion_from_folder(folder):
    folder_lower = folder.lower()
    parts = folder_lower.split("_")

    speaker_id = parts[0]
    raw_emotion = "_".join(parts[1:])

    emotion = EMOTION_MAP.get(raw_emotion, None)
    return speaker_id, raw_emotion, emotion


def extract_text_from_filename(filename):
    """
    TESS filenames are usually like:
    OAF_back_angry.wav
    YAF_ditch_ps.wav

    Here:
    speaker = OAF/YAF
    text/transcript word = back/ditch/etc.
    emotion = angry/ps/etc.
    """

    name = os.path.splitext(filename)[0]
    parts = name.split("_")

    if len(parts) < 3:
        return None

    # Remove speaker prefix and emotion suffix
    text_parts = parts[1:-1]

    if len(text_parts) == 0:
        return None

    text = " ".join(text_parts)
    text = text.replace("-", " ").replace("_", " ")
    text = text.lower().strip()

    return text


def main():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_PATH}\n"
            "Place the TESS dataset under the project-root 'dataset/' folder."
        )

    if os.path.exists(METADATA_PATH):
        os.remove(METADATA_PATH)

    metadata = []
    all_files = []

    for folder in os.listdir(DATASET_PATH):
        folder_path = os.path.join(DATASET_PATH, folder)

        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.lower().endswith(".wav"):
                    all_files.append((folder, file))

    print(f"\nFound {len(all_files)} wav files\n")

    skipped = []

    for folder, file in tqdm(all_files):
        folder_path = os.path.join(DATASET_PATH, folder)
        file_path = os.path.join(folder_path, file)

        try:
            speaker_id, raw_emotion, emotion = get_emotion_from_folder(folder)
            text = extract_text_from_filename(file)

            if emotion is None:
                skipped.append((file_path, "Unknown emotion", raw_emotion))
                continue

            if text is None or text == "":
                skipped.append((file_path, "Text extraction failed", raw_emotion))
                continue

            metadata.append({
                "file_path": file_path,
                "text": text,
                "emotion": emotion,
                "speaker_id": speaker_id,
                "raw_emotion": raw_emotion,
                "original_folder": folder,
                "original_file": file,
                "dataset": "TESS",
            })

        except Exception as e:
            print(f"\nError Processing: {file_path}")
            print(e)

    df = pd.DataFrame(metadata)

    if len(df) == 0:
        raise ValueError("No valid text samples found. Check your TESS folder structure.")

    df.to_csv(METADATA_PATH, index=False)

    print("\nText Preprocessing Complete\n")
    print(f"Total valid samples: {len(df)}")

    print("\nClass counts:")
    print(df["emotion"].value_counts().sort_index())

    print("\nSpeaker IDs:")
    print(df["speaker_id"].unique())

    print("\nSample rows:")
    print(df[["text", "emotion", "speaker_id", "original_file"]].head(10))

    if skipped:
        print("\nSkipped files:")
        for item in skipped[:20]:
            print(item)

    print(f"\nMetadata Saved -> {METADATA_PATH}")


if __name__ == "__main__":
    main()