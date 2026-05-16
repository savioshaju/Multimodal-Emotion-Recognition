import os
import pandas as pd
from tqdm import tqdm

DATASET_PATH = "../../dataset"
METADATA_PATH = "metadata.csv"

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
    "ps": "surprise"
}

def get_emotion_from_folder(folder):
    folder_lower = folder.lower()
    parts = folder_lower.split("_")
    speaker_id = parts[0]
    raw_emotion = "_".join(parts[1:])
    emotion = EMOTION_MAP.get(raw_emotion, None)
    return speaker_id, raw_emotion, emotion

def main():
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

        if not os.path.exists(file_path):
            continue

        try:
            speaker_id, raw_emotion, emotion = get_emotion_from_folder(folder)

            if emotion is None:
                skipped.append((file_path, raw_emotion))
                continue

            metadata.append({
                "file_path": file_path,
                "emotion": emotion,
                "speaker_id": speaker_id,
                "raw_emotion": raw_emotion,
                "original_folder": folder,
                "dataset": "TESS"
            })

        except Exception as e:
            print(f"\nError Processing: {file_path}")
            print(e)

    df = pd.DataFrame(metadata)
    df.to_csv(METADATA_PATH, index=False)

    print("\nPreprocessing Complete\n")
    print(f"Total valid samples: {len(df)}")
    print("\nClass counts:")
    print(df["emotion"].value_counts().sort_index())

    print("\nSpeaker IDs:")
    print(df["speaker_id"].unique())

    if skipped:
        print("\nSkipped folders/emotions:")
        for item in skipped[:20]:
            print(item)

    print("\nMetadata Saved -> metadata.csv")

if __name__ == "__main__":
    main()