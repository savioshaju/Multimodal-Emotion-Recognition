import os
import librosa
import numpy as np
import pandas as pd

from tqdm import tqdm

DATASET_PATH = "../../dataset"

OUTPUT_DIR = "processed_data"

SR = 16000

DURATION = 3

MAX_LEN = SR * DURATION

N_MELS = 128

os.makedirs(OUTPUT_DIR, exist_ok=True)

metadata = []

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
    "surprise": "surprise"
}

def preprocess_audio(path):

    audio, _ = librosa.load(
        path,
        sr=SR
    )

    audio, _ = librosa.effects.trim(
        audio,
        top_db=30
    )

    if len(audio) > MAX_LEN:

        audio = audio[:MAX_LEN]

    else:

        pad_amount = MAX_LEN - len(audio)

        audio = np.pad(
            audio,
            (0, pad_amount)
        )

    audio = librosa.util.normalize(audio)

    return audio

def extract_features(audio):

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

    delta1 = librosa.feature.delta(
        mel_db
    )

    delta2 = librosa.feature.delta(
        mel_db,
        order=2
    )

    stacked = np.stack(
        [mel_db, delta1, delta2],
        axis=0
    )

    mean = stacked.mean(
        axis=(1, 2),
        keepdims=True
    )

    std = stacked.std(
        axis=(1, 2),
        keepdims=True
    )

    stacked = (
        stacked - mean
    ) / (std + 1e-8)

    return stacked.astype(np.float32)

all_files = []

for folder in os.listdir(DATASET_PATH):

    folder_path = os.path.join(
        DATASET_PATH,
        folder
    )

    if os.path.isdir(folder_path):

        for file in os.listdir(folder_path):

            if file.endswith(".wav"):

                all_files.append(
                    (
                        folder,
                        file
                    )
                )

print(f"\nFound {len(all_files)} wav files.\n")

for folder, file in tqdm(all_files):

    folder_path = os.path.join(
        DATASET_PATH,
        folder
    )

    file_path = os.path.join(
        folder_path,
        file
    )

    try:

        parts = folder.lower().split("_")

        speaker_id = parts[0]

        raw_emotion = "_".join(parts[1:])

        raw_emotion = raw_emotion.lower()

        if (
            "pleasant" in raw_emotion
            or
            "surprise" in raw_emotion
        ):

            emotion = "surprise"

        else:

            emotion = EMOTION_MAP.get(
                raw_emotion,
                None
            )

        if emotion is None:

            continue

        audio = preprocess_audio(
            file_path
        )

        features = extract_features(
            audio
        )

        save_name = (
            f"{speaker_id}_{file[:-4]}.npy"
        )

        save_path = os.path.join(
            OUTPUT_DIR,
            save_name
        )

        np.save(
            save_path,
            features
        )

        metadata.append({

            "file_path": file_path,

            "feature_path": save_path,

            "emotion": emotion,

            "speaker_id": speaker_id,

            "dataset": "TESS"
        })

    except Exception as e:

        print(f"\nError processing: {file_path}")

        print(e)

df = pd.DataFrame(metadata)

df.to_csv(
    "metadata.csv",
    index=False
)

print("\nPreprocessing Complete\n")

print(
    df["emotion"].value_counts()
)

print("\nUnique Speakers:")

print(
    df["speaker_id"].unique()
)
print("\nMetadata saved to: metadata.csv")