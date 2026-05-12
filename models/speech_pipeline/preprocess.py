import os
import shutil
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

DATASET_PATH="../../dataset"
OUTPUT_DIR="processed_data"
METADATA_PATH="metadata.csv"

SR=16000
DURATION=3
MAX_LEN=SR*DURATION
N_MELS=128

EMOTION_MAP={
    "angry":"anger",
    "disgust":"disgust",
    "fear":"fear",
    "happy":"happiness",
    "neutral":"neutral",
    "sad":"sadness",
    "pleasant_surprise":"surprise",
    "pleasant_surprised":"surprise",
    "pleasant":"surprise",
    "surprise":"surprise",
    "ps":"surprise"
}

def reset_output():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR,exist_ok=True)
    if os.path.exists(METADATA_PATH):
        os.remove(METADATA_PATH)

def preprocess_audio(path):
    audio,_=librosa.load(path,sr=SR)
    audio,_=librosa.effects.trim(audio,top_db=30)
    if len(audio)>MAX_LEN:
        audio=audio[:MAX_LEN]
    else:
        audio=np.pad(audio,(0,MAX_LEN-len(audio)))
    audio=librosa.util.normalize(audio)
    return audio.astype(np.float32)

def normalize_feature(x):
    return (x-x.mean())/(x.std()+1e-8)

def extract_features(audio):
    mel=librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_fft=1024,
        hop_length=256,
        n_mels=N_MELS
    )

    mel_db=librosa.power_to_db(
        mel,
        ref=np.max
    )

    delta1=librosa.feature.delta(
        mel_db
    )

    mfcc=librosa.feature.mfcc(
        y=audio,
        sr=SR,
        n_mfcc=40,
        n_fft=1024,
        hop_length=256
    )

    mfcc=librosa.util.fix_length(
        mfcc,
        size=mel_db.shape[1],
        axis=1
    )

    mfcc=np.repeat(
        mfcc,
        4,
        axis=0
    )[:128]

    mel_db=normalize_feature(mel_db)
    delta1=normalize_feature(delta1)
    mfcc=normalize_feature(mfcc)

    stacked=np.stack(
        [mel_db,delta1,mfcc],
        axis=0
    )

    return stacked.astype(np.float32)

def get_emotion_from_folder(folder):
    folder_lower=folder.lower()
    parts=folder_lower.split("_")
    speaker_id=parts[0]
    raw_emotion="_".join(parts[1:])
    emotion=EMOTION_MAP.get(raw_emotion,None)
    return speaker_id,raw_emotion,emotion

def main():
    reset_output()

    metadata=[]
    all_files=[]

    for folder in os.listdir(DATASET_PATH):
        folder_path=os.path.join(DATASET_PATH,folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.lower().endswith(".wav"):
                    all_files.append((folder,file))

    print(f"\nFound {len(all_files)} wav files\n")

    skipped=[]

    for folder,file in tqdm(all_files):
        folder_path=os.path.join(DATASET_PATH,folder)
        file_path=os.path.join(folder_path,file)

        try:
            speaker_id,raw_emotion,emotion=get_emotion_from_folder(folder)

            if emotion is None:
                skipped.append((file_path,raw_emotion))
                continue

            audio=preprocess_audio(file_path)
            features=extract_features(audio)

            save_name=f"{speaker_id}_{emotion}_{file[:-4]}.npy"
            save_path=os.path.join(OUTPUT_DIR,save_name)

            np.save(save_path,features)

            metadata.append({
                "file_path":file_path,
                "feature_path":save_path,
                "emotion":emotion,
                "speaker_id":speaker_id,
                "raw_emotion":raw_emotion,
                "original_folder":folder,
                "dataset":"TESS"
            })

        except Exception as e:
            print(f"\nError Processing: {file_path}")
            print(e)

    df=pd.DataFrame(metadata)
    df.to_csv(METADATA_PATH,index=False)

    print("\nPreprocessing Complete\n")
    print(df["emotion"].value_counts().sort_index())

    print("\nSpeakers:")
    print(df["speaker_id"].unique())

    if skipped:
        print("\nSkipped folders/emotions:")
        for item in skipped[:20]:
            print(item)

    print("\nMetadata Saved -> metadata.csv")

if __name__=="__main__":
    main()