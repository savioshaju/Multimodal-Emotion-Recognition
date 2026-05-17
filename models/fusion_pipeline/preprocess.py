import os
import glob
import pandas as pd

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(PIPELINE_DIR, "..", ".."))
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset")
METADATA_PATH = os.path.join(PIPELINE_DIR, "metadata.csv")

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

def preprocess_dataset():
    print(f"Scanning dataset at: {DATASET_PATH}")
    
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset path does not exist: {DATASET_PATH}")
        return

    wav_files = glob.glob(os.path.join(DATASET_PATH, "**", "*.wav"), recursive=True)
    
    if not wav_files:
        print(f"Error: No .wav files found in {DATASET_PATH}")
        return
        
    data = []
    skipped_files = []
    
    for file_path in wav_files:
        original_file = os.path.basename(file_path)
        original_folder = os.path.basename(os.path.dirname(file_path))
        
        folder_parts = original_folder.split('_')
        if len(folder_parts) < 2:
            skipped_files.append(file_path)
            continue
            
        speaker_id = folder_parts[0].lower()
        if speaker_id not in ["oaf", "yaf"]:
            skipped_files.append(file_path)
            continue
            
        raw_emotion = "_".join(folder_parts[1:]).lower()
        
        name_without_ext = os.path.splitext(original_file)[0]
        file_parts = name_without_ext.split('_')
        
        if len(file_parts) >= 3:
            text = "_".join(file_parts[1:-1]).lower()
        else:
            text = "unknown"
            
        emotion = EMOTION_MAP.get(raw_emotion)
        if emotion is None:
            skipped_files.append(file_path)
            continue
            
        data.append({
            "file_path": file_path,
            "text": text,
            "emotion": emotion,
            "speaker_id": speaker_id,
            "raw_emotion": raw_emotion,
            "original_folder": original_folder,
            "original_file": original_file,
            "dataset": "TESS"
        })
        
    df = pd.DataFrame(data)
    
    if skipped_files:
        print(f"\nSkipped {len(skipped_files)} files.")
        print("First 5 skipped files:")
        for f in skipped_files[:5]:
            print(f)
    
    print(f"Total valid samples: {len(df)}")
    print("\nClass counts:")
    print(df['emotion'].value_counts())
    print("\nSpeaker IDs:")
    print(df['speaker_id'].value_counts())
    print("\nSample rows:")
    print(df.head())
    
    df.to_csv(METADATA_PATH, index=False)
    print(f"\nSaved metadata to: {METADATA_PATH}")

if __name__ == "__main__":
    preprocess_dataset()
