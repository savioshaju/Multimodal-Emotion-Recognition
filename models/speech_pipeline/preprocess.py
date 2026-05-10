import os
import librosa
import soundfile as sf
import pandas as pd
from tqdm import tqdm
import warnings

# Suppress librosa warnings
warnings.filterwarnings('ignore')

# --------------------------
# Multi-Dataset Unified Configuration
# --------------------------
RAW_DATA_DIR = "../../datasets" # Expected root containing TESS, RAVDESS, CREMAD, SAVEE
FALLBACK_DATA_DIR = "../../dataset" # Fallback if user only has TESS in 'dataset'
PROCESSED_DIR = "processed_dataset"
METADATA_CSV = "metadata.csv"
TARGET_SR = 16000 # 16kHz is standard for modern SER (Wav2Vec2 uses 16kHz)

# Unified Emotion Mapper
EMOTION_MAP = {
    # TESS & General
    "angry": "anger", "ang": "anger", "a": "anger", "05": "anger",
    "disgust": "disgust", "dis": "disgust", "d": "disgust", "07": "disgust",
    "fear": "fear", "fea": "fear", "f": "fear", "06": "fear", "fearful": "fear",
    "happy": "happiness", "hap": "happiness", "h": "happiness", "03": "happiness",
    "neutral": "neutral", "neu": "neutral", "n": "neutral", "01": "neutral", 
    "calm": "neutral", "02": "neutral",
    "sad": "sadness", "sa": "sadness", "04": "sadness",
    "surprise": "surprise", "su": "surprise", "08": "surprise", "ps": "surprise", "pleasant": "surprise"
}

os.makedirs(PROCESSED_DIR, exist_ok=True)

# --------------------------
# Parsers for specific datasets
# --------------------------
def parse_tess(filename):
    parts = filename.lower().replace('.wav', '').split('_')
    if len(parts) >= 2:
        return "TESS", parts[0], parts[-1]
    return None, None, None

def parse_ravdess(filename):
    # RAVDESS: 03-01-06-01-02-01-12.wav (Emotion is 3rd, Actor is 7th)
    parts = filename.replace('.wav', '').split('-')
    if len(parts) == 7:
        return "RAVDESS", f"ravdess_{parts[6]}", parts[2]
    return None, None, None

def parse_cremad(filename):
    # CREMA-D: 1001_DFA_ANG_XX.wav (Actor is 1st, Emotion is 3rd)
    parts = filename.split('_')
    if len(parts) >= 3:
        return "CREMAD", f"crema_{parts[0]}", parts[2].lower()
    return None, None, None

def parse_savee(filename):
    # SAVEE: DC_a01.wav (Actor is DC, Emotion is a)
    parts = filename.split('_')
    if len(parts) == 2:
        emotion_code = ''.join([c for c in parts[1] if not c.isdigit()])
        return "SAVEE", f"savee_{parts[0]}", emotion_code.lower()
    return None, None, None

# --------------------------
# Audio Processing
# --------------------------
def process_audio(input_path, output_path):
    """
    Loads, trims silence via VAD, resamples, and saves audio.
    This shifts heavy CPU processing out of the PyTorch DataLoader.
    """
    if os.path.exists(output_path):
        return True # Skip if already processed

    try:
        audio, _ = librosa.load(input_path, sr=TARGET_SR)
        audio, _ = librosa.effects.trim(audio, top_db=30)
        
        # Save clean standardized audio
        sf.write(output_path, audio, TARGET_SR)
        return True
    except Exception as e:
        print(f"Failed to process {input_path}: {e}")
        return False

# --------------------------
# Main Execution
# --------------------------
def main():
    search_dir = RAW_DATA_DIR if os.path.exists(RAW_DATA_DIR) else FALLBACK_DATA_DIR
    if not os.path.exists(search_dir):
        print(f"Error: Could not find '{RAW_DATA_DIR}' or '{FALLBACK_DATA_DIR}'.")
        print("Please place your dataset folders (TESS, RAVDESS, etc.) inside a 'datasets/' directory.")
        return

    metadata = []
    
    print(f"Scanning '{search_dir}' for audio files...")
    
    # Recursively find all wav files
    wav_files = []
    for root, _, files in os.walk(search_dir):
        for f in files:
            if f.endswith('.wav'):
                wav_files.append(os.path.join(root, f))

    print(f"Found {len(wav_files)} total .wav files. Processing & Trimming...")

    for file_path in tqdm(wav_files):
        filename = os.path.basename(file_path)
        dataset_name, speaker_id, raw_emotion = None, None, None

        # Determine dataset format heuristically
        if "OAF" in filename or "YAF" in filename:
            dataset_name, speaker_id, raw_emotion = parse_tess(filename)
        elif len(filename.split('-')) == 7:
            dataset_name, speaker_id, raw_emotion = parse_ravdess(filename)
        elif len(filename.split('_')) == 4 and filename.split('_')[0].isdigit():
            dataset_name, speaker_id, raw_emotion = parse_cremad(filename)
        elif len(filename.split('_')) == 2 and filename.split('_')[0].isalpha():
            dataset_name, speaker_id, raw_emotion = parse_savee(filename)
        else:
            # Generic fallback (e.g., standard TESS folder structure)
            dataset_name = "UNKNOWN"
            speaker_id = "unknown"
            raw_emotion = os.path.basename(os.path.dirname(file_path)).lower()
            if "_" in raw_emotion:
                speaker_id = raw_emotion.split("_")[0]
                raw_emotion = raw_emotion.split("_")[1]

        # Map emotion to unified taxonomy
        emotion = EMOTION_MAP.get(raw_emotion, None)
        if not emotion:
            continue # Skip files we can't classify

        # Generate output path
        out_filename = f"{dataset_name}_{speaker_id}_{emotion}_{filename}"
        out_path = os.path.join(PROCESSED_DIR, out_filename)

        # Process and save audio
        if process_audio(file_path, out_path):
            metadata.append({
                "file_path": out_path,
                "dataset": dataset_name,
                "speaker_id": speaker_id,
                "emotion": emotion
            })

    # Save Unified Metadata
    df = pd.DataFrame(metadata)
    df.to_csv(METADATA_CSV, index=False)
    
    print("\nPreprocessing Complete!")
    print(f"Unified Metadata saved to '{METADATA_CSV}'")
    print(df['emotion'].value_counts())
    print(f"\nTotal Datasets: {df['dataset'].unique()}")
    print(f"Total Unique Speakers: {df['speaker_id'].nunique()}")

if __name__ == "__main__":
    main()