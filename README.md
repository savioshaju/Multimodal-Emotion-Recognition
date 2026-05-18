# Multimodal Emotion Recognition

This project performs emotion recognition using independent and combined machine learning pipelines focusing on speech, text, and multimodal fusion. 

The dataset utilized is the **Toronto Emotional Speech Set (TESS)**. The core objective of this project is to robustly compare unimodal methods (processing acoustic properties or semantic transcripts individually) against an multimodal method. 

## Table of Contents

1. [How to Run the Project](#how-to-run-the-project)
2. [Project Objective](#project-objective)
3. [Dataset — TESS](#dataset--tess)
4. [Overall Architecture](#overall-architecture)
5. [Complete Project Folder Structure](#complete-project-folder-structure)
6. [Speech Emotion Recognition Pipeline](#speech-emotion-recognition-pipeline)
7. [Text Emotion Recognition Pipeline](#text-emotion-recognition-pipeline)
8. [Multimodal Fusion Emotion Recognition Pipeline](#multimodal-fusion-emotion-recognition-pipeline)
9. [Final Comparison Table](#final-comparison-table)
10. [Result Visualizations](#result-visualizations)
11. [Evaluation Metrics](#evaluation-metrics)
12. [Limitations](#limitations)
13. [Future Improvements](#future-improvements)
14. [Conclusion](#conclusion)

---

## How to Run the Project

### 1. Clone the Repository
Clone the project to your local machine and open the project directory.
```powershell
git clone <repository_url>
cd "Multimodal Emotion Recognition"
```

### 2. Setup the Environment
The project requires Python 3.11.x. A setup script is provided to automatically create a virtual environment, activate it, upgrade pip, and install all dependencies.

**Using `setup.bat` (Windows)**
Run the following command from the project root:
```powershell
setup.bat
```
This script will:
1. Verify that Python 3.11.x is installed.
2. Create a virtual environment named `venv` if it does not exist.
3. Activate the virtual environment.
4. Upgrade `pip` to the latest version.
5. Install all required dependencies from `requirements.txt`.

**Manual Setup**
If you prefer to set up the environment manually, run:
```powershell
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Running the Pipelines

This project is divided into three separate pipelines. Each must be run from its respective directory within `models/`.

#### Speech Pipeline
This pipeline predicts emotion from audio waveforms using WavLM.

```powershell
cd models\speech_pipeline
python preprocess.py
python train.py
python test.py
```
* `python preprocess.py`: Generates the `metadata.csv` from the dataset.
* `python train.py`: Trains the WavLM model, evaluates it, and generates results.
* `python test.py`: Opens the graphical user interface (GUI) to test the model.
* `python test.py path/to/audio.wav`: Runs a direct CLI prediction without opening the GUI.

#### Text Pipeline
This pipeline predicts emotion from text transcripts using DistilBERT.

```powershell
cd models\text_pipeline
python preprocess.py
python train.py
python test.py
```
* `python preprocess.py`: Generates the text-focused `metadata.csv`.
* `python train.py`: Trains the DistilBERT model, evaluates it, and generates results.
* `python test.py`: Opens the graphical user interface (GUI) to test the text model. (This pipeline does not support CLI predictions).

#### Fusion Pipeline
This pipeline predicts emotion by fusing audio (WavLM) and text (DistilBERT).

```powershell
cd models\fusion_pipeline
python preprocess.py
python train.py
python test.py
```
* `python preprocess.py`: Generates the dual-modal `metadata.csv`.
* `python train.py`: Trains the multimodal fusion model, evaluates it, and generates results.
* `python test.py`: Opens the graphical user interface (GUI) to test the fusion model.
* `python test.py path/to/audio.wav "your transcript here"`: Runs a direct CLI prediction using both audio and text without opening the GUI.

---

## Project Objective

The goal of the project is to classify emotional states from speech and textual data. 

The system categorizes input into one of seven distinct emotion classes:
- anger
- disgust
- fear
- happiness
- neutral
- sadness
- surprise

The project fundamentally compares speech, text, and fusion methods to isolate which data modality provides the richest emotional signal and to observe if a multimodal fusion strategy successfully leverages both modalities simultaneously.

---

## Dataset — TESS

The primary dataset is the **Toronto Emotional Speech Set (TESS)**.
- It contains acted emotional speech recordings.
- It features two female speakers: OAF and YAF.
- Audio files are organized precisely by speaker and emotion folders.
- Emotion labels are extracted programmatically from the parent folder names.
- Transcript text is extracted by parsing the raw audio filenames (e.g., `OAF_back_angry.wav` extracts the word `back`).

**Important Note on Modality Strength**: TESS is incredibly strong for speech emotion recognition because the actors clearly vocally project the emotions. However, TESS is inherently weak for text emotion recognition because the spoken words are emotionally neutral target words (such as "bar", "ditch", "jar") that repeat identically across every emotional state.

### TESS Folder Structure
```text
dataset/
├── OAF_angry/
├── OAF_disgust/
├── OAF_Fear/
├── OAF_happy/
├── OAF_neutral/
├── OAF_Pleasant_surprise/
├── OAF_Sad/
├── YAF_angry/
├── YAF_disgust/
├── YAF_fear/
├── YAF_happy/
├── YAF_neutral/
├── YAF_pleasant_surprised/
└── YAF_sad/
```

---

## Overall Architecture

```mermaid
flowchart TD
    A[Speech Audio] --> B[Speech Preprocessing]
    B --> C[WavLM Feature Extraction]
    C --> D[Speech Representation]
    D --> H[Fusion Layer]

    E[Text Transcript] --> F[Text Preprocessing]
    F --> G[DistilBERT Text Representation]
    G --> H[Fusion Layer]

    H --> I[Classifier]
    I --> J[Emotion Label]
```

The architecture utilizes independent processing branches. Audio is handled via WavLM to extract acoustic features like pitch and tone. Text is tokenized and processed by DistilBERT to capture semantics. The fusion layer concatenates these representations and feeds them into an MLP Classifier to generate the final emotion label.

---

## Complete Project Folder Structure

```text
Multimodal Emotion Recognition/
├── dataset/
│   ├── OAF_angry/
│   └── (remaining emotion folders)
│
├── models/
│   ├── speech_pipeline/
│   │   ├── preprocess.py
│   │   ├── train.py
│   │   ├── test.py
│   │   ├── metadata.csv
│   │   ├── train_split.csv
│   │   ├── val_split.csv
│   │   ├── test_split.csv
│   │   └── saved_models/
│   │       ├── best_model.pth
│   │       └── model_config.json
│   │
│   ├── text_pipeline/
│   │   ├── preprocess.py
│   │   ├── train.py
│   │   ├── test.py
│   │   ├── metadata.csv
│   │   ├── train_split.csv
│   │   ├── val_split.csv
│   │   ├── test_split.csv
│   │   └── saved_models/
│   │       ├── best_model.pth
│   │       ├── model_config.json
│   │       ├── tokenizer_config.json
│   │       ├── tokenizer.json
│   │       ├── special_tokens_map.json
│   │       └── vocab.txt
│   │
│   └── fusion_pipeline/
│       ├── preprocess.py
│       ├── train.py
│       ├── test.py
│       ├── metadata.csv
│       ├── train_split.csv
│       ├── val_split.csv
│       ├── test_split.csv
│       └── saved_models/
│           ├── best_model.pth
│           ├── model_config.json
│           ├── tokenizer_config.json
│           ├── tokenizer.json
│           ├── special_tokens_map.json
│           └── vocab.txt
│
├── results/
│   ├── speech_pipeline/
│   │   ├── metrics/
│   │   │   └── speech_metrics.json
│   │   ├── plots/
│   │   │   ├── confusion_matrix.png
│   │   │   └── training_curve.png
│   │   └── results/
│   │       ├── classification_report.csv
│   │       ├── classification_report.txt
│   │       ├── confusion_matrix.csv
│   │       └── summary.csv
│   │
│   ├── text_pipeline/
│   │   ├── metrics/
│   │   │   └── text_metrics.json
│   │   ├── plots/
│   │   │   ├── confusion_matrix.png
│   │   │   └── training_curve.png
│   │   └── results/
│   │       ├── classification_report.csv
│   │       ├── classification_report.txt
│   │       ├── confusion_matrix.csv
│   │       └── summary.csv
│   │
│   └── fusion_pipeline/
│       ├── metrics/
│       │   ├── fusion_metrics.json
│       │   └── training_metrics.csv
│       ├── plots/
│       │   ├── confusion_matrix.png
│       │   ├── confusion_matrix_test.png
│       │   └── training_curve.png
│       └── results/
│           ├── classification_report.csv
│           ├── classification_report.txt
│           ├── confusion_matrix.csv
│           └── summary.csv
│
└── README.md
```

---

# Speech Emotion Recognition Pipeline

## Purpose

This pipeline predicts emotion from speech audio only. It isolates the effect of acoustic cues and assesses whether a dedicated feature-extraction approach is effective on the TESS dataset.

## Dataset and Input

The input originates from `.wav` audio files in the TESS dataset.

The dataset directory is structured into subfolders based on speaker and emotion:
```text
dataset/
├── OAF_angry/
├── YAF_happy/
└── ...
```

The script scans these folders for `.wav` files and extracts the `speaker_id` (e.g., `oaf`, `yaf`) and the `raw_emotion` (e.g., `angry`, `happy`) directly from the folder names. The raw emotions are then mapped to seven standardized emotion labels.

## Preprocessing

The `preprocess.py` script performs audio processing, feature extraction, and metadata generation.

1. **Audio Loading**: Files are loaded at 16 kHz using `librosa`, silence is trimmed, and the waveform is fixed to a length of 3 seconds.
2. **Feature Extraction**: It extracts three distinct acoustic features:
   - Mel Spectrogram (in dB)
   - Delta Mel Spectrogram
   - MFCCs (expanded to match the Mel frequency dimension)
3. **Saving Features**: These three channels are stacked into a 3D NumPy array of shape `(3, 128, time_frames)` and saved as `.npy` files in the `models/speech_pipeline/processed_data/` directory.

The script creates `models/speech_pipeline/metadata.csv` with the following columns:
- `file_path`: Original `.wav` path
- `feature_path`: Path to the extracted `.npy` file
- `emotion`: Standardized emotion label
- `speaker_id`: Speaker identifier (`oaf` or `yaf`)
- `raw_emotion`: Original TESS emotion string
- `original_folder`: TESS folder name
- `original_file`: TESS filename
- `dataset`: Fixed as `TESS`
- `sample_rate`: Sampling rate (`16000`)
- `duration`: Duration in seconds (`3`)
- `feature_type`: Type of feature (`mel_delta_mfcc`)
- `feature_shape`: Shape of the saved feature array

## Model Architecture

The `train.py` script implements a **CNN + BiLSTM + Attention** model (`CNN_BiLSTM_Attention_MelDeltaMFCC`).

| Component | Description |
|---|---|
| CNN Feature Extraction | 4 Residual Blocks (`ResBlock`) with `MaxPool2d` to extract spatial and frequency features from the 3-channel input. |
| Temporal Modelling | A 2-layer Bidirectional LSTM (`BiLSTM`) to capture temporal dynamics across time frames. |
| Attention Pooling | An attention mechanism using `Tanh` and `Softmax` to compute context weights and pool the LSTM outputs into a single context vector. |
| Classifier | An MLP classifier head using `LayerNorm`, `Linear`, `ReLU`, and `Dropout` layers mapping to the 7 emotion classes. |

## Speech Pipeline Architecture

```mermaid
flowchart BT
    A[Speech Audio] --> B[Preprocessing & Feature Extraction]
    B --> C[CNN Feature Extraction]
    C --> D[BiLSTM Temporal Modelling]
    D --> E[Attention Pooling]
    E --> F[Classifier]
    F --> G[Emotion Label]
```

## Training Strategy

The model uses a **Speaker-Aware Adaptation** split strategy to ensure realistic evaluation across speakers.

- **Base Train Speaker**: All samples from `oaf` are used for training.
- **Target Speaker Adaptation**: A small portion (`ADAPT_RATIO = 0.05`) of the `yaf` speaker's data is included in the training set for adaptation.
- **Validation/Test**: The remaining target speaker data is split into validation (30%) and testing (70%).

The splits are explicitly saved to:
- `models/speech_pipeline/train_split.csv`
- `models/speech_pipeline/val_split.csv`
- `models/speech_pipeline/test_split.csv`

**Training Hyperparameters:**
- **Optimizer**: `AdamW` (Learning Rate: 1e-4, Weight Decay: 1e-4)
- **Loss Function**: `CrossEntropyLoss` with label smoothing (0.03)
- **Batch Size**: 16
- **Epochs**: 40
- **Early Stopping**: Patience of 7 epochs based on Validation Macro F1.
- **Best Model Saving**: The model with the highest Validation Macro F1 is saved to `models/speech_pipeline/saved_models/best_model.pth`.

## Testing 

The testing and inference module is located in `test.py`. It supports both Graphical User Interface and Command-Line Interface (CLI) workflows.

Running `python test.py` directly opens the CustomTkinter GUI. The GUI allows you to upload audio files, visualize predictions and confidence scores, and open tables for the classification report, confusion matrix, and metrics summary.

Running `python test.py path/to/audio.wav` bypasses the GUI entirely and outputs the emotion prediction directly in the terminal (CLI prediction).

## How to Run Speech Pipeline

Run the commands from the project root:

```powershell
cd "models\speech_pipeline"
python preprocess.py
python train.py
python test.py
```

| Command | Purpose |
|---|---|
| `python preprocess.py` | Extracts acoustic features into `.npy` files and creates `metadata.csv` |
| `python train.py` | Trains the CNN-BiLSTM-Attention model, evaluates it, and saves results |
| `python test.py` | Opens the CustomTkinter GUI for prediction and viewing reports |
| `python test.py path/to/audio.wav` | Performs CLI emotion prediction on a specific audio file |

## Speech Pipeline Results

| Metric        |                  Value |
| ------------- | ---------------------: |
| Test Accuracy |                 99.89% |
| Test UAR      |                 99.89% |
| Test Macro F1 |                 99.89% |
| Model Name    | `CNN_BiLSTM_Attention` |

## Speech Classification Report

| Emotion   | Precision | Recall | F1-score | Support |
| --------- | --------: | -----: | -------: | ------: |
| anger     |      1.00 |   0.99 |     1.00 |     133 |
| disgust   |      1.00 |   1.00 |     1.00 |     133 |
| fear      |      1.00 |   1.00 |     1.00 |     133 |
| happiness |      1.00 |   1.00 |     1.00 |     133 |
| neutral   |      1.00 |   1.00 |     1.00 |     133 |
| sadness   |      0.99 |   1.00 |     1.00 |     133 |
| surprise  |      1.00 |   1.00 |     1.00 |     133 |

## Speech Confusion Matrix

| Actual \ Predicted | anger | disgust | fear | happiness | neutral | sadness | surprise |
| ------------------ | ----: | ------: | ---: | --------: | ------: | ------: | -------: |
| anger              |   132 |       0 |    0 |         0 |       0 |       1 |        0 |
| disgust            |     0 |     133 |    0 |         0 |       0 |       0 |        0 |
| fear               |     0 |       0 |  133 |         0 |       0 |       0 |        0 |
| happiness          |     0 |       0 |    0 |       133 |       0 |       0 |        0 |
| neutral            |     0 |       0 |    0 |         0 |     133 |       0 |        0 |
| sadness            |     0 |       0 |    0 |         0 |       0 |     133 |        0 |
| surprise           |     0 |       0 |    0 |         0 |       0 |       0 |      133 |

## Speech Result Images

![Speech Training Curve](results/speech_pipeline/plots/training_curve.png)

![Speech Confusion Matrix](results/speech_pipeline/plots/confusion_matrix.png)

## Speech Result Interpretation

The speech pipeline achieves near-perfect performance on the TESS test split, with a test accuracy of 99.89%. This indicates that acoustic information is highly discriminative for this dataset and that the extracted features (Mel Spectrogram, Delta, MFCC) coupled with the CNN-BiLSTM-Attention architecture are extremely effective.

The confusion matrix shows that almost all samples are correctly classified. The only error is a single `anger` sample predicted as `sadness`. The model has learned strong class separation for the controlled TESS recordings.

The result is strong because TESS is an acted emotional speech dataset where emotional cues are clearly expressed through pitch, tone, energy, rhythm, and prosody. 

However, this result should not be overclaimed. TESS is clean, studio-recorded, acted, and limited to two speakers. Performance on real-world data with background noise and spontaneous emotion is expected to be lower.

---

# Text Emotion Recognition Pipeline

## Purpose

This pipeline predicts emotion from text only. It is designed to test whether the spoken word content extracted from the TESS filenames contains enough semantic information to identify emotion without using the audio signal.


## Input

The input is text extracted from TESS audio filenames.

Example:

```text
OAF_back_angry.wav
```

is converted into:

```text
back
```

The extracted word is used as the text input, and the emotion label is extracted from the folder name.

## Model Used

| Component           | Description                                                               |
| ------------------- | ------------------------------------------------------------------------- |
| Base model          | `distilbert-base-uncased`                                                 |
| Tokenizer           | Hugging Face `AutoTokenizer`                                              |
| Encoder             | DistilBERT transformer encoder                                            |
| Text representation | CLS token representation                                                  |
| Classifier          | MLP classification head                                                   |
| Output classes      | `anger`, `disgust`, `fear`, `happiness`, `neutral`, `sadness`, `surprise` |

## Text Pipeline Architecture

```mermaid
flowchart BT
    A[Text Input] --> B[Preprocessing]
    B --> C[Feature Extraction]
    C --> D[Contextual Modelling]
    D --> E[Classifier]
    E --> F[Emotion Label]
```

## Implementation Mapping

| Diagram Block        | Code Implementation                                                      |
| -------------------- | ------------------------------------------------------------------------ |
| Text Input           | Word extracted from TESS filename                                        |
| Preprocessing        | Filename parsing, lowercasing, text cleaning, metadata generation        |
| Feature Extraction   | DistilBERT tokenizer converts text into `input_ids` and `attention_mask` |
| Contextual Modelling | DistilBERT transformer encoder                                           |
| Classifier           | Dropout → Linear → ReLU → Dropout → Linear                               |
| Emotion Label        | Predicted class among seven emotion labels                               |

## Preprocessing

The `preprocess.py` script scans the TESS dataset directory and creates a text-focused `metadata.csv`.

The script extracts:

| Field             | Source                                |
| ----------------- | ------------------------------------- |
| `file_path`       | Full path of the `.wav` file          |
| `text`            | Spoken word extracted from filename   |
| `emotion`         | Standardized emotion label            |
| `speaker_id`      | Speaker prefix such as `OAF` or `YAF` |
| `raw_emotion`     | Original emotion string               |
| `original_folder` | Original TESS folder name             |
| `original_file`   | Original `.wav` filename              |
| `dataset`         | Fixed value: `TESS`                   |

Example filename parsing:

```text
OAF_back_angry.wav
```

| Part    | Meaning           |
| ------- | ----------------- |
| `OAF`   | Speaker ID        |
| `back`  | Extracted text    |
| `angry` | Raw emotion label |

The preprocessing script maps raw emotion labels into seven standard classes:

| Raw Label                                                               | Standard Label |
| ----------------------------------------------------------------------- | -------------- |
| `angry`                                                                 | `anger`        |
| `disgust`                                                               | `disgust`      |
| `fear`                                                                  | `fear`         |
| `happy`                                                                 | `happiness`    |
| `neutral`                                                               | `neutral`      |
| `sad`                                                                   | `sadness`      |
| `pleasant_surprise`, `pleasant_surprised`, `pleasant`, `surprise`, `ps` | `surprise`     |

The preprocessing stage does not perform model training. It only creates the metadata required by the training script.

## Training Method

The `train.py` script trains a DistilBERT-based text emotion classifier.

Training uses a speaker-aware split strategy:

| Setting                 | Value                                                              |
| ----------------------- | ------------------------------------------------------------------ |
| Dataset                 | TESS                                                               |
| Split strategy          | Speaker-aware                                                      |
| Base train speaker      | `oaf`                                                              |
| Target speaker          | `yaf`                                                              |
| Adaptation ratio        | `0.05`                                                             |
| Model                   | `distilbert-base-uncased`                                          |
| Architecture            | DistilBERT Transformer Encoder + CLS Pooling + Classification Head |
| Maximum sequence length | `32`                                                               |
| Batch size              | `16`                                                               |
| Epochs                  | `20`                                                               |
| Learning rate           | `2e-5`                                                             |
| Optimizer               | AdamW                                                              |
| Loss function           | CrossEntropyLoss                                                   |
| Label smoothing         | `0.03`                                                             |
| Weight decay            | `1e-4`                                                             |
| Early stopping patience | `5`                                                                |
| Seed                    | `42`                                                               |

The model uses the CLS token representation from DistilBERT:

```text
outputs.last_hidden_state[:, 0, :]
```

This CLS representation is passed into an MLP classifier head:

```text
Dropout(0.3)
Linear(hidden_size, 256)
ReLU
Dropout(0.3)
Linear(256, num_classes)
```

The `train.py` script also evaluates the trained model on the held-out `test_split.csv` and produces the final test results.

The evaluation produces:

| Output                | Description                                         |
| --------------------- | --------------------------------------------------- |
| Accuracy              | Overall percentage of correct predictions           |
| UAR                   | Macro recall across all classes                     |
| Macro F1              | Average F1-score across all emotion classes         |
| Classification report | Class-wise precision, recall, F1-score, and support |
| Confusion matrix      | Actual-vs-predicted class distribution              |
| Training curve        | Training loss and validation Macro F1 across epochs |

## Testing and Inference Method

The GUI prediction functionality is implemented inside `test.py`.

The script loads:

```text
models/text_pipeline/saved_models/best_model.pth
```

and uses:

```text
models/text_pipeline/saved_models/model_config.json
```

Running `python test.py` opens the CustomTkinter GUI. The GUI supports:

| Feature               | Description                                          |
| --------------------- | ---------------------------------------------------- |
| Text input            | User enters text manually                            |
| Predict Emotion       | Runs model inference on the entered text             |
| Classification Report | Opens the saved classification report                |
| Confusion Matrix      | Opens the saved confusion matrix                     |
| Metrics Summary       | Opens the saved metrics JSON                         |
| View Plots            | Opens the training curve and confusion matrix images |

## How to Run Text Pipeline

Run the following commands from the project root:

```powershell
cd "models\text_pipeline"
python preprocess.py
python train.py
python test.py
```

Command purpose:

| Command                | Purpose                                                                         |
| ---------------------- | ------------------------------------------------------------------------------- |
| `python preprocess.py` | Extracts text from TESS filenames and creates `metadata.csv`                    |
| `python train.py`      | Trains the DistilBERT text emotion classifier, evaluates it, and saves outputs  |
| `python test.py`       | Opens the CustomTkinter GUI for text emotion prediction and viewing reports     |


## Text Pipeline Results

| Metric                   |                     Value |
| ------------------------ | ------------------------: |
| Test Accuracy            |                    14.93% |
| Test UAR                 |                    14.93% |
| Test Macro F1            |                     7.29% |
| Best Epoch               |                         5 |
| Best Validation Macro F1 |                     6.18% |
| Model Name               | `distilbert-base-uncased` |

## Text Training Metrics

| Epoch | Train Loss | Val Accuracy | Val UAR | Val Macro F1 |
| ----: | ---------: | -----------: | ------: | -----------: |
|     1 |     1.9528 |       14.29% |  14.29% |        3.57% |
|     2 |     1.9543 |       15.04% |  15.04% |        5.95% |
|     3 |     1.9522 |       14.29% |  14.29% |        3.60% |
|     4 |     1.9522 |       14.29% |  14.29% |        3.64% |
|     5 |     1.9493 |       12.78% |  12.78% |        6.18% |
|     6 |     1.9486 |       14.29% |  14.29% |        3.57% |
|     7 |     1.9486 |       14.29% |  14.29% |        3.57% |
|     8 |     1.9470 |       14.29% |  14.29% |        3.57% |
|     9 |     1.9503 |       14.54% |  14.54% |        4.07% |
|    10 |     1.9490 |       14.29% |  14.29% |        3.57% |

## Text Classification Report

| Emotion      | Precision | Recall | F1-score | Support |
| ------------ | --------: | -----: | -------: | ------: |
| anger        |      0.15 |   0.68 |     0.24 |     133 |
| disgust      |      0.15 |   0.32 |     0.21 |     133 |
| fear         |      0.14 |   0.04 |     0.06 |     133 |
| happiness    |      0.00 |   0.00 |     0.00 |     133 |
| neutral      |      0.00 |   0.00 |     0.00 |     133 |
| sadness      |      0.00 |   0.00 |     0.00 |     133 |
| surprise     |      0.00 |   0.00 |     0.00 |     133 |
| accuracy     |      0.15 |   0.15 |     0.15 |     931 |
| macro avg    |      0.06 |   0.15 |     0.07 |     931 |
| weighted avg |      0.06 |   0.15 |     0.07 |     931 |

## Text Confusion Matrix

| Actual \ Predicted | anger | disgust | fear | happiness | neutral | sadness | surprise |
| ------------------ | ----: | ------: | ---: | --------: | ------: | ------: | -------: |
| anger              |    91 |      37 |    5 |         0 |       0 |       0 |        0 |
| disgust            |    85 |      43 |    5 |         0 |       0 |       0 |        0 |
| fear               |    90 |      38 |    5 |         0 |       0 |       0 |        0 |
| happiness          |    85 |      43 |    5 |         0 |       0 |       0 |        0 |
| neutral            |    88 |      40 |    5 |         0 |       0 |       0 |        0 |
| sadness            |    88 |      40 |    5 |         0 |       0 |       0 |        0 |
| surprise           |    86 |      40 |    7 |         0 |       0 |       0 |        0 |

## Text Result Images

![Text Training Curve](results/text_pipeline/plots/training_curve.png)

![Text Confusion Matrix](results/text_pipeline/plots/confusion_matrix.png)

## Text Result Interpretation

The text pipeline performs poorly on the TESS dataset.

The test accuracy is 14.93%, while random chance for seven balanced classes is approximately 14.28%. This means the model is performing almost at random-chance level.

The Macro F1-score is only 7.29%, which confirms that the model is not learning meaningful class separation across all seven emotion categories.

The confusion matrix shows that the model predicts mostly `anger`, `disgust`, and a small number of `fear` samples. It completely fails to predict `happiness`, `neutral`, `sadness`, and `surprise`. This is why the precision, recall, and F1-score for those classes are all zero.

This result is expected because TESS is primarily an audio emotion dataset. The emotional information in TESS is carried mainly by vocal tone, pitch, energy, rhythm, and prosody. The extracted text is only a short spoken word such as `back`, `bar`, `goose`, or `ditch`.

The same word appears across multiple emotion classes. For example, the word `back` can appear as angry, happy, sad, neutral, fearful, disgusted, or surprised depending on how it is spoken. When audio is removed, the text alone does not contain enough emotional information for reliable classification.

Therefore, the low performance is not necessarily a coding failure. It is an experimental finding showing that TESS is not suitable for strong text-only emotion recognition when text is extracted only from filenames/transcripts.

## Why Performance Is Low When TESS Is Used for Text

TESS is designed for speech emotion recognition, not natural-language emotion recognition.

In normal text emotion datasets, emotion is often expressed through sentence meaning. For example:

```text
I am so happy today.
```

or:

```text
I feel terrible and disappointed.
```

In TESS, the text is usually only a neutral word:

```text
back
goose
ditch
bar
```

These words do not carry emotional meaning by themselves. The emotion exists in the way the word is spoken, not in the word itself.

This creates a label conflict problem:

| Same Text | Possible Labels                                             |
| --------- | ----------------------------------------------------------- |
| `back`    | anger, disgust, fear, happiness, neutral, sadness, surprise |
| `goose`   | anger, disgust, fear, happiness, neutral, sadness, surprise |
| `ditch`   | anger, disgust, fear, happiness, neutral, sadness, surprise |

Because the same input text maps to multiple different labels, the text-only model receives contradictory training signals. DistilBERT cannot reliably infer emotion when the semantic input is almost identical across classes.

## Limitations

| Limitation                     | Explanation                                                                  |
| ------------------------------ | ---------------------------------------------------------------------------- |
| Short text input               | Each sample contains only one or very few words                              |
| Weak semantic emotion          | The extracted words are mostly neutral and do not express emotion directly   |
| Repeated text across labels    | The same word appears in multiple emotion categories                         |
| Dataset mismatch               | TESS is built for audio emotion recognition, not text emotion recognition    |
| No acoustic cues               | Pitch, tone, rhythm, and intensity are removed                               |
| Poor real-world generalization | This model should not be treated as a strong general text emotion classifier |
| Low class coverage             | The model fails to predict several classes completely                        |

## Key Inference

The text pipeline demonstrates that text-only emotion recognition is ineffective on TESS when the text is extracted from filenames or short transcripts.

The result supports the main multimodal conclusion: for TESS, speech carries the dominant emotional signal, while text contributes very little because the textual content is semantically weak.

This makes the text pipeline useful as an experimental baseline. It proves that the poor text result is caused by dataset limitations, not simply by model architecture.

---

# Multimodal Fusion Emotion Recognition Pipeline

## Purpose
This pipeline aggressively combines the acoustic representation with semantic text representation to generate a unified prediction.

## Input
* 16 kHz audio waveform.
* Literal text transcript.

## Model Used
* WavLM (speech encoder)
* DistilBERT (text encoder)
* Linear projection layers reducing both modalities
* Concatenation layer (to 512-dim)
* MLP classifier head

## Preprocessing
* Constructs a dual-purpose `metadata.csv` containing both `file_path` and `text`.
* Prepares dual dataloaders bridging Hugging Face tokenization and Librosa audio tensor generation simultaneously.

## Training Method
* Synchronously fine-tunes WavLM and DistilBERT branches.
* Backpropagates errors dynamically across the fusion layer.
* Optimizer: AdamW
* Loss: CrossEntropyLoss
* Batch Size: 8
* Epochs: 30
* Saves combined weights and essential tokenizers.

## Testing Method
* `test.py` boots the large combined architecture.
* Loads audio waveforms and tokens simultaneously.
* Evaluates on test_split.csv.
* Writes deep cross-modal reports, metrics, and plots.

## GUI / Inference Usage
* `test.py` accepts both a literal audio upload and corresponding text transcript box.
* Generates the multi-modal confidence array.
* Provides report buttons: Classification Report, Confusion Matrix, Metrics Summary, View Plots.

## How to Run Fusion Pipeline
```powershell
cd "models\fusion_pipeline"
python preprocess.py
python train.py
python test.py
```

## Fusion Pipeline Generated Files

| File/Folder | Created By | Purpose |
|---|---|---|
| metadata.csv | preprocess.py | Stores audio path, text, labels |
| train_split.csv | train.py | Training split |
| val_split.csv | train.py | Validation split |
| test_split.csv | train.py | Test split |
| saved_models/best_model.pth | train.py | Trained fusion model |
| saved_models/model_config.json | train.py | Model configuration |
| saved_models/tokenizer.json | train.py | Tokenizer dict |
| saved_models/vocab.txt | train.py | Vocabulary list |
| results/fusion_pipeline/metrics/fusion_metrics.json | train.py/test.py | Final JSON metrics |
| results/fusion_pipeline/metrics/training_metrics.csv | train.py | Epoch-wise metrics |
| results/fusion_pipeline/plots/training_curve.png | train.py | Training curve |
| results/fusion_pipeline/plots/confusion_matrix.png | train.py/test.py | Basic confusion matrix |
| results/fusion_pipeline/plots/confusion_matrix_test.png | test.py | Test evaluation confusion matrix |
| results/fusion_pipeline/results/classification_report.csv | test.py/train.py | Classification report CSV |
| results/fusion_pipeline/results/confusion_matrix.csv | test.py/train.py | Confusion matrix raw values |
| results/fusion_pipeline/results/summary.csv | test.py/train.py | Final summary values |

## Fusion Pipeline Results

| Metric | Value |
|---|---:|
| Test Accuracy | 99.68% |
| Test UAR | 99.68% |
| Test Macro F1 | 99.68% |
| Model Name | WavLM + DistilBERT |

## Fusion Classification Report

| Emotion | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| anger | 0.99 | 0.99 | 0.99 | 133 |
| disgust | 1.00 | 1.00 | 1.00 | 133 |
| fear | 1.00 | 0.99 | 1.00 | 133 |
| happiness | 0.99 | 1.00 | 1.00 | 133 |
| neutral | 0.99 | 1.00 | 1.00 | 133 |
| sadness | 1.00 | 1.00 | 1.00 | 133 |
| surprise | 1.00 | 0.99 | 1.00 | 133 |

## Fusion Confusion Matrix

| Actual \ Predicted | anger | disgust | fear | happiness | neutral | sadness | surprise |
|---|---:|---:|---:|---:|---:|---:|---:|
| anger | 132 | 0 | 0 | 0 | 1 | 0 | 0 |
| disgust | 0 | 133 | 0 | 0 | 0 | 0 | 0 |
| fear | 1 | 0 | 132 | 0 | 0 | 0 | 0 |
| happiness | 0 | 0 | 0 | 133 | 0 | 0 | 0 |
| neutral | 0 | 0 | 0 | 0 | 133 | 0 | 0 |
| sadness | 0 | 0 | 0 | 0 | 0 | 133 | 0 |
| surprise | 0 | 0 | 0 | 1 | 0 | 0 | 132 |

## Fusion Result Images
![Fusion Training Curve](results/fusion_pipeline/plots/training_curve.png)
![Fusion Confusion Matrix](results/fusion_pipeline/plots/confusion_matrix_test.png)

## Fusion Result Interpretation
* Fusion performs incredibly strongly (99.68%).
* Because the TESS text is so structurally weak, it is evident that WavLM aggressively captures acoustic emotion cues, overriding DistilBERT.
* The speech branch likely contributes over 90% of the useful discriminative signal.
* The text branch is structurally included and successfully tokenized, but contributes less because the text modality is fundamentally constrained in TESS.
* The high result is scientifically valid for TESS, but must absolutely not be overclaimed. TESS is clean, acted, controlled, and has only two speakers.
* Real-world performance on complex, conversational datasets will be substantially lower.

---

# FINAL COMPARISON SECTION

| Pipeline | Input | Model | Accuracy | UAR | Macro F1 | Main Inference |
|---|---|---|---:|---:|---:|---|
| Speech Pipeline | Audio | microsoft/wavlm-base | 99.89% | 99.89% | 99.89% | Speech captures strong emotion cues in TESS. |
| Text Pipeline | Text | distilbert-base-uncased | 14.93% | 14.93% | 7.29% | Text is entirely weak for TESS and stays close to random chance. |
| Fusion Pipeline | Audio + Text | WavLM + DistilBERT | 99.68% | 99.68% | 99.68% | Fusion performs exceptionally well, mainly due to the acoustic speech branch dominating. |

---

# GLOBAL RESULT IMAGE SECTION

## Speech Pipeline Visualizations
![Speech Training Curve](results/speech_pipeline/plots/training_curve.png)
![Speech Confusion Matrix](results/speech_pipeline/plots/confusion_matrix.png)

## Text Pipeline Visualizations
![Text Training Curve](results/text_pipeline/plots/training_curve.png)
![Text Confusion Matrix](results/text_pipeline/plots/confusion_matrix.png)

## Fusion Pipeline Visualizations
![Fusion Training Curve](results/fusion_pipeline/plots/training_curve.png)
![Fusion Confusion Matrix](results/fusion_pipeline/plots/confusion_matrix_test.png)

---

# EVALUATION METRICS SECTION
* **Accuracy:** Overall percentage of correct emotion classifications.
* **Precision:** Calculates how many of the positively predicted instances actually belonged to the class.
* **Recall:** Calculates how many actual positives the model successfully captured.
* **F1-score:** Harmonic mean of precision and recall.
* **Macro F1:** An unweighted mean of all per-class F1-scores, heavily utilized because it does not skew for class imbalance.
* **UAR (Unweighted Average Recall):** Used heavily in Speech Emotion Recognition to accurately judge cross-class generalization.
* **Confusion matrix:** A tabular matrix showing exactly which classes are confused/mispredicted against their true label.

---

# LIMITATIONS SECTION
* TESS has only two female speakers (OAF and YAF). Generalization to male voices is untested.
* TESS is acted and controlled in a noiseless studio setting.
* Text is not emotionally meaningful (controlled neutral target vocabulary).
* The fusion methodology heavily pivots towards being speech-dominated.
* Real-world noise, microphone degradation, and overlapping chatter are not represented.
* Cross-dataset generalization is completely untested.

---

# FUTURE IMPROVEMENTS SECTION
* Add datasets like RAVDESS, CREMA-D, IEMOCAP, and SAVEE.
* Utilize robust ASR-generated transcripts dynamically instead of relying on dataset filenames.
* Pre-train text models on legitimate emotion corpora like GoEmotions, ISEAR, and DailyDialog.
* Implement aggressive cross-dataset evaluations.
* Introduce random white-noise and ambient environmental audio augmentation.
* Implement real-time inference streaming pipelines.
* Integrate model explainability techniques (e.g., attention head visualization).
* Port the GUI into a live web application via Flask, FastAPI, or Streamlit.

---

# CONCLUSION SECTION
The project comprehensively implements and evaluates speech, text, and multimodal fusion pipelines. Ultimately, speech is exceptionally strong for TESS due to pristine acoustic variance, while text-only methods fall exactly to random chance because TESS text is intrinsically neutral. The fusion model performs spectacularly because it inherently absorbs the power of the speech modality. The robust workflow gracefully establishes preprocessing, training, automated testing, result generation, strict visualizations, serializations, and user-facing GUI interfaces suitable for expansion.
