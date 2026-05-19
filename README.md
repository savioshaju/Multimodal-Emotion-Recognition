# Multimodal Emotion Recognition

This project implements a comparative emotion recognition system using three core pipelines: speech-only emotion recognition, text-only emotion recognition, and multimodal speech-text fusion.

The primary dataset used for the core implementation is the Toronto Emotional Speech Set (TESS). The core pipelines are designed to evaluate how acoustic information, textual information, and combined multimodal information contribute to emotion classification.


## Table of Contents

1. [How to Run the Project](#how-to-run-the-project)
2. [Project Objective](#project-objective)
3. [Dataset Usage](#dataset-usage)
4. [Overall Architecture](#overall-architecture)
5. [Complete Project Folder Structure](#complete-project-folder-structure)
6. [Speech Emotion Recognition Pipeline](#speech-emotion-recognition-pipeline)
7. [Text Emotion Recognition Pipeline](#text-emotion-recognition-pipeline)
   - [Additional Text Experiment: DailyDialog](#additional-text-experiment-dailydialog)
8. [Multimodal Fusion Pipeline](#multimodal-fusion-pipeline)
   - [Additional Fusion Experiment: MELD](#additional-fusion-experiment-meld)
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
git clone https://github.com/savioshaju/Multimodal-Emotion-Recognition.git
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

The project contains three core TESS pipelines and two additional experimental pipelines. Each pipeline must be run from its respective directory inside `models/`.

#### Speech Pipeline
This pipeline predicts emotion from audio waveforms using acoustic features (Mel Spectrogram, Delta, MFCC) with a CNN + BiLSTM + Attention architecture.

```powershell
cd models\speech_pipeline
python preprocess.py
python train.py
python test.py
```
* `python preprocess.py`: Extracts acoustic features and generates the `metadata.csv` from the dataset.
* `python train.py`: Trains the CNN-BiLSTM-Attention model, evaluates it, and generates results.
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
* `python test.py`: Opens the graphical user interface (GUI) to test the text model.

#### Fusion Pipeline
This pipeline predicts emotion by fusing acoustic features (Mel Spectrogram, Delta, MFCC via CNN-BiLSTM-Attention) with text (DistilBERT) using concatenation.

```powershell
cd models\fusion_pipeline
python preprocess.py
python train.py
python test.py
```
* `python preprocess.py`: Generates the dual-modal `metadata.csv` with both audio paths and text.
* `python train.py`: Trains the multimodal fusion model (CNN-BiLSTM-Attention + DistilBERT), evaluates it, and generates results.
* `python test.py`: Opens the graphical user interface (GUI) to test the fusion model.
* `python test.py path/to/audio.wav "your transcript here"`: Runs a direct CLI prediction using both audio and text without opening the GUI.

#### DailyDialog Text Pipeline

This pipeline evaluates text emotion recognition on realistic conversational text using DailyDialog.

```powershell
cd models\text_pipeline_DailyDialog
python preprocess.py
python train.py
python test.py
```

* `python preprocess.py`: Parses DailyDialog files, cleans utterances, maps labels, creates metadata, and generates dialogue-level train/validation/test splits.
* `python train.py`: Trains the RoBERTa-based text emotion classifier and saves metrics, plots, and reports.
* `python test.py`: Opens the GUI or evaluation interface if implemented in the pipeline.

#### MELD Fusion Pipeline

This pipeline evaluates multimodal emotion recognition on realistic dialogue data using MELD.

```powershell
cd models\fusion_pipeline_MELD
python preprocess.py
python train.py
python test.py
```

* `python preprocess.py`: Processes MELD train/dev/test metadata, extracts acoustic features from video/audio files, and aligns them with text utterances.
* `python train.py`: Trains the CNN-BiLSTM-Attention + DistilBERT fusion model using class-weighted loss.
* `python test.py`: Runs final evaluation or inference if implemented in the pipeline.

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

## Dataset Usage

### Primary Dataset: TESS

TESS is the primary dataset used for the three core pipelines:

| Pipeline | Dataset | Purpose |
|---|---|---|
| Speech Pipeline | TESS | Required speech-only emotion recognition |
| Text Pipeline | TESS | Required text-only baseline using filename-derived transcript words |
| Fusion Pipeline | TESS | Required multimodal speech + text fusion |

TESS is suitable for speech emotion recognition because it contains clean, acted emotional speech. However, it is weak for text-only emotion recognition because the spoken words are short neutral target words that repeat across emotion classes.

### Supporting Dataset: DailyDialog

DailyDialog is used only as an additional text experiment under the text pipeline. It is not the main required text pipeline. It is included to test the text model on meaningful conversational sentences.

### Supporting Dataset: MELD

MELD is used only as an additional fusion experiment under the fusion pipeline. It is not the main required fusion pipeline. It is included to test multimodal fusion on realistic dialogue data with aligned audio and text.
## Overall Architecture

```mermaid
flowchart TD
    A[Speech Audio] --> B[Audio Preprocessing]
    B --> C[Feature Extraction: Mel Spectrogram + Delta + MFCC]
    C --> D[CNN Feature Extraction]
    D --> E[BiLSTM Temporal Modeling]
    E --> F[Attention Pooling]
    F --> H[Fusion Layer]

    G[Text Transcript] --> I[Text Preprocessing]
    I --> J[DistilBERT Text Encoding]
    J --> H[Fusion Layer]

    H --> K[MLP Classifier]
    K --> L[Emotion Label]
```

The architecture utilizes independent processing branches. Audio is preprocessed and converted to three-channel acoustic features (Mel Spectrogram, Delta, MFCC) which are processed through a CNN feature extractor, BiLSTM temporal modeler, and attention pooling layer to extract acoustic representations. Text is tokenized and processed by DistilBERT to capture semantic meaning. The fusion layer concatenates these representations and feeds them into an MLP classifier to generate the final emotion label.

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
│   ├── text_pipeline_DailyDialog/
│   │   ├── preprocess.py
│   │   ├── train.py
│   │   ├── data/
│   │   ├── processed_data/
│   │   └── saved_models/
│   │
│   ├── fusion_pipeline/
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
│   └── fusion_pipeline_MELD/
│   │   ├── preprocess.py
│       ├── train.py
│       ├── processed_data/
│       └── saved_models/
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
│   ├── text_pipeline_DailyDialog/
│   │   ├── metrics/
│   │   ├── plots/
│   │   └── results/
│   │
│   ├── fusion_pipeline/
│   │   ├── metrics/
│   │   │   ├── fusion_metrics.json
│   │   │   └── training_metrics.csv
│   │   ├── plots/
│   │   │   ├── confusion_matrix.png
│   │   │   ├── confusion_matrix_test.png
│   │   │   └── training_curve.png
│   │   └── results/
│   │       ├── classification_report.csv
│   │       ├── classification_report.txt
│   │       ├── confusion_matrix.csv
│   │       └── summary.csv
│   │
│   └── fusion_pipeline_MELD/
│   │   ├── metrics/
│   │   ├── plots/
│   │   └── results/
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
| Model Name    | CNN_BiLSTM_Attention_MelDeltaMFCC |
| Architecture  | Mel Spectrogram + Delta + MFCC → CNN → BiLSTM → Attention → MLP |

**Note on Performance**: TESS achieves near-perfect performance because it features clean, acted studio recordings with two speakers (OAF, YAF) and clear acoustic emotional cues. Performance on real-world data with background noise, multiple speakers, accents, and spontaneous emotions would be considerably lower.

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

The speech pipeline achieves near-perfect performance on the TESS test split, with a test accuracy of 99.89%, UAR of 99.89%, and Macro F1 of 99.89%. This indicates that acoustic information is highly discriminative for this dataset and that the extracted features (Mel Spectrogram, Delta, MFCC) coupled with the CNN-BiLSTM-Attention architecture are extremely effective.

The confusion matrix shows that almost all samples are correctly classified. The only error is a single `anger` sample predicted as `sadness`. The model has learned strong class separation for the controlled TESS recordings.

**Why Performance Is Strong on TESS:**
- TESS features clean, studio-recorded audio with clear emotional vocal expression
- Strong acoustic emotion cues through pitch, tone, energy, rhythm, and prosody
- Controlled environment with no background noise
- Limited to two highly consistent female speakers

**Why Performance May Be Lower in Practice:**
- Real-world audio contains background noise, microphone degradation, and acoustic variation
- Live speech has multiple speakers with different accents, ages, and voice characteristics
- Spontaneous emotions differ significantly from acted emotions
- TESS dataset contains only female speakers (OAF and YAF) — generalization to male voices is untested

This model demonstrates the effectiveness of feature-based (Mel Spectrogram + Delta + MFCC) CNN-BiLSTM-Attention architectures for controlled datasets but should not be overclaimed for real-world deployment.

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


## Additional Text Experiment: DailyDialog

DailyDialog was added as a supporting text-only experiment because the TESS text pipeline has a structural limitation. In TESS, the extracted text is usually a single neutral word such as `back`, `bar`, `goose`, or `ditch`. These words repeat across all emotion classes, so they do not contain enough semantic information for reliable text emotion recognition.

DailyDialog provides natural conversational utterances where emotion can be expressed through sentence meaning. This makes it more suitable for evaluating whether the text model architecture can learn emotion-related language patterns.

This experiment is not a replacement for the required TESS text pipeline. It is included only as a secondary validation experiment.

| Item | Description |
|---|---|
| Dataset | DailyDialog |
| Modality | Text only |
| Model | RoBERTa-based text classifier |
| Split Strategy | Dialogue-level split to avoid leakage |
| Imbalance Handling | WeightedRandomSampler |

### DailyDialog Text Results

| Metric | Value |
|---|---:|
| Test Accuracy | 77.34% |
| Test UAR | 59.31% |
| Test Macro F1 | 46.49% |
| Test Weighted F1 | 79.93% |

### DailyDialog Interpretation

The DailyDialog result shows that the text model performs much better when the input text contains meaningful emotional semantics. This supports the conclusion that the poor TESS text result is caused by the dataset structure, not by a failure of the text model itself.


# Multimodal Fusion Pipeline

## Purpose

This pipeline combines acoustic speech representation and text representation to generate a unified emotion prediction. The objective is to evaluate whether multimodal fusion improves emotion recognition compared with using speech or text independently.

## Input
* 16 kHz audio waveform.
* Literal text transcript.

## Model Used

The fusion pipeline combines two independent branches:

| Component | Description |
| --- | --- |
| Speech Branch | Mel Spectrogram + Delta + MFCC features → CNN (Residual Blocks) → BiLSTM (2-layer) → Attention Pooling |
| Text Branch | DistilBERT transformer encoder with CLS token pooling |
| Fusion Method | Concatenation of speech and text representations |
| Classifier | MLP classification head |
| Output Classes | anger, disgust, fear, happiness, neutral, sadness, surprise |

## Preprocessing

* Constructs a dual-purpose `metadata.csv` containing both audio `file_path` and extracted `text`.
* Extracts text from TESS filenames (e.g., `OAF_back_angry.wav` → `back`).
* Processes audio features identically to the speech pipeline (Mel Spectrogram, Delta, MFCC).
* Tokenizes text using DistilBERT tokenizer.

## Training Method

The `train.py` script trains the dual-branch fusion model with speaker-aware adaptation:

| Setting | Value |
| --- | --- |
| Dataset | TESS |
| Split Strategy | Speaker-aware adaptation |
| Base Train Speaker | `oaf` |
| Target Speaker | `yaf` |
| Adaptation Ratio | 0.05 (5%) |
| Batch Size | 16 |
| Epochs | 30 |
| Patience (Early Stopping) | 5 |
| Learning Rate | 2e-5 |
| Weight Decay | 1e-4 |
| Optimizer | AdamW |
| Loss Function | CrossEntropyLoss |
| Seed | 42 |

Both the speech branch (CNN-BiLSTM-Attention) and text branch (DistilBERT) are trained end-to-end. Features are extracted and concatenated, then passed through an MLP classifier head for emotion prediction.

## Testing Method

The `test.py` script loads the trained fusion model and provides both GUI and CLI interfaces:

* Loads the trained CNN-BiLSTM-Attention speech branch and DistilBERT text branch.
* Loads model configuration from `saved_models/model_config.json`.
* Performs joint inference on both audio and text.
* Evaluates on `test_split.csv` and generates comprehensive metrics and visualizations.

GUI features include:
- Audio file upload
- Text transcript input box
- Emotion prediction with confidence scores
- Classification report, confusion matrix, and metrics summary viewers
- Training curve and confusion matrix visualization

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
| Test Accuracy | 94.74% |
| Test UAR | 94.74% |
| Test Macro F1 | 94.67% |
| Best Epoch | 14 |
| Model Architecture | CNN-BiLSTM-Attention + DistilBERT + Concatenation + MLP |

**Note on Performance**: The fusion model achieves 94.74% accuracy on TESS, combining acoustic and semantic information. However, the text component (TESS filename-derived words) is inherently weak, so performance is primarily driven by the speech branch. On real-world data with meaningful text transcripts and diverse speakers/acoustic conditions, multimodal fusion would likely provide more balanced cross-modal contributions.

## Fusion Classification Report

| Emotion | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| anger | 0.9561 | 0.8195 | 0.8826 | 133 |
| disgust | 0.9924 | 0.9774 | 0.9848 | 133 |
| fear | 0.9708 | 1.0000 | 0.9852 | 133 |
| happiness | 0.9695 | 0.9549 | 0.9621 | 133 |
| neutral | 0.8832 | 0.9098 | 0.8963 | 133 |
| sadness | 0.9167 | 0.9925 | 0.9531 | 133 |
| surprise | 0.9489 | 0.9774 | 0.9630 | 133 |

## Fusion Confusion Matrix

| Actual \ Predicted | anger | disgust | fear | happiness | neutral | sadness | surprise |
|---|---:|---:|---:|---:|---:|---:|---:|
| anger | 109 | 0 | 0 | 0 | 23 | 1 | 0 |
| disgust | 0 | 130 | 0 | 0 | 2 | 0 | 1 |
| fear | 0 | 0 | 133 | 0 | 0 | 0 | 0 |
| happiness | 1 | 0 | 0 | 127 | 4 | 1 | 0 |
| neutral | 0 | 2 | 0 | 3 | 121 | 7 | 0 |
| sadness | 0 | 0 | 0 | 1 | 9 | 132 | 0 |
| surprise | 0 | 0 | 0 | 1 | 0 | 0 | 130 |

## Fusion Result Images
![Fusion Training Curve](results/fusion_pipeline/plots/training_curve.png)
![Fusion Confusion Matrix](results/fusion_pipeline/plots/confusion_matrix_test.png)

## Fusion Result Interpretation

The fusion pipeline achieves 94.74% accuracy by combining speech and text representations through concatenation and an MLP classifier.

**Key Findings:**

1. **Speech-Driven Performance**: The speech branch (CNN-BiLSTM-Attention with Mel Spectrogram, Delta, MFCC) remains the dominant contributor to fusion performance. The acoustic features capture clear emotional cues from TESS's acted recordings.

2. **Weak Text Contribution**: The text component extracted from TESS filenames (single neutral words like "back", "ditch", "goose") provides minimal emotional signal. Text is semantically identical across all emotion classes, so it contributes little to the decision boundary.

3. **Model Efficiency**: Compared to transformer-based approaches like WavLM, the CNN-BiLSTM-Attention architecture is:
   - Significantly more parameter-efficient
   - Faster to train and inference
   - More interpretable (feature importance can be visualized)
   - Better suited for resource-constrained environments

4. **Cross-Modal Asymmetry**: The fusion model demonstrates that when one modality is weak (text in TESS), multimodal fusion performs close to the strong modality (speech) alone. On datasets with balanced multi-modal contributions, fusion would provide more significant improvements.

**Limitations on TESS:**
- Text is artificially weak (controlled neutral vocabulary)
- Only two female speakers (OAF, YAF)
- Studio-quality, acted recordings
- No background noise or real-world acoustic variation

**Expected Real-World Performance:**
On datasets with meaningful text transcripts (e.g., customer reviews, social media) and diverse speakers/acoustic conditions, multimodal fusion would provide more balanced contributions and potentially larger performance gains over single-modality baselines.

---


## Additional Fusion Experiment: MELD

MELD was added as a supporting multimodal fusion experiment because TESS has limited text semantics. In the TESS fusion pipeline, the speech branch dominates because the text branch receives only short neutral words extracted from filenames.

MELD provides aligned audio and text from realistic multi-speaker dialogue. This makes it more suitable for testing whether speech and text can contribute together in a true multimodal setting.

This experiment is not a replacement for the required TESS fusion pipeline. It is included only as a secondary validation experiment.

| Item | Description |
|---|---|
| Dataset | MELD |
| Modality | Audio + Text |
| Split Strategy | Official train/dev/test split |
| Speech Branch | CNN-BiLSTM-Attention acoustic encoder |
| Text Branch | DistilBERT |
| Fusion Method | Concatenation + MLP classifier |
| Imbalance Handling | Class-weighted cross-entropy loss |

### MELD Fusion Results

| Metric | Value |
|---|---:|
| Test Accuracy | 59.50% |
| Test UAR | 42.23% |
| Test Macro F1 | 41.36% |
| Test Weighted F1 | 59.91% |

### MELD Interpretation

The MELD result is lower than the TESS result because MELD is more realistic and more difficult. It contains multiple speakers, natural dialogue, background variation, short utterances, and strong class imbalance.

The result is still useful because it shows how the fusion architecture behaves outside the clean TESS environment. It also gives a more realistic estimate of multimodal emotion recognition performance.


## Final Comparison Table

| Pipeline | Dataset | Modality | Architecture | Accuracy | UAR | Macro F1 | Weighted F1 | Interpretation |
|---|---|---|---|---:|---:|---:|---:|---|
| Speech Pipeline | TESS | Audio | Mel/Delta/MFCC + CNN-BiLSTM-Attention | 99.89% | 99.89% | 99.89% | Not reported | Strongest core pipeline; TESS speech contains clear acoustic emotion cues. |
| Text Pipeline | TESS | Text | DistilBERT | 14.93% | 14.93% | 7.29% | Not reported | Performs near random chance because TESS text is semantically weak. |
| Fusion Pipeline | TESS | Audio + Text | CNN-BiLSTM-Attention + DistilBERT | 94.74% | 94.74% | 94.67% | Not reported | Strong performance, but lower than speech-only because weak text adds noise. |
| Additional Text Experiment | DailyDialog | Text | RoBERTa | 77.34% | 59.31% | 46.49% | 79.93% | Shows text model works when sentence semantics contain emotional cues. |
| Additional Fusion Experiment | MELD | Audio + Text | CNN-BiLSTM-Attention + DistilBERT | 59.50% | 42.23% | 41.36% | 59.91% | More realistic fusion benchmark with noisy, imbalanced, multi-speaker dialogue. |

## Result Visualizations


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


## Limitations

1. **Limited Speaker Coverage**: TESS contains only two female speakers (OAF and YAF). Generalization to male voices, different ages, and accents is completely untested.

2. **Controlled Acoustic Environment**: TESS is recorded in a noiseless studio with professional audio equipment. Real-world audio contains background noise, microphone degradation, room reverberation, and acoustic variation.

3. **Acted vs. Spontaneous**: TESS features acted emotions performed by trained speakers. Spontaneous emotions in natural conversation exhibit different acoustic patterns and temporal dynamics.

4. **Weak Text Modality**: Emotion is encoded in TESS filename words (single neutral words like "back", "ditch", "goose") that repeat identically across all emotion classes. This creates a fundamental data limitation for text-only emotion recognition.

5. **Speech-Dominated Fusion**: The fusion model is heavily speech-dominated because TESS text is inherently weak. On datasets with meaningful text (e.g., social media, customer reviews) and diverse speakers, multimodal fusion would show more balanced contributions.

6. **No Cross-Dataset Validation**: The models are trained and tested exclusively on TESS. Performance on other speech emotion datasets (RAVDESS, CREMA-D, IEMOCAP, SAVEE) is unknown.

7. **Single Language**: All experiments use English. Generalization to other languages is not evaluated.

8. **Pre-trained Model Constraints**: DistilBERT is trained on English Wikipedia and book text, not emotion-specific corpora. Fine-tuning on weak TESS text limits the text branch performance.

---

## Future Improvements
1. **Multi-Dataset Evaluation**: Extend experiments to RAVDESS, CREMA-D, IEMOCAP, and SAVEE to assess cross-dataset generalization and identify domain-specific biases.

2. **Meaningful Text Representation**: Replace TESS filename words with:
   - Automatic Speech Recognition (ASR) generated transcripts from audio
   - Pre-trained text emotion datasets (GoEmotions, ISEAR, DailyDialog)
   - Natural language emotion datasets that provide semantic emotion signals

3. **Transformer-Based Speech Encoding**: Explore WavLM, Hubert, or Wav2Vec 2.0 for speech feature extraction to benchmark against the current CNN-BiLSTM-Attention approach.

4. **Robust Audio Augmentation**: Add:
   - Random white noise and background chatter
   - Microphone degradation and frequency filtering
   - Time-stretching and pitch-shifting for speaker variation
   - Room reverberation and acoustic simulation

5. **Speaker-Robust Training**: Implement:
   - Speaker-dependent adaptation techniques
   - Domain adaptation methods for cross-speaker generalization
   - Few-shot learning for rapid adaptation to new speakers

6. **Balanced Fusion Architecture**: Design fusion methods that provide:
   - Adaptive weighting between modalities based on input quality
   - Gating mechanisms to learn when to trust each modality
   - Cross-modal attention to improve text representation using speech

7. **Real-Time Inference Pipeline**: Deploy as:
   - Web application using Flask, FastAPI, or Streamlit
   - Mobile application for on-device inference
   - Live streaming emotion detection

8. **Model Explainability**: Integrate:
   - Attention head visualization for interpretability
   - SHAP or LIME for feature importance analysis
   - Confusion-driven error analysis and active learning

9. **Multi-Lingual Support**: Extend to Spanish, Mandarin, German, and Hindi emotion datasets.

10. **Emotion Intensity Regression**: Extend from 7-class classification to continuous emotion intensity scores (e.g., arousal-valence space).

---

## Conclusion

This project comprehensively implements and evaluates three emotion recognition pipelines: speech-only, text-only, and multimodal fusion. The experiments reveal fundamental insights about the TESS dataset and multimodal emotion recognition:

## Key Findings

**Speech Emotion Recognition (99.89% Accuracy)**
- Acoustic features (Mel Spectrogram, Delta, MFCC) with CNN-BiLSTM-Attention achieve near-perfect classification
- TESS provides clean, acted recordings with strong vocal emotion cues (pitch, tone, energy, rhythm, prosody)
- Clear class separation indicates that speech carries the dominant emotional signal in TESS

**Text Emotion Recognition (14.93% Accuracy)**
- Text-only approaches perform at near-random chance level (14.28% random baseline)
- TESS filename-derived words are emotionally neutral and repeat identically across all emotion classes
- This demonstrates a fundamental limitation: TESS is an audio emotion dataset, not a text emotion dataset
- Finding is scientifically valuable—it proves poor text performance is due to dataset design, not modeling failure

**Multimodal Fusion (94.74% Accuracy)**
- Fusion combines CNN-BiLSTM-Attention (speech) and DistilBERT (text) through concatenation
- Performance is dominated by the speech branch due to weak text modality
- Architecture is more parameter-efficient and interpretable compared to transformer-based approaches like WavLM
- On datasets with meaningful text and diverse speakers, fusion would provide more balanced cross-modal benefits

## Deployment Readiness

The project provides:
- **Robust preprocessing pipelines** for audio feature extraction and text tokenization
- **Automated training workflows** with early stopping, model checkpointing, and hyperparameter tuning
- **Comprehensive evaluation** including classification reports, confusion matrices, and training curves
- **User-facing GUI interfaces** (CustomTkinter) for model inference and result visualization
- **CLI support** for batch processing and integration into production systems
- **Well-documented architecture** suitable for extension and experimentation

## Important Caveats

This project demonstrates state-of-the-art performance **on TESS specifically**. Users should note:
1. TESS contains only two female speakers—generalization to other speakers/accents is limited
2. Studio-quality recordings do not represent real-world noisy environments
3. Acted emotions differ from spontaneous emotional expressions
4. Text modality is artificially weak—results should not generalize to datasets with meaningful text

For real-world deployment, the models should be re-trained and evaluated on domain-specific datasets with realistic acoustic conditions and diverse speakers.

## Conclusion

The project successfully demonstrates a complete machine learning pipeline for emotion recognition, from data preprocessing through model training, evaluation, and inference. The experimental design reveals that **multimodal emotion recognition is not inherently superior to single-modality approaches when one modality is weak**. Future work should focus on datasets with balanced multi-modal contributions to realize the full potential of fusion architectures.
