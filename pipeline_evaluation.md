# Multimodal Emotion Recognition — Critical Pipeline Evaluation

---

## SPEECH PIPELINE EVALUATION

---

### 1. Overall Verdict

**⚠️ PARTIALLY CORRECT — Strong architecture but results are not trustworthy.**

The Speech pipeline uses WavLM (`microsoft/wavlm-base`) with mean pooling and a classification head — a correct and modern transformer-based approach. Preprocessing, feature extraction, and inference code are well-structured. However, the reported test accuracy of **99.88%** is inflated and unreliable due to a fundamental split design problem: the test set comes **only from YAF speaker**, which the model was specifically trained to generalize to using only 14% adapt ratio. The model memorized TESS dataset's highly constrained acoustic patterns, producing near-perfect results on a toy evaluation set. This does not generalize to real speech.

---

### 2. Pipeline Block Verification Table

| Pipeline Block | Expected Function | Evidence from Files | Status | Issue |
|---|---|---|---|---|
| **Preprocessing** | Handle varying audio length, 16kHz, trim/pad, normalize | `preprocess_audio()` in train.py: loads at SR=16000, trims with top_db=30, pads/crops to 3s, normalizes | ✅ Pass | None |
| **Augmentation** | Data augmentation during training | `augment_audio()`: noise, gain, roll, pitch_shift, time_stretch — applied to train split only | ✅ Pass | None |
| **Feature Extraction** | WavLM feature extractor applied per sample | `AutoFeatureExtractor` with `padding=max_length`, `truncation=True`, `max_length=48000` per sample | ⚠️ Partial | Feature extraction happens inside `__getitem__` — no caching. Runs on every epoch. Extremely slow. |
| **Temporal Modelling** | Transformer encoder learns temporal audio patterns | `AutoModel.from_pretrained("microsoft/wavlm-base")` + `mean(hidden_states, dim=1)` pooling | ✅ Pass | None |
| **Classifier** | Linear classifier predicts 7-class emotion | `Dropout(0.3) → Linear(768, 256) → ReLU → Dropout(0.3) → Linear(256, 7)` | ✅ Pass | None |
| **Speaker-based Split** | OAF trains, YAF evaluates | `BASE_TRAIN_SPEAKER=oaf`, `TARGET_SPEAKER=yaf`, 14% adapt_ratio from YAF added to train | ⚠️ Partial | Test set = only YAF. No cross-speaker test included. Model may not generalize to unseen speakers. |
| **Checkpointing** | Best model saved, config saved | `torch.save(model.state_dict(), best_model.pth)` + `model_config.json` | ✅ Pass | None |
| **Metrics** | Accuracy, UAR, Macro F1 reported | `evaluate()` computes all three correctly | ✅ Pass | None |
| **Plots** | Training curve, confusion matrix | `training_curve.png`, `confusion_matrix.png` in plots/ | ✅ Pass | None |

---

### 3. Code Review

#### preprocess.py
- **Correct:** Parses TESS folder structure, maps folder names to emotion labels including all variants of surprise (ps, pleasant_surprise, pleasant_surprised)
- **Correct:** Records speaker_id, raw_emotion, file_path, emotion — all needed fields
- **⚠️ Issue:** Does NOT perform audio loading or validation during preprocessing. No check for corrupted/empty wav files. These errors only surface during training.
- **⚠️ Issue:** No train/val/test split in preprocess.py — all splitting is done inside train.py's `build_training_data()`. This couples preprocessing to training logic.

#### train.py
- **Correct:** WavLM architecture correctly initialized from HuggingFace
- **Correct:** `AutoFeatureExtractor` handles normalization for WavLM input
- **Correct:** `mean(hidden_states, dim=1)` pooling is a valid temporal pooling approach
- **Correct:** Augmentation applied only to training set
- **Correct:** Gradient clipping at 1.0, AdamW optimizer
- **Correct:** Early stopping (patience=5), best model saved per F1
- **🔴 CRITICAL:** Feature extraction inside `__getitem__` → runs on every batch, every epoch. This is not cached. For 30+ epochs on 1800+ audio files, this is extremely slow and wasteful.
- **⚠️ Issue:** Class weights are hardcoded: `[1.1, 1.0, 1.0, 1.15, 1.0, 1.0, 1.05]`. TESS is a balanced dataset (~200 samples per class), so weights are not critical here, but they are not computed dynamically.
- **⚠️ Issue:** No LR scheduler used. A linear warmup or cosine decay scheduler would improve fine-tuning stability for transformer models.
- **⚠️ Issue:** `model.load_state_dict(torch.load(...))` uses default `weights_only=False` — deprecated warning in newer PyTorch.
- **⚠️ Issue:** The confusion matrix in results/ uses `test_actuals`/`test_predictions` as integer-encoded values without mapping to class names in the labels — this works because CLASS_NAMES ordering matches encoder.

#### test.py (speech)
- Not directly reviewed here, but the GUI loads from `model_config.json` + `best_model.pth` (state_dict only format) — both confirmed present.
- **⚠️ Issue:** test.py in speech pipeline was not shown, but based on checkpoints this should be compatible with state_dict-only loading (`torch.load(best_model.pth, map_location=device)`).

---

### 4. Output Folder Review

| Artifact | File | Size | Status |
|---|---|---|---|
| Model weights | `saved_models/best_model.pth` | 378 MB | ✅ Present |
| Model config | `saved_models/model_config.json` | 154 B | ✅ Present |
| Training metrics | `metrics/training_metrics.csv` | 522 B | ✅ Present |
| Classification report | `results/classification_report.csv` | 519 B | ✅ Present |
| Confusion matrix | `results/confusion_matrix.csv` | 229 B | ✅ Present |
| Summary metrics | `results/summary.csv` | 125 B | ✅ Present |
| Training curve | `plots/training_curve.png` | 21 KB | ✅ Present |
| Confusion matrix plot | `plots/confusion_matrix.png` | 34 KB | ✅ Present |
| **Classification report TXT** | — | — | ❌ Missing |
| **Text metrics JSON** | — | — | ❌ Missing |

> [!WARNING]
> Speech pipeline is missing a plaintext `classification_report.txt` and a `speech_metrics.json`. The text pipeline has both. For submission consistency, these should be added.

---

### 5. Performance Review

| Metric | Training Epoch 9 (val) | Test |
|---|---|---|
| Accuracy | 100% | 99.88% |
| UAR / Macro Recall | 100% | 99.88% |
| Macro F1 | 100% | 99.88% |

**Class-wise test performance (from classification_report.csv):**
- All 7 classes: precision=1.0, recall=1.0, F1=1.0 (except happiness and surprise which have ~0.991-0.996)

> [!CAUTION]
> **These results are not credible for a real-world use case.** TESS is a controlled dataset with only 2 speakers, studio-recorded, single-word utterances with highly distinct acoustic profiles per emotion. Getting 99.88% on YAF-only test set is expected — the model learned speaker-specific features. **This is NOT generalizable SER performance.** In your report, you MUST note this as a dataset-specific result and mention that the model is tuned for this specific dataset distribution.

**Training curve analysis:** Val F1 reached 100% at epoch 4 and stayed there for 5 epochs until early stopping. Train loss continued dropping. This confirms **memorization**, not generalization.

---

### 6. Data Leakage Review

| Check | Details | Status |
|---|---|---|
| Train/Val/Test split by speaker | OAF → train, YAF → val+test with 86/14 split | ✅ No cross-speaker leakage |
| Same word in multiple splits? | Possible — TESS uses same words across emotions (e.g., "back" said in all 7 emotions). Only emotion differs, not the word itself | ⚠️ Structural overlap |
| Adapt ratio | 14% of YAF per class added to train | ✅ Controlled |
| Remaining YAF → val (30%) + test (70%) | Verified in build_training_data() | ✅ Correct |

> [!NOTE]
> Since TESS uses the **same 200 carrier words** across all speakers and emotions, the model hears the WORD "back" spoken by OAF-angry during training, and then the word "back" spoken by YAF-angry during test. This means the model can pick up on lexical anchors even in acoustic features. This is a **dataset design limitation**, not a code error.

---

### 7. Problems Found

| # | Problem | Severity |
|---|---|---|
| 1 | Feature extraction (WavLM AutoFeatureExtractor) runs inside `__getitem__` without caching — extremely slow over 30 epochs | High |
| 2 | Test results (99.88%) reflect TESS dataset memorization, not generalizable SER | High |
| 3 | No LR scheduler for transformer fine-tuning — training stability is suboptimal | Medium |
| 4 | Hardcoded class weights not computed from actual class distribution | Low |
| 5 | Missing `classification_report.txt` and `speech_metrics.json` in results for submission completeness | Low |
| 6 | `torch.load` without `weights_only=True` produces deprecation warning in PyTorch >= 2.0 | Low |
| 7 | Audio validation (corrupted files) not done during preprocess — errors only surface during training | Low |

---

### 8. Recommended Fixes

1. **Cache features:** Precompute WavLM features and save as `.npy` files in `processed_data/`. Load from disk in `__getitem__`. This is exactly what the `processed_data/` folder (currently containing `.npy` files) was intended for — but the current `train.py` does not use them.

2. **Add LR scheduler:** Add `get_linear_schedule_with_warmup` from transformers (same as text pipeline) with `warmup_ratio=0.1`.

3. **Add plaintext report:** At the end of train.py, write `classification_report.txt` to results/ and a `speech_metrics.json` to metrics/.

4. **Fix torch.load warning:** Use `torch.load(path, map_location=device, weights_only=True)` — or wrap with `weights_only=False` explicitly if loading legacy checkpoint format.

5. **Report limitation in submission:** State clearly that 99.88% accuracy is on the TESS controlled dataset and is expected to be lower on spontaneous/uncontrolled speech.

---

---

## TEXT PIPELINE EVALUATION

---

### 1. Overall Verdict

**✅ CORRECT — Well-structured, honest performance, dialogue-level leakage prevention in place.**

The Text pipeline correctly implements a RoBERTa-base transformer encoder with mean pooling and a classification head. The preprocessing uses dialogue-level group splits to prevent leakage. The performance is honest and consistent with DailyDialog benchmark results: high accuracy is dominated by the neutral class, while minority classes (disgust, fear, anger) have lower F1 scores — which is accurate and expected for this dataset.

---

### 2. Pipeline Block Verification Table

| Pipeline Block | Expected Function | Evidence from Files | Status | Issue |
|---|---|---|---|---|
| **Preprocessing** | Clean text, handle empty inputs, 7-class labels, no leakage | `preprocess.py`: strips, removes URLs, maps emotion IDs, group splits by dialogue_id, leakage verified | ✅ Pass | None |
| **Feature Extraction** | RoBERTa tokenization, max_length, attention mask, padding, truncation | `tokenize_dataframe()` in train.py: batch tokenize all data once before training | ✅ Pass | Tokenized once, not per-batch |
| **Contextual Modelling** | RoBERTa encoder processes token sequences with self-attention | `AutoModel.from_pretrained("roberta-base")` applied to `input_ids` + `attention_mask` | ✅ Pass | None |
| **Mean Pooling** | Weighted mean over token embeddings | `mean_pooling()`: mask × hidden_states / clamp(sum, min=1e-9) | ✅ Pass | Numerically safe |
| **Classifier** | 7-class emotion output | `Dropout(0.3) → Linear(768, 256) → ReLU → Dropout(0.3) → Linear(256, 7)` | ✅ Pass | None |
| **Class Imbalance** | Weighted loss, capped class weights | Dynamic `create_class_weights()` capped at 2.5, label smoothing 0.03 | ✅ Pass | None |
| **Data Leakage** | No dialogue appears in multiple splits | `GroupShuffleSplit` by dialogue_id, verified with `check_dialogue_leakage()` | ✅ Pass | None |
| **Checkpoint** | Best model by val macro F1 saved | `torch.save({model_state_dict, label_classes, config, best_epoch, best_val_macro_f1})` | ✅ Pass | Saves full dict, not state_dict only |
| **Mixed Precision** | AMP for GPU training | `torch.amp.GradScaler` + `autocast` | ✅ Pass | Enabled only if CUDA available |
| **Early Stopping** | Patience=2 by val macro F1 | Confirmed in train.py — stopped at epoch 8 | ✅ Pass | None |

---

### 3. Code Review

#### preprocess.py
- **Correct:** Parses DailyDialog's `__eou__` delimited text and space-separated emotion IDs
- **Correct:** Skips utterances with mismatched lengths, invalid IDs, or too-short text
- **Correct:** Deduplication by `(dialogue_id, utterance_id, text, emotion)` — safe deduplication that doesn't collapse valid variations
- **Correct:** Group split by `dialogue_id` — prevents dialogue context leakage
- **Correct:** Saves `split_info.json` documenting split sizes and leakage check result
- **⚠️ Minor:** `clean_text()` only removes URLs and extra whitespace. It does NOT lowercase or remove punctuation. This is actually FINE for RoBERTa (which is cased and handles punctuation), but contrasts with the old `clean_text()` in test.py that previously did lowercase + regex strip.

#### train.py
- **Correct:** `EncodedTextDataset` stores pre-tokenized tensors — no per-batch tokenization
- **Correct:** Warmup ratio (6%), linear LR scheduler, AdamW with weight_decay=0.01
- **Correct:** Gradient clipping at `GRAD_CLIP=1.0`
- **Correct:** `val_loader` and `test_loader` use `batch_size * 2` for faster evaluation
- **Correct:** Dialogue leakage verified at start of main()
- **Correct:** `evaluate()` now returns weighted_f1 in addition to macro_f1, UAR
- **⚠️ Issue:** Training metrics show val_macro_f1 peaked at **45.31%** at epoch 8 but test macro F1 is **49.77%** — test is 4.5% HIGHER than the best validation F1. This is unusual. Usually test F1 ≤ val F1. Possible explanation: DailyDialog's group split is approximate, not exact 70/15/15, and the test split may happen to have a slightly easier distribution.
- **⚠️ Issue:** `best_val_f1` in checkpoint dict is saved as the value *before* the epoch that triggered saving — compare epoch 8 in training_metrics.csv: val_macro_f1 = 45.31%, saved as `best_validation_macro_f1 = 45.31` ✅ — this is consistent.
- **⚠️ Issue:** `num_workers = 2 if os.name == 'nt' else 4`. On Windows, `num_workers > 0` with PyTorch can cause multiprocessing issues if not run inside `if __name__ == '__main__'`. This is already satisfied since train.py uses `if __name__ == "__main__": main()`.

#### test.py (GUI)
- **Correct:** Loads from both `MODEL_PATH` and `CONFIG_PATH`, raises separate errors if either is missing
- **Correct:** Reads `dropout_rate` from config and passes to model init — consistent with training
- **Correct:** Handles both old (state_dict only) and new (dict with `model_state_dict` key) checkpoint formats
- **Correct:** Uncertainty logic: marks prediction as "uncertain" if confidence < 50% or gap to 2nd < 8%
- **Correct:** Top-3 predictions displayed in the confidence label
- **⚠️ Issue:** `clean_text()` in test.py does NOT apply the same cleaning as the old `predict_emotion` in the previous version (which lowercased). The current version matches `preprocess.py`'s `clean_text()` — just strip + URL removal. **This is correct and consistent now.**
- **⚠️ Issue:** `saved_models/text_emotion_model.pkl` is still present (19.9 KB — old scikit-learn model). This is a leftover from a previous version and should be deleted to avoid confusion.

#### y.py (Batch Terminal Inference)
- **Correct:** Mirrors `test.py` load logic exactly
- **Correct:** Outputs to `metrics/batch_text_predictions.csv`
- **Correct:** Top-3 + confidence gap + final_label (with uncertainty rule) all computed
- **✅ Bonus:** This is a clean standalone inference script that doesn't require the GUI.

---

### 4. Output Folder Review

| Artifact | File | Status |
|---|---|---|
| Model weights | `saved_models/text_emotion_model.pth` (499 MB) | ✅ Present |
| Model config | `saved_models/model_config.json` | ✅ Present |
| **Stale file** | `saved_models/text_emotion_model.pkl` (19.9 KB) | ⚠️ Should be deleted |
| Training metrics | `metrics/training_metrics.csv` | ✅ Present |
| Classification report TXT | `metrics/classification_report.txt` | ✅ Present |
| Classification report CSV | `metrics/classification_report.csv` | ✅ Present |
| Confusion matrix CSV | `metrics/confusion_matrix.csv` | ✅ Present |
| Text metrics JSON | `metrics/text_metrics.json` | ✅ Present |
| Split info | `processed_data/split_info.json` | ✅ Present |
| Training curve | `plots/training_curve.png` | ✅ Present |
| Confusion matrix plot | `plots/confusion_matrix.png` | ✅ Present |
| Class distribution | `plots/class_distribution.png` | ✅ Present |
| Batch predictions | `metrics/batch_text_predictions.csv` | ✅ Present |

---

### 5. Performance Review

| Metric | Value | Notes |
|---|---|---|
| Test Accuracy | **81.30%** | High because neutral dominates (83% of test set) |
| Test UAR (Macro Recall) | **56.68%** | More honest — minority classes recall is weak |
| Test Macro F1 | **49.77%** | Honest and expected for DailyDialog |
| Test Weighted F1 | **82.82%** | Neutral-weighted, inflated |

**Class-wise analysis (from classification_report.txt):**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| anger | 0.36 | 0.40 | 0.38 | 154 |
| disgust | 0.21 | 0.34 | 0.26 | 50 |
| fear | 0.44 | 0.50 | 0.47 | 32 |
| happiness | 0.47 | 0.74 | 0.57 | 1942 |
| **neutral** | **0.94** | **0.84** | **0.89** | **12944** |
| sadness | 0.29 | 0.49 | 0.36 | 170 |
| surprise | 0.47 | 0.67 | 0.55 | 289 |

**Key observations:**
- Neutral F1 = 89% drags up accuracy and weighted F1 — this is **misleading**
- Disgust and fear have very low F1 due to tiny support (50 and 32 samples) — **not reliable estimates**
- Confusion matrix confirms: neutral is a "vacuum" class absorbing off-emotion predictions (11,944 correct but also 1,571 happiness and 89 anger misclassified as neutral)
- **Macro F1 of 49.77% is the honest number to report** — it gives equal weight to all classes

**Imbalance handling:**
- Capped weighted loss ✅
- Label smoothing 0.03 ✅
- No undersampling or oversampling — minority classes (fear=32, disgust=50) have too few test samples to evaluate reliably

---

### 6. Data Leakage Review

| Check | Details | Status |
|---|---|---|
| Dialogue-level split | `GroupShuffleSplit` by `dialogue_id` — same dialogue never in two splits | ✅ Pass |
| Leakage verification code | `check_dialogue_leakage()` called in both preprocess.py and train.py | ✅ Pass |
| Split info documented | `split_info.json` records dialogue counts per split and leakage_check=passed | ✅ Pass |
| Total samples | 102,965 total → 71,913 train / 15,471 val / 15,581 test | ✅ Correct proportions |
| Dialogue counts | 9,181 train / 1,968 val / 1,968 test — **no overlap possible** | ✅ Pass |

> [!NOTE]
> This is the correct approach for DailyDialog. The earlier version used stratified random split without dialogue grouping — that version had potential leakage because consecutive utterances from the same dialogue could appear in both train and test. The current implementation correctly prevents this.

---

### 7. Problems Found

| # | Problem | Severity |
|---|---|---|
| 1 | `text_emotion_model.pkl` stale file (old sklearn model) still exists in saved_models/ — could confuse GUI if loaded accidentally | Medium |
| 2 | Test macro F1 (49.77%) > Best val macro F1 (45.31%) — suggests possible evaluation inconsistency or favorable test split distribution | Medium |
| 3 | Disgust (n=50) and fear (n=32) test support are too small for reliable per-class metrics — F1 estimates for these classes have high variance | Medium |
| 4 | val_loss increases every epoch (1.30 → 1.41) while train_loss decreases — model is **overfitting** progressively | Medium |
| 5 | Patience = 2 is very tight — best model was saved at epoch 8 and stopped immediately. With patience=3, one more epoch might have been attempted | Low |
| 6 | No `num_workers` cap on GPU machines — `num_workers=2` on Windows is fine but may be slow on multicore systems | Low |
| 7 | `class_distribution.png` exists in plots/ but is not generated by the current train.py — likely a leftover from a previous version | Low |

---

### 8. Recommended Fixes

1. **Delete stale file:** Remove `saved_models/text_emotion_model.pkl` — it is a leftover from the old scikit-learn pipeline.

2. **Note overfitting in report:** val_loss increases every epoch while train_loss decreases from 1.49 to 0.93. Mention label smoothing and dropout as regularization strategies used, and note that early stopping at patience=2 helped prevent further overfit.

3. **Add class distribution plot generation in train.py:** Either generate `class_distribution.png` in train.py or remove it from the plots folder if it was generated separately.

4. **In report, use Macro F1 (49.77%) as the primary metric** — not accuracy (81.30%), as the dataset is highly imbalanced by neutral.

5. **For fear/disgust discussion:** Note in the report that these classes have very low test support (<50 samples) making their per-class metrics unreliable estimates.

---

---

## 9. Final Submission Readiness

### Speech Pipeline

| Item | Status |
|---|---|
| Architecture correct (WavLM + classifier) | ✅ |
| Preprocessing correct | ✅ |
| Model saved and loadable | ✅ |
| Results consistent | ✅ |
| Performance credible | ⚠️ — inflate by dataset |
| Missing plaintext classification report | ❌ |
| Missing speech_metrics.json | ❌ |

**Speech pipeline is functionally complete but needs two output files added.**

### Text Pipeline

| Item | Status |
|---|---|
| Architecture correct (RoBERTa + mean pool + classifier) | ✅ |
| Preprocessing correct with leakage prevention | ✅ |
| Model saved and loadable | ✅ |
| All output files present | ✅ |
| Performance honest and consistent with literature | ✅ |
| Stale .pkl file | ⚠️ should delete |

**Text pipeline is submission-ready.**

---

### What MUST be fixed before submission

1. ✅ **Text pipeline:** Delete `saved_models/text_emotion_model.pkl`
2. ❌ **Speech pipeline:** Add `results/classification_report.txt` and `metrics/speech_metrics.json`

### What can be mentioned as limitations in the report

1. **Speech:** "Test accuracy of 99.88% is specific to the TESS controlled dataset (2 speakers, studio-recorded, single-word utterances). Real-world SER on spontaneous speech would yield significantly lower performance."
2. **Speech:** "Feature extraction runs per sample per epoch without caching, which slows training. Future work would cache WavLM embeddings."
3. **Text:** "DailyDialog has severe class imbalance (neutral ~83% of dataset). Macro F1 of 49.77% is the reliable metric; accuracy of 81.30% is inflated by the neutral class."
4. **Text:** "Minority classes (fear: 32 test samples, disgust: 50 test samples) have insufficient test support for reliable per-class metric estimation."
5. **Both:** "No fusion layer implemented — speech and text pipelines are evaluated independently."
