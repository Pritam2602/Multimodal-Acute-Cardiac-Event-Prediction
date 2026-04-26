# Project File Descriptions

Generated on 2026-04-23.

This document gives a detailed, file-by-file explanation of the repository for **Multimodal Acute Cardiac Event Prediction**. The project predicts Acute Myocardial Infarction (AMI) from two data streams: raw 12-lead ECG waveforms and structured clinical EHR features. The current implemented model is an early-fusion PyTorch model, with supporting preprocessing scripts, training utilities, prediction storage, explainability, and a FastAPI serving layer.

## Repository-Level Summary

The repository is organized around an end-to-end AMI prediction workflow:

1. Raw MIMIC-IV, MIMIC-IV-ECG, clinical, diagnosis, and machine-measurement files are transformed into a fused parquet dataset.
2. ECG records are downloaded from S3 and cached locally as WFDB `.hea` and `.dat` files.
3. Clinical features are cleaned, imputed, standardized, and paired with ECG paths and AMI labels.
4. The early-fusion model combines projected clinical channels with ECG channels before shared temporal modeling.
5. Training saves checkpoints, metrics, model weights, and diagnostic plots.
6. A prediction-storage script evaluates registered models, selects the best model, computes explanations, and writes frontend-ready records into SQLite.
7. The FastAPI app serves patients, ECG waveforms, metrics, comparisons, and explainability data.

## Root Files

### `.gitignore`

This file defines which files should be excluded from version control. It ignores common generated, private, or large artifacts such as documents, archives, logs, local environment files, IDE folders, virtual environments, parquet data, raw ECG waveform files, NumPy arrays, and ECG cache directories. It explicitly un-ignores `requirements.txt`, ensuring dependency information remains versioned even though generic `.txt` files are ignored.

Important role:

- Protects large MIMIC-derived datasets and waveform files from being committed.
- Prevents accidental exposure of `.env` credentials.
- Keeps generated caches and local development clutter out of the repository.

### `README.md`

The README is the high-level project introduction. It explains that the system predicts AMI using a multimodal approach combining raw ECG signals and clinical EHR features. It outlines the core methodology: data collection, preprocessing, temporal feature engineering, ECG processing, AMI label creation, early/late fusion comparison, and evaluation metrics.

Important role:

- Gives new readers the project motivation and intended workflow.
- Describes the two modeling strategies conceptually.
- Notes that training can save resumable checkpoints and best validation-F1 weights.

Notable detail:

- The file contains mojibake/encoding artifacts in several headings and symbols, likely caused by UTF-8 text being interpreted with the wrong encoding.

### `architecture_summary.md`

This is the main architecture reference. It documents the early-fusion model in detail and proposes a late-fusion design for future implementation. It describes input shapes, model blocks, fusion points, fair comparison rules, data splitting, training setup, recommended metrics, and the API/frontend context.

Important role:

- Serves as the design document for early-vs-late fusion experiments.
- Clarifies why the current model is genuinely early fusion: clinical features are projected into temporal channels and concatenated with ECG before shared CNN and BiLSTM processing finishes.
- Defines the intended late-fusion contrast: separate ECG and clinical branches that combine only near the prediction head.

### `walkthrough.md`

This file is an operational walkthrough of training, comparison, prediction storage, explainability, and serving. It explains the pipeline from `python -m early_fusion.train` through `store_predictions.py` and `api.py`.

Important role:

- Helps users run the project end to end.
- Explains current model status and planned late-fusion status.
- Describes how trained models are compared and how predictions are stored for the dashboard.

Notable detail:

- Some links point to `D:\MINI_PROJECT`, which may be a previous project location rather than the current local path.

### `requirements.txt`

This file pins the Python dependencies needed for data handling, ECG processing, machine learning, deep learning, visualization, utilities, and API serving.

Main dependency groups:

- Data: `pandas`, `numpy`, `pyarrow`
- AWS and ECG: `boto3`, `wfdb`
- ML/DL: `scikit-learn`, `torch`, `torchvision`, `torchaudio`
- Visualization: `matplotlib`, `seaborn`
- API: `fastapi`, `uvicorn`

Important role:

- Recreates the expected runtime environment.
- Makes the project more reproducible by pinning exact package versions.

### `LICENSE`

This file contains the Apache License 2.0 text. It defines permissions and conditions for use, modification, distribution, patent grant, and warranty disclaimers.

Important role:

- Provides the legal terms under which the project code may be used and redistributed.

### `api.py`

This is the FastAPI serving layer for the frontend/dashboard. It reads from a shared SQLite database located at `early_fusion/artifacts/predictions.db` and exposes model-agnostic endpoints for patients, ECG signals, explanations, metrics, statistics, and model comparisons.

Main responsibilities:

- Creates the FastAPI app and enables permissive CORS.
- Opens SQLite connections through `get_db()`.
- Serves paginated patient lists via `GET /api/patients`.
- Serves full patient records via `GET /api/patients/{patient_id}`.
- Loads local WFDB ECG records on demand via `GET /api/patients/{patient_id}/ecg`.
- Serves ranked feature attribution explanations via `GET /api/patients/{patient_id}/insights`.
- Returns aggregate metrics via `GET /api/metrics`.
- Returns dashboard summary counts via `GET /api/stats`.
- Returns model comparison results via `GET /api/comparison`.

Inputs:

- SQLite tables created by `store_predictions.py`.
- Local ECG WFDB paths stored in the database.

Outputs:

- JSON responses consumed by a dashboard or other client.

Important implementation details:

- ECG waveforms are loaded lazily with `wfdb.rdrecord`.
- Signals are cleaned for NaN and infinity before JSON serialization.
- ECG leads are padded to 12 leads if fewer are present.
- Patient gender is converted from encoded values to display strings in some responses.

Potential caution:

- CORS is fully open with `allow_origins=["*"]`, which is fine for local development but should be restricted for production.
- The database path is under `early_fusion/artifacts`, even though the API is intended to be model-agnostic.

### `explain.py`

This file contains shared explainability logic for any PyTorch model with a `forward(ecg, clinical)` signature. It computes gradient-times-input attributions for clinical features and converts them into human-readable reasoning entries.

Main responsibilities:

- Defines the 24 clinical feature names used by the model.
- Defines normal clinical ranges for frontend explanation text.
- Maps internal feature names to friendly display names.
- Computes clinical feature attributions with Gradient x Input.
- Builds ranked explanation dictionaries for top contributing clinical features.
- Produces a short summary string for each patient.

Inputs:

- A trained PyTorch model.
- One ECG tensor shaped like `(1, 12, 5000)`.
- One clinical tensor shaped like `(1, N_features)`.
- Raw clinical values for natural-language explanation.

Outputs:

- Attribution scores.
- Sigmoid AMI probability.
- Ranked reasoning entries and a concise summary.

Important implementation details:

- Only clinical feature attributions are computed, not ECG waveform attributions.
- Attributions are normalized by the maximum attribution value.
- Features with known normal ranges are classified as normal, abnormal, or critical based on deviation.

Potential caution:

- Explanations use raw values from the dataframe, while the model receives standardized clinical inputs. This is appropriate for user-facing text but means attribution magnitude is based on standardized inputs.
- `logits.backward()` assumes a single forward pass with scalar output. That matches current per-patient usage.

### `store_predictions.py`

This script is the bridge between trained models and the serving database. It evaluates all registered models on the same held-out test set, picks the best model by F1 score, computes explainability for the selected model, and stores patient-level predictions in SQLite.

Main responsibilities:

- Defines a model registry with import paths, class names, weight paths, constructor configs, metrics paths, and dataset modules.
- Creates the SQLite schema for `patients`, `feature_importances`, `model_metrics`, and `model_comparison`.
- Dynamically loads registered models.
- Reads the saved best threshold from metrics JSON when available.
- Evaluates each model on the same test set.
- Selects the best model by F1 score.
- Runs gradient-based clinical explanations for each test patient.
- Stores frontend-ready predictions, risk levels, clinical values, explanations, and aggregate metrics.

Inputs:

- Data from `early_fusion.dataset.load_and_prepare_data`.
- Trained model weights from `early_fusion/artifacts/models/early_fusion_model.pth`.
- Optional threshold from `early_fusion/artifacts/metrics/metrics.json`.

Outputs:

- SQLite database at `early_fusion/artifacts/predictions.db`.

Important implementation details:

- Uses dynamic imports so future models can be added to `MODEL_REGISTRY`.
- Normalizes ECGs per lead at inference in the same style as training.
- Falls back to zero ECG arrays if waveform loading fails.
- Stores model comparison rows even though only early fusion is currently registered.

Potential caution:

- `evaluate_model()` imports `early_fusion.dataset.load_ecg_signal` and `early_fusion.config` directly, so full model-agnostic support would need slight refactoring when late fusion or other modules diverge.
- The script deletes the existing predictions database before recreating it.

## `Dataset/` Preprocessing Scripts

### `Dataset/ECG_Machine_Dataset.py`

This script processes machine-generated ECG measurement data. It starts from `machine_measurements.csv`, removes free-text report columns, engineers numeric ECG tabular features, and writes `ecg_features.csv`.

Main responsibilities:

- Drops `report_0` through `report_17`.
- Computes heart rate from RR interval.
- Computes PR interval, QRS duration, QT interval, and QTc.
- Renames/keeps axis features and RR interval.
- Saves a compact CSV containing subject/study IDs and engineered ECG machine features.

Inputs:

- `machine_measurements.csv`.

Outputs:

- `ecg_features.csv`.

Important implementation details:

- QTc uses Bazett correction: `QT_interval / sqrt(rr_interval / 1000)`.
- The output is later merged with ECG and clinical metadata by `ecg_dataset_combine_with_clinical.py`.

### `Dataset/ecg_dataset_combine_with_clinical.py`

This script builds an intermediate ECG-plus-clinical dataset by listing ECG files from S3, extracting IDs from file paths, merging machine ECG features, matching ECGs to admissions, and combining them with clinical data.

Main responsibilities:

- Lists `.hea` ECG files from the MIMIC-IV-ECG S3 access point.
- Derives `subject_id`, `study_id`, `hea_key`, and `dat_key`.
- Merges engineered ECG features from `ecg_features.csv`.
- Loads `record_list.csv` and `admissions.csv`.
- Keeps ECGs that occur during the admission window.
- Merges with `mimic_clinical_dataset.parquet`.
- Cleans duplicate columns and selects final fields.
- Saves `final_ecg_clinical_dataset.parquet`.

Inputs:

- MIMIC-IV-ECG S3 object listing.
- `ecg_features.csv`.
- `record_list.csv`.
- `admissions.csv`.
- `mimic_clinical_dataset.parquet`.

Outputs:

- `final_ecg_clinical_dataset.parquet`.

Potential caution:

- AWS credentials are hard-coded as placeholders. In production or shared code, credentials should come from environment variables or IAM roles.
- The script uses local filenames without a central config, so it depends heavily on being run from the expected working directory.

### `Dataset/temporal_clinical_extraction_with_mapping.py`

This script extracts temporally valid clinical features from MIMIC events by using item mapping tables and retaining only observations recorded before each ECG time.

Main responsibilities:

- Loads admissions, chartevents, labevents, ECG record list, `d_items`, and `d_labitems`.
- Converts admission, discharge, ECG, chart, and lab times to datetime.
- Finds relevant item IDs by keyword matching in mapping tables.
- Merges chart/lab events with ECG records.
- Filters events to those at or before the ECG timestamp.
- Extracts the latest pre-ECG value for each feature.
- Merges vital and lab features.
- Saves `temporal_clinical_features.parquet`.

Inputs:

- MIMIC admissions, chartevents, labevents, record list, item dictionaries, and lab item dictionaries.

Outputs:

- `temporal_clinical_features.parquet`.

Important implementation details:

- The temporal filter is central to avoiding future-information leakage.
- The script groups by `subject_id`, `hadm_id`, and `study_id`, then takes the last available value before ECG.

Potential caution:

- Several path strings use backslashes without raw-string prefixes. In Python, sequences like `\m` are usually tolerated but can create warnings or confusion. Raw strings or `Path` objects would be safer.
- Keyword matching such as `"troponin"` or `"blood pressure"` may pull multiple item IDs and should be reviewed for clinical precision.

### `Dataset/dataset_preprocessing.py`

This script turns temporal clinical features into the final model-ready parquet dataset. It imputes missing values, creates missingness indicators, encodes gender, removes unused columns, creates AMI labels from ICD diagnoses, and attaches the ECG path.

Main responsibilities:

- Loads `temporal_clinical_features.parquet`.
- Creates `_missing` indicator columns for numeric clinical variables.
- Fills numeric missing values with medians.
- Encodes gender as `M = 1`, `F = 0`.
- Drops unused columns.
- Loads diagnosis and diagnosis-dictionary CSV files.
- Flags AMI diagnoses by searching diagnosis descriptions for myocardial infarction.
- Aggregates AMI labels per admission.
- Merges AMI labels into the feature dataset.
- Drops identifier columns not used by the model.
- Creates `ecg_path` from `dat_key`.
- Saves `final_preprocessed_fusion_dataset.parquet`.

Inputs:

- `temporal_clinical_features.parquet`.
- `diagnoses_icd.csv.gz`.
- `d_icd_diagnoses.csv.gz`.

Outputs:

- `final_preprocessed_fusion_dataset.parquet`.

Potential caution:

- AMI labeling by description text is simple and readable, but ICD-code based label definitions may be more reproducible for research reporting.
- The script assumes `dat_key` still exists when creating `ecg_path`, even though it is dropped from the final modeling table afterward.

## `early_fusion/` Package

### `early_fusion/__init__.py`

This file marks `early_fusion` as a Python package and contains a short package description. Its presence enables module-style imports such as `python -m early_fusion.train`.

### `early_fusion/config.py`

This is the central configuration file for early-fusion training and evaluation. It defines project paths, artifact directories, clinical feature names, ECG settings, training hyperparameters, threshold-search settings, reproducibility seed, ECG lead names, and frontend/explainability normal ranges.

Main responsibilities:

- Defines `PROJECT_ROOT` and paths to dataset, models, plots, metrics, and database.
- Lists the 24 clinical features expected by the model.
- Defines target column `AMI`.
- Sets ECG shape constants: 12 leads and 5000 samples.
- Sets training hyperparameters such as batch size, learning rate, epochs, dropout, focal loss parameters, split ratios, and threshold-search range.
- Adjusts DataLoader worker settings for Windows vs non-Windows platforms.
- Defines normal ranges used by the UI/explainability layer.

Important implementation details:

- The model expects clinical features in exactly the order listed here.
- `TEST_SPLIT` creates a held-out test set for later dashboard and model comparison.
- Normal ranges are explicitly documented as UI/explainability ranges, not preprocessing filters.

### `early_fusion/dataset.py`

This file implements ECG downloading, ECG loading, lazy PyTorch dataset behavior, DataLoader creation, clinical cleaning, train/validation/test splitting, standardization, and class-weight computation.

Main responsibilities:

- Loads `.env` credentials for AWS S3.
- Creates thread-local S3 clients.
- Converts S3 ECG paths into local cache paths.
- Downloads `.hea` and `.dat` files in parallel.
- Loads WFDB ECG records and normalizes them to shape `(12, 5000)`.
- Implements `MultimodalCardiacDataset`, which lazy-loads ECGs per sample.
- Builds DataLoaders with OS-sensitive worker settings.
- Loads the final preprocessed parquet dataset.
- Cleans invalid clinical values using wide physiological ranges.
- Imputes clinical missing values with medians.
- Splits data into train pool, held-out test, train, and validation sets.
- Standardizes clinical values using train-pool statistics only.
- Returns train/validation loaders, positive class weight, and test metadata.

Inputs:

- `mimic_data/final_preprocessed_fusion_dataset.parquet`.
- ECG S3 paths from the parquet dataset.
- Local `.env` AWS credentials.

Outputs:

- `train_loader`.
- `val_loader`.
- `pos_weight`.
- `test_metadata` with test dataframe, local paths, standardized clinical arrays, labels, and standardization statistics.

Important implementation details:

- ECG loading pads/truncates all records to 12 leads and 5000 samples.
- ECG normalization is per lead using z-score normalization.
- Failed ECG loads return zero tensors, making training robust to some missing/corrupt records.
- Standardization is fit on the train pool before train/validation splitting to avoid test leakage.

Potential caution:

- `download_all_ecgs()` may trigger large network activity and long runtime for full datasets.
- Median imputation occurs before splitting, which is a small possible leakage point because medians are computed over all rows before standardization. Standardization itself avoids test leakage.

### `early_fusion/model.py`

This file defines the current early-fusion neural network architecture. It contains an ECG/clinical shared feature extractor and the full AMI classifier.

Main classes:

- `EarlyFusionFeatureExtractor`
- `EarlyFusionModel`

Architecture:

- Clinical features are projected from 24 values into `clinical_channels` channels.
- The projected clinical channels are broadcast across the ECG time axis.
- The broadcast clinical representation is concatenated with the raw ECG channels.
- The fused sequence passes through three Conv1D, BatchNorm, ReLU, and MaxPool blocks.
- The convolution output is transposed into sequence format and passed through a bidirectional LSTM.
- The final LSTM timestep is transformed through dense layers.
- A classifier outputs one AMI logit.

Inputs:

- `ecg`: tensor shaped `(batch, 12, 5000)`.
- `clinical`: tensor shaped `(batch, 24)`.

Outputs:

- Logits shaped `(batch,)`.

Why it is early fusion:

- Clinical information is merged into the ECG channel dimension before shared CNN and BiLSTM temporal modeling is complete. The temporal encoder therefore learns from ECG and clinical context jointly.

Potential caution:

- The model uses the final BiLSTM timestep rather than pooling across all timesteps. That is simple and efficient, but temporal attention or pooling could be explored later.

### `early_fusion/losses.py`

This file defines `FocalLoss`, a binary classification loss designed to reduce the dominance of easy examples and help with class imbalance.

Main responsibilities:

- Wraps `binary_cross_entropy_with_logits`.
- Supports optional positive class weighting.
- Applies focal modulation using `(1 - pt) ** gamma`.
- Returns mean loss over the batch.

Inputs:

- Raw logits.
- Binary targets.

Outputs:

- Scalar focal loss.

Important implementation details:

- `alpha` globally scales the focal loss.
- `gamma` controls how strongly easy examples are down-weighted.
- `pos_weight` can increase the contribution of the positive AMI class.

### `early_fusion/engine.py`

This file contains reusable training/evaluation functions and threshold-search logic.

Main responsibilities:

- Searches for the best validation threshold by F1 score.
- Computes accuracy, precision, recall, F1, ROC-AUC, average precision, and predictions.
- Trains one epoch with gradient clipping.
- Evaluates a model with optional automatic threshold search.
- Optionally returns raw labels, predictions, and probabilities for plots and metrics JSON.

Inputs:

- PyTorch model, DataLoader, criterion, optimizer, device, threshold settings.

Outputs:

- Average loss.
- Metrics dictionary.
- Optional outputs dictionary with labels, predictions, and probabilities.

Important implementation details:

- Training skips batches that produce NaN loss.
- Gradients are clipped using `max_grad_norm=1.0`.
- Threshold search defaults to the config range and step count.

### `early_fusion/train.py`

This is the main training entry point for the early-fusion model.

Main responsibilities:

- Parses command-line arguments for subset training and DataLoader worker overrides.
- Seeds Python, NumPy, and PyTorch for reproducibility.
- Creates artifact directories.
- Loads train/validation data.
- Builds `EarlyFusionModel`.
- Creates `FocalLoss` with positive class weighting.
- Creates Adam optimizer.
- Resumes from `latest_checkpoint.pth` when compatible.
- Trains epoch by epoch.
- Searches validation threshold automatically each epoch.
- Saves best validation-F1 model weights.
- Saves a resumable checkpoint after every epoch.
- Asks the user whether to continue after each epoch.
- Saves final metrics JSON, comparison CSV, and plot images.

Inputs:

- Final preprocessed dataset.
- Cached/downloadable ECG records.
- Optional existing checkpoint.

Outputs:

- `early_fusion/artifacts/models/early_fusion_model.pth`.
- `early_fusion/artifacts/models/latest_checkpoint.pth`.
- `early_fusion/artifacts/metrics/metrics.json`.
- `early_fusion/artifacts/metrics/comparison_table.csv`.
- Plot PNG files in `early_fusion/artifacts/plots`.

Important implementation details:

- Checkpoint payload includes model state, optimizer state, history, best F1, best threshold, args, and RNG states.
- If checkpoint loading fails due to architecture mismatch, training starts fresh.
- Best model saving occurs at epoch 1 for a new run or whenever validation F1 improves.

Potential caution:

- Interactive epoch prompts are helpful for manual training but can block automated runs unless input is provided.

### `early_fusion/plots.py`

This file contains plotting utilities for training diagnostics and model evaluation summaries.

Main responsibilities:

- Uses the non-interactive Matplotlib `Agg` backend.
- Saves train/validation loss curves.
- Saves accuracy, precision, recall, and F1 curves.
- Saves confusion matrix.
- Saves ROC curve.
- Saves precision-recall curve.
- Saves a compact model/split comparison table as a PNG.

Inputs:

- Training history dictionaries.
- Validation labels, predictions, and probabilities.
- Metric values such as AUC and average precision.

Outputs:

- PNG plot files under `early_fusion/artifacts/plots`.

Important implementation details:

- The functions close figures after saving, preventing memory leaks in long runs.
- The comparison-table function turns metric rows into a visual report-ready table.

## Generated Artifacts

### `early_fusion/artifacts/metrics/metrics.json`

This JSON file stores the latest training and validation metrics, epoch history, and validation outputs. It is consumed by `store_predictions.py` to retrieve the saved best threshold when available.

Current contents include:

- Train loss, accuracy, precision, recall, F1, AUC, and average precision.
- Validation loss, accuracy, precision, recall, F1, AUC, and average precision.
- Training history over two completed epochs.
- Validation labels, predictions, and probabilities.

Notable detail:

- The file currently reports `epochs_completed: 2` and `target_epochs: 25`.
- It does not currently include `best_threshold`, although newer `train.py` code writes that field. In this case, `store_predictions.py` will fall back to threshold `0.5`.

### `early_fusion/artifacts/models/early_fusion_model.pth`

This is the saved PyTorch state dictionary for the best early-fusion model according to validation F1 during training. It is loaded by `store_predictions.py` for evaluation and prediction storage.

Important role:

- Represents the deployable trained model weights.
- Does not include optimizer state or training history.

### `early_fusion/artifacts/models/latest_checkpoint.pth`

This is the resumable training checkpoint saved after each epoch. It contains more than model weights: model state, optimizer state, epoch number, best validation F1, best threshold, training history, command-line args, and random number generator states.

Important role:

- Allows interrupted training to continue from the next epoch.
- Preserves enough state to make resumed training closer to a continuous run.

### `early_fusion/artifacts/plots/accuracy.png`

This plot visualizes training and validation accuracy across epochs. It is generated by `save_accuracy_plot()` in `early_fusion/plots.py`.

### `early_fusion/artifacts/plots/loss.png`

This plot visualizes training and validation loss across epochs. It is generated by `save_loss_plot()` in `early_fusion/plots.py`.

### `early_fusion/artifacts/plots/precision.png`

This plot visualizes training and validation precision across epochs. Precision measures how many predicted AMI cases are true AMI cases.

### `early_fusion/artifacts/plots/recall.png`

This plot visualizes training and validation recall across epochs. Recall is especially important for AMI prediction because missed AMI cases are clinically high risk.

### `early_fusion/artifacts/plots/f1.png`

This plot visualizes training and validation F1 score across epochs. F1 balances precision and recall and is used as the main model-selection metric in this project.

### `early_fusion/artifacts/plots/confusion_matrix.png`

This image shows the validation confusion matrix, separating true non-AMI, false AMI, missed AMI, and correctly predicted AMI counts.

### `early_fusion/artifacts/plots/roc_curve.png`

This image shows the ROC curve and AUC for validation predictions. It visualizes the tradeoff between true positive rate and false positive rate across probability thresholds.

### `early_fusion/artifacts/plots/pr_curve.png`

This image shows the precision-recall curve and average precision. It is especially useful for imbalanced AMI classification because it focuses on positive-class detection behavior.

### `early_fusion/artifacts/plots/model_comparison_table.png`

This image is a compact table comparing train and validation metrics. It is generated from the final comparison rows in `train.py`.

## Data Flow Across Files

The main data and model flow is:

```text
Dataset/ECG_Machine_Dataset.py
    -> ecg_features.csv

Dataset/temporal_clinical_extraction_with_mapping.py
    -> temporal_clinical_features.parquet

Dataset/ecg_dataset_combine_with_clinical.py
    -> final_ecg_clinical_dataset.parquet

Dataset/dataset_preprocessing.py
    -> final_preprocessed_fusion_dataset.parquet

early_fusion/dataset.py
    -> train_loader, val_loader, test_metadata

early_fusion/train.py
    -> model weights, checkpoint, metrics, plots

store_predictions.py
    -> predictions.db

api.py
    -> REST API responses for frontend/dashboard
```

## Key Design Decisions

- The project uses **early fusion** as the current implemented model: clinical features are injected into the ECG stream before shared temporal modeling.
- ECG data is **lazy-loaded** to avoid holding many gigabytes of waveform data in memory.
- ECG records are standardized **per lead**, while clinical features are standardized using train-pool statistics.
- Focal loss and positive class weighting are used to handle AMI class imbalance.
- Validation threshold search is used because a fixed `0.5` threshold may not optimize F1 on imbalanced data.
- Prediction serving is decoupled from training through a SQLite database.
- Explainability is currently focused on clinical feature attributions, not waveform attribution.

## Suggested Follow-Up Improvements

- Move hard-coded preprocessing paths into a shared config file or command-line arguments.
- Replace placeholder/hard-coded AWS credentials in preprocessing scripts with environment variables.
- Add `best_threshold` to existing `metrics.json` by rerunning training or updating the metrics artifact format.
- Refactor `store_predictions.py` so model evaluation can call each registered model's own dataset/config utilities instead of importing early-fusion utilities directly.
- Consider computing clinical imputation medians from the train pool only to avoid small leakage from global median imputation.
- Add a true `late_fusion/` package following the architecture described in `architecture_summary.md`.
- Add ECG explainability or waveform saliency if the dashboard needs modality-level reasoning beyond clinical features.
