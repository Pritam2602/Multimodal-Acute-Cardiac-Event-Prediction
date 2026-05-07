# Changelog

## 2026-05-07 (Session 2) — Error Analysis & Label Refinement

### Error Analysis
- Ran `analysis/error_analysis.py` on the best model (`strict_ami_early_focal_ecgtiming`, F1=0.7224)
- Identified **3 root causes** for the F1 plateau:
  1. **False Positives (1,284)**: driven by `age_x_log1p_Troponin_T` — Type 2 MI patients (demand ischemia from sepsis/PE/HF) with elevated troponin but NOT acute coronary syndrome
  2. **False Negatives (1,055)**: low-troponin AMI cases (mean 0.15 vs TP mean 1.96), plus heavy patient duplication (e.g. p11062890 appears 7x as FN)
  3. **Dead features**: 15 zero-variance features (8 invalid flags, 6 missing flags, ecg_time_missing) adding noise

### Fix 1: Remove Zero-Variance Features
- Removed 15 features from `CLINICAL_FEATURES` in `config.py` (59 -> 44 features)

### Fix 2: Troponin Delta Feature
- Added `troponin_delta_24h = troponin_max_24h - troponin_first_24h` (computed on-the-fly in `dataset.py`)
- Captures rising troponin trajectory to separate true AMI from chronic elevation

### Fix 3: Troponin-to-Creatinine Ratio
- Added `troponin_creatinine_ratio = log1p_Troponin_T / Creatinine.clip(0.1)` (computed on-the-fly in `dataset.py`)
- Normalises troponin by kidney function to reduce CKD false positives

### Fix 4: Per-Patient ECG Cap
- Added `--max-ecgs-per-patient N` CLI flag (dataset.py `_apply_data_filters()`)
- Limits repeated ECGs from the same patient to reduce FN dominance by edge cases

### Fix 5: ECG Timing Window
- Added `--ecg-time-window N` CLI flag to keep ECGs within +/-N hours of admission
- ECGs taken days after admission contain less acute signal

### Fix 6: Type 1 MI Label Refinement (Parquet Rebuild)
- Modified `create_ami_flag()` in `dataset_preprocessing.py` to **exclude ICD-10 I21A subcodes**:
  - I21A1 = "Myocardial infarction type 2" (1,465 cases)
  - I21A9 = "Other myocardial infarction type" (13 cases)
- Rebuilt parquet: AMI positives changed from 23,921 -> 22,013 (removed 1,908 Type 2 MI cases)
- These were demand-ischemia cases presenting with elevated troponin from non-cardiac causes

### Experiment Results

| Run | Arch | Key Changes | Best F1 | Precision | Recall |
|---|---|---|---|---|---|
| Previous best | baseline | 59 features, old labels | **0.7256** | 0.691 | 0.764 |
| `baseline_engineered_v1` | baseline | New features, old labels | 0.7228 | 0.693 | 0.756 |
| `filmed_filtered_v1` | filmed | +/-72h filter + 3-ECG cap | 0.7112 | 0.681 | 0.745 |
| `type1_baseline_v1` | baseline | New features + Type 1 labels | 0.7183 | **0.719** | 0.718 |
| `type1_filmed_v1` | filmed | FiLM + Type 1 labels | 0.7130 | **0.731** | 0.695 |

**Key finding**: The Type 1 label refinement significantly improved **precision** (0.719-0.731 vs 0.691) by removing confusing Type 2 MI cases. However, F1 stayed in the 0.71-0.72 range because we now have fewer positive examples and the remaining task is harder (pure Type 1 MI detection).

### Files Modified
- `early_fusion/config.py` — removed 15 dead features, added 2 engineered features, added filter constants
- `early_fusion/dataset.py` — `_compute_engineered_features()`, `_apply_data_filters()`, updated `load_and_prepare_data()`
- `early_fusion/train.py` — added `--ecg-time-window`, `--max-ecgs-per-patient` CLI args
- `Dataset_preproceesing/dataset_preprocessing.py` — excluded I21A (Type 2 MI) from AMI label

## 2026-05-07

### Training Pipeline Improvements (Tier 1)
- **OneCycleLR scheduler** — replaced `CosineAnnealingLR` (epoch-level) with `OneCycleLR` (per-batch stepping, 10% warmup → peak → cosine decay). Added `--scheduler` CLI flag (`onecycle` | `cosine`).
- **Label smoothing** — added label smoothing in `engine.py:train_one_epoch()` that softens 0/1 targets to `0.05`/`0.95`. Controlled via `--label-smoothing` CLI flag (default `0.05`).
- **Stronger ECG augmentation** — added three new augmentations to `dataset.py:_augment_ecg()`:
  - Random lead dropout (30% chance, zeros out 1–2 leads)
  - Random temporal cutout (20% chance, masks 100–500 contiguous samples)
  - Random baseline wander (25% chance, sinusoidal drift)
- **Config changes** (`config.py`):
  - `EARLY_STOPPING_PATIENCE`: `5` → `10`
  - `THRESHOLD_SEARCH_STEPS`: `50` → `200`
  - Added: `LABEL_SMOOTHING = 0.05`, `SCHEDULER_TYPE = "onecycle"`
  - Added: `AUG_LEAD_DROP_PROB`, `AUG_CUTOUT_PROB`, `AUG_WANDER_PROB`
- **Experiment: `onecycle_aug_v2`**
  - arch=baseline, scheduler=onecycle, label_smoothing=0.05, lr=1e-3
  - Best Val F1: `0.7256` at epoch 3 (early stopped at epoch 13)
  - Result: marginal improvement over previous best (`0.7224`), still in the same plateau.

### FiLM-Conditioned Multi-Scale Architecture (Tier 2)
- Added `FiLMedEarlyFusionModel` in `model.py` (`--model-arch filmed`):
  - **FiLM fusion**: clinical features generate per-channel scale/shift (gamma/beta) to modulate ECG feature maps at 4 layers, replacing the crude channel-broadcast approach.
  - **Multi-scale convolutions**: parallel conv branches at kernel sizes 3, 7, 15 capture both fine QRS detail and broad ST-segment morphology.
  - **Squeeze-and-Excitation**: channel attention blocks after each multi-scale conv.
  - **2-layer BiLSTM** (bidirectional, hidden=128) replaces 1-layer BiLSTM.
  - **Dual-path classifier**: concatenates attention-pooled ECG features with clinical encoder output before final MLP.
  - Parameters: `2,108,082` (vs `286,386` baseline, 7.4× larger).
- Added supporting modules: `SqueezeExcitation`, `FiLMLayer`, `MultiScaleConvBlock`.
- Old architectures (`baseline`, `resnet`) are fully preserved for backward compatibility.
- **Experiment: `filmed_focal_v1`**
  - arch=filmed, scheduler=onecycle, label_smoothing=0.05, lr=3e-4
  - Best Val F1: `0.7250` at epoch 9 (13 epochs completed)
  - Val loss was still decreasing through epoch 8 (`0.2912`) — model had more capacity to learn compared to baseline.
  - Result: comparable F1 to baseline despite fundamentally different fusion. The 0.72–0.73 ceiling appears to be a data-level bottleneck, not purely architectural.

### Files Modified
- `early_fusion/config.py` — new constants for augmentation, scheduler, label smoothing, patience
- `early_fusion/dataset.py` — enhanced `_augment_ecg()` with lead dropout, cutout, wander
- `early_fusion/engine.py` — label smoothing + per-batch scheduler stepping in `train_one_epoch()`
- `early_fusion/train.py` — OneCycleLR support, `--scheduler`, `--label-smoothing`, `--model-arch filmed`
- `early_fusion/model.py` — added `FiLMedEarlyFusionModel` and supporting modules

## 2026-05-06

### ECG-Admission Timing Features
- Downloaded the full MIMIC-IV-ECG `machine_measurements.csv` using `boto3` and the local `.env` AWS credentials.
- Added `download_machine_measurements.py` for repeatable download without hardcoding secrets in source code.
- Added ECG timing features in `Dataset_preproceesing/dataset_preprocessing.py` by extracting `study_id` from `ecg_path`, merging `ecg_time`, and aligning to `admissions.csv.gz`:
  `hours_admit_to_ecg`, `ecg_within_first_6h`, `ecg_within_first_24h`,
  `ecg_before_admission`, `ecg_after_discharge`, and `ecg_time_missing`.
- Updated `early_fusion.config`, `late_fusion.config`, `explain.py`, and `store_predictions.py` for the new timing features.
- Rebuilt `mimic_data/final_preprocessed_fusion_dataset.parquet`:
  - shape: `125757 x 63`
  - configured model features: `59`
  - model feature nulls/infs: `0`
- Recommended run:
  ```powershell
  python -m early_fusion.train --loss-name focal --auto-continue --run-name strict_ami_early_focal_ecgtiming
  ```

### First-24h Lab Timing Features
- Added admission-window lab features from `labevents.csv.gz`, aligned to `admissions.csv.gz` `admittime` by `hadm_id`.
- New features:
  `troponin_first_24h`, `troponin_max_24h`, `troponin_count_24h`,
  `hours_admit_to_first_troponin`, `troponin_24h_missing`,
  `creatinine_max_24h`, and `potassium_min_24h`.
- Updated `early_fusion.config` and `late_fusion.config`.
- Rebuilt `mimic_data/final_preprocessed_fusion_dataset.parquet`:
  - shape: `125757 x 57`
  - configured model features: `53`
  - model feature nulls/infs: `0`
- Updated explanation/prediction code so old 46-feature runs and new 53-feature runs can coexist.
- Recommended run:
  ```powershell
  python -m early_fusion.train --loss-name focal --auto-continue --run-name strict_ami_early_focal_labtiming
  ```

### Optional Stronger ECG Encoder
- Added optional `--model-arch resnet` to `early_fusion.train` and `early_fusion.cross_validate`.
- Default remains `--model-arch baseline`, preserving compatibility with the previous best model.
- Current best model still remains `strict_ami_grouped_early_focal` unless a newer run beats validation F1 `0.7153`.

## 2026-04-28

### Dataset Preprocessing
- Added engineered clinical and ECG-machine features in `Dataset_preproceesing/dataset_preprocessing.py`.
- Added invalid-value indicators for ECG-machine measurements before median imputation:
  `PR_interval_invalid`, `QRS_duration_invalid`, `QT_interval_invalid`, `QTc_invalid`,
  `P_axis_invalid`, `QRS_axis_invalid`, `T_axis_invalid`, and `RR_interval_invalid`.
- Added troponin-derived features:
  `log1p_Troponin_T`, `Troponin_T_positive`, `Troponin_T_high`,
  and `age_x_log1p_Troponin_T`.
- Added clinical abnormality flags:
  `Creatinine_high`, `Potassium_low`, and `Potassium_high`.
- Added ECG-machine abnormality flags:
  `QRS_wide`, `QTc_prolonged`, `PR_prolonged`,
  `QRS_axis_deviation`, and `T_axis_abnormal`.
- Added heart-rate consistency features:
  `HR_from_RR_interval` and `HR_RR_disagreement`.
- Replaced hard-coded Windows string paths with `Path`-based project-relative paths.
- Added fallback behavior so the script can enrich the existing
  `final_preprocessed_fusion_dataset.parquet` when the source temporal parquet is unavailable.
- Re-ran preprocessing on the current dataset:
  final parquet shape changed from `125757 x 28` to `125757 x 50`.

### Model Feature Configuration
- Updated `early_fusion/config.py` to include the engineered features.
- Updated `late_fusion/config.py` to include the engineered features.
- `NUM_CLINICAL_FEATURES` now resolves dynamically from `len(CLINICAL_FEATURES)`.
- Early/late fusion clinical input size changed from `24` to `46`, so older checkpoints are no longer compatible with the updated configs.

### Training Experiments
- Ran `engineered_weighted_bce`:
  - `weighted_sampling=true`
  - validation F1: `0.599048`
  - result: worse than previous `focal_tune`; weighted sampling plus BCE `pos_weight` was too aggressive.
- Ran `engineered_bce_no_sampler`:
  - `weighted_sampling=false`
  - validation F1: `0.610790`
  - validation AUC: `0.853243`
  - result: new best early-fusion validation F1 so far.

### Cross-Validation and Sampling
- Confirmed `--weighted-sampling` is available for early and late fusion cross-validation.
- Fixed missing `import copy` in:
  - `early_fusion/cross_validate.py`
  - `late_fusion/cross_validate.py`
- Added `--pos-weight-scale` to early-fusion training and cross-validation so the
  computed positive-class weight can be reduced or increased per experiment.
- Stored raw and effective positive-class weights in early-fusion metrics outputs.

### Leakage-Safe Grouped Splitting
- Added patient-level grouped split support in `early_fusion/dataset.py`.
- Since the final parquet no longer stores `subject_id`, subject groups are recovered from `ecg_path`.
- Replaced row-level train/validation/test splitting with grouped splitting for early-fusion training.
- Updated `early_fusion/cross_validate.py` to use `StratifiedGroupKFold` when group IDs are available.
- Added split-overlap logging to verify that the same patient does not appear in multiple splits.
- Dry split check:
  - total subject groups: `34,135`
  - train-pool/test group overlap: `0`
  - train/validation group overlap: `0`
  - grouped train prevalence: `21.39%`
  - grouped validation prevalence: `19.63%`
  - grouped test prevalence: `21.61%`

### Intermediate Fusion Package
- Added a new `intermediate_fusion/` package with files mirroring the early-fusion training pipeline:
  `config.py`, `dataset.py`, `engine.py`, `losses.py`, `model.py`, `plots.py`,
  `train.py`, `cross_validate.py`, and `__init__.py`.
- Added `IntermediateFusionModel`, a two-branch architecture:
  ECG waveform encoder and clinical MLP encoder are trained separately, then their learned embeddings are concatenated before final classification.
- Set intermediate-fusion artifacts to save under `intermediate_fusion/artifacts`.
- Added grouped-split support, weighted sampling, and `--pos-weight-scale` support through the copied training and cross-validation pipelines.
- Verified the model forward pass with `46` clinical features and output shape `(batch,)`.

### Recommended Next Runs
- Grouped split baseline:
  ```powershell
  python -m early_fusion.train --loss-name bce --auto-continue --run-name grouped_engineered_bce
  ```
- Grouped cross-validation:
  ```powershell
  python -m early_fusion.cross_validate --loss-name bce --run-name grouped_crossval_engineered_bce
  ```
- Lower positive-weight BCE test:
  ```powershell
  python -m early_fusion.train --loss-name bce --pos-weight-scale 0.5 --auto-continue --run-name grouped_engineered_bce_pw05
  ```
- Intermediate fusion baseline:
  ```powershell
  python -m intermediate_fusion.train --loss-name focal --auto-continue --run-name grouped_intermediate_focal
  ```

### Clinical-Only Baseline
- Added `clinical_baseline/` with a grouped clinical-only training script using the same engineered `CLINICAL_FEATURES`.
- Models included:
  `logistic_balanced` and `hist_gradient_boosting`.
- Ran grouped clinical-only baseline:
  - best model: `hist_gradient_boosting`
  - validation F1: `0.5305`
  - validation AUC: `0.7959`
  - validation AP: `0.5281`
- Result: clinical-only performance is below the best grouped early-fusion run, so ECG appears to add useful signal under patient-level splitting.

### Diagnosis Label Audit
- Added `analysis/diagnosis_label_audit.py` to compare the old broad text-match AMI label against a stricter acute/current ICD-code AMI label.
- Updated `Dataset_preproceesing/dataset_preprocessing.py` so future dataset rebuilds use acute/current MI ICD rules instead of broad `"myocardial infarction"` text matching.
- Audit result:
  - broad text-match MI admissions: `30,496`
  - acute/current ICD-code MI admissions: `12,172`
  - broad-only admissions: `22,412`
- Main broad-only source:
  - ICD-9 `412`: `Old myocardial infarction`
  - ICD-10 `I252`: `Old myocardial infarction`
- Interpretation: the old label likely mixed acute AMI with prior/history MI, which can confuse training and evaluation.
- Re-ran preprocessing after confirming `hadm_id` is present in the final parquet.
- Current `final_preprocessed_fusion_dataset.parquet` status:
  - shape: `125757 x 50`
  - `hadm_id`: present
  - engineered features: present
  - model feature nulls/infs: `0`
  - strict AMI positives: `23,921`
  - strict AMI prevalence: `19.02%`

### Strict AMI Error Analysis
- Ran `analysis.error_analysis` on `early_fusion/artifacts/runs/strict_ami_grouped_early_focal`.
- Validation outcome counts at threshold `0.687755`:
  - TP: `3,172`
  - FP: `1,598`
  - TN: `15,970`
  - FN: `927`
- Threshold search confirmed the saved threshold is already the best validation-F1 point.
- Main FP-vs-TN shifts were troponin-heavy:
  `age_x_log1p_Troponin_T`, `Troponin_T_high`, `Troponin_T`, plus ECG interval changes.
- Main FN-vs-TP shifts showed false negatives often have much weaker troponin signal despite positive strict AMI labels.
- Added `--exclude-clinical-features` to `early_fusion.train` for controlled ablation runs without changing the global dataset.
- Recommended ablation:
  ```powershell
  python -m early_fusion.train --loss-name focal --exclude-clinical-features "Troponin_T,log1p_Troponin_T,Troponin_T_positive,Troponin_T_high,age_x_log1p_Troponin_T" --auto-continue --run-name strict_ami_early_focal_no_troponin_features
  ```

### Reverted Interaction Feature Experiment
- Tested error-driven interaction features in `strict_ami_early_focal_interactions`.
- Result: best validation F1 was `0.7088`, below the previous strict 46-feature focal run (`0.7153`).
- Reverted the active dataset/model feature list back to the previous `46` clinical features.
- Rebuilt `mimic_data/final_preprocessed_fusion_dataset.parquet`:
  - shape: `125757 x 50`
  - configured model features: `46`
  - model feature nulls/infs: `0`
  - strict AMI positives: `23,921`

### Optional Stronger ECG Encoder
- Added optional `--model-arch resnet` to `early_fusion.train` and `early_fusion.cross_validate`.
- Default remains `--model-arch baseline`, preserving compatibility with the previous best model.
- New ResNet ECG extractor uses residual 1D convolution blocks plus attention pooling.
- Smoke check:
  - baseline model: `285,554` parameters
  - resnet model: `2,023,314` parameters
  - both produce output shape `(batch,)` with the current `46` clinical features.
- Recommended run:
  ```powershell
  python -m early_fusion.train --loss-name focal --model-arch resnet --auto-continue --run-name strict_ami_early_resnet_focal
  ```

### First-24h Lab Timing Features
- Added admission-window lab features from `labevents.csv.gz`, aligned to `admissions.csv.gz` `admittime` by `hadm_id`.
- New features:
  - `troponin_first_24h`
  - `troponin_max_24h`
  - `troponin_count_24h`
  - `hours_admit_to_first_troponin`
  - `troponin_24h_missing`
  - `creatinine_max_24h`
  - `potassium_min_24h`
- Updated `early_fusion.config` and `late_fusion.config`.
- Rebuilt `mimic_data/final_preprocessed_fusion_dataset.parquet`:
  - shape: `125757 x 57`
  - configured model features: `53`
  - model feature nulls/infs: `0`
- Updated explanation/prediction code so old 46-feature runs and new 53-feature runs can coexist.
- Recommended run:
  ```powershell
  python -m early_fusion.train --loss-name focal --auto-continue --run-name strict_ami_early_focal_labtiming
  ```
