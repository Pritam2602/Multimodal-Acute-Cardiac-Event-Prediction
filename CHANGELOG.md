# Changelog

## 2026-04-28

### Dataset Preprocessing
- Moved preprocessing-owned feature/schema definitions into `Dataset/feature_schema.py`.
- Kept feature engineering and strict-label preprocessing in `Dataset/dataset_preprocessing.py`.
- Updated model-side code to import shared preprocessing definitions from `Dataset/` instead of defining them directly in `late_fusion/`.
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
- Added a safe fallback so existing parquet-only runs keep the current `AMI` column when diagnosis CSVs are unavailable.
- Moved diagnosis-audit outputs under `Dataset/analysis/outputs/diagnosis_label_audit`.
- Re-ran preprocessing on the current dataset:
  final parquet shape changed from `125757 x 28` to `125757 x 50`.

### Model Feature Configuration
- Updated `late_fusion/config.py` to import the engineered feature list from `Dataset/feature_schema.py`.
- `NUM_CLINICAL_FEATURES` now resolves dynamically from `len(CLINICAL_FEATURES)`.
- Late-fusion clinical input size changed from `24` to `46`, so older checkpoints are no longer compatible with the updated config.

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

### Recommended Next Runs
- Grouped split baseline:
  ```powershell
  python -m early_fusion.train --loss-name bce --auto-continue --run-name grouped_engineered_bce
  ```
- Grouped cross-validation:
  ```powershell
  python -m early_fusion.cross_validate --loss-name bce --run-name grouped_crossval_engineered_bce
  ```
