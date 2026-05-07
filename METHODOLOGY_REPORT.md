# AMI Prediction Methodology Report

## 1. Project Objective

The goal of this project is to predict Acute Myocardial Infarction (AMI) using a multimodal model that combines:

- 12-lead ECG waveform signal
- Clinical features such as vitals, labs, demographics, diagnoses, and ECG machine measurements

AMI is a clinically important but imbalanced classification problem. In the final strict-label dataset, AMI prevalence is approximately `19.02%`, meaning most samples are non-AMI. Because of this imbalance, the project focuses mainly on F1 score, AUC, average precision, precision, and recall instead of accuracy alone.

## 2. Data Sources Used

The project uses MIMIC-style ECG and hospital data stored under `mimic_data/`.

Main files used:

| File | Purpose |
|---|---|
| `final_preprocessed_fusion_dataset.parquet` | Main model-ready multimodal dataset |
| `diagnoses_icd.csv.gz` | ICD diagnosis codes used for AMI label creation |
| `d_icd_diagnoses.csv.gz` | Diagnosis code descriptions |
| `admissions.csv.gz` | Admission and discharge times |
| `labevents.csv.gz` | Lab timing and lab value extraction |
| `machine_measurements.csv` | ECG machine measurements and ECG acquisition time |

The final active dataset after the latest preprocessing has:

- Rows: `125,757`
- Columns: `63`
- Configured model clinical features: `59`
- Strict AMI positives: `23,921`
- AMI prevalence: `19.02%`
- Missing values in configured model features: `0`

## 3. Label Creation

Initially, AMI labels were based on broad diagnosis text matching for `"myocardial infarction"`. This was too broad because it included historical MI codes, such as:

- ICD-9 `412`: Old myocardial infarction
- ICD-10 `I25.2`: Old myocardial infarction

This can confuse the model because a patient with a previous MI is not necessarily having an acute MI during the current ECG/admission.

### Final AMI Label Algorithm

A stricter acute/current AMI label was created from ICD codes:

- ICD-9: codes starting with `410`, using fifth digit `1` or `2`
- ICD-10: codes starting with `I21` or `I22`
- Old/history MI codes are excluded

This made the target clinically cleaner and reduced label noise.

Audit result:

| Label Type | Count |
|---|---:|
| Broad text-match MI admissions | `30,496` |
| Strict acute/current MI admissions | `12,172` |
| Broad-only admissions | `22,412` |

## 4. Preprocessing Pipeline

The preprocessing script is implemented in:

`Dataset_preproceesing/dataset_preprocessing.py`

### 4.1 Missing Value Handling

Missing indicators were created before imputation for important clinical variables:

- `Troponin_T_missing`
- `Creatinine_missing`
- `Sodium_missing`
- `Potassium_missing`
- `Heart_Rate_missing`
- `Respiratory_Rate_missing`

This is useful because missingness itself can contain clinical signal. For example, whether troponin was ordered or missing can reflect clinical suspicion or workflow.

After missing flags were created, numeric missing values were filled using median imputation.

### 4.2 Invalid Value Cleaning

Wide physiological validity ranges were used to detect impossible or sentinel values. Invalid values were converted to missing and then imputed.

For ECG machine measurements, invalid-value flags were also created:

- `PR_interval_invalid`
- `QRS_duration_invalid`
- `QT_interval_invalid`
- `QTc_invalid`
- `P_axis_invalid`
- `QRS_axis_invalid`
- `T_axis_invalid`
- `RR_interval_invalid`

This protects the model from extreme machine placeholder values such as impossible ECG intervals or axes.

### 4.3 Categorical Encoding

Gender was encoded numerically:

- Male: `1`
- Female: `0`

### 4.4 Clinical Feature Engineering

Troponin-derived features:

- `log1p_Troponin_T`
- `Troponin_T_positive`
- `Troponin_T_high`
- `age_x_log1p_Troponin_T`

Why useful:

Troponin is one of the strongest biomarkers for myocardial injury. Log transformation helps reduce the effect of extreme values, while binary flags give the model direct clinical threshold information.

Renal/electrolyte abnormality features:

- `Creatinine_high`
- `Potassium_low`
- `Potassium_high`

Why useful:

Renal dysfunction and electrolyte imbalance can affect cardiac risk, ECG interpretation, and patient acuity.

ECG machine abnormality features:

- `QRS_wide`
- `QTc_prolonged`
- `PR_prolonged`
- `QRS_axis_deviation`
- `T_axis_abnormal`

Why useful:

These encode clinically interpretable ECG abnormalities that may be related to ischemia, conduction delay, or cardiac stress.

Heart-rate consistency features:

- `HR_from_RR_interval`
- `HR_RR_disagreement`

Why useful:

The model gets both recorded clinical heart rate and ECG-derived heart rate consistency. Large disagreement may indicate measurement noise, arrhythmia, or timing mismatch.

### 4.5 First-24h Lab Timing Features

From `labevents.csv.gz`, labs were aligned to admission time using `hadm_id`.

Added features:

- `troponin_first_24h`
- `troponin_max_24h`
- `troponin_count_24h`
- `hours_admit_to_first_troponin`
- `troponin_24h_missing`
- `creatinine_max_24h`
- `potassium_min_24h`

Why useful:

AMI evidence is time-sensitive. A high troponin near admission is more relevant than a late abnormal lab. These features help the model distinguish early AMI-like patterns from late or nonspecific abnormalities.

### 4.6 ECG-Admission Timing Features

The full MIMIC-IV-ECG `machine_measurements.csv` was downloaded using `boto3`. The preprocessing now extracts `study_id` from `ecg_path`, merges `ecg_time`, and aligns it with admission/discharge times.

Added features:

- `hours_admit_to_ecg`
- `ecg_within_first_6h`
- `ecg_within_first_24h`
- `ecg_before_admission`
- `ecg_after_discharge`
- `ecg_time_missing`

Why useful:

The ECG waveform should be interpreted in clinical time. An ECG taken early in admission is usually more relevant for acute presentation than an ECG taken much later. Timing features help reduce ambiguity between acute and historical findings.

## 5. Data Splitting Strategy

Initially, random row-level splitting was used. This can be misleading if the same patient has multiple ECGs, because ECGs from the same patient can appear in both train and validation sets.

The final approach uses patient-level grouped splitting:

- Group ID is recovered from `ecg_path`
- Same patient does not appear across train, validation, and test
- `StratifiedGroupKFold` is used for cross-validation when possible

Grouped split verification:

| Check | Result |
|---|---:|
| Subject groups | `34,135` |
| Train/test group overlap | `0` |
| Train/validation group overlap | `0` |

This makes validation more realistic and reduces leakage.

## 6. Models and Algorithms Considered

### 6.1 Clinical-Only Baseline

Algorithms:

- Balanced logistic regression
- Histogram gradient boosting

Purpose:

To test how much predictive signal exists in tabular clinical data alone.

Best result:

| Model | Validation F1 | Validation AUC | Validation AP |
|---|---:|---:|---:|
| Histogram Gradient Boosting | `0.5305` | `0.7959` | `0.5281` |

This became the approximate starting point: F1 around `0.53`.

### 6.2 Early Fusion Model

Early fusion combines ECG waveform features and clinical features inside one neural network.

Architecture:

- ECG branch:
  - 1D convolution layers
  - BiLSTM sequence modeling
  - Attention pooling
- Clinical branch:
  - Fully connected projection layers
- Fusion:
  - ECG embedding and clinical embedding are concatenated
  - Final classifier predicts AMI probability

Why early fusion:

AMI diagnosis often depends on both ECG patterns and clinical context. Early fusion allows the network to learn interactions between ECG waveform morphology and clinical/lab features.

### 6.3 Intermediate Fusion

An intermediate fusion package was created with separate ECG and clinical encoders whose embeddings are concatenated before classification.

Purpose:

To compare a two-branch representation-learning approach against the existing early-fusion model.

Observed result:

Intermediate fusion reached validation F1 around `0.5507` in the tested focal-loss run, which was below the best early-fusion result.

### 6.4 ResNet ECG Encoder

A stronger optional ECG encoder was added:

```powershell
--model-arch resnet
```

Architecture:

- Residual 1D convolution blocks
- Attention pooling
- Clinical fusion head

Purpose:

To test whether a deeper waveform encoder improves ECG representation.

Results:

| Run | Validation F1 |
|---|---:|
| `strict_ami_early_resnet_focal` | `0.7102` |
| `strict_ami_early_resnet_focal_lr3e4_do02` | `0.7122` |

The ResNet version improved representation capacity but did not beat the best baseline early-fusion model.

## 7. Loss Functions and Imbalance Handling

### 7.1 Binary Cross Entropy With Positive Class Weight

BCE was tested with `pos_weight` to compensate for fewer AMI positives.

Also tested:

```powershell
--pos-weight-scale
```

This allowed reducing or increasing the positive-class weight.

### 7.2 Weighted Sampling

Weighted sampling was tested so AMI-positive samples appear more often during training.

Observation:

Weighted sampling plus BCE positive weighting was too aggressive. It increased pressure toward positives and did not improve validation F1.

### 7.3 Focal Loss

Focal loss was used as the main final loss:

```text
FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
```

Why useful:

Focal loss reduces the effect of easy examples and focuses training on harder cases. This is useful in imbalanced medical classification where the model may otherwise learn mostly non-AMI patterns.

Final commonly used settings:

- `focal_alpha = 1.0`
- `focal_gamma = 2.0`
- `pos_weight_scale = 1.0`

## 8. Threshold Optimization

The model outputs probabilities, but F1 depends on the classification threshold.

Instead of using a fixed `0.5` threshold, validation predictions were searched over a threshold range. The threshold with best validation F1 was saved.

Example:

Best strict early-fusion run:

- Best threshold: `0.6878`
- Best validation F1: `0.7153`

This is important because for imbalanced data, the best decision threshold is often not `0.5`.

## 9. Evaluation Metrics

### Accuracy

Accuracy measures the fraction of correct predictions:

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Problem:

The dataset is imbalanced. If only about `19%` are AMI, a naive model can get high accuracy by predicting mostly non-AMI. That would be clinically unsafe because it may miss AMI cases.

### Precision

Precision measures how many predicted AMI cases were truly AMI:

```text
Precision = TP / (TP + FP)
```

High precision means fewer false alarms.

### Recall

Recall measures how many true AMI cases were detected:

```text
Recall = TP / (TP + FN)
```

High recall means fewer missed AMI cases.

### F1 Score

F1 is the harmonic mean of precision and recall:

```text
F1 = 2 * Precision * Recall / (Precision + Recall)
```

Why F1 is suitable here:

- AMI is imbalanced.
- Accuracy can hide poor AMI detection.
- Precision alone may ignore missed AMI cases.
- Recall alone may create too many false alarms.
- F1 balances false positives and false negatives.

Because this project is a binary medical risk prediction task with class imbalance, F1 is a more meaningful primary metric than accuracy.

### AUC

AUC measures ranking quality across thresholds. It shows whether AMI patients generally receive higher predicted probabilities than non-AMI patients.

### Average Precision

Average precision summarizes the precision-recall curve. It is especially useful for imbalanced datasets because it focuses on positive-class detection performance.

## 10. Result Progression

| Stage | Main Change | Validation F1 | Notes |
|---|---|---:|---|
| Clinical-only baseline | Histogram Gradient Boosting on tabular features | `0.5305` | Baseline without raw ECG waveform |
| Engineered BCE, no sampler | Clinical/ECG engineered features, BCE | `0.6108` | Better than clinical-only, but row-level/non-final comparison |
| Grouped BCE | Patient-level grouped split | `0.5448` | More realistic split reduced inflated performance |
| Grouped focal baseline | Strict AMI label + grouped split + focal early fusion | `0.7153` | Major reliable improvement |
| 3-fold grouped CV | Same strict early-fusion setup | `0.7132 ± 0.0056` | Confirms stability |
| No troponin ablation | Removed troponin-derived features | `~0.5268` | Troponin features are important |
| Interaction features | Added many manually designed interactions | `0.7088` | Did not improve; reverted |
| ResNet ECG encoder | Larger ECG encoder | `0.7102 - 0.7122` | Did not beat baseline |
| First-24h lab timing | Added early lab timing features | `0.7189` | Small improvement over `0.7153` |

Best validation result so far:

- Run: `strict_ami_early_focal_labtiming`
- Validation F1: `0.7189`
- Validation AUC: `0.9173`
- Validation AP: `0.7776`

Best strict grouped cross-validation result:

- Mean F1: `0.7132`
- Std F1: `0.0056`
- Mean AUC: `0.9053`
- Mean AP: `0.7645`

Held-out test result for the strict early-fusion model:

- Accuracy: `0.8759`
- Precision: `0.6477`
- Recall: `0.7545`
- F1: `0.6970`
- AUC: `0.9098`

## 11. How the Result Improved From 0.53 to 0.71

The improvement did not come from one single trick. It came from a sequence of corrections and model improvements:

1. Clinical-only baseline established a starting F1 of about `0.5305`.
2. ECG waveform was added through early fusion, allowing the model to learn signal patterns beyond clinical features.
3. Clinical and ECG machine features were cleaned and engineered, improving usable signal quality.
4. Invalid ECG-machine values were flagged instead of silently treated as normal values.
5. Strict acute/current AMI labels reduced label noise from old/history MI diagnoses.
6. Patient-level grouped splitting made validation more realistic and prevented leakage.
7. Focal loss helped handle class imbalance and hard examples.
8. Threshold optimization selected the best validation-F1 decision threshold.
9. First-24h lab timing features added time-aware clinical context.

Together, these steps improved validation F1 from approximately `0.53` to approximately `0.71-0.72`.

## 12. Error Analysis Summary

Error analysis on the strict early-fusion model showed:

Validation outcome counts at threshold `0.6878`:

| Outcome | Count |
|---|---:|
| True Positive | `3,172` |
| False Positive | `1,598` |
| True Negative | `15,970` |
| False Negative | `927` |

Important observations:

- False positives were often troponin-heavy, meaning the model learned that high troponin strongly indicates AMI.
- False negatives often had weaker troponin signal despite strict AMI labels.
- Removing troponin features caused F1 to fall to about `0.5268`, confirming that troponin is highly predictive.
- Manual interaction features did not help and were reverted.
- Time-aware lab features were more useful than manually created interaction features.

## 13. Explainability

The project includes gradient x input attribution for clinical features.

Purpose:

- Identify which clinical features influenced each prediction.
- Generate human-readable reasoning for predictions.
- Support frontend/database storage of predictions and explanations.

The explainability code supports both old 46-feature models and newer 53/59-feature models by reading the feature list stored in model metrics.

## 14. Current Best Model Direction

The strongest validated direction is:

- Strict AMI ICD label
- Patient-level grouped split
- Early fusion model
- Focal loss
- Threshold optimization
- Time-aware lab features
- ECG-admission timing features ready for the next run

Recommended next experiment:

```powershell
python -m early_fusion.train --loss-name focal --auto-continue --run-name strict_ami_early_focal_ecgtiming
```

After training, the ECG-timing model should be compared against:

- `strict_ami_grouped_early_focal`: validation F1 `0.7153`
- `strict_ami_early_focal_labtiming`: validation F1 `0.7189`
- Grouped CV mean: F1 `0.7132 ± 0.0056`

## 15. Conclusion

This project evolved from a clinical-only baseline to a leakage-safe multimodal AMI prediction pipeline. The largest improvements came from combining ECG waveform signal with clinical features, improving preprocessing quality, cleaning the AMI label, using grouped patient-level validation, and optimizing for F1 rather than accuracy.

The project improved from approximately `0.53` validation F1 in the clinical-only baseline to approximately `0.71-0.72` validation F1 in the multimodal early-fusion pipeline. Accuracy is reported, but F1 is the more suitable primary metric because AMI is an imbalanced medical classification task where both missed AMI cases and false alarms matter.
