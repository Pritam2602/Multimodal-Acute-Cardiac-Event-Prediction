# ==========================================
# DATASET PREPROCESSING SCRIPT
# AMI PREDICTION (ECG + CLINICAL)
# ==========================================

from pathlib import Path

import numpy as np
import pandas as pd

# ==========================================
# FILE PATHS (UPDATE THESE)
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = PROJECT_ROOT / "mimic_data" / "temporal_clinical_features.parquet"

DIAG_PATH = PROJECT_ROOT / "mimic_data" / "diagnoses_icd.csv.gz"
D_ICD_PATH = PROJECT_ROOT / "mimic_data" / "d_icd_diagnoses.csv.gz"

OUTPUT_PATH = PROJECT_ROOT / "mimic_data" / "final_preprocessed_fusion_dataset.parquet"
USING_EXISTING_MODEL_DATASET = False

if not INPUT_PATH.exists() and OUTPUT_PATH.exists():
    print(f"Source temporal dataset not found: {INPUT_PATH}")
    print(f"Falling back to existing model dataset: {OUTPUT_PATH}")
    INPUT_PATH = OUTPUT_PATH
    USING_EXISTING_MODEL_DATASET = True

# ==========================================
# LOAD DATA
# ==========================================
print("Loading dataset...")

df = pd.read_parquet(INPUT_PATH)

print("Initial shape:", df.shape)

# ==========================================
# DEFINE COLUMNS
# ==========================================
numeric_cols = [
    'Troponin_T', 'Creatinine', 'Sodium', 'Potassium',
    'Heart_Rate', 'Respiratory_Rate', 'anchor_age'
]

ecg_machine_cols = [
    'PR_interval', 'QRS_duration', 'QT_interval', 'QTc',
    'P_axis', 'QRS_axis', 'T_axis', 'RR_interval'
]

categorical_cols = ['gender']

VALID_RANGES = {
    'anchor_age': (0, 120),
    'Heart_Rate': (20, 250),
    'Respiratory_Rate': (5, 60),
    'Troponin_T': (0, 50),
    'Creatinine': (0.1, 15),
    'Sodium': (100, 180),
    'Potassium': (1.5, 10),
    'PR_interval': (50, 500),
    'QRS_duration': (40, 300),
    'QT_interval': (200, 800),
    'QTc': (200, 800),
    'P_axis': (-180, 360),
    'QRS_axis': (-180, 360),
    'T_axis': (-180, 360),
    'RR_interval': (200, 3000),
}

# ==========================================
# CREATE MISSING INDICATORS
# ==========================================
print("Creating missing indicators...")

for col in numeric_cols:
    if col in df.columns:
        df[f"{col}_missing"] = df[col].isnull().astype(int)

# ==========================================
# CLEAN INVALID VALUES + CREATE INDICATORS
# ==========================================
print("Cleaning invalid values and creating invalid indicators...")

for col, (lo, hi) in VALID_RANGES.items():
    if col not in df.columns:
        continue

    df[col] = pd.to_numeric(df[col], errors='coerce')
    invalid_mask = df[col].isna() | np.isinf(df[col]) | (df[col] < lo) | (df[col] > hi)

    if col in ecg_machine_cols:
        df[f"{col}_invalid"] = invalid_mask.astype(int)

    df.loc[invalid_mask, col] = np.nan

# ==========================================
# HANDLE MISSING VALUES
# ==========================================
print("Filling missing values...")

fill_cols = [col for col in numeric_cols + ecg_machine_cols if col in df.columns]
df[fill_cols] = df[fill_cols].fillna(df[fill_cols].median())

# Fill num_diagnoses separately
if 'num_diagnoses' in df.columns:
    df['num_diagnoses'] = df['num_diagnoses'].fillna(df['num_diagnoses'].median())

# ==========================================
# ENCODE CATEGORICAL VARIABLES
# ==========================================
print("Encoding categorical features...")

if 'gender' in df.columns and df['gender'].dtype == object:
    df['gender'] = df['gender'].map({'M': 1, 'F': 0})

# ==========================================
# FEATURE ENGINEERING
# ==========================================
print("Creating engineered clinical and ECG-machine features...")

if 'Troponin_T' in df.columns:
    troponin = df['Troponin_T'].clip(lower=0)
    df['log1p_Troponin_T'] = np.log1p(troponin)
    df['Troponin_T_positive'] = (df['Troponin_T'] > 0.04).astype(int)
    df['Troponin_T_high'] = (df['Troponin_T'] > 0.10).astype(int)

if {'anchor_age', 'log1p_Troponin_T'}.issubset(df.columns):
    df['age_x_log1p_Troponin_T'] = df['anchor_age'] * df['log1p_Troponin_T']

if 'QRS_duration' in df.columns:
    df['QRS_wide'] = (df['QRS_duration'] > 120).astype(int)

if 'QTc' in df.columns:
    df['QTc_prolonged'] = (df['QTc'] > 460).astype(int)

if 'PR_interval' in df.columns:
    df['PR_prolonged'] = (df['PR_interval'] > 200).astype(int)

if 'QRS_axis' in df.columns:
    df['QRS_axis_deviation'] = ((df['QRS_axis'] < -30) | (df['QRS_axis'] > 90)).astype(int)

if 'T_axis' in df.columns:
    df['T_axis_abnormal'] = ((df['T_axis'] < 15) | (df['T_axis'] > 75)).astype(int)

if 'Creatinine' in df.columns:
    df['Creatinine_high'] = (df['Creatinine'] > 1.2).astype(int)

if 'Potassium' in df.columns:
    df['Potassium_low'] = (df['Potassium'] < 3.5).astype(int)
    df['Potassium_high'] = (df['Potassium'] > 5.0).astype(int)

if {'Heart_Rate', 'RR_interval'}.issubset(df.columns):
    rr_interval = df['RR_interval'].replace(0, np.nan)
    df['HR_from_RR_interval'] = 60000 / rr_interval
    df['HR_RR_disagreement'] = (df['Heart_Rate'] - df['HR_from_RR_interval']).abs()
    df[['HR_from_RR_interval', 'HR_RR_disagreement']] = (
        df[['HR_from_RR_interval', 'HR_RR_disagreement']]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(df[['HR_from_RR_interval', 'HR_RR_disagreement']].median())
    )

# ==========================================
# DROP UNUSED COLUMNS
# ==========================================
print("Dropping unnecessary columns...")

drop_cols = [
    'Hemoglobin', 'Lactate', 'SBP', 'DBP',
    'charttime'
]

df = df.drop(columns=drop_cols, errors='ignore')

# ==========================================
# LOAD DIAGNOSIS DATA (FOR AMI LABEL)
# ==========================================
if 'AMI' not in df.columns:
    print("Loading diagnosis data...")

    diag = pd.read_csv(DIAG_PATH)
    d_icd = pd.read_csv(D_ICD_PATH)

    # Merge diagnosis descriptions
    diag = diag.merge(
        d_icd,
        on=['icd_code', 'icd_version'],
        how='left'
    )

    # Create AMI flag
    diag['AMI_flag'] = diag['long_title'].str.contains(
        'myocardial infarction',
        case=False,
        na=False
    ).astype(int)

    # Aggregate per admission
    ami_labels = diag.groupby('hadm_id')['AMI_flag'].max().reset_index()
    ami_labels = ami_labels.rename(columns={'AMI_flag': 'AMI'})
else:
    print("AMI label already present; skipping diagnosis label merge.")
    ami_labels = None

# ==========================================
# MERGE AMI LABEL
# ==========================================
print("Merging AMI labels...")

if ami_labels is not None:
    df = df.merge(ami_labels, on='hadm_id', how='left')

df['AMI'] = df['AMI'].fillna(0)

print("AMI distribution:")
print(df['AMI'].value_counts())

# ==========================================
# CREATE MODEL DATASET
# ==========================================
print("Preparing model dataset...")

drop_cols_final = [
    'subject_id', 'study_id', 'hea_key',
    'hadm_id', 'ecg_time', 'dat_key'
]

df_model = df.drop(columns=drop_cols_final, errors='ignore')

# Create ECG path
if 'dat_key' in df.columns:
    df_model['ecg_path'] = df['dat_key'].str.replace('.dat', '', regex=False)

# ==========================================
# FINAL CHECK
# ==========================================
print("\nFinal dataset shape:", df_model.shape)
print("\nMissing values:")
print(df_model.isnull().sum())

print("\nDataset info:")
print(df_model.info())

# ==========================================
# SAVE DATASET
# ==========================================
print("\nSaving dataset...")

df_model.to_parquet(OUTPUT_PATH, index=False)

print("Saved successfully at:", OUTPUT_PATH)
