# ==========================================
# DATASET PREPROCESSING SCRIPT
# AMI PREDICTION (ECG + CLINICAL)
# ==========================================

from pathlib import Path

import numpy as np
import pandas as pd

from Dataset.feature_schema import ECG_MACHINE_COLS, NUMERIC_COLS, VALID_RANGES

# ==========================================
# FILE PATHS (UPDATE THESE)
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = PROJECT_ROOT / "mimic_data" / "temporal_clinical_features.parquet"

DIAG_PATH = PROJECT_ROOT / "mimic_data" / "diagnoses_icd.csv.gz"
D_ICD_PATH = PROJECT_ROOT / "mimic_data" / "d_icd_diagnoses.csv.gz"

OUTPUT_PATH = PROJECT_ROOT / "mimic_data" / "final_preprocessed_fusion_dataset.parquet"
USING_EXISTING_MODEL_DATASET = False
AMI_LABEL_MODE = "acute_current_mi"

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
numeric_cols = NUMERIC_COLS

ecg_machine_cols = ECG_MACHINE_COLS

categorical_cols = ['gender']

def create_ami_flag(diag: pd.DataFrame) -> pd.Series:
    """
    Create a stricter acute/current MI label from ICD codes.

    The previous text search for "myocardial infarction" also captured history
    codes such as ICD-9 412 and ICD-10 I25.2 ("Old myocardial infarction"),
    which are not acute AMI targets.
    """
    code = (
        diag['icd_code']
        .astype(str)
        .str.upper()
        .str.replace('.', '', regex=False)
        .str.strip()
    )
    version = diag['icd_version'].astype(int)

    icd9_acute_mi = (
        (version == 9)
        & code.str.startswith('410')
        & code.str.len().ge(5)
        & code.str[4].isin(['1', '2'])
    )
    icd10_acute_mi = (
        (version == 10)
        & (
            code.str.startswith('I21')
            | code.str.startswith('I22')
        )
    )

    return (icd9_acute_mi | icd10_acute_mi).astype(int)

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
# LOAD DIAGNOSIS DATA (FOR STRICT AMI LABEL)
# ==========================================
ami_labels = None
can_relabel_from_icd = (
    'hadm_id' in df.columns
    and DIAG_PATH.exists()
    and D_ICD_PATH.exists()
)

if can_relabel_from_icd:
    print("Loading diagnosis data for strict AMI label...")

    diag = pd.read_csv(DIAG_PATH)
    d_icd = pd.read_csv(D_ICD_PATH)

    # Merge diagnosis descriptions
    diag = diag.merge(
        d_icd,
        on=['icd_code', 'icd_version'],
        how='left'
    )

    # Create AMI flag using acute/current ICD code rules.
    diag['AMI_flag'] = create_ami_flag(diag)
    print(f"AMI label mode: {AMI_LABEL_MODE}")

    # Aggregate per admission
    ami_labels = diag.groupby('hadm_id')['AMI_flag'].max().reset_index()
    ami_labels = ami_labels.rename(columns={'AMI_flag': 'AMI'})
elif 'hadm_id' not in df.columns:
    print("hadm_id unavailable; using existing AMI label.")
else:
    print("Diagnosis CSVs not found; keeping existing AMI labels from the current parquet.")
    print(f"Missing: {DIAG_PATH if not DIAG_PATH.exists() else D_ICD_PATH}")

# ==========================================
# MERGE AMI LABEL
# ==========================================
print("Merging AMI labels...")

if ami_labels is not None:
    old_ami_counts = df['AMI'].value_counts().to_dict() if 'AMI' in df.columns else None
    df = df.drop(columns=['AMI'], errors='ignore')
    df = df.merge(ami_labels, on='hadm_id', how='left')
    if old_ami_counts is not None:
        print("Previous AMI distribution:", old_ami_counts)
elif 'AMI' not in df.columns:
    raise ValueError(
        "AMI column is missing and strict relabeling cannot run because diagnosis files are unavailable."
    )

df['AMI'] = df['AMI'].fillna(0)

print("AMI distribution:")
print(df['AMI'].value_counts())

# ==========================================
# CREATE MODEL DATASET
# ==========================================
print("Preparing model dataset...")

drop_cols_final = [
    'hea_key', 'ecg_time', 'dat_key'
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
