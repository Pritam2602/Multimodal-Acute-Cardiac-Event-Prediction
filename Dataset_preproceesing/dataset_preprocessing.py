# ==========================================
# DATASET PREPROCESSING SCRIPT
# AMI PREDICTION (ECG + CLINICAL)
# ==========================================

import pandas as pd

# ==========================================
# FILE PATHS (UPDATE THESE)
# ==========================================
INPUT_PATH = "D:\MINI_PROJECT\mimic_data\temporal_clinical_features.parquet"

DIAG_PATH = "diagnoses_icd.csv.gz"
D_ICD_PATH = "d_icd_diagnoses.csv.gz"

OUTPUT_PATH = "D:\MINI_PROJECT\mimic_data\final_preprocessed_fusion_dataset.parquet"

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

categorical_cols = ['gender']

# ==========================================
# CREATE MISSING INDICATORS
# ==========================================
print("Creating missing indicators...")

for col in numeric_cols:
    df[f"{col}_missing"] = df[col].isnull().astype(int)

# ==========================================
# HANDLE MISSING VALUES
# ==========================================
print("Filling missing values...")

df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Fill num_diagnoses separately
if 'num_diagnoses' in df.columns:
    df['num_diagnoses'] = df['num_diagnoses'].fillna(df['num_diagnoses'].median())

# ==========================================
# ENCODE CATEGORICAL VARIABLES
# ==========================================
print("Encoding categorical features...")

df['gender'] = df['gender'].map({'M': 1, 'F': 0})

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

# ==========================================
# MERGE AMI LABEL
# ==========================================
print("Merging AMI labels...")

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