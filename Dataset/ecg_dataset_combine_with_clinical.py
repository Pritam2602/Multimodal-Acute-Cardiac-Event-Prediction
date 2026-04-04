# ==========================================
# ECG + CLINICAL DATA MERGING PIPELINE
# ==========================================

import pandas as pd
import boto3
import wfdb
import os

# ==========================================
# AWS CONFIG (Replace with your credentials)
# ==========================================
AWS_ACCESS_KEY = "YOUR_ACCESS_KEY"
AWS_SECRET_KEY = "YOUR_SECRET_KEY"

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name="us-east-1"
)

bucket = "arn:aws:s3:us-east-1:724665945834:accesspoint/mimic-iv-ecg-v1-0-01"
prefix = "mimic-iv-ecg/1.0/files/"

# ==========================================
# STEP 1: GET ECG FILE KEYS
# ==========================================
print("Fetching ECG file keys...")

paginator = s3.get_paginator("list_objects_v2")
pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

hea_files = []

for page in pages:
    for obj in page.get("Contents", []):
        key = obj["Key"]
        if key.endswith(".hea"):
            hea_files.append(key)

print(f"Total ECG records: {len(hea_files)}")

# ==========================================
# STEP 2: CREATE ECG METADATA
# ==========================================
dataset = []

for hea_key in hea_files:
    dat_key = hea_key.replace(".hea", ".dat")

    parts = hea_key.split("/")
    subject_id = parts[4].replace("p", "")
    study_id = parts[5].replace("s", "")

    dataset.append({
        "subject_id": subject_id,
        "study_id": study_id,
        "hea_key": hea_key,
        "dat_key": dat_key
    })

ecg_meta = pd.DataFrame(dataset)

print("ECG metadata created:", ecg_meta.shape)

# ==========================================
# STEP 3: LOAD MACHINE ECG FEATURES
# ==========================================
machine_df = pd.read_csv("ecg_features.csv")

machine_df['study_id'] = machine_df['study_id'].astype(str)
ecg_meta['study_id'] = ecg_meta['study_id'].astype(str)

ecg_meta = ecg_meta.merge(machine_df, on="study_id", how="left")

# ==========================================
# STEP 4: LOAD RECORD + ADMISSIONS
# ==========================================
record_list = pd.read_csv("record_list.csv")
admissions = pd.read_csv("admissions.csv")

record_list["ecg_time"] = pd.to_datetime(record_list["ecg_time"])
admissions["admittime"] = pd.to_datetime(admissions["admittime"])
admissions["dischtime"] = pd.to_datetime(admissions["dischtime"])

ecg_studies = set(ecg_meta["study_id"])

record_list_filtered = record_list[
    record_list["study_id"].astype(str).isin(ecg_studies)
]

# Merge ECG with admissions
ecg_adm = record_list_filtered.merge(admissions, on="subject_id", how="inner")

# Keep ECG within admission time
ecg_adm = ecg_adm[
    (ecg_adm["ecg_time"] >= ecg_adm["admittime"]) &
    (ecg_adm["ecg_time"] <= ecg_adm["dischtime"])
]

# ==========================================
# STEP 5: LOAD CLINICAL DATA
# ==========================================
clinical_df = pd.read_parquet("mimic_clinical_dataset.parquet")

ecg_clinical = ecg_adm.merge(
    clinical_df,
    on=["subject_id", "hadm_id"],
    how="inner"
)

# ==========================================
# STEP 6: FINAL MERGE
# ==========================================
ecg_clinical["subject_id"] = ecg_clinical["subject_id"].astype(str)
ecg_clinical["study_id"] = ecg_clinical["study_id"].astype(str)
ecg_meta["subject_id"] = ecg_meta["subject_id"].astype(str)
ecg_meta["study_id"] = ecg_meta["study_id"].astype(str)

final_dataset = ecg_clinical.merge(
    ecg_meta,
    on=["subject_id", "study_id"],
    how="inner"
)

print("Merged dataset shape:", final_dataset.shape)

# ==========================================
# STEP 7: CLEAN DATA
# ==========================================
df = final_dataset.copy()

# Remove duplicate subject columns
df["subject_id"] = df["subject_id"].astype(str)
df = df.drop(columns=["subject_id_x", "subject_id_y"], errors="ignore")

# Fix machine feature duplicates
df = df.rename(columns={
    "heart_rate_x": "heart_rate",
    "PR_interval_x": "PR_interval",
    "QRS_duration_x": "QRS_duration",
    "QT_interval_x": "QT_interval",
    "QTc_x": "QTc"
})

df = df.drop(columns=[
    "heart_rate_y", "PR_interval_y", "QRS_duration_y",
    "QT_interval_y", "QTc_y"
], errors="ignore")

# Fix clinical duplicates
df = df.rename(columns={
    "hospital_expire_flag_y": "hospital_expire_flag"
})

df = df.drop(columns=["hospital_expire_flag_x"], errors="ignore")

# Remove unnecessary columns
drop_cols = [
    "file_name", "path", "language",
    "insurance_x", "insurance_y"
]

df = df.drop(columns=drop_cols, errors="ignore")

# ==========================================
# STEP 8: SELECT FINAL FEATURES
# ==========================================
keep_cols = [
    "subject_id", "hadm_id", "study_id", "hea_key", "dat_key",
    "anchor_age", "gender",
    "Heart Rate", "Respiratory Rate",
    "Creatinine", "Hemoglobin",
    "Sodium", "Potassium",
    "Lactate", "Troponin T",
    "num_diagnoses", "los",
    "heart_rate", "PR_interval",
    "QRS_duration", "QT_interval", "QTc"
]

final_dataset = df[keep_cols]

print("Final dataset shape:", final_dataset.shape)

# ==========================================
# STEP 9: SAVE DATASET
# ==========================================
output_path = "final_ecg_clinical_dataset.parquet"

final_dataset.to_parquet(
    output_path,
    engine="pyarrow",
    compression="snappy"
)

print("Dataset saved successfully!")
print("Location:", output_path)

# File size
size_gb = os.path.getsize(output_path) / (1024**3)
print(f"File size: {size_gb:.2f} GB")