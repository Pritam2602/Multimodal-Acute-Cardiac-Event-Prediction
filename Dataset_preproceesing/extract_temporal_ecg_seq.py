import pandas as pd
import numpy as np
import os

MAIN_DATA_PATH = r"D:\MINI_PROJECT\mimic_data\final_preprocessed_fusion_dataset.parquet"
OUTPUT_PATH = r"D:\MINI_PROJECT\mimic_data\temporal_fusion_dataset.parquet"
MAX_ECG_SEQ = 3

print("Loading existing dataset (which has static clinical features & temporal troponin)...")
df_main = pd.read_parquet(MAIN_DATA_PATH)

print(f"Initial rows: {len(df_main)}")
print(f"Unique admissions: {df_main['hadm_id'].nunique()}")

# Sort by hadm_id and ecg time (we need to extract ecg time first)
# Wait, df_main doesn't explicitly have ecg_time as a datetime column, but we can get it from machine_measurements
print("Loading machine measurements to get explicit ECG times...")
df_records = pd.read_csv(r"D:\MINI_PROJECT\mimic_data\machine_measurements.csv", usecols=["study_id", "ecg_time"])
df_records["ecg_time"] = pd.to_datetime(df_records["ecg_time"])

# Extract study_id from ecg_path
df_main["study_id"] = df_main["ecg_path"].astype(str).str.split('/').str[-1].str.replace('.hea', '', regex=False)
df_main["study_id"] = pd.to_numeric(df_main["study_id"], errors="coerce")

# Merge time
df_main = df_main.merge(df_records, on="study_id", how="left")

# Sort chronologically per admission
df_main = df_main.sort_values(["hadm_id", "ecg_time"])

print("Aggregating temporal ECG sequences per admission...")

def build_temporal_row(group):
    # Base clinical features (take from the first row since they are static per admission)
    row = group.iloc[0].to_dict()
    
    # Extract temporal ECG paths and times
    paths = group["ecg_path"].values[:MAX_ECG_SEQ]
    times = group["ecg_time"].values[:MAX_ECG_SEQ]
    
    # Fill sequence
    for i in range(MAX_ECG_SEQ):
        if i < len(paths):
            row[f"ecg_path_{i}"] = paths[i]
            row[f"ecg_time_{i}"] = times[i]
        else:
            row[f"ecg_path_{i}"] = None
            row[f"ecg_time_{i}"] = pd.NaT
            
    row["ecg_seq_len"] = min(len(paths), MAX_ECG_SEQ)
    
    # We no longer need the single 'ecg_path' or 'study_id'
    if "ecg_path" in row: del row["ecg_path"]
    if "study_id" in row: del row["study_id"]
    if "ecg_time" in row: del row["ecg_time"]
    
    return pd.Series(row)

# Apply grouping
df_temporal = df_main.groupby("hadm_id").apply(build_temporal_row).reset_index(drop=True)

print(f"Final Temporal Dataset rows (1 per admission): {len(df_temporal)}")
print(f"Admissions with >=1 ECG: {len(df_temporal[df_temporal['ecg_seq_len'] >= 1])}")
print(f"Admissions with >=2 ECGs: {len(df_temporal[df_temporal['ecg_seq_len'] >= 2])}")
print(f"Admissions with >=3 ECGs: {len(df_temporal[df_temporal['ecg_seq_len'] >= 3])}")

print(f"Saving to {OUTPUT_PATH}...")
df_temporal.to_parquet(OUTPUT_PATH, index=False)
print("Done! We now have a strictly Temporal Multimodal Dataset.")
