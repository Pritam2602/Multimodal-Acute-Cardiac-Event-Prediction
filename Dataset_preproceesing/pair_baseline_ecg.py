import pandas as pd
import numpy as np
import os

MAIN_DATA_PATH = r"D:\MINI_PROJECT\mimic_data\final_preprocessed_fusion_dataset.parquet"
RECORD_LIST_PATH = r"D:\MINI_PROJECT\mimic_data\machine_measurements.csv"

print("1. Loading datasets...")
df_main = pd.read_parquet(MAIN_DATA_PATH)
df_records = pd.read_csv(RECORD_LIST_PATH, usecols=["subject_id", "study_id", "ecg_time"])

# Reconstruct path for baseline ECGs
# files/p100/p10000032/s50411002/50411002
df_records["path"] = "files/p" + df_records["subject_id"].astype(str).str[:2] + "/p" + df_records["subject_id"].astype(str) + "/s" + df_records["study_id"].astype(str) + "/" + df_records["study_id"].astype(str)

# Convert times
print("2. Parsing timestamps...")
df_records["ecg_time"] = pd.to_datetime(df_records["ecg_time"])

# Extract study_id from ecg_path in main dataset
# Path format: .../p100/p10000032/s50411002/50411002
print("3. Extracting identifiers from main dataset...")
df_main["study_id_str"] = df_main["ecg_path"].astype(str).str.split('/').str[-1]
# Some might have .hea, so strip it just in case
df_main["study_id_str"] = df_main["study_id_str"].str.replace('.hea', '', regex=False)
df_main["study_id_int"] = pd.to_numeric(df_main["study_id_str"], errors="coerce")

# Join with records to get subject_id and anchor ecg_time
df_anchor = df_main[["hadm_id", "ecg_path", "study_id_int"]].merge(
    df_records, 
    left_on="study_id_int", 
    right_on="study_id", 
    how="left"
).rename(columns={"ecg_time": "anchor_time"})

# Keep only necessary anchor columns
df_anchor = df_anchor[["hadm_id", "ecg_path", "subject_id", "anchor_time"]].dropna(subset=["subject_id", "anchor_time"])

print("4. Finding Baseline ECGs per admission...")
# Sort records by subject and time descending so we can easily pick the most recent
df_records_sorted = df_records.sort_values(["subject_id", "ecg_time"], ascending=[True, False])

# We will iterate and find the latest prior ECG for each admission
results = []

# Group records by subject for fast lookup
records_by_subject = dict(tuple(df_records_sorted.groupby("subject_id")))

for _, row in df_anchor.iterrows():
    subj = row["subject_id"]
    anchor_t = row["anchor_time"]
    
    baseline_path = None
    hours_since = 0.0
    has_base = 0
    
    if subj in records_by_subject:
        subj_records = records_by_subject[subj]
        
        # Filter for strictly before anchor_time
        # We subtract a small buffer (e.g., 1 hour) to ensure it's truly a *previous* baseline
        # and not a duplicate taken at the exact same admission moment.
        # Actually, let's just use < anchor_t
        prior_ecgs = subj_records[subj_records["ecg_time"] < anchor_t]
        
        if len(prior_ecgs) > 0:
            # Already sorted descending, so the first one is the most recent baseline
            latest_prior = prior_ecgs.iloc[0]
            baseline_path = latest_prior["path"]
            
            # Calculate hours since
            time_diff = anchor_t - latest_prior["ecg_time"]
            hours_since = time_diff.total_seconds() / 3600.0
            
            has_base = 1
            
    results.append({
        "ecg_path": row["ecg_path"],
        "baseline_ecg_path": baseline_path,
        "hours_since_baseline_ecg": hours_since,
        "has_baseline_ecg": has_base
    })

df_results = pd.DataFrame(results).drop_duplicates("ecg_path")

print("5. Merging back into main dataset...")
df_merged = pd.merge(df_main, df_results, on="ecg_path", how="left")

# Fill missing for patients who didn't even match an anchor (rare but possible)
df_merged["has_baseline_ecg"] = df_merged["has_baseline_ecg"].fillna(0).astype(int)
df_merged["hours_since_baseline_ecg"] = df_merged["hours_since_baseline_ecg"].fillna(0.0)

print(f"Total admissions: {len(df_merged)}")
print(f"Admissions WITH a baseline ECG: {df_merged['has_baseline_ecg'].sum()}")

# Cleanup temp columns
df_merged = df_merged.drop(columns=["study_id_str", "study_id_int"])

print(f"6. Saving to {MAIN_DATA_PATH}...")
df_merged.to_parquet(MAIN_DATA_PATH, index=False)
print("Complete! Baseline ECG pairing successful.")
