import pandas as pd
import shutil
import os

MAIN_DATA_PATH = r"D:\MINI_PROJECT\mimic_data\final_preprocessed_fusion_dataset.parquet"
SERIAL_DATA_PATH = r"D:\MINI_PROJECT\mimic_data\serial_troponin_features.parquet"
BACKUP_PATH = r"D:\MINI_PROJECT\mimic_data\final_preprocessed_fusion_dataset_v1.parquet"

print("Backing up original dataset...")
if not os.path.exists(BACKUP_PATH):
    shutil.copy2(MAIN_DATA_PATH, BACKUP_PATH)
    print(f"Backed up to: {BACKUP_PATH}")
else:
    print("Backup already exists.")

print("Loading datasets...")
df_main = pd.read_parquet(MAIN_DATA_PATH)
df_serial = pd.read_parquet(SERIAL_DATA_PATH)

print(f"Main dataset rows: {len(df_main)}")
print(f"Serial features rows: {len(df_serial)}")

print("Merging...")
# Left join to preserve every single row in the main dataset
df_merged = pd.merge(df_main, df_serial, on="hadm_id", how="left")

print("Filling missing longitudinal features with 0 (since they represent no temporal change)...")
serial_cols = [
    "troponin_count", "troponin_initial", "troponin_peak", 
    "troponin_delta", "troponin_velocity", "rise_fall_slope", 
    "time_to_peak_hours", "normalized_delta"
]

for col in serial_cols:
    df_merged[col] = df_merged[col].fillna(0)

# Check for duplicates or missing data
assert len(df_merged) == len(df_main), f"Merge caused row explosion! {len(df_main)} -> {len(df_merged)}"

print("\nFinal missing value check on new features:")
print(df_merged[serial_cols].isna().sum())

print(f"\nSaving updated dataset to {MAIN_DATA_PATH}...")
df_merged.to_parquet(MAIN_DATA_PATH, index=False)
print("Done! Dataset successfully updated with serial troponin features.")
