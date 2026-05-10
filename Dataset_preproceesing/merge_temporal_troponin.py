import pandas as pd
import shutil
import os

MAIN_DATA_PATH = r"D:\MINI_PROJECT\mimic_data\final_preprocessed_fusion_dataset.parquet"
TEMP_TROP_PATH = r"D:\MINI_PROJECT\mimic_data\temporal_troponin_sequence.parquet"
BACKUP_PATH = r"D:\MINI_PROJECT\mimic_data\final_preprocessed_fusion_dataset_v2.parquet"

print("Backing up main dataset...")
if not os.path.exists(BACKUP_PATH):
    shutil.copy2(MAIN_DATA_PATH, BACKUP_PATH)
    print(f"Backed up to: {BACKUP_PATH}")

print("Loading datasets...")
df_main = pd.read_parquet(MAIN_DATA_PATH)
df_seq = pd.read_parquet(TEMP_TROP_PATH)

print("Merging temporal troponin sequences into main dataset...")
df_merged = pd.merge(df_main, df_seq, on="hadm_id", how="left")

# List of sequence columns
seq_cols = ["trop_0", "trop_0_time", "trop_1", "trop_1_time", "trop_2", "trop_2_time", "trop_seq_len"]

# Impute missing values
print("Imputing missing temporal values...")
# If an admission has 0 troponin measurements, trop_seq_len is NaN. Fill with 0.
df_merged["trop_seq_len"] = df_merged["trop_seq_len"].fillna(0)

# Fill missing troponin values with 0
# Fill missing times with -1 (since legitimate time is usually >= -24 and <= 48)
for col in ["trop_0", "trop_1", "trop_2"]:
    df_merged[col] = df_merged[col].fillna(0.0)

for col in ["trop_0_time", "trop_1_time", "trop_2_time"]:
    df_merged[col] = df_merged[col].fillna(-1.0)

# Add explicit missing flags just in case the model needs them
df_merged["has_trop_0"] = (df_merged["trop_seq_len"] >= 1).astype(float)
df_merged["has_trop_1"] = (df_merged["trop_seq_len"] >= 2).astype(float)
df_merged["has_trop_2"] = (df_merged["trop_seq_len"] >= 3).astype(float)

assert len(df_merged) == len(df_main), "Merge caused row explosion!"

print(f"Saving merged dataset back to {MAIN_DATA_PATH}...")
df_merged.to_parquet(MAIN_DATA_PATH, index=False)
print("Done! Dataset successfully updated with True Temporal Trajectories.")
