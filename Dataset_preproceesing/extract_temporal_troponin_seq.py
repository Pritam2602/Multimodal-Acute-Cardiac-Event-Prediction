import pandas as pd
import numpy as np
import os

LABEVENTS_PATH = r"D:\MINI_PROJECT\mimic_data\labevents.csv.gz"
ADMISSIONS_PATH = r"D:\MINI_PROJECT\mimic_data\admissions.csv.gz"
D_LABITEMS_PATH = r"D:\MINI_PROJECT\mimic_data\d_labitems.csv.gz"
OUTPUT_PATH = r"D:\MINI_PROJECT\mimic_data\temporal_troponin_sequence.parquet"
MAX_SEQ_LEN = 3 # We will extract up to 3 troponin measurements

# 1. LOAD DATA
print("Loading admissions to get admittime...")
admissions = pd.read_csv(ADMISSIONS_PATH, usecols=["hadm_id", "admittime"])
admissions["admittime"] = pd.to_datetime(admissions["admittime"])

print("Loading labitems to find Troponin...")
d_labitems = pd.read_csv(D_LABITEMS_PATH)
d_labitems["label"] = d_labitems["label"].str.lower()
trop_matches = d_labitems[d_labitems["label"].str.contains("troponin", na=False)]
TROPONIN_ITEMIDS = trop_matches["itemid"].unique().tolist()
print(f"Found Troponin itemids: {TROPONIN_ITEMIDS}")

print("Extracting Troponin events...")
iter_csv = pd.read_csv(LABEVENTS_PATH, chunksize=1000000, usecols=["subject_id", "hadm_id", "itemid", "charttime", "valuenum"])
trop_chunks = []
for chunk in iter_csv:
    chunk = chunk[chunk["itemid"].isin(TROPONIN_ITEMIDS)].dropna(subset=["valuenum", "hadm_id"])
    trop_chunks.append(chunk)

trop_events = pd.concat(trop_chunks, ignore_index=True)
trop_events["charttime"] = pd.to_datetime(trop_events["charttime"])
trop_events["valuenum"] = pd.to_numeric(trop_events["valuenum"], errors="coerce")
trop_events = trop_events.dropna(subset=["valuenum"])

# Join with admissions to get admittime
trop_events = trop_events.merge(admissions, on="hadm_id", how="inner")
print(f"Total Troponin measurements linked to admissions: {len(trop_events)}")

# 2. EXTRACT TEMPORAL SEQUENCES
print(f"Extracting up to {MAX_SEQ_LEN} sequential troponin measurements per admission...")
# Sort by hadm_id and charttime
trop_events = trop_events.sort_values(["hadm_id", "charttime"])

# Calculate hours since admission for each measurement
trop_events["hours_since_admit"] = (trop_events["charttime"] - trop_events["admittime"]) / np.timedelta64(1, 'h')

# Keep only measurements taken within the first 48 hours to represent the acute evolution
trop_events = trop_events[trop_events["hours_since_admit"] <= 48.0]
# Also allow some measurements just before admission (ED)
trop_events = trop_events[trop_events["hours_since_admit"] >= -24.0]

# Group by admission and take first MAX_SEQ_LEN
def extract_seq(group):
    vals = group["valuenum"].values[:MAX_SEQ_LEN]
    times = group["hours_since_admit"].values[:MAX_SEQ_LEN]
    
    # Pad with NaNs if less than MAX_SEQ_LEN
    pad_len = MAX_SEQ_LEN - len(vals)
    if pad_len > 0:
        vals = np.pad(vals, (0, pad_len), constant_values=np.nan)
        times = np.pad(times, (0, pad_len), constant_values=np.nan)
        
    return pd.Series({
        "trop_0": vals[0],
        "trop_0_time": times[0],
        "trop_1": vals[1],
        "trop_1_time": times[1],
        "trop_2": vals[2],
        "trop_2_time": times[2],
        "trop_seq_len": len(group)
    })

seq_features = trop_events.groupby("hadm_id").apply(extract_seq).reset_index()

print(f"Extracted sequences for {len(seq_features)} admissions.")
print(f"Admissions with >=2 measurements: {len(seq_features[seq_features['trop_seq_len'] >= 2])}")
print(f"Admissions with >=3 measurements: {len(seq_features[seq_features['trop_seq_len'] >= 3])}")

# Save
seq_features.to_parquet(OUTPUT_PATH, index=False)
print(f"Temporal Troponin Sequences successfully saved to: {OUTPUT_PATH}")
