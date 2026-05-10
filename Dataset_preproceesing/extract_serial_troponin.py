import pandas as pd
import numpy as np

# ==========================================
# FILE PATHS
# ==========================================
LABEVENTS_PATH = r"D:\MINI_PROJECT\mimic_data\labevents.csv.gz"
ADMISSIONS_PATH = r"D:\MINI_PROJECT\mimic_data\admissions.csv.gz"
D_LABITEMS_PATH = r"D:\MINI_PROJECT\mimic_data\d_labitems.csv.gz"
OUTPUT_PATH = r"D:\MINI_PROJECT\mimic_data\serial_troponin_features.parquet"

# ==========================================
# 1. LOAD DATA
# ==========================================
print("Loading labevents (this may take a moment)...")
d_labitems = pd.read_csv(D_LABITEMS_PATH)
d_labitems["label"] = d_labitems["label"].str.lower()

# Find Troponin itemids
trop_matches = d_labitems[d_labitems["label"].str.contains("troponin", na=False)]
TROPONIN_ITEMIDS = trop_matches["itemid"].unique().tolist()
print(f"Found Troponin itemids: {TROPONIN_ITEMIDS}")

# Load only troponin rows to save massive amounts of RAM
print("Filtering labevents for Troponin specifically...")
iter_csv = pd.read_csv(LABEVENTS_PATH, chunksize=1000000, usecols=["subject_id", "hadm_id", "itemid", "charttime", "valuenum"])
trop_chunks = []
for chunk in iter_csv:
    chunk = chunk[chunk["itemid"].isin(TROPONIN_ITEMIDS)].dropna(subset=["valuenum", "hadm_id"])
    trop_chunks.append(chunk)

trop_events = pd.concat(trop_chunks, ignore_index=True)
trop_events["charttime"] = pd.to_datetime(trop_events["charttime"])
trop_events["valuenum"] = pd.to_numeric(trop_events["valuenum"], errors="coerce")
trop_events = trop_events.dropna(subset=["valuenum"])
print(f"Total Troponin measurements extracted: {len(trop_events)}")

# ==========================================
# 2. COMPUTE SERIAL TROPONIN FEATURES
# ==========================================
print("Computing trajectory features per admission...")

# Sort chronologically per admission
trop_events = trop_events.sort_values(["hadm_id", "charttime"])

def compute_trajectory(group):
    if len(group) == 0:
        return pd.Series()
    
    vals = group["valuenum"].values
    times = group["charttime"].values
    
    # Basic metrics
    trop_initial = vals[0]
    trop_peak = vals.max()
    trop_final = vals[-1]
    
    peak_idx = vals.argmax()
    time_to_peak_hours = (times[peak_idx] - times[0]) / np.timedelta64(1, 'h')
    
    # Delta
    trop_delta_absolute = trop_peak - trop_initial
    normalized_delta = trop_delta_absolute / (trop_initial + 1e-6)  # Avoid div zero
    
    # Velocity (Highest ROI Feature)
    if time_to_peak_hours > 0:
        trop_velocity = trop_delta_absolute / time_to_peak_hours
    else:
        trop_velocity = 0.0
        
    # Rise/Fall Classification Slope (overall from start to end)
    total_time_hours = (times[-1] - times[0]) / np.timedelta64(1, 'h')
    if total_time_hours > 0:
        overall_slope = (trop_final - trop_initial) / total_time_hours
    else:
        overall_slope = 0.0
        
    return pd.Series({
        "troponin_count": len(group),
        "troponin_initial": trop_initial,
        "troponin_peak": trop_peak,
        "troponin_delta": trop_delta_absolute,
        "troponin_velocity": trop_velocity,
        "rise_fall_slope": overall_slope,
        "time_to_peak_hours": time_to_peak_hours,
        "normalized_delta": normalized_delta
    })

# Apply grouped computation
serial_features = trop_events.groupby("hadm_id").apply(compute_trajectory).reset_index()

# Filter for admissions that actually have serial data (count >= 2)
# Or keep all and let single-measurements have 0 velocity. Keeping all is safer for joining.
print(f"Computed trajectories for {len(serial_features)} unique admissions.")
print(f"Admissions with serial (>=2) measurements: {len(serial_features[serial_features['troponin_count'] >= 2])}")

# ==========================================
# 3. SAVE TO DISK
# ==========================================
serial_features.to_parquet(OUTPUT_PATH, index=False)
print(f"Serial Troponin features successfully saved to: {OUTPUT_PATH}")
