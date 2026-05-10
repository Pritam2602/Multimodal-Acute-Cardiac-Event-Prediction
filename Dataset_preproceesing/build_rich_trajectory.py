import pandas as pd
import numpy as np
import os
import math

MAIN_DATA_PATH = r"D:\MINI_PROJECT\mimic_data\final_preprocessed_fusion_dataset.parquet"
OUTPUT_PATH = r"D:\MINI_PROJECT\mimic_data\temporal_fusion_dataset.parquet"

print("Loading existing dataset...")
df_main = pd.read_parquet(MAIN_DATA_PATH)

print("Loading machine measurements to get explicit ECG times...")
df_records = pd.read_csv(r"D:\MINI_PROJECT\mimic_data\machine_measurements.csv", usecols=["study_id", "ecg_time"])
df_records["ecg_time"] = pd.to_datetime(df_records["ecg_time"])

df_main["study_id"] = df_main["ecg_path"].astype(str).str.split('/').str[-1].str.replace('.hea', '', regex=False)
df_main["study_id"] = pd.to_numeric(df_main["study_id"], errors="coerce")

df_main = df_main.merge(df_records, on="study_id", how="left")
df_main = df_main.sort_values(["hadm_id", "ecg_time"])

def calculate_troponin_dynamics(row):
    # Extract trop times and values
    times = []
    vals = []
    
    for i in range(3):
        t = row.get(f"trop_{i}_time")
        v = row.get(f"trop_{i}")
        if pd.notnull(t) and pd.notnull(v) and v >= 0:
            # t might be NaT or a datetime, or numeric
            if isinstance(t, pd.Timestamp):
                t_val = t.timestamp() / 3600.0
            else:
                try:
                    t_val = pd.to_datetime(t).timestamp() / 3600.0
                except:
                    t_val = float(t)
            times.append(t_val)
            vals.append(float(v))
            
    if len(times) < 2:
        return pd.Series({
            "trop_dyn_slope": 0.0,
            "trop_dyn_accel": 0.0,
            "trop_dyn_rise_ratio": 1.0,
            "trop_dyn_delta": 0.0
        })
        
    t0, v0 = times[0], vals[0]
    t_last, v_last = times[-1], vals[-1]
    
    delta_t = max(t_last - t0, 0.1) # prevent div by zero
    delta_v = v_last - v0
    slope = delta_v / delta_t
    
    rise_ratio = max(vals) / (v0 + 1e-5)
    
    accel = 0.0
    if len(times) == 3:
        t1, v1 = times[1], vals[1]
        dt1 = max(t1 - t0, 0.1)
        dt2 = max(t_last - t1, 0.1)
        slope1 = (v1 - v0) / dt1
        slope2 = (v_last - v1) / dt2
        accel = slope2 - slope1
        
    return pd.Series({
        "trop_dyn_slope": slope,
        "trop_dyn_accel": accel,
        "trop_dyn_rise_ratio": rise_ratio,
        "trop_dyn_delta": delta_v
    })


def build_rich_temporal_row(group):
    row = group.iloc[0].to_dict()
    
    # We want maximally spaced ECGs
    # ECG 0: Admission (first available)
    # ECG 1: Evolving phase (~3-6h, must be >2h from ECG 0)
    # ECG 2: Late phase (~12-24h, must be >6h from ECG 1)
    
    ecgs = []
    times = []
    
    for _, r in group.iterrows():
        p = r["ecg_path"]
        t = r["ecg_time"]
        if pd.isnull(p) or pd.isnull(t): continue
        if len(ecgs) == 0:
            ecgs.append(p)
            times.append(t)
        else:
            t0 = times[0]
            hours_diff_0 = (t - t0).total_seconds() / 3600.0
            
            if len(ecgs) == 1:
                # We want something > 2h
                if hours_diff_0 >= 2.0:
                    ecgs.append(p)
                    times.append(t)
            elif len(ecgs) == 2:
                # We want something > 6h from ECG 1, or just significantly later
                t1 = times[1]
                hours_diff_1 = (t - t1).total_seconds() / 3600.0
                if hours_diff_1 >= 6.0:
                    ecgs.append(p)
                    times.append(t)
                    break # We have our 3 spaced ECGs
                    
    # Fill sequence
    for i in range(3):
        if i < len(ecgs):
            row[f"ecg_path_{i}"] = ecgs[i]
            row[f"ecg_time_{i}"] = times[i]
        else:
            row[f"ecg_path_{i}"] = None
            row[f"ecg_time_{i}"] = pd.NaT
            
    row["ecg_seq_len"] = len(ecgs)
    
    # Clean up single-row keys
    for k in ["ecg_path", "study_id", "ecg_time"]:
        if k in row: del row[k]
        
    return pd.Series(row)

print("Aggregating strictly spaced temporal ECG sequences per admission...")
df_temporal = df_main.groupby("hadm_id").apply(build_rich_temporal_row).reset_index(drop=True)

print("Calculating Troponin Kinematics (slope, acceleration, rise_ratio)...")
trop_dyn = df_temporal.apply(calculate_troponin_dynamics, axis=1)
df_temporal = pd.concat([df_temporal, trop_dyn], axis=1)

print(f"Final Temporal Dataset rows: {len(df_temporal)}")
print(f"Admissions with >=1 ECG: {len(df_temporal[df_temporal['ecg_seq_len'] >= 1])}")
print(f"Admissions with >=2 ECGs (Properly spaced >2h): {len(df_temporal[df_temporal['ecg_seq_len'] >= 2])}")
print(f"Admissions with >=3 ECGs (Properly spaced >6h): {len(df_temporal[df_temporal['ecg_seq_len'] >= 3])}")

print(f"Saving to {OUTPUT_PATH}...")
df_temporal.to_parquet(OUTPUT_PATH, index=False)
print("Done! Temporal Dataset is now physiologically rich.")
