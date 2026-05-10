import pandas as pd
import numpy as np

def refine_dataset():
    print("Loading temporal dataset...")
    df = pd.read_parquet(r'D:\MINI_PROJECT\mimic_data\temporal_fusion_dataset.parquet')
    initial_len = len(df)
    
    # Replace 0.0 with NaN for troponin calculations so we don't mess up min/max
    trop_cols = ['trop_0', 'trop_1', 'trop_2']
    df_trop = df[trop_cols].replace(0.0, np.nan)
    
    trop_max = df_trop.max(axis=1)
    trop_min = df_trop.min(axis=1)
    
    print("1. Label Quality Filtering...")
    # Weak Positives: AMI=1 but max troponin is < 0.04 (standard MI threshold)
    # If it's NaN, we keep it because it might have been diagnosed via other means (e.g. echo, STEMI)
    weak_pos_mask = (df['AMI'] == 1) & (trop_max < 0.04)
    
    # Weak Negatives: AMI=0 but max troponin > 0.5 and no major confounders
    weak_neg_mask = (df['AMI'] == 0) & (trop_max > 0.5) & (df['has_ckd'] == 0) & (df['has_sepsis'] == 0)
    
    df = df[~(weak_pos_mask | weak_neg_mask)].copy()
    
    print("2. Trajectory Richness Filtering...")
    # Remove trajectories where multiple ECGs are spaced < 30 mins apart (noise/lead adjustments)
    def check_spacing(row):
        t0, t1, t2 = row['ecg_time_0'], row['ecg_time_1'], row['ecg_time_2']
        times = pd.to_datetime([t0, t1, t2], errors='coerce').dropna()
        if len(times) >= 2:
            times = times.sort_values()
            diffs = pd.Series(times).diff().dt.total_seconds().dropna() / 3600.0
            if diffs.min() < 0.5:
                return False
        return True

    mask_spacing = df.apply(check_spacing, axis=1)
    df = df[mask_spacing].copy()
    
    print("3. Troponin Context Normalization...")
    # Recalculate after filtering
    df_trop = df[trop_cols].replace(0.0, np.nan)
    trop_max = df_trop.max(axis=1)
    trop_min = df_trop.min(axis=1)
    
    # If trop_min is < 0.01, clip it to 0.01 to avoid massive ratios
    trop_min_safe = trop_min.clip(lower=0.01)
    
    df['trop_peak_baseline_ratio'] = (trop_max / trop_min_safe).fillna(0.0)
    df['trop_fold_rise'] = ((trop_max - trop_min_safe) / trop_min_safe).fillna(0.0)
    
    print("4. Cohort Refinement (Confounders)...")
    # Drop Non-AMI patients heavily confounded by overlapping pathologies
    confounding_count = df[['has_ckd', 'has_hf', 'has_sepsis']].sum(axis=1)
    extreme_confounder_mask = (df['AMI'] == 0) & (confounding_count >= 2) & (trop_max > 0.1)
    df = df[~extreme_confounder_mask].copy()
    
    out_path = r'D:\MINI_PROJECT\mimic_data\refined_temporal_fusion_dataset.parquet'
    df.to_parquet(out_path)
    
    print(f"Dataset successfully refined.")
    print(f"Original size: {initial_len}")
    print(f"Refined size:  {len(df)}")
    print(f"Removed {initial_len - len(df)} highly ambiguous admissions.")

if __name__ == "__main__":
    refine_dataset()
