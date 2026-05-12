import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import (
    ECG_LENGTH, ECG_LEADS,
    CLINICAL_FEATURES, TARGET_COLUMN
)

ECG_CACHE_DIR = Path(r"D:\MINI_PROJECT\mimic_data\ecg_cache")
from .dataset import (
    _get_local_record_path, _is_cached, load_ecg_signal,
    download_all_ecgs, _apply_data_filters
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

TEMPORAL_DATA_PATH = r"D:\MINI_PROJECT\mimic_data\refined_temporal_fusion_dataset.parquet"
MAX_SEQ_LEN = 3

class TemporalFusionDataset(Dataset):
    def __init__(
        self,
        memmap_path: str,
        memmap_indices: np.ndarray,
        clinical_static: np.ndarray,
        trop_seq: np.ndarray,       # (N, 3)
        trop_times: np.ndarray,     # (N, 3)
        ecg_times: np.ndarray,      # (N, 3)
        seq_masks: np.ndarray,      # (N, 3)
        labels: np.ndarray
    ):
        self.memmap_path = memmap_path
        self.memmap_indices = memmap_indices
        self.clinical_static = torch.tensor(clinical_static, dtype=torch.float32)
        self.trop_seq = torch.tensor(trop_seq, dtype=torch.float32)
        
        # Calculate time deltas (relative to step 0)
        # Handle -1 or NaN by replacing with 0 for the delta computation
        self.ecg_time_deltas = self._compute_deltas(ecg_times)
        self.trop_time_deltas = self._compute_deltas(trop_times)
        
        self.seq_masks = torch.tensor(seq_masks, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

        # Open memmap in read mode
        n_samples = len(memmap_indices)  # Wait, memmap contains ALL dataset rows. 
        # But we only need access to it. We need the full shape.
        # Actually, the memmap is constructed on the original un-split dataframe.
        # So we pass the shape and read the specific index.
        pass # We will open it lazily in __getitem__ for thread safety
        self.n_total_memmap = None # Will set after finding shape
        
    def _compute_deltas(self, times):
        # times shape: (N, 3)
        # base time is times[:, 0]
        # if times[:, i] is missing (indicated by -1 or NaN), we set delta to 0
        deltas = np.zeros_like(times, dtype=np.float32)
        base = times[:, 0]
        for i in range(MAX_SEQ_LEN):
            valid = (times[:, i] != -1.0) & (~np.isnan(times[:, i])) & (base != -1.0) & (~np.isnan(base))
            deltas[valid, i] = (times[valid, i] - base[valid])
            
        # Clamp deltas to prevent float16 overflow in PyTorch AMP (max float16 is ~65504)
        deltas = np.clip(deltas, -2000.0, 2000.0)
        return torch.tensor(deltas, dtype=torch.float32)

    def bind_memmap_shape(self, n_total):
        self.n_total_memmap = n_total

    def __len__(self):
        return len(self.memmap_indices)

    def __getitem__(self, idx):
        real_idx = self.memmap_indices[idx]
        
        # Lazy open memmap to avoid DataLoader worker issues
        memmap = np.memmap(
            self.memmap_path, 
            dtype=np.float32, 
            mode='r', 
            shape=(self.n_total_memmap, MAX_SEQ_LEN, ECG_LEADS, ECG_LENGTH)
        )
        
        ecg_seq = torch.tensor(memmap[real_idx].copy(), dtype=torch.float32)
        del memmap
        
        return {
            "ecg_seq": ecg_seq,
            "trop_seq": self.trop_seq[idx],
            "trop_time_deltas": self.trop_time_deltas[idx],
            "ecg_time_deltas": self.ecg_time_deltas[idx],
            "seq_mask": self.seq_masks[idx],
            "clinical_static": self.clinical_static[idx],
            "label": self.labels[idx]
        }


def load_and_prepare_temporal_data(
    batch_size=32,
    train_workers=4,
    val_workers=2,
    ecg_time_window=0,
    max_ecgs_per_patient=0
):
    print("======================================================================")
    print(" TEMPORAL DATA LOADING")
    print("======================================================================")
    print(f"[DATA] Loading: {TEMPORAL_DATA_PATH}")
    df = pd.read_parquet(TEMPORAL_DATA_PATH)
    print(f"[DATA] Full temporal dataset: {len(df)} episodes (admissions)")
    
    # Exclude engineered sequential features from static clinical
    temporal_engineered = [
        "troponin_delta", "troponin_velocity", "rise_fall_slope", 
        "time_to_peak_hours", "normalized_delta"
    ]
    static_clinical_cols = [c for c in CLINICAL_FEATURES if c not in temporal_engineered and c in df.columns]
    print(f"[DATA] Static Clinical features: {len(static_clinical_cols)}")
    
    # 1. Fill NaNs in clinical
    clinical_df = df[static_clinical_cols].copy()
    clinical_df = clinical_df.replace([np.inf, -np.inf], np.nan)
    clinical_df = clinical_df.fillna(clinical_df.median())
    clinical_static_raw = clinical_df.values.astype(np.float32)
    clinical_static_raw = np.nan_to_num(clinical_static_raw, nan=0.0)
    
    # 2. Extract Labels
    labels = df[TARGET_COLUMN].values.astype(np.float32)
    
    # 3. Extract Trop Sequences
    trop_seq = df[["trop_0", "trop_1", "trop_2"]].values.astype(np.float32)
    trop_times = df[["trop_0_time", "trop_1_time", "trop_2_time"]].values.astype(np.float32)
    
    # 4. Extract ECG Times (convert to hours relative to something, or just unix timestamp)
    # The dataframe has ecg_time_0, 1, 2 as datetime.
    ecg_times = np.zeros((len(df), MAX_SEQ_LEN), dtype=np.float32)
    for i in range(MAX_SEQ_LEN):
        col = f"ecg_time_{i}"
        # convert to hours from epoch to easily compute differences
        times = pd.to_datetime(df[col], errors='coerce').astype(np.int64) / 1e9 / 3600.0
        # NaT becomes negative (like -250000), we'll replace with -1.0
        times = np.where(times < 0, -1.0, times)
        ecg_times[:, i] = times
        
    # 5. Extract Sequence Masks
    seq_masks = np.zeros((len(df), MAX_SEQ_LEN), dtype=np.float32)
    for i in range(len(df)):
        seq_len = int(df.iloc[i]["ecg_seq_len"])
        seq_masks[i, :seq_len] = 1.0
        
    # 6. Build ECG Memmap by copying from the original static memmap
    n_samples = len(df)
    memmap_path = ECG_CACHE_DIR / "temporal_ecg_dataset.dat"
    memmap_shape = (n_samples, MAX_SEQ_LEN, ECG_LEADS, ECG_LENGTH)
    
    expected_bytes = n_samples * MAX_SEQ_LEN * ECG_LEADS * ECG_LENGTH * 4 # 4 bytes per float32
    
    if memmap_path.exists() and memmap_path.stat().st_size == expected_bytes:
        print(f"[ECG] Temporal Memmap already exists and size matches: {memmap_path}")
    else:
        print("[ECG] Building 4D Temporal ECG Memmap from existing static memmap...")
        
        # Load the original mapping to find the indices
        old_df = pd.read_parquet(r"D:\MINI_PROJECT\mimic_data\final_preprocessed_fusion_dataset.parquet")
        path_to_idx = {p: i for i, p in enumerate(old_df["ecg_path"].values)}
        
        old_memmap_path = ECG_CACHE_DIR / f"ecg_memmap_{len(old_df)}.dat"
        if not old_memmap_path.exists():
            raise FileNotFoundError(f"Cannot find original memmap at {old_memmap_path}")
            
        old_memmap = np.memmap(old_memmap_path, dtype=np.float32, mode='r', shape=(len(old_df), ECG_LEADS, ECG_LENGTH))
        new_memmap = np.memmap(memmap_path, dtype=np.float32, mode='w+', shape=memmap_shape)
        
        for i, (_, row) in enumerate(tqdm(df.iterrows(), total=n_samples, desc="Writing Sequences")):
            for t in range(MAX_SEQ_LEN):
                p = row[f"ecg_path_{t}"]
                if pd.notnull(p) and p in path_to_idx:
                    idx = path_to_idx[p]
                    # The old memmap is already z-scored!
                    new_memmap[i, t] = old_memmap[idx]
                else:
                    new_memmap[i, t] = np.zeros((ECG_LEADS, ECG_LENGTH), dtype=np.float32)
                    
        new_memmap.flush()
        del new_memmap
        del old_memmap
        
    # 7. Splits (Grouped by subject_id)
    # df has hadm_id, maybe subject_id. Let's assume hadm_id is episode-centric.
    # Grouping by subject_id prevents patient overlap.
    if "subject_id" not in df.columns:
        print("[WARNING] 'subject_id' not found! Falling back to 'hadm_id' for splitting.")
        groups = df["hadm_id"].values
    else:
        groups = df["subject_id"].values
        
    indices = np.arange(n_samples)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_val_idx, test_idx = next(gss.split(indices, groups=groups))

    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, val_idx = next(gss_val.split(train_val_idx, groups=groups[train_val_idx]))
    train_idx = train_val_idx[train_idx]
    val_idx = train_val_idx[val_idx]
    
    # 8. Scale Static Clinical Features
    scaler = StandardScaler()
    clinical_static_raw[train_idx] = scaler.fit_transform(clinical_static_raw[train_idx])
    clinical_static_raw[val_idx] = scaler.transform(clinical_static_raw[val_idx])
    clinical_static_raw[test_idx] = scaler.transform(clinical_static_raw[test_idx])
    
    # (Trop sequences can also be scaled, but since they are absolute physiological values, 
    # letting the model learn from absolute Troponin (0-50 ng/mL) is often fine, 
    # or we can apply log1p. We will just log1p the troponin sequence to compress massive outliers.)
    trop_seq = np.log1p(trop_seq)

    # 9. Create PyTorch Datasets
    def make_ds(idxs):
        ds = TemporalFusionDataset(
            memmap_path=str(memmap_path),
            memmap_indices=idxs,
            clinical_static=clinical_static_raw[idxs],
            trop_seq=trop_seq[idxs],
            trop_times=trop_times[idxs],
            ecg_times=ecg_times[idxs],
            seq_masks=seq_masks[idxs],
            labels=labels[idxs]
        )
        ds.bind_memmap_shape(n_samples)
        return ds

    train_ds = make_ds(train_idx)
    val_ds = make_ds(val_idx)
    test_ds = make_ds(test_idx)

    # 10. DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=train_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=val_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=val_workers, pin_memory=True)

    pos_weight = (len(train_idx) - labels[train_idx].sum()) / labels[train_idx].sum()
    print(f"[DATA] Positive class weight: {pos_weight:.4f}")

    return train_loader, val_loader, test_loader, pos_weight, len(static_clinical_cols)
