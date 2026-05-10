# ==========================================
# DATASET — Multimodal Cardiac Dataset
# ==========================================
# • Downloads ECG waveforms from S3 using
#   multithreaded parallel downloads (fast)
# • Lazy-loads ECGs from local cache during
#   training (memory-efficient)
# ==========================================

import os
import numpy as np
import pandas as pd
import torch
import boto3
import wfdb
from pathlib import Path
from dotenv import load_dotenv
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import (
    DATASET_PATH, CLINICAL_FEATURES, TARGET_COLUMN,
    ECG_LEADS, ECG_LENGTH, BATCH_SIZE, VAL_SPLIT, TEST_SPLIT, SEED,
    PROJECT_ROOT, NUM_WORKERS, VAL_NUM_WORKERS, PREFETCH_FACTOR,
    AUG_SCALE_MIN, AUG_SCALE_MAX, AUG_NOISE_STD, AUG_SHIFT_MAX,
    AUG_LEAD_DROP_PROB, AUG_CUTOUT_PROB, AUG_WANDER_PROB,
    ECG_TIME_WINDOW_HOURS, MAX_ECGS_PER_PATIENT,
)

# ── Load .env from project root ──────────────────────────────────────────────
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH)

# ── AWS S3 Configuration (loaded from .env) ──────────────────────────────────
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION     = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET      = os.getenv("S3_BUCKET", "arn:aws:s3:us-east-1:724665945834:accesspoint/mimic-iv-ecg-v1-0-01")

# ── Local cache directory for downloaded ECG signals ─────────────────────────
ECG_CACHE_DIR = PROJECT_ROOT / "mimic_data" / "ecg_cache"

# ── Download settings ────────────────────────────────────────────────────────
MAX_DOWNLOAD_WORKERS = 64   # parallel S3 download threads


# ==========================================
# ECG LOADING UTILITIES
# ==========================================

def get_s3_client():
    """Create a boto3 S3 client (each thread needs its own)."""
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION,
    )


def _get_local_record_path(ecg_path: str, cache_dir: Path) -> str:
    """Convert an S3 ecg_path to its local cache path (without extension)."""
    record_name = ecg_path.rsplit("/", 1)[-1]
    sub_dir     = ecg_path.rsplit("/", 1)[0]
    local_dir   = cache_dir / sub_dir
    return str(local_dir / record_name)


def _is_cached(local_record_path: str) -> bool:
    """Check if both .hea and .dat exist locally."""
    return (os.path.exists(local_record_path + ".hea") and
            os.path.exists(local_record_path + ".dat"))


def _download_single_ecg(ecg_path: str, cache_dir: Path) -> bool:
    """
    Download a single ECG record (.hea + .dat) from S3.
    Returns True on success, False on failure.
    Each call creates its own S3 client (thread-safe).
    """
    local_record_path = _get_local_record_path(ecg_path, cache_dir)

    # Skip if already cached
    if _is_cached(local_record_path):
        return True

    # Create local directory
    local_dir = os.path.dirname(local_record_path)
    os.makedirs(local_dir, exist_ok=True)

    try:
        s3 = get_s3_client()
        s3.download_file(S3_BUCKET, ecg_path + ".hea", local_record_path + ".hea")
        s3.download_file(S3_BUCKET, ecg_path + ".dat", local_record_path + ".dat")
        return True
    except Exception:
        # Clean up partial downloads
        for ext in [".hea", ".dat"]:
            path = local_record_path + ext
            if os.path.exists(path):
                os.remove(path)
        return False


def download_all_ecgs(ecg_paths: list, cache_dir: Path = ECG_CACHE_DIR,
                      max_workers: int = MAX_DOWNLOAD_WORKERS):
    """
    Download all ECG records from S3 using multithreaded parallelism.

    With 64 threads, ~125K records takes ~1-2 hours instead of 70+ hours.
    Already-cached records are skipped automatically (resume-safe).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check how many are already cached
    uncached = [p for p in ecg_paths if not _is_cached(_get_local_record_path(p, cache_dir))]
    n_cached = len(ecg_paths) - len(uncached)

    if not uncached:
        print(f"[ECG] All {len(ecg_paths):,} records already cached (OK)")
        return

    print(f"[ECG] Already cached: {n_cached:,} / {len(ecg_paths):,}")
    print(f"[ECG] Downloading {len(uncached):,} records with {max_workers} parallel threads ...")

    success = 0
    failed  = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_download_single_ecg, path, cache_dir): path
            for path in uncached
        }

        with tqdm(total=len(uncached), desc="Downloading ECGs") as pbar:
            for future in as_completed(futures):
                if future.result():
                    success += 1
                else:
                    failed += 1
                pbar.update(1)

    print(f"[ECG] Download complete -- success: {success:,} | failed: {failed:,}")


def load_ecg_signal(record_path: str, target_length: int = ECG_LENGTH) -> np.ndarray:
    """
    Load a single 12-lead ECG signal from a local wfdb record.

    Parameters
    ----------
    record_path   : str — path without .hea/.dat extension
    target_length : int — desired signal length (pads or truncates)

    Returns
    -------
    signal : np.ndarray of shape (12, target_length)
    """
    record = wfdb.rdrecord(record_path)
    signal = record.p_signal  # (num_samples, num_leads)

    if signal is None:
        raise ValueError(f"No signal in record: {record_path}")

    # Transpose to (leads, samples)
    signal = signal.T  # (num_leads, num_samples)

    # Ensure exactly 12 leads
    n_leads = signal.shape[0]
    if n_leads < ECG_LEADS:
        pad = np.zeros((ECG_LEADS - n_leads, signal.shape[1]), dtype=np.float32)
        signal = np.vstack([signal, pad])
    elif n_leads > ECG_LEADS:
        signal = signal[:ECG_LEADS, :]

    # Truncate or pad to target_length
    n_samples = signal.shape[1]
    if n_samples >= target_length:
        signal = signal[:, :target_length]
    else:
        pad = np.zeros((ECG_LEADS, target_length - n_samples), dtype=np.float32)
        signal = np.hstack([signal, pad])

    return signal.astype(np.float32)


# ==========================================
# ECG DATA AUGMENTATION
# ==========================================

def _augment_ecg(ecg: np.ndarray) -> np.ndarray:
    """
    Apply random augmentations to a z-scored ECG signal during training.

    Augmentations:
      1. Random amplitude scaling (±10%)
      2. Additive Gaussian noise (small)
      3. Random temporal shift (up to ±100 samples)
      4. Random lead dropout (zero out 1–2 leads)
      5. Random temporal cutout / masking
      6. Random baseline wander (sinusoidal drift)

    Parameters
    ----------
    ecg : np.ndarray of shape (12, target_length) — already z-scored

    Returns
    -------
    ecg : np.ndarray of same shape, augmented
    """
    # 1. Random amplitude scaling
    scale = np.random.uniform(AUG_SCALE_MIN, AUG_SCALE_MAX)
    ecg = ecg * scale

    # 2. Additive Gaussian noise
    noise = np.random.normal(0, AUG_NOISE_STD, ecg.shape).astype(np.float32)
    ecg = ecg + noise

    # 3. Random temporal shift (wrap-around)
    shift = np.random.randint(-AUG_SHIFT_MAX, AUG_SHIFT_MAX + 1)
    if shift != 0:
        ecg = np.roll(ecg, shift, axis=1)

    # 4. Random lead dropout — zero out 1–2 leads to prevent single-lead reliance
    if np.random.random() < AUG_LEAD_DROP_PROB:
        n_drop = np.random.randint(1, 3)
        drop_leads = np.random.choice(ecg.shape[0], n_drop, replace=False)
        ecg[drop_leads, :] = 0.0

    # 5. Random temporal cutout — mask a contiguous segment (simulates artifacts)
    if np.random.random() < AUG_CUTOUT_PROB:
        mask_len = np.random.randint(100, 500)
        max_start = max(ecg.shape[1] - mask_len, 1)
        start = np.random.randint(0, max_start)
        ecg[:, start:start + mask_len] = 0.0

    # 6. Random baseline wander — low-frequency sinusoidal drift
    if np.random.random() < AUG_WANDER_PROB:
        freq = np.random.uniform(0.1, 0.5)
        amp = np.random.uniform(0.05, 0.2)
        t = np.linspace(0, 2 * np.pi * freq, ecg.shape[1])
        wander = (amp * np.sin(t)).astype(np.float32)
        ecg = ecg + wander[np.newaxis, :]

    return ecg


# ==========================================
# MEMORY-MAPPED ECG CACHE
# ==========================================
# Instead of reading 125K wfdb files per epoch (~20 min), we pre-process
# all ECGs once into a single memory-mapped file. Each subsequent epoch
# reads pre-z-scored signals via memmap (microseconds per sample).
# ==========================================

def _ecg_memmap_path(n_records: int) -> Path:
    """Return path to the memmap file for a given dataset size."""
    return ECG_CACHE_DIR / f"ecg_memmap_{n_records}.dat"


def _build_ecg_memmap(local_paths: list) -> tuple:
    """
    Build (or load) a memory-mapped numpy array of all z-scored ECG signals.

    First run  : loads each ECG via wfdb, z-scores, writes to memmap (one-time).
    Later runs : just opens the existing memmap (instant).

    Returns
    -------
    memmap_path : Path  — stored so datasets can reopen lazily per-worker
    shape       : tuple — (n_records, ECG_LEADS, ECG_LENGTH)
    """
    n = len(local_paths)
    shape = (n, ECG_LEADS, ECG_LENGTH)
    memmap_path = _ecg_memmap_path(n)

    if memmap_path.exists():
        # Validate the file size matches expected shape
        expected_bytes = n * ECG_LEADS * ECG_LENGTH * 4  # float32
        actual_bytes = memmap_path.stat().st_size
        if actual_bytes == expected_bytes:
            print(f"[ECG] Memmap cache loaded: {memmap_path} ({n:,} records, {actual_bytes / 1e9:.1f} GB)")
            return memmap_path, shape
        else:
            print(f"[ECG] Memmap size mismatch ({actual_bytes} vs {expected_bytes}), rebuilding ...")
            memmap_path.unlink()

    size_gb = n * ECG_LEADS * ECG_LENGTH * 4 / 1e9
    print(f"[ECG] Building memmap cache for {n:,} records (~{size_gb:.1f} GB, one-time operation) ...")
    mmap = np.memmap(str(memmap_path), dtype=np.float32, mode='w+', shape=shape)

    for i in tqdm(range(n), desc="Pre-caching ECG signals"):
        try:
            ecg = load_ecg_signal(local_paths[i])
        except Exception:
            ecg = np.zeros((ECG_LEADS, ECG_LENGTH), dtype=np.float32)

        # Clean and z-score (same logic as the old __getitem__)
        ecg = np.nan_to_num(ecg, nan=0.0, posinf=0.0, neginf=0.0)
        lead_mean = ecg.mean(axis=1, keepdims=True)
        lead_std  = ecg.std(axis=1, keepdims=True) + 1e-8
        ecg = (ecg - lead_mean) / lead_std

        mmap[i] = ecg

        # Flush periodically to avoid memory buildup
        if (i + 1) % 10000 == 0:
            mmap.flush()

    mmap.flush()
    del mmap  # close the write handle

    print(f"[ECG] Memmap cache saved: {memmap_path}")
    return memmap_path, shape


# ==========================================
# FAST MEMMAP-BACKED PYTORCH DATASET
# ==========================================

class MultimodalCardiacDataset(Dataset):
    """
    Fast Dataset that reads pre-processed ECG signals from a memory-mapped file.

    The memmap is opened lazily in each DataLoader worker process to avoid
    pickling issues on Windows multiprocessing.
    """
    def __init__(self, memmap_path, memmap_shape: tuple,
                 ecg_indices: np.ndarray, clinical_data: np.ndarray,
                 labels: np.ndarray, augment: bool = False,
                 ecg_base_memmap_path=None, has_baseline: np.ndarray = None):
        self.memmap_path   = str(memmap_path)
        self.memmap_shape  = memmap_shape
        self.ecg_indices   = np.asarray(ecg_indices)
        self.clinical_data = torch.tensor(clinical_data, dtype=torch.float32)
        self.labels        = torch.tensor(labels, dtype=torch.float32)
        self.augment       = augment
        self._ecg_memmap   = None  # opened lazily per worker
        
        self.ecg_base_memmap_path = str(ecg_base_memmap_path) if ecg_base_memmap_path else None
        self.has_baseline = torch.tensor(has_baseline, dtype=torch.float32) if has_baseline is not None else None
        self._ecg_base_memmap = None

    def _get_memmap(self):
        """Open the memmap lazily — each worker gets its own file handle."""
        if self._ecg_memmap is None:
            self._ecg_memmap = np.memmap(
                self.memmap_path, dtype=np.float32, mode='r',
                shape=self.memmap_shape,
            )
        return self._ecg_memmap

    def _get_base_memmap(self):
        if self._ecg_base_memmap is None and self.ecg_base_memmap_path is not None:
            self._ecg_base_memmap = np.memmap(
                self.ecg_base_memmap_path, dtype=np.float32, mode='r',
                shape=self.memmap_shape,
            )
        return self._ecg_base_memmap

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Fast memmap read — microseconds instead of ~10ms per wfdb call
        mmap = self._get_memmap()
        memmap_idx = self.ecg_indices[idx]
        ecg = np.array(mmap[memmap_idx])  # copy to writable array

        # Baseline ECG
        base_mmap = self._get_base_memmap()
        if base_mmap is not None and self.has_baseline is not None:
            ecg_base = np.array(base_mmap[memmap_idx])
            has_base = self.has_baseline[idx]
        else:
            ecg_base = np.zeros_like(ecg)
            has_base = torch.tensor(0.0, dtype=torch.float32)

        # Apply augmentation during training
        if self.augment:
            ecg = _augment_ecg(ecg)

        ecg_tensor = torch.tensor(ecg, dtype=torch.float32)
        ecg_base_tensor = torch.tensor(ecg_base, dtype=torch.float32)
        
        return ecg_tensor, self.clinical_data[idx], self.labels[idx], ecg_base_tensor, has_base


def _build_weighted_sampler(labels: np.ndarray, hard_negative_mask: np.ndarray | None = None) -> WeightedRandomSampler | None:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size == 0:
        return None

    if hard_negative_mask is None:
        class_counts = np.bincount(labels, minlength=2).astype(np.float64)
        if np.any(class_counts == 0):
            return None
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels]
    else:
        sample_weights = np.ones_like(labels, dtype=np.float64)
        # Class 0 (Negatives): default weight 0.2
        sample_weights[labels == 0] = 0.2
        # Class 0 Hard Negatives: weight 2.0
        sample_weights[(labels == 0) & hard_negative_mask] = 2.0
        # Class 1 (Positives): weight 1.0
        sample_weights[labels == 1] = 1.0

    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
    )


def infer_group_ids(df: pd.DataFrame) -> tuple[str | None, np.ndarray | None]:
    """
    Return stable patient/admission groups for leakage-safe splitting.

    Prefer subject-level grouping when available. The current model parquet keeps
    only ecg_path, so we recover subject_id from paths like:
    mimic-iv-ecg/.../p1579/p15797206/s41845173/41845173
    """
    if "subject_id" in df.columns:
        return "subject_id", df["subject_id"].astype(str).to_numpy()

    if "ecg_path" in df.columns:
        extracted = df["ecg_path"].astype(str).str.extract(r"/p\d+/p(\d+)/")[0]
        if extracted.notna().any():
            return "subject_id_from_ecg_path", extracted.fillna(df["ecg_path"].astype(str)).to_numpy()

    if "hadm_id" in df.columns:
        return "hadm_id", df["hadm_id"].astype(str).to_numpy()

    return None, None


def stratified_group_train_test_split(
    indices: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray | None,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stratified split that keeps all rows from a group in the same side.

    StratifiedGroupKFold controls split size by number of folds, so the resulting
    row count can be near, not exactly equal to, the requested fraction.
    """
    indices = np.asarray(indices)

    if groups is None:
        return train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=labels[indices],
        )

    n_splits = max(2, round(1.0 / test_size))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    local_y = labels[indices]
    local_groups = groups[indices]

    train_local, test_local = next(
        splitter.split(np.zeros(len(indices)), local_y, groups=local_groups)
    )
    return indices[train_local], indices[test_local]


def _make_dataloader(
    dataset: Dataset,
    shuffle: bool,
    num_workers: int,
    sampler: WeightedRandomSampler | None = None,
) -> DataLoader:
    """
    Build a DataLoader with settings that are stable for the current OS/device.

    On Windows, multiprocessing workers can fail with
    "Couldn't open shared file mapping" for large batches. Using
    num_workers=0 avoids shared-memory file mappings entirely.
    """
    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "shuffle": shuffle if sampler is None else False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    if sampler is not None:
        loader_kwargs["sampler"] = sampler

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR

    return DataLoader(dataset, **loader_kwargs)


def prepare_full_dataset(subset: int = None, clinical_features: list[str] | None = None):
    """
    Load, clean, and cache the full dataset once for reuse across splits.
    """
    print(f"[DATA] Loading: {DATASET_PATH}")
    df = pd.read_parquet(DATASET_PATH)
    print(f"[DATA] Full dataset: {df.shape[0]:,} rows x {df.shape[1]} cols")

    if subset is not None:
        df = df.head(subset).copy()
        print(f"[DATA] Using subset: {len(df):,} samples")

    selected_features = list(clinical_features or CLINICAL_FEATURES)
    missing_features = [col for col in selected_features if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing clinical feature columns: {missing_features}")

    clinical_df = df[selected_features].copy()
    labels = df[TARGET_COLUMN].values.astype(np.float32)
    ecg_paths = df["ecg_path"].tolist()

    clinical_df = clinical_df.replace([np.inf, -np.inf], np.nan)

    valid_ranges = {
        "anchor_age": (0, 120),
        "Heart_Rate": (20, 250),
        "Respiratory_Rate": (5, 60),
        "Troponin_T": (0, 50),
        "Creatinine": (0.1, 15),
        "Sodium": (100, 180),
        "Potassium": (1.5, 10),
        "PR_interval": (50, 500),
        "QRS_duration": (40, 300),
        "QT_interval": (200, 800),
        "QTc": (200, 800),
        "P_axis": (-180, 360),
        "QRS_axis": (-180, 360),
        "T_axis": (-180, 360),
        "RR_interval": (200, 3000),
    }
    for col, (lo, hi) in valid_ranges.items():
        if col in clinical_df.columns:
            mask = (clinical_df[col] < lo) | (clinical_df[col] > hi)
            clinical_df.loc[mask, col] = np.nan

    clinical_df = clinical_df.fillna(clinical_df.median())
    clinical_raw = clinical_df.values.astype(np.float32)
    clinical_raw = np.nan_to_num(clinical_raw, nan=0.0, posinf=0.0, neginf=0.0)

    n_samples = len(labels)
    print(f"[DATA] Clinical features: {clinical_raw.shape[1]} (cleaned)")
    print(f"[DATA] AMI prevalence: {labels.mean():.4%}  ({int(labels.sum()):,} / {n_samples:,})")

    local_paths = [_get_local_record_path(p, ECG_CACHE_DIR) for p in ecg_paths]
    candidate_memmap = _ecg_memmap_path(n_samples)
    expected_bytes = n_samples * ECG_LEADS * ECG_LENGTH * 4

    if candidate_memmap.exists() and candidate_memmap.stat().st_size == expected_bytes:
        print("[ECG] Memmap cache valid, skipping individual file checks")
        ecg_memmap_path = candidate_memmap
        ecg_memmap_shape = (n_samples, ECG_LEADS, ECG_LENGTH)
    else:
        download_all_ecgs(ecg_paths)

        valid_mask = np.array([_is_cached(p) for p in local_paths])
        if not valid_mask.all():
            n_before = n_samples
            valid_indices = np.where(valid_mask)[0]
            local_paths = [local_paths[i] for i in valid_indices]
            clinical_raw = clinical_raw[valid_mask]
            labels = labels[valid_mask]
            df = df.iloc[valid_indices].reset_index(drop=True)
            n_samples = len(labels)
            print(f"[DATA] Filtered: {n_before:,} -> {n_samples:,} (removed {n_before - n_samples:,} failed)")

        ecg_memmap_path, ecg_memmap_shape = _build_ecg_memmap(local_paths)

    return {
        "df": df,
        "clinical_raw": clinical_raw,
        "clinical_features": selected_features,
        "labels": labels,
        "local_paths": local_paths,
        "ecg_memmap_path": ecg_memmap_path,
        "ecg_memmap_shape": ecg_memmap_shape,
    }


def make_split_dataloaders(
    prepared: dict,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    train_num_workers: int | None = None,
    val_num_workers: int | None = None,
    augment_train: bool = True,
    weighted_sampling: bool = False,
):
    """
    Standardize clinical features on the training indices and build loaders.
    """
    clinical_raw = prepared["clinical_raw"]
    labels = prepared["labels"]

    train_idx = np.asarray(train_idx)
    val_idx = np.asarray(val_idx)

    train_clinical = clinical_raw[train_idx]
    col_mean = train_clinical.mean(axis=0, keepdims=True)
    col_std = train_clinical.std(axis=0, keepdims=True) + 1e-8
    clinical_standardized = (clinical_raw - col_mean) / col_std

    train_ds = MultimodalCardiacDataset(
        prepared["ecg_memmap_path"],
        prepared["ecg_memmap_shape"],
        train_idx,
        clinical_standardized[train_idx],
        labels[train_idx],
        augment=augment_train,
    )
    val_ds = MultimodalCardiacDataset(
        prepared["ecg_memmap_path"],
        prepared["ecg_memmap_shape"],
        val_idx,
        clinical_standardized[val_idx],
        labels[val_idx],
        augment=False,
    )

    effective_train_workers = NUM_WORKERS if train_num_workers is None else train_num_workers
    effective_val_workers = VAL_NUM_WORKERS if val_num_workers is None else val_num_workers

    train_sampler = None
    if weighted_sampling:
        df = prepared.get("df")
        if df is not None and "comorbidity_count" in df.columns and "Troponin_T_high" in df.columns:
            # hard negative: troponin > 0.1 AND has comorbidities
            hard_negative_mask = (df["Troponin_T_high"].values == 1) & (df["comorbidity_count"].values > 0)
            train_sampler = _build_weighted_sampler(labels[train_idx], hard_negative_mask[train_idx])
        else:
            train_sampler = _build_weighted_sampler(labels[train_idx])
    train_loader = _make_dataloader(
        train_ds,
        shuffle=True,
        num_workers=effective_train_workers,
        sampler=train_sampler,
    )
    val_loader = _make_dataloader(val_ds, shuffle=False, num_workers=effective_val_workers)

    n_pos = labels[train_idx].sum()
    n_neg = len(train_idx) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "pos_weight": pos_weight,
        "clinical_standardized": clinical_standardized,
        "col_mean": col_mean,
        "col_std": col_std,
        "train_workers": effective_train_workers,
        "val_workers": effective_val_workers,
        "weighted_sampling": weighted_sampling,
    }


# ==========================================
# MAIN DATA LOADING FUNCTION
# ==========================================

def _compute_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute error-analysis-driven features on-the-fly.

    These features target the two root causes of the F1 plateau:
      - troponin_delta_24h: separates rising troponin (true AMI) from
        chronically elevated troponin (Type 2 MI / CKD false positives)
      - troponin_creatinine_ratio: normalises troponin by kidney function
        to reduce false positives from CKD patients
    """
    # Troponin trajectory — rise from first to max in 24h
    if "troponin_max_24h" in df.columns and "troponin_first_24h" in df.columns:
        df["troponin_delta_24h"] = (
            df["troponin_max_24h"] - df["troponin_first_24h"]
        ).fillna(0.0)
        
        # Multiplicative rise
        df["troponin_rise_ratio"] = (
            df["troponin_max_24h"] / df["troponin_first_24h"].clip(lower=0.001)
        ).fillna(1.0)
    else:
        df["troponin_delta_24h"] = 0.0
        df["troponin_rise_ratio"] = 1.0

    # Troponin-to-ECG timing alignment
    if "hours_admit_to_first_troponin" in df.columns and "hours_admit_to_ecg" in df.columns:
        # Avoid treating 999 (missing) as valid timing gaps
        valid_timing = (df["hours_admit_to_first_troponin"] < 900) & (df["hours_admit_to_ecg"] < 900)
        
        df["troponin_ecg_time_gap"] = np.where(
            valid_timing,
            np.abs(df["hours_admit_to_first_troponin"] - df["hours_admit_to_ecg"]),
            999.0
        )
        
        df["troponin_before_ecg"] = np.where(
            valid_timing,
            (df["hours_admit_to_first_troponin"] < df["hours_admit_to_ecg"]).astype(int),
            0
        )
    else:
        df["troponin_ecg_time_gap"] = 999.0
        df["troponin_before_ecg"] = 0

    # Troponin normalised by kidney function
    if "log1p_Troponin_T" in df.columns and "Creatinine" in df.columns:
        df["troponin_creatinine_ratio"] = (
            df["log1p_Troponin_T"] / (df["Creatinine"].clip(lower=0.1))
        ).fillna(0.0)
    else:
        df["troponin_creatinine_ratio"] = 0.0

    # Clinical Acuity Score (composite risk indicator)
    req_cols = ["Troponin_T_high", "Creatinine_high", "QTc_prolonged", "QRS_wide", "Heart_Rate", "Respiratory_Rate"]
    if all(c in df.columns for c in req_cols):
        df["acuity_score"] = (
            df["Troponin_T_high"] + 
            df["Creatinine_high"] + 
            df["QTc_prolonged"] + 
            df["QRS_wide"] + 
            (df["Heart_Rate"] > 100).astype(int) + 
            (df["Respiratory_Rate"] > 22).astype(int)
        ).fillna(0.0)
    else:
        df["acuity_score"] = 0.0

    return df


def _apply_data_filters(
    indices: np.ndarray,
    df: pd.DataFrame,
    groups: np.ndarray | None,
    ecg_time_window: int,
    max_ecgs_per_patient: int,
) -> np.ndarray:
    """
    Apply error-analysis-driven data quality filters.

    Parameters
    ----------
    indices : array of valid row indices
    df : full DataFrame (used for column access)
    groups : patient group array (or None)
    ecg_time_window : keep ECGs within ±N hours of admission (0 = off)
    max_ecgs_per_patient : cap repeated ECGs per patient (0 = off)

    Returns
    -------
    filtered_indices : array of indices that pass all filters
    """
    n_before = len(indices)

    # Fix 5: Filter ECGs taken far from admission
    if ecg_time_window > 0 and "hours_admit_to_ecg" in df.columns:
        hours = df["hours_admit_to_ecg"].values
        time_valid = np.abs(hours[indices]) <= ecg_time_window
        indices = indices[time_valid]
        print(
            f"[FILTER] ECG timing +/-{ecg_time_window}h: "
            f"{n_before:,} -> {len(indices):,} "
            f"(removed {n_before - len(indices):,})"
        )

    # Fix 4: Cap per-patient ECGs
    if max_ecgs_per_patient > 0 and groups is not None:
        from collections import Counter
        group_counts: dict[int, int] = Counter()
        keep = []
        for idx in indices:
            g = groups[idx]
            group_counts[g] += 1
            if group_counts[g] <= max_ecgs_per_patient:
                keep.append(idx)
        n_before2 = len(indices)
        indices = np.array(keep, dtype=indices.dtype)
        print(
            f"[FILTER] Per-patient cap ({max_ecgs_per_patient}): "
            f"{n_before2:,} -> {len(indices):,} "
            f"(removed {n_before2 - len(indices):,})"
        )

    return indices


def load_and_prepare_data(
    subset: int = None,
    train_num_workers: int | None = None,
    val_num_workers: int | None = None,
    weighted_sampling: bool = False,
    clinical_features: list[str] | None = None,
    ecg_time_window: int = ECG_TIME_WINDOW_HOURS,
    max_ecgs_per_patient: int = MAX_ECGS_PER_PATIENT,
):
    """
    Load preprocessed parquet, download ECGs from S3 (multithreaded),
    and return train/val DataLoaders with lazy ECG loading.

    Performs a TWO-STAGE split:
      Stage 1: Full data → train_pool (85%) + test_set (15%)
      Stage 2: train_pool → train (80%) + val (20%)

    Standardization is fitted ONLY on train_pool (no data leakage).

    Parameters
    ----------
    subset : int or None
        If set, use only the first N samples (for quick testing).

    Returns
    -------
    train_loader : DataLoader
    val_loader   : DataLoader
    pos_weight   : torch.Tensor
    test_metadata : dict — test set data for store_predictions.py
    """

    # ── Load preprocessed clinical + label data ──────────────────────────
    print(f"[DATA] Loading: {DATASET_PATH}")
    df = pd.read_parquet(DATASET_PATH)
    print(f"[DATA] Full dataset: {df.shape[0]:,} rows x {df.shape[1]} cols")

    # ── Optional subset for quick testing ────────────────────────────────
    if subset is not None:
        df = df.head(subset).copy()
        print(f"[DATA] Using subset: {len(df):,} samples")

    # ── Compute engineered features on-the-fly ────────────────────────────
    df = _compute_engineered_features(df)

    # ── Extract features, labels, paths ──────────────────────────────────
    selected_features = list(clinical_features or CLINICAL_FEATURES)
    missing_features = [col for col in selected_features if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing clinical feature columns: {missing_features}")

    clinical_df = df[selected_features].copy()
    labels      = df[TARGET_COLUMN].values.astype(np.float32)
    ecg_paths   = df["ecg_path"].tolist()
    
    # ── Siamese Baseline ECG Paths ──────────────────────────────────────
    has_baseline_mask = df["has_baseline_ecg"].values.astype(np.float32) if "has_baseline_ecg" in df.columns else np.zeros(len(df), dtype=np.float32)
    base_ecg_paths = df["baseline_ecg_path"].tolist() if "baseline_ecg_path" in df.columns else [None]*len(df)

    # ── Clean clinical features (sentinel / extreme values) ──────────────
    # Many ECG machine columns contain sentinel values like -29999, 32767,
    # which are invalid and cause NaN loss during training.

    # 1. Replace inf with NaN
    clinical_df = clinical_df.replace([np.inf, -np.inf], np.nan)

    # 2. Replace impossible values with NaN (wide PHYSIOLOGICAL ranges)
    #    ⚠️ These are NOT normal ranges — we KEEP abnormal values (critical for AMI detection).
    #    We only remove garbage: sensor errors, sentinel values (-29999, 32767), etc.
    VALID_RANGES = {
        # Demographics
        "anchor_age":       (0, 120),       # years
        # Vitals — keep abnormal (tachycardia, bradycardia matter for AMI)
        "Heart_Rate":       (20, 250),      # bpm
        "Respiratory_Rate": (5, 60),        # breaths/min
        # Labs — keep elevated values (elevated troponin IS the AMI signal!)
        "Troponin_T":       (0, 50),        # ng/mL — AMI patients can have very high values
        "Creatinine":       (0.1, 15),      # mg/dL
        "Sodium":           (100, 180),     # mEq/L
        "Potassium":        (1.5, 10),      # mEq/L
        # ECG machine measurements
        "PR_interval":      (50, 500),      # ms
        "QRS_duration":     (40, 300),      # ms
        "QT_interval":      (200, 800),     # ms
        "QTc":              (200, 800),     # ms
        "P_axis":           (-180, 360),    # degrees
        "QRS_axis":         (-180, 360),    # degrees
        "T_axis":           (-180, 360),    # degrees
        "RR_interval":      (200, 3000),    # ms
    }
    for col, (lo, hi) in VALID_RANGES.items():
        if col in clinical_df.columns:
            mask = (clinical_df[col] < lo) | (clinical_df[col] > hi)
            clinical_df.loc[mask, col] = np.nan

    # 3. Fill NaN with column median
    clinical_df = clinical_df.fillna(clinical_df.median())

    # 4. Convert to numpy (standardization deferred until after split)
    clinical_raw = clinical_df.values.astype(np.float32)
    clinical_raw = np.nan_to_num(clinical_raw, nan=0.0, posinf=0.0, neginf=0.0)

    n_samples = len(labels)
    print(f"[DATA] Clinical features: {clinical_raw.shape[1]} (cleaned)")
    print(f"[DATA] AMI prevalence: {labels.mean():.4%}  ({int(labels.sum()):,} / {n_samples:,})")

    # ── ECG loading: use memmap if available, else download + build ────
    # Build local cache paths (cheap string manipulation, no disk I/O)
    local_paths = [_get_local_record_path(p, ECG_CACHE_DIR) for p in ecg_paths]

    # Check if a valid memmap already exists for this dataset size.
    # If so, skip S3 download and individual file validation entirely.
    # This allows users to delete the individual .hea/.dat files after
    # the memmap has been built, freeing ~28 GB of disk space.
    candidate_memmap = _ecg_memmap_path(n_samples)
    expected_bytes = n_samples * ECG_LEADS * ECG_LENGTH * 4  # float32

    if candidate_memmap.exists() and candidate_memmap.stat().st_size == expected_bytes:
        print(f"[ECG] Memmap cache valid, skipping individual file checks")
        ecg_memmap_path = candidate_memmap
        ecg_memmap_shape = (n_samples, ECG_LEADS, ECG_LENGTH)
    else:
        # First run (or memmap missing): download from S3 + validate + build memmap
        download_all_ecgs(ecg_paths)

        # Filter out records that failed to download
        valid_mask = np.array([_is_cached(p) for p in local_paths])
        if not valid_mask.all():
            n_before = n_samples
            valid_indices = np.where(valid_mask)[0]
            local_paths  = [local_paths[i] for i in valid_indices]
            clinical_raw = clinical_raw[valid_mask]
            labels       = labels[valid_mask]
            has_baseline_mask = has_baseline_mask[valid_mask]
            base_ecg_paths = [base_ecg_paths[i] for i in valid_indices]
            df           = df.iloc[valid_indices].reset_index(drop=True)
            n_samples    = len(labels)
            print(f"[DATA] Filtered: {n_before:,} -> {n_samples:,} (removed {n_before - n_samples:,} failed)")

        # Build memory-mapped ECG cache (one-time)
        ecg_memmap_path, ecg_memmap_shape = _build_ecg_memmap(local_paths)

    # ── Baseline ECG Memmap ──────────────────────────────────────────────────
    candidate_base_memmap = ECG_CACHE_DIR / f"base_dataset_ecg_{n_samples}.dat"
    ecg_base_memmap_path = None
    if "has_baseline_ecg" in df.columns:
        if candidate_base_memmap.exists() and candidate_base_memmap.stat().st_size == expected_bytes:
            print(f"[ECG] Baseline Memmap cache valid")
            ecg_base_memmap_path = candidate_base_memmap
        else:
            print("[ECG] Building Baseline ECG Memmap...")
            valid_base_paths = [p for p in base_ecg_paths if p is not None]
            download_all_ecgs(valid_base_paths)
            
            # Map None paths to a dummy string so _build_ecg_memmap gracefully handles them
            local_base_paths = []
            for p in base_ecg_paths:
                if p is not None:
                    local_base_paths.append(_get_local_record_path(p, ECG_CACHE_DIR))
                else:
                    local_base_paths.append(None)
            
            # Custom memmap build for baselines (filling None with zeros)
            base_shape = (n_samples, ECG_LEADS, ECG_LENGTH)
            base_memmap = np.memmap(candidate_base_memmap, dtype=np.float32, mode='w+', shape=base_shape)
            
            for i, p in enumerate(tqdm(local_base_paths, desc="Writing Base Memmap")):
                if p is not None and _is_cached(p):
                    try:
                        sig = load_ecg_signal(p)
                        # Z-score exactly like admit ECG
                        sig_mean = np.mean(sig)
                        sig_std = np.std(sig) + 1e-8
                        sig = (sig - sig_mean) / sig_std
                        base_memmap[i] = sig
                    except Exception:
                        base_memmap[i] = np.zeros((ECG_LEADS, ECG_LENGTH))
                else:
                    base_memmap[i] = np.zeros((ECG_LEADS, ECG_LENGTH))
            
            base_memmap.flush()
            del base_memmap
            ecg_base_memmap_path = candidate_base_memmap

    # ══════════════════════════════════════════════════════════════════════
    # DATA QUALITY FILTERS (error-analysis-driven)
    # ══════════════════════════════════════════════════════════════════════
    indices = np.arange(n_samples)
    group_name, groups = infer_group_ids(df)

    indices = _apply_data_filters(
        indices, df, groups, ecg_time_window, max_ecgs_per_patient,
    )

    # Update labels/groups/clinical to only use filtered indices
    if len(indices) < n_samples:
        labels_filtered = labels[indices]
        clinical_filtered = clinical_raw[indices]
        has_baseline_filtered = has_baseline_mask[indices]
        groups_filtered = groups[indices] if groups is not None else None
        # Re-index: after filtering, indices become 0..len-1 for the split,
        # but we keep the original memmap indices for ECG access.
        original_memmap_indices = indices.copy()
        n_filtered = len(indices)
        print(f"[DATA] After filters: {n_samples:,} -> {n_filtered:,} samples")
        print(f"[DATA] AMI prevalence (filtered): {labels_filtered.mean():.4%}")
    else:
        labels_filtered = labels
        clinical_filtered = clinical_raw
        has_baseline_filtered = has_baseline_mask
        groups_filtered = groups
        original_memmap_indices = indices.copy()
        n_filtered = n_samples

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 1: Split off held-out TEST SET (15%)
    # ══════════════════════════════════════════════════════════════════════
    split_indices = np.arange(n_filtered)
    if groups_filtered is not None:
        print(f"[DATA] Grouped split enabled: {group_name} ({len(np.unique(groups_filtered)):,} groups)")
    else:
        print("[DATA] Grouped split unavailable; falling back to row-level stratified split")

    train_pool_idx, test_idx = stratified_group_train_test_split(
        split_indices,
        labels_filtered,
        groups_filtered,
        test_size=TEST_SPLIT,
        seed=SEED,
    )

    print(f"[DATA] Stage 1 split: train_pool={len(train_pool_idx):,}  |  test={len(test_idx):,}")
    if groups_filtered is not None:
        overlap = len(set(groups_filtered[train_pool_idx]) & set(groups_filtered[test_idx]))
        print(f"[DATA] Stage 1 group overlap: {overlap}")

    # ── Standardize using ONLY train_pool stats (prevent data leakage) ───
    train_pool_clinical = clinical_filtered[train_pool_idx]
    col_mean = train_pool_clinical.mean(axis=0, keepdims=True)
    col_std  = train_pool_clinical.std(axis=0, keepdims=True) + 1e-8

    # Apply same standardization to all splits
    clinical_standardized = (clinical_filtered - col_mean) / col_std

    # ── Map split indices → original memmap indices for ECG access ────────
    test_memmap_idx = original_memmap_indices[test_idx]

    # ── Build test metadata (for store_predictions.py) ───────────────────
    test_metadata = {
        "test_df":          df.iloc[original_memmap_indices[test_idx]].reset_index(drop=True),
        "clinical_features": selected_features,
        "test_local_paths": [local_paths[i] for i in test_memmap_idx],
        "test_clinical":    clinical_standardized[test_idx],
        "test_labels":      labels_filtered[test_idx],
        "col_mean":         col_mean,
        "col_std":          col_std,
        # Memmap info so store_predictions.py can read ECGs without
        # needing the individual .hea/.dat wfdb files on disk.
        "ecg_memmap_path":  ecg_memmap_path,
        "ecg_memmap_shape": ecg_memmap_shape,
        "test_ecg_indices": test_memmap_idx,
    }

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 2: Split train_pool → train (80%) + val (20%)
    # ══════════════════════════════════════════════════════════════════════
    train_idx, val_idx = stratified_group_train_test_split(
        train_pool_idx,
        labels_filtered,
        groups_filtered,
        test_size=VAL_SPLIT,
        seed=SEED,
    )
    if groups_filtered is not None:
        overlap = len(set(groups_filtered[train_idx]) & set(groups_filtered[val_idx]))
        print(f"[DATA] Stage 2 group overlap: {overlap}")

    # Map to memmap indices for ECG dataset
    train_memmap_idx = original_memmap_indices[train_idx]
    val_memmap_idx = original_memmap_indices[val_idx]

    train_ds = MultimodalCardiacDataset(
        ecg_memmap_path, ecg_memmap_shape,
        train_memmap_idx, clinical_standardized[train_idx], labels_filtered[train_idx],
        augment=True,   # ECG augmentation for training only
        ecg_base_memmap_path=ecg_base_memmap_path,
        has_baseline=has_baseline_filtered[train_idx],
    )
    val_ds = MultimodalCardiacDataset(
        ecg_memmap_path, ecg_memmap_shape,
        val_memmap_idx, clinical_standardized[val_idx], labels_filtered[val_idx],
        augment=False,  # no augmentation for validation
        ecg_base_memmap_path=ecg_base_memmap_path,
        has_baseline=has_baseline_filtered[val_idx],
    )

    effective_train_workers = NUM_WORKERS if train_num_workers is None else train_num_workers
    effective_val_workers = VAL_NUM_WORKERS if val_num_workers is None else val_num_workers

    train_sampler = _build_weighted_sampler(labels_filtered[train_idx]) if weighted_sampling else None
    train_loader = _make_dataloader(
        train_ds,
        shuffle=True,
        num_workers=effective_train_workers,
        sampler=train_sampler,
    )
    val_loader = _make_dataloader(val_ds, shuffle=False, num_workers=effective_val_workers)

    print(f"[DATA] Stage 2 split: Train={len(train_ds):,}  |  Val={len(val_ds):,}  |  Test={len(test_metadata['test_labels']):,}")
    print(
        f"[DATA] DataLoader config: train_workers={effective_train_workers} | "
        f"val_workers={effective_val_workers} | "
        f"weighted_sampling={weighted_sampling} | "
        f"pin_memory={torch.cuda.is_available()} | "
        f"prefetch_factor={PREFETCH_FACTOR if effective_train_workers > 0 or effective_val_workers > 0 else 'n/a'}"
    )

    # ── Class weight for imbalanced target ───────────────────────────────
    n_pos = labels_filtered[train_idx].sum()
    n_neg = len(train_idx) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
    print(f"[DATA] Positive class weight: {pos_weight.item():.4f}")

    return train_loader, val_loader, pos_weight, test_metadata

