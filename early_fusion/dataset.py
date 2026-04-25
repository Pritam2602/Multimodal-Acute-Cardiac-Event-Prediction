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
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import (
    DATASET_PATH, CLINICAL_FEATURES, TARGET_COLUMN,
    ECG_LEADS, ECG_LENGTH, BATCH_SIZE, VAL_SPLIT, TEST_SPLIT, SEED,
    PROJECT_ROOT, NUM_WORKERS, VAL_NUM_WORKERS, PREFETCH_FACTOR,
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

    Parameters
    ----------
    ecg : np.ndarray of shape (12, target_length) — already z-scored

    Returns
    -------
    ecg : np.ndarray of same shape, augmented
    """
    # 1. Random amplitude scaling
    scale = np.random.uniform(0.9, 1.1)
    ecg = ecg * scale

    # 2. Additive Gaussian noise
    noise = np.random.normal(0, 0.02, ecg.shape).astype(np.float32)
    ecg = ecg + noise

    # 3. Random temporal shift (wrap-around)
    shift = np.random.randint(-100, 101)
    if shift != 0:
        ecg = np.roll(ecg, shift, axis=1)

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

    Parameters
    ----------
    memmap_path   : Path — file path to the ECG memmap
    memmap_shape  : tuple — (n_total_records, 12, 5000)
    ecg_indices   : np.ndarray — global indices into the memmap for this split
    clinical_data : np.ndarray of shape (N, num_features)
    labels        : np.ndarray of shape (N,)
    augment       : bool — whether to apply ECG augmentation (training only)
    """

    def __init__(self, memmap_path, memmap_shape: tuple,
                 ecg_indices: np.ndarray, clinical_data: np.ndarray,
                 labels: np.ndarray, augment: bool = False):
        self.memmap_path   = str(memmap_path)
        self.memmap_shape  = memmap_shape
        self.ecg_indices   = np.asarray(ecg_indices)
        self.clinical_data = torch.tensor(clinical_data, dtype=torch.float32)
        self.labels        = torch.tensor(labels, dtype=torch.float32)
        self.augment       = augment
        self._ecg_memmap   = None  # opened lazily per worker

    def _get_memmap(self):
        """Open the memmap lazily — each worker gets its own file handle."""
        if self._ecg_memmap is None:
            self._ecg_memmap = np.memmap(
                self.memmap_path, dtype=np.float32, mode='r',
                shape=self.memmap_shape,
            )
        return self._ecg_memmap

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Fast memmap read — microseconds instead of ~10ms per wfdb call
        mmap = self._get_memmap()
        ecg = np.array(mmap[self.ecg_indices[idx]])  # copy to writable array

        # Apply augmentation during training
        if self.augment:
            ecg = _augment_ecg(ecg)

        ecg_tensor = torch.tensor(ecg, dtype=torch.float32)
        return ecg_tensor, self.clinical_data[idx], self.labels[idx]


def _make_dataloader(dataset: Dataset, shuffle: bool, num_workers: int) -> DataLoader:
    """
    Build a DataLoader with settings that are stable for the current OS/device.

    On Windows, multiprocessing workers can fail with
    "Couldn't open shared file mapping" for large batches. Using
    num_workers=0 avoids shared-memory file mappings entirely.
    """
    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR

    return DataLoader(dataset, **loader_kwargs)


# ==========================================
# MAIN DATA LOADING FUNCTION
# ==========================================

def load_and_prepare_data(
    subset: int = None,
    train_num_workers: int | None = None,
    val_num_workers: int | None = None,
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

    # ── Extract features, labels, paths ──────────────────────────────────
    clinical_df = df[CLINICAL_FEATURES].copy()
    labels      = df[TARGET_COLUMN].values.astype(np.float32)
    ecg_paths   = df["ecg_path"].tolist()

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
            df           = df.iloc[valid_indices].reset_index(drop=True)
            n_samples    = len(labels)
            print(f"[DATA] Filtered: {n_before:,} → {n_samples:,} (removed {n_before - n_samples:,} failed)")

        # Build memory-mapped ECG cache (one-time)
        ecg_memmap_path, ecg_memmap_shape = _build_ecg_memmap(local_paths)

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 1: Split off held-out TEST SET (15%)
    # ══════════════════════════════════════════════════════════════════════
    indices = np.arange(n_samples)
    train_pool_idx, test_idx = train_test_split(
        indices, test_size=TEST_SPLIT, random_state=SEED, stratify=labels
    )

    print(f"[DATA] Stage 1 split: train_pool={len(train_pool_idx):,}  |  test={len(test_idx):,}")

    # ── Standardize using ONLY train_pool stats (prevent data leakage) ───
    train_pool_clinical = clinical_raw[train_pool_idx]
    col_mean = train_pool_clinical.mean(axis=0, keepdims=True)
    col_std  = train_pool_clinical.std(axis=0, keepdims=True) + 1e-8

    # Apply same standardization to all splits
    clinical_standardized = (clinical_raw - col_mean) / col_std

    # ── Build test metadata (for store_predictions.py) ───────────────────
    test_metadata = {
        "test_df":          df.iloc[test_idx].reset_index(drop=True),
        "test_local_paths": [local_paths[i] for i in test_idx],
        "test_clinical":    clinical_standardized[test_idx],
        "test_labels":      labels[test_idx],
        "col_mean":         col_mean,
        "col_std":          col_std,
        # Memmap info so store_predictions.py can read ECGs without
        # needing the individual .hea/.dat wfdb files on disk.
        "ecg_memmap_path":  ecg_memmap_path,
        "ecg_memmap_shape": ecg_memmap_shape,
        "test_ecg_indices": test_idx,
    }

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 2: Split train_pool → train (80%) + val (20%)
    # ══════════════════════════════════════════════════════════════════════
    pool_labels = labels[train_pool_idx]
    pool_indices = np.arange(len(train_pool_idx))
    train_sub_idx, val_sub_idx = train_test_split(
        pool_indices, test_size=VAL_SPLIT, random_state=SEED, stratify=pool_labels
    )

    # Map back to global indices
    train_idx = train_pool_idx[train_sub_idx]
    val_idx   = train_pool_idx[val_sub_idx]

    train_ds = MultimodalCardiacDataset(
        ecg_memmap_path, ecg_memmap_shape,
        train_idx, clinical_standardized[train_idx], labels[train_idx],
        augment=True,   # ECG augmentation for training only
    )
    val_ds = MultimodalCardiacDataset(
        ecg_memmap_path, ecg_memmap_shape,
        val_idx, clinical_standardized[val_idx], labels[val_idx],
        augment=False,  # no augmentation for validation
    )

    effective_train_workers = NUM_WORKERS if train_num_workers is None else train_num_workers
    effective_val_workers = VAL_NUM_WORKERS if val_num_workers is None else val_num_workers

    train_loader = _make_dataloader(train_ds, shuffle=True, num_workers=effective_train_workers)
    val_loader = _make_dataloader(val_ds, shuffle=False, num_workers=effective_val_workers)

    print(f"[DATA] Stage 2 split: Train={len(train_ds):,}  |  Val={len(val_ds):,}  |  Test={len(test_metadata['test_labels']):,}")
    print(
        f"[DATA] DataLoader config: train_workers={effective_train_workers} | "
        f"val_workers={effective_val_workers} | "
        f"pin_memory={torch.cuda.is_available()} | "
        f"prefetch_factor={PREFETCH_FACTOR if effective_train_workers > 0 or effective_val_workers > 0 else 'n/a'}"
    )

    # ── Class weight for imbalanced target ───────────────────────────────
    n_pos = labels[train_idx].sum()
    n_neg = len(train_idx) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
    print(f"[DATA] Positive class weight: {pos_weight.item():.4f}")

    return train_loader, val_loader, pos_weight, test_metadata
