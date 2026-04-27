import json
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import torch
import wfdb
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import (
    BATCH_SIZE,
    CLINICAL_FEATURES,
    DATASET_PATH,
    ECG_CACHE_DIR,
    ECG_LENGTH,
    ECG_LEADS,
    ECG_MEMMAP_DIR,
    NUM_WORKERS,
    PREFETCH_FACTOR,
    PROJECT_ROOT,
    SEED,
    TARGET_COLUMN,
    TEST_SPLIT,
    VAL_NUM_WORKERS,
    VAL_SPLIT,
)

ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH)

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv(
    "S3_BUCKET",
    "arn:aws:s3:us-east-1:724665945834:accesspoint/mimic-iv-ecg-v1-0-01",
)

MAX_DOWNLOAD_WORKERS = 64


def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION,
    )


def _get_local_record_path(ecg_path: str, cache_dir: Path) -> str:
    record_name = ecg_path.rsplit("/", 1)[-1]
    sub_dir = ecg_path.rsplit("/", 1)[0]
    local_dir = cache_dir / sub_dir
    return str(local_dir / record_name)


def _is_cached(local_record_path: str) -> bool:
    return os.path.exists(local_record_path + ".hea") and os.path.exists(local_record_path + ".dat")


def _download_single_ecg(ecg_path: str, cache_dir: Path) -> bool:
    local_record_path = _get_local_record_path(ecg_path, cache_dir)
    if _is_cached(local_record_path):
        return True

    os.makedirs(os.path.dirname(local_record_path), exist_ok=True)
    try:
        s3 = get_s3_client()
        s3.download_file(S3_BUCKET, ecg_path + ".hea", local_record_path + ".hea")
        s3.download_file(S3_BUCKET, ecg_path + ".dat", local_record_path + ".dat")
        return True
    except Exception:
        for ext in [".hea", ".dat"]:
            partial_path = local_record_path + ext
            if os.path.exists(partial_path):
                os.remove(partial_path)
        return False


def download_all_ecgs(ecg_paths: list[str], cache_dir: Path = ECG_CACHE_DIR, max_workers: int = MAX_DOWNLOAD_WORKERS):
    cache_dir.mkdir(parents=True, exist_ok=True)
    uncached = [p for p in ecg_paths if not _is_cached(_get_local_record_path(p, cache_dir))]
    n_cached = len(ecg_paths) - len(uncached)

    if not uncached:
        print(f"[ECG] All {len(ecg_paths):,} records already cached (OK)")
        return

    print(f"[ECG] Already cached: {n_cached:,} / {len(ecg_paths):,}")
    print(f"[ECG] Downloading {len(uncached):,} records with {max_workers} parallel threads ...")

    success = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_download_single_ecg, path, cache_dir): path for path in uncached}
        with tqdm(total=len(uncached), desc="Downloading ECGs") as pbar:
            for future in as_completed(futures):
                if future.result():
                    success += 1
                else:
                    failed += 1
                pbar.update(1)

    print(f"[ECG] Download complete -- success: {success:,} | failed: {failed:,}")


def load_ecg_signal(record_path: str, target_length: int = ECG_LENGTH) -> np.ndarray:
    record = wfdb.rdrecord(record_path)
    signal = record.p_signal
    if signal is None:
        raise ValueError(f"No signal in record: {record_path}")

    signal = signal.T
    n_leads = signal.shape[0]
    if n_leads < ECG_LEADS:
        pad = np.zeros((ECG_LEADS - n_leads, signal.shape[1]), dtype=np.float32)
        signal = np.vstack([signal, pad])
    elif n_leads > ECG_LEADS:
        signal = signal[:ECG_LEADS, :]

    n_samples = signal.shape[1]
    if n_samples >= target_length:
        signal = signal[:, :target_length]
    else:
        pad = np.zeros((ECG_LEADS, target_length - n_samples), dtype=np.float32)
        signal = np.hstack([signal, pad])

    return signal.astype(np.float32)


def normalize_ecg(ecg: np.ndarray) -> np.ndarray:
    ecg = np.nan_to_num(ecg, nan=0.0, posinf=0.0, neginf=0.0)
    lead_mean = ecg.mean(axis=1, keepdims=True)
    lead_std = ecg.std(axis=1, keepdims=True) + 1e-8
    return ((ecg - lead_mean) / lead_std).astype(np.float32)


def _get_memmap_shape(dataset_size: int) -> tuple[int, int, int]:
    return (dataset_size, ECG_LEADS, ECG_LENGTH)


def _memmap_data_path(dataset_size: int) -> Path:
    return ECG_MEMMAP_DIR / f"ecg_memmap_{dataset_size}.dat"


def _memmap_meta_path(dataset_size: int) -> Path:
    return ECG_MEMMAP_DIR / f"ecg_memmap_{dataset_size}.json"


def _load_memmap(ecg_memmap_path: str | Path, ecg_memmap_shape) -> np.memmap:
    return np.memmap(
        str(ecg_memmap_path),
        dtype=np.float32,
        mode="r",
        shape=tuple(ecg_memmap_shape),
    )


def _memmap_is_valid(memmap_path: Path, meta_path: Path, shape: tuple[int, int, int]) -> bool:
    if not memmap_path.exists() or not meta_path.exists():
        return False

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return False

    expected_size = int(np.prod(shape, dtype=np.int64)) * np.dtype(np.float32).itemsize
    return (
        tuple(meta.get("shape", ())) == tuple(shape)
        and meta.get("dtype") == "float32"
        and memmap_path.stat().st_size == expected_size
    )


def build_or_load_ecg_memmap(local_paths: list[str]):
    dataset_size = len(local_paths)
    shape = _get_memmap_shape(dataset_size)
    memmap_path = _memmap_data_path(dataset_size)
    meta_path = _memmap_meta_path(dataset_size)
    ECG_MEMMAP_DIR.mkdir(parents=True, exist_ok=True)

    if _memmap_is_valid(memmap_path, meta_path, shape):
        print(f"[ECG] Using memmap cache: {memmap_path}")
        return memmap_path, shape

    print(f"[ECG] Building memmap cache: {memmap_path}")
    memmap = np.memmap(str(memmap_path), dtype=np.float32, mode="w+", shape=shape)
    for idx, record_path in enumerate(tqdm(local_paths, desc="Building ECG memmap")):
        try:
            ecg = normalize_ecg(load_ecg_signal(record_path))
        except Exception:
            ecg = np.zeros((ECG_LEADS, ECG_LENGTH), dtype=np.float32)
        memmap[idx] = ecg
    memmap.flush()
    del memmap

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"shape": list(shape), "dtype": "float32"}, f, indent=2)

    return memmap_path, shape


def load_ecg_from_metadata(index: int, test_metadata: dict) -> np.ndarray:
    memmap_path = test_metadata.get("ecg_memmap_path")
    memmap_shape = test_metadata.get("ecg_memmap_shape")
    test_ecg_indices = test_metadata.get("test_ecg_indices")
    if memmap_path and memmap_shape is not None and test_ecg_indices is not None:
        ecg_memmap = _load_memmap(memmap_path, memmap_shape)
        return np.array(ecg_memmap[int(test_ecg_indices[index])], copy=True)

    local_paths = test_metadata.get("test_local_paths", [])
    try:
        return normalize_ecg(load_ecg_signal(local_paths[index]))
    except Exception:
        return np.zeros((ECG_LEADS, ECG_LENGTH), dtype=np.float32)


class MultimodalCardiacDataset(Dataset):
    def __init__(
        self,
        ecg_paths: list[str],
        clinical_data: np.ndarray,
        labels: np.ndarray,
        ecg_indices: np.ndarray | list | None = None,
        ecg_memmap_path: str | Path | None = None,
        ecg_memmap_shape=None,
    ):
        self.ecg_paths = ecg_paths
        self.clinical_data = torch.tensor(clinical_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.ecg_indices = None if ecg_indices is None else np.asarray(ecg_indices, dtype=np.int64)
        self.ecg_memmap_path = None if ecg_memmap_path is None else str(ecg_memmap_path)
        self.ecg_memmap_shape = None if ecg_memmap_shape is None else tuple(ecg_memmap_shape)
        self._ecg_memmap = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.ecg_memmap_path and self.ecg_indices is not None:
            if self._ecg_memmap is None:
                self._ecg_memmap = _load_memmap(self.ecg_memmap_path, self.ecg_memmap_shape)
            ecg = np.array(self._ecg_memmap[int(self.ecg_indices[idx])], copy=True)
        else:
            try:
                ecg = normalize_ecg(load_ecg_signal(self.ecg_paths[idx]))
            except Exception:
                ecg = np.zeros((ECG_LEADS, ECG_LENGTH), dtype=np.float32)

        ecg_tensor = torch.tensor(ecg, dtype=torch.float32)
        return ecg_tensor, self.clinical_data[idx], self.labels[idx]


def _make_dataloader(dataset: Dataset, shuffle: bool, num_workers: int) -> DataLoader:
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


def load_and_prepare_data(
    subset: int = None,
    train_num_workers: int | None = None,
    val_num_workers: int | None = None,
):
    print(f"[DATA] Loading: {DATASET_PATH}")
    df = pd.read_parquet(DATASET_PATH)
    df["_memmap_index"] = np.arange(len(df), dtype=np.int64)
    print(f"[DATA] Full dataset: {df.shape[0]:,} rows x {df.shape[1]} cols")

    full_ecg_paths = df["ecg_path"].tolist()
    full_local_paths = [_get_local_record_path(p, ECG_CACHE_DIR) for p in full_ecg_paths]
    full_memmap_shape = _get_memmap_shape(len(full_local_paths))
    full_memmap_path = _memmap_data_path(len(full_local_paths))
    full_memmap_meta_path = _memmap_meta_path(len(full_local_paths))
    full_memmap_exists = _memmap_is_valid(full_memmap_path, full_memmap_meta_path, full_memmap_shape)

    if subset is not None:
        df = df.head(subset).copy()
        print(f"[DATA] Using subset: {len(df):,} samples")

    clinical_df = df[CLINICAL_FEATURES].copy()
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
    requested_shape = _get_memmap_shape(len(local_paths))
    memmap_path = _memmap_data_path(len(local_paths))
    memmap_meta_path = _memmap_meta_path(len(local_paths))

    if not full_memmap_exists and not _memmap_is_valid(memmap_path, memmap_meta_path, requested_shape):
        download_all_ecgs(ecg_paths)

    if full_memmap_exists:
        valid_mask = np.ones(len(local_paths), dtype=bool)
    elif _memmap_is_valid(memmap_path, memmap_meta_path, requested_shape):
        valid_mask = np.ones(len(local_paths), dtype=bool)
    else:
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

    if full_memmap_exists:
        ecg_memmap_path, ecg_memmap_shape = full_memmap_path, full_memmap_shape
        print(f"[ECG] Using full-dataset memmap cache: {ecg_memmap_path}")
    else:
        ecg_memmap_path, ecg_memmap_shape = build_or_load_ecg_memmap(local_paths)

    indices = np.arange(n_samples)
    train_pool_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SPLIT,
        random_state=SEED,
        stratify=labels,
    )
    print(f"[DATA] Stage 1 split: train_pool={len(train_pool_idx):,}  |  test={len(test_idx):,}")

    train_pool_clinical = clinical_raw[train_pool_idx]
    col_mean = train_pool_clinical.mean(axis=0, keepdims=True)
    col_std = train_pool_clinical.std(axis=0, keepdims=True) + 1e-8
    clinical_standardized = (clinical_raw - col_mean) / col_std

    test_metadata = {
        "test_df": df.iloc[test_idx].reset_index(drop=True),
        "test_local_paths": [local_paths[i] for i in test_idx],
        "test_clinical": clinical_standardized[test_idx],
        "test_labels": labels[test_idx],
        "ecg_memmap_path": str(ecg_memmap_path),
        "ecg_memmap_shape": list(ecg_memmap_shape),
        "test_ecg_indices": df["_memmap_index"].iloc[test_idx].astype(int).tolist(),
        "col_mean": col_mean,
        "col_std": col_std,
    }

    pool_labels = labels[train_pool_idx]
    pool_indices = np.arange(len(train_pool_idx))
    train_sub_idx, val_sub_idx = train_test_split(
        pool_indices,
        test_size=VAL_SPLIT,
        random_state=SEED,
        stratify=pool_labels,
    )

    train_idx = train_pool_idx[train_sub_idx]
    val_idx = train_pool_idx[val_sub_idx]
    train_paths = [local_paths[i] for i in train_idx]
    val_paths = [local_paths[i] for i in val_idx]

    train_ds = MultimodalCardiacDataset(
        train_paths,
        clinical_standardized[train_idx],
        labels[train_idx],
        ecg_indices=df["_memmap_index"].iloc[train_idx].to_numpy(dtype=np.int64),
        ecg_memmap_path=ecg_memmap_path,
        ecg_memmap_shape=ecg_memmap_shape,
    )
    val_ds = MultimodalCardiacDataset(
        val_paths,
        clinical_standardized[val_idx],
        labels[val_idx],
        ecg_indices=df["_memmap_index"].iloc[val_idx].to_numpy(dtype=np.int64),
        ecg_memmap_path=ecg_memmap_path,
        ecg_memmap_shape=ecg_memmap_shape,
    )

    effective_train_workers = NUM_WORKERS if train_num_workers is None else train_num_workers
    effective_val_workers = VAL_NUM_WORKERS if val_num_workers is None else val_num_workers
    train_loader = _make_dataloader(train_ds, shuffle=True, num_workers=effective_train_workers)
    val_loader = _make_dataloader(val_ds, shuffle=False, num_workers=effective_val_workers)

    print(
        f"[DATA] Stage 2 split: Train={len(train_ds):,}  |  Val={len(val_ds):,}  |  "
        f"Test={len(test_metadata['test_labels']):,}"
    )
    print(
        f"[DATA] DataLoader config: train_workers={effective_train_workers} | "
        f"val_workers={effective_val_workers} | "
        f"pin_memory={torch.cuda.is_available()} | "
        f"prefetch_factor={PREFETCH_FACTOR if effective_train_workers > 0 or effective_val_workers > 0 else 'n/a'}"
    )

    n_pos = labels[train_idx].sum()
    n_neg = len(train_idx) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
    print(f"[DATA] Positive class weight: {pos_weight.item():.4f}")

    return train_loader, val_loader, pos_weight, test_metadata
