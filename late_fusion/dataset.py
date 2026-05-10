# ==========================================
# DATASET - Multimodal Cardiac Dataset
# ==========================================
# - Downloads ECG waveforms from S3 using parallel downloads
# - Builds a reusable ECG memmap cache
# - Uses grouped patient splits recovered from ecg_path
# - Optionally uses weighted sampling on the training split
# ==========================================

import hashlib
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import torch
import wfdb
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from Dataset.feature_schema import MODEL_INPUT_VALID_RANGES

from .config import (
    BATCH_SIZE,
    CLINICAL_FEATURES,
    DATASET_PATH,
    ECG_LENGTH,
    ECG_LEADS,
    MEMMAP_DIR,
    NUM_WORKERS,
    PREFETCH_FACTOR,
    PREPROCESSED_ECG_DIR,
    PROJECT_ROOT,
    SEED,
    TARGET_COLUMN,
    TEST_SPLIT,
    USE_PREPROCESSED_ECG,
    VAL_NUM_WORKERS,
    VAL_SPLIT,
)

# Load .env from project root
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH)

# AWS S3 configuration
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv(
    "S3_BUCKET",
    "arn:aws:s3:us-east-1:724665945834:accesspoint/mimic-iv-ecg-v1-0-01",
)

# Local cache directory for downloaded ECG signals
ECG_CACHE_DIR = PROJECT_ROOT / "mimic_data" / "ecg_cache"

# Download settings
MAX_DOWNLOAD_WORKERS = 64


def get_s3_client():
    """Create a boto3 S3 client. Each thread needs its own client."""
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION,
    )


def _get_local_record_path(ecg_path: str, cache_dir: Path) -> str:
    """Convert an S3 ecg_path to its local cache path without extension."""
    record_name = ecg_path.rsplit("/", 1)[-1]
    sub_dir = ecg_path.rsplit("/", 1)[0]
    local_dir = cache_dir / sub_dir
    return str(local_dir / record_name)


def _is_cached(local_record_path: str) -> bool:
    """Check whether both WFDB files exist locally."""
    return os.path.exists(local_record_path + ".hea") and os.path.exists(local_record_path + ".dat")


def _get_preprocessed_ecg_path(ecg_path: str, preprocessed_dir: Path = PREPROCESSED_ECG_DIR) -> Path:
    return preprocessed_dir / f"{ecg_path}.npy"


def _is_preprocessed_cached(ecg_path: str, preprocessed_dir: Path = PREPROCESSED_ECG_DIR) -> bool:
    return _get_preprocessed_ecg_path(ecg_path, preprocessed_dir).exists()


def _download_single_ecg(ecg_path: str, cache_dir: Path) -> bool:
    """
    Download a single ECG record (.hea + .dat) from S3.

    Returns True on success and False on failure.
    """
    local_record_path = _get_local_record_path(ecg_path, cache_dir)

    if _is_cached(local_record_path):
        return True

    local_dir = os.path.dirname(local_record_path)
    os.makedirs(local_dir, exist_ok=True)

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


def download_all_ecgs(
    ecg_paths: list[str],
    cache_dir: Path = ECG_CACHE_DIR,
    max_workers: int = MAX_DOWNLOAD_WORKERS,
):
    """
    Download all ECG records from S3 using multithreaded parallelism.

    Already-cached records are skipped automatically.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    uncached = [path for path in ecg_paths if not _is_cached(_get_local_record_path(path, cache_dir))]
    n_cached = len(ecg_paths) - len(uncached)

    if not uncached:
        print(f"[ECG] All {len(ecg_paths):,} records already cached")
        return

    print(f"[ECG] Already cached: {n_cached:,} / {len(ecg_paths):,}")
    print(f"[ECG] Downloading {len(uncached):,} records with {max_workers} threads")

    success = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_download_single_ecg, path, cache_dir): path for path in uncached}
        with tqdm(total=len(uncached), desc="Downloading ECGs") as progress:
            for future in as_completed(futures):
                if future.result():
                    success += 1
                else:
                    failed += 1
                progress.update(1)

    print(f"[ECG] Download complete: success={success:,} failed={failed:,}")


def load_ecg_signal(record_path: str, target_length: int = ECG_LENGTH) -> np.ndarray:
    """
    Load a single 12-lead ECG signal from a local WFDB record or preprocessed .npy.

    Returns an array of shape (12, target_length).
    """
    record_path_obj = Path(record_path)
    if record_path_obj.suffix.lower() == ".npy":
        signal = np.load(record_path_obj).astype(np.float32, copy=False)
        if signal.ndim != 2:
            raise ValueError(f"Expected 2D ECG array in {record_path_obj}, got shape={signal.shape}")
        if signal.shape[0] != ECG_LEADS:
            raise ValueError(f"Expected {ECG_LEADS} leads in {record_path_obj}, got {signal.shape[0]}")

        n_samples = signal.shape[1]
        if n_samples >= target_length:
            signal = signal[:, :target_length]
        else:
            pad = np.zeros((ECG_LEADS, target_length - n_samples), dtype=np.float32)
            signal = np.hstack([signal, pad])
        return signal.astype(np.float32, copy=False)

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


def _normalize_ecg(ecg: np.ndarray) -> np.ndarray:
    ecg = np.asarray(ecg, dtype=np.float32)
    ecg = np.nan_to_num(ecg, nan=0.0, posinf=0.0, neginf=0.0)

    # Protect against extreme spikes before z-scoring.
    clip_scale = np.percentile(np.abs(ecg), 99.5, axis=1, keepdims=True)
    clip_scale = np.where(clip_scale > 1e-6, clip_scale, 1.0)
    ecg = np.clip(ecg, -5.0 * clip_scale, 5.0 * clip_scale)

    lead_mean = ecg.mean(axis=1, keepdims=True)
    lead_std = ecg.std(axis=1, keepdims=True)
    safe_std = np.where(lead_std > 1e-6, lead_std, 1.0)
    normalized = (ecg - lead_mean) / safe_std

    flat_mask = (lead_std <= 1e-6).astype(np.float32)
    if flat_mask.any():
        normalized = normalized * (1.0 - flat_mask[:, None])

    return normalized.astype(np.float32, copy=False)


def _extract_subject_group_id(ecg_path: str) -> str:
    """
    Recover a patient-level group id from the ECG path.

    Typical MIMIC ECG paths contain shard folders plus a patient folder like
    `p10000032`. The longest `p<digits>` token is used as the subject id.
    """
    matches = re.findall(r"(p\d+)", ecg_path)
    if matches:
        return max(matches, key=len)

    path_parts = [part for part in Path(ecg_path).parts if part not in ("/", "\\")]
    if len(path_parts) >= 2:
        return path_parts[-2]
    return ecg_path


def _group_labels_from_rows(labels: np.ndarray, group_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique_groups, inverse = np.unique(group_ids, return_inverse=True)
    group_labels = np.zeros(len(unique_groups), dtype=np.int64)
    np.maximum.at(group_labels, inverse, labels.astype(np.int64))
    return unique_groups, group_labels


def infer_group_ids(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Recover per-row subject groups from the dataframe ECG paths.

    Returns the row indices together with a subject-group id per row so
    analysis code can reproduce the same leakage-safe grouped splits used by
    late-fusion training.
    """
    indices = np.arange(len(df))
    groups = np.array([_extract_subject_group_id(path) for path in df["ecg_path"]], dtype=object)
    return indices, groups


def _split_groups(
    group_ids: np.ndarray,
    labels: np.ndarray,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unique_groups, group_labels = _group_labels_from_rows(labels, group_ids)
    class_counts = np.bincount(group_labels, minlength=2)
    stratify_labels = group_labels if np.count_nonzero(class_counts) == 2 and class_counts.min() >= 2 else None
    if stratify_labels is None:
        print("[DATA] Group stratification unavailable for this split; falling back to unstratified grouped split")
    train_groups, held_out_groups = train_test_split(
        unique_groups,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels,
    )
    train_idx = np.flatnonzero(np.isin(group_ids, train_groups))
    held_out_idx = np.flatnonzero(np.isin(group_ids, held_out_groups))
    return train_idx, held_out_idx, train_groups, held_out_groups


def stratified_group_train_test_split(
    indices: np.ndarray,
    labels: np.ndarray,
    group_ids: np.ndarray,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a provided subset of rows with patient-level grouping.

    The returned indices stay in the same index space as the provided
    ``indices`` array so callers can chain splits deterministically.
    """
    subset_indices = np.asarray(indices)
    subset_labels = np.asarray(labels)[subset_indices]
    subset_groups = np.asarray(group_ids, dtype=object)[subset_indices]
    train_rel, test_rel, _, _ = _split_groups(
        subset_groups,
        subset_labels,
        test_size=test_size,
        random_state=seed,
    )
    return subset_indices[train_rel], subset_indices[test_rel]


def _log_split_overlap(stage_name: str, left_name: str, left_groups: np.ndarray, right_name: str, right_groups: np.ndarray):
    overlap = len(set(left_groups).intersection(set(right_groups)))
    print(f"[DATA] {stage_name} group overlap ({left_name}/{right_name}): {overlap}")


def _log_group_prevalence(split_name: str, labels: np.ndarray, group_ids: np.ndarray):
    _, group_labels = _group_labels_from_rows(labels, group_ids)
    prevalence = float(group_labels.mean()) if len(group_labels) else 0.0
    positives = int(group_labels.sum())
    print(
        f"[DATA] {split_name} grouped prevalence: {prevalence:.2%} "
        f"({positives:,} / {len(group_labels):,} subject groups)"
    )


def _memmap_cache_path(source_paths: list[str], source_kind: str) -> Path:
    digest = hashlib.sha1(
        (f"{source_kind}|{ECG_LENGTH}|\n" + "\n".join(source_paths)).encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()[:16]
    return MEMMAP_DIR / f"late_fusion_ecg_{source_kind}_{digest}.npy"


def _legacy_wfdb_memmap_cache_path(source_paths: list[str]) -> Path:
    digest = hashlib.sha1(
        ("\n".join(source_paths)).encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()[:16]
    return MEMMAP_DIR / f"late_fusion_ecg_{digest}.npy"


def build_ecg_memmap(source_paths: list[str], source_kind: str = "wfdb") -> Path:
    """
    Convert ECG sources into a reusable read-only memmap.
    """
    MEMMAP_DIR.mkdir(parents=True, exist_ok=True)
    memmap_path = _memmap_cache_path(source_paths, source_kind)

    if memmap_path.exists():
        print(f"[ECG] Reusing memmap cache: {memmap_path}")
        return memmap_path

    if source_kind == "wfdb":
        legacy_memmap_path = _legacy_wfdb_memmap_cache_path(source_paths)
        if legacy_memmap_path.exists():
            print(f"[ECG] Reusing legacy memmap cache: {legacy_memmap_path}")
            return legacy_memmap_path

    print(f"[ECG] Building memmap cache: {memmap_path}")
    memmap = np.lib.format.open_memmap(
        memmap_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(source_paths), ECG_LEADS, ECG_LENGTH),
    )

    try:
        for idx, source_path in enumerate(tqdm(source_paths, desc="Caching ECG memmap")):
            try:
                ecg = load_ecg_signal(source_path)
            except Exception:
                ecg = np.zeros((ECG_LEADS, ECG_LENGTH), dtype=np.float32)
            memmap[idx] = _normalize_ecg(ecg)
        memmap.flush()
    except Exception:
        del memmap
        if memmap_path.exists():
            memmap_path.unlink()
        raise

    del memmap
    return memmap_path


class MultimodalCardiacDataset(Dataset):
    """
    Memory-efficient dataset with lazy ECG loading from memmap or WFDB files.
    """

    def __init__(
        self,
        clinical_data: np.ndarray,
        labels: np.ndarray,
        ecg_paths: list[str] | None = None,
        ecg_memmap_path: str | None = None,
        ecg_indices: np.ndarray | None = None,
    ):
        self.ecg_paths = ecg_paths
        self.ecg_memmap_path = ecg_memmap_path
        self.ecg_indices = np.arange(len(labels)) if ecg_indices is None else np.asarray(ecg_indices)
        self._ecg_memmap = None
        self.clinical_data = torch.tensor(clinical_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def _get_memmap(self):
        if self._ecg_memmap is None and self.ecg_memmap_path is not None:
            self._ecg_memmap = np.load(self.ecg_memmap_path, mmap_mode="r")
        return self._ecg_memmap

    def __getitem__(self, idx):
        memmap = self._get_memmap()
        if memmap is not None:
            ecg = np.array(memmap[self.ecg_indices[idx]], dtype=np.float32, copy=True)
        else:
            try:
                ecg = _normalize_ecg(load_ecg_signal(self.ecg_paths[idx]))
            except Exception:
                ecg = np.zeros((ECG_LEADS, ECG_LENGTH), dtype=np.float32)

        ecg_tensor = torch.tensor(ecg, dtype=torch.float32)
        return ecg_tensor, self.clinical_data[idx], self.labels[idx]


def _make_dataloader(
    dataset: Dataset,
    shuffle: bool,
    num_workers: int,
    batch_size: int,
    sampler: WeightedRandomSampler | None = None,
) -> DataLoader:
    """
    Build a DataLoader with settings that are stable for the current OS/device.
    """
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if sampler is not None:
        loader_kwargs["sampler"] = sampler
    else:
        loader_kwargs["shuffle"] = shuffle

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR

    return DataLoader(dataset, **loader_kwargs)


def load_and_prepare_data(
    subset: int | None = None,
    train_num_workers: int | None = None,
    val_num_workers: int | None = None,
    weighted_sampling: bool = False,
    batch_size: int = BATCH_SIZE,
):
    """
    Load preprocessed data and return train/val loaders plus test metadata.

    Splitting is leakage-safe at the patient level:
    Stage 1: full data -> train_pool + test
    Stage 2: train_pool -> train + val
    """
    print(f"[DATA] Loading: {DATASET_PATH}")
    df = pd.read_parquet(DATASET_PATH)
    print(f"[DATA] Full dataset: {df.shape[0]:,} rows x {df.shape[1]} cols")

    if subset is not None:
        df = df.head(subset).copy()
        print(f"[DATA] Using subset: {len(df):,} samples")

    missing_feature_columns = [column for column in CLINICAL_FEATURES if column not in df.columns]
    if missing_feature_columns:
        raise ValueError(
            "The late-fusion dataset is missing engineered feature columns: "
            + ", ".join(missing_feature_columns)
        )

    clinical_df = df[CLINICAL_FEATURES].copy()
    labels = df[TARGET_COLUMN].values.astype(np.float32)
    ecg_paths = df["ecg_path"].tolist()
    group_ids = np.array([_extract_subject_group_id(path) for path in ecg_paths], dtype=object)

    clinical_df = clinical_df.replace([np.inf, -np.inf], np.nan)

    for column, (lo, hi) in MODEL_INPUT_VALID_RANGES.items():
        if column in clinical_df.columns:
            invalid_mask = (clinical_df[column] < lo) | (clinical_df[column] > hi)
            clinical_df.loc[invalid_mask, column] = np.nan

    clinical_df = clinical_df.fillna(clinical_df.median())
    clinical_raw = clinical_df.values.astype(np.float32)
    clinical_raw = np.nan_to_num(clinical_raw, nan=0.0, posinf=0.0, neginf=0.0)

    n_samples = len(labels)
    print(f"[DATA] Clinical features: {clinical_raw.shape[1]} (cleaned)")
    print(f"[DATA] AMI prevalence: {labels.mean():.4%} ({int(labels.sum()):,} / {n_samples:,})")

    if USE_PREPROCESSED_ECG:
        ecg_source_paths = [str(_get_preprocessed_ecg_path(path)) for path in ecg_paths]
        valid_mask = np.array([_is_preprocessed_cached(path) for path in ecg_paths], dtype=bool)
        source_kind = "preprocessed"
        print(f"[ECG] Using preprocessed ECG directory: {PREPROCESSED_ECG_DIR}")
    else:
        download_all_ecgs(ecg_paths)
        ecg_source_paths = [_get_local_record_path(path, ECG_CACHE_DIR) for path in ecg_paths]
        valid_mask = np.array([_is_cached(path) for path in ecg_source_paths], dtype=bool)
        source_kind = "wfdb"

    if not valid_mask.all():
        n_before = n_samples
        valid_indices = np.flatnonzero(valid_mask)
        ecg_source_paths = [ecg_source_paths[idx] for idx in valid_indices]
        clinical_raw = clinical_raw[valid_mask]
        labels = labels[valid_mask]
        group_ids = group_ids[valid_mask]
        df = df.iloc[valid_indices].reset_index(drop=True)
        n_samples = len(labels)
        print(f"[DATA] Filtered after ECG source check: {n_before:,} -> {n_samples:,}")

    train_pool_idx, test_idx, train_pool_groups, test_groups = _split_groups(
        group_ids,
        labels,
        test_size=TEST_SPLIT,
        random_state=SEED,
    )
    print(f"[DATA] Total subject groups: {len(np.unique(group_ids)):,}")
    print(f"[DATA] Stage 1 split: train_pool={len(train_pool_idx):,} | test={len(test_idx):,}")
    _log_split_overlap("Stage 1", "train_pool", train_pool_groups, "test", test_groups)

    train_pool_clinical = clinical_raw[train_pool_idx]
    col_mean = train_pool_clinical.mean(axis=0, keepdims=True)
    col_std = train_pool_clinical.std(axis=0, keepdims=True) + 1e-8
    clinical_standardized = (clinical_raw - col_mean) / col_std

    test_metadata = {
        "test_df": df.iloc[test_idx].reset_index(drop=True),
        "test_local_paths": [ecg_source_paths[idx] for idx in test_idx],
        "test_clinical": clinical_standardized[test_idx],
        "test_labels": labels[test_idx],
        "col_mean": col_mean,
        "col_std": col_std,
    }

    train_idx_rel, val_idx_rel, train_groups, val_groups = _split_groups(
        group_ids[train_pool_idx],
        labels[train_pool_idx],
        test_size=VAL_SPLIT,
        random_state=SEED,
    )
    train_idx = train_pool_idx[train_idx_rel]
    val_idx = train_pool_idx[val_idx_rel]
    _log_split_overlap("Stage 2", "train", train_groups, "val", val_groups)
    _log_group_prevalence("Train", labels[train_idx], group_ids[train_idx])
    _log_group_prevalence("Validation", labels[val_idx], group_ids[val_idx])
    _log_group_prevalence("Test", labels[test_idx], group_ids[test_idx])

    ecg_memmap_path = build_ecg_memmap(ecg_source_paths, source_kind=source_kind)

    train_ds = MultimodalCardiacDataset(
        clinical_standardized[train_idx],
        labels[train_idx],
        ecg_memmap_path=str(ecg_memmap_path),
        ecg_indices=train_idx,
    )
    val_ds = MultimodalCardiacDataset(
        clinical_standardized[val_idx],
        labels[val_idx],
        ecg_memmap_path=str(ecg_memmap_path),
        ecg_indices=val_idx,
    )

    effective_train_workers = NUM_WORKERS if train_num_workers is None else train_num_workers
    effective_val_workers = VAL_NUM_WORKERS if val_num_workers is None else val_num_workers

    train_sampler = None
    if weighted_sampling:
        train_labels = labels[train_idx].astype(np.int64)
        class_counts = np.bincount(train_labels, minlength=2)
        class_weights = np.zeros(2, dtype=np.float64)
        nonzero_mask = class_counts > 0
        class_weights[nonzero_mask] = len(train_labels) / class_counts[nonzero_mask]
        sample_weights = class_weights[train_labels]
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(train_labels),
            replacement=True,
        )
        print(
            f"[DATA] Weighted sampling enabled | "
            f"class_counts={class_counts.tolist()} | "
            f"class_weights={[round(value, 4) for value in class_weights.tolist()]}"
        )
    else:
        print("[DATA] Weighted sampling disabled")

    train_loader = _make_dataloader(
        train_ds,
        shuffle=train_sampler is None,
        num_workers=effective_train_workers,
        batch_size=batch_size,
        sampler=train_sampler,
    )
    val_loader = _make_dataloader(
        val_ds,
        shuffle=False,
        num_workers=effective_val_workers,
        batch_size=batch_size,
    )

    print(
        f"[DATA] Stage 2 split: Train={len(train_ds):,} | "
        f"Val={len(val_ds):,} | Test={len(test_metadata['test_labels']):,}"
    )
    print(
        f"[DATA] DataLoader config: train_workers={effective_train_workers} | "
        f"val_workers={effective_val_workers} | "
        f"batch_size={batch_size} | "
        f"pin_memory={torch.cuda.is_available()} | "
        f"memmap={ecg_memmap_path.name} | "
        f"source_kind={source_kind} | "
        f"prefetch_factor={PREFETCH_FACTOR if effective_train_workers > 0 or effective_val_workers > 0 else 'n/a'}"
    )

    n_pos = labels[train_idx].sum()
    n_neg = len(train_idx) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
    print(f"[DATA] Positive class weight: {pos_weight.item():.4f}")

    return train_loader, val_loader, pos_weight, test_metadata
