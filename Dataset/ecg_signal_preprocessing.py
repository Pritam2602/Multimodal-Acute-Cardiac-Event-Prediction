"""
Standalone ECG signal preprocessing for late-fusion experiments.

This script:
1. Reads the fusion parquet to discover ECG records in use.
2. Loads cached WFDB records from mimic_data/ecg_cache.
3. Applies bandpass filtering (default 0.5-40 Hz).
4. Downsamples each 12-lead ECG to a fixed target length (default 2500).
5. Saves one .npy file per record plus a manifest CSV.

It does not build or rely on the late_fusion memmap cache.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt, iirnotch, resample, sosfiltfilt
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = PROJECT_ROOT / "mimic_data" / "final_preprocessed_fusion_dataset.parquet"
DEFAULT_ECG_CACHE_DIR = PROJECT_ROOT / "mimic_data" / "ecg_cache"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "mimic_data" / "ecg_preprocessed_2500_bp"
DEFAULT_MANIFEST_PATH = DEFAULT_OUTPUT_DIR / "manifest.csv"

ECG_LEADS = 12


def _get_local_record_path(ecg_path: str, cache_dir: Path) -> Path:
    record_name = ecg_path.rsplit("/", 1)[-1]
    sub_dir = ecg_path.rsplit("/", 1)[0]
    return cache_dir / sub_dir / record_name


def _load_wfdb_signal(record_path: Path) -> tuple[np.ndarray, float]:
    record = wfdb.rdrecord(str(record_path))
    signal = record.p_signal
    if signal is None:
        raise ValueError(f"No ECG signal found in record: {record_path}")

    fs = float(getattr(record, "fs", 500.0) or 500.0)
    signal = signal.T.astype(np.float32, copy=False)

    n_leads = signal.shape[0]
    if n_leads < ECG_LEADS:
        pad = np.zeros((ECG_LEADS - n_leads, signal.shape[1]), dtype=np.float32)
        signal = np.vstack([signal, pad])
    elif n_leads > ECG_LEADS:
        signal = signal[:ECG_LEADS, :]

    return signal, fs


def _bandpass_filter(signal: np.ndarray, fs: float, lowcut: float, highcut: float, order: int) -> np.ndarray:
    nyquist = fs * 0.5
    safe_highcut = min(highcut, nyquist - 1e-3)
    if not (0 < lowcut < safe_highcut):
        raise ValueError(
            f"Invalid bandpass bounds for fs={fs}: lowcut={lowcut}, highcut={highcut}, nyquist={nyquist}"
        )

    sos = butter(order, [lowcut, safe_highcut], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, signal, axis=1).astype(np.float32, copy=False)


def _notch_filter(signal: np.ndarray, fs: float, notch_hz: float, quality_factor: float) -> np.ndarray:
    if notch_hz <= 0 or notch_hz >= (fs * 0.5):
        return signal
    b, a = iirnotch(w0=notch_hz, Q=quality_factor, fs=fs)
    return filtfilt(b, a, signal, axis=1).astype(np.float32, copy=False)


def _sanitize_and_clip(signal: np.ndarray) -> np.ndarray:
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    clip_scale = np.percentile(np.abs(signal), 99.5, axis=1, keepdims=True)
    clip_scale = np.where(clip_scale > 1e-6, clip_scale, 1.0)
    return np.clip(signal, -5.0 * clip_scale, 5.0 * clip_scale).astype(np.float32, copy=False)


def _downsample_to_length(signal: np.ndarray, target_length: int) -> np.ndarray:
    if signal.shape[1] == target_length:
        return signal.astype(np.float32, copy=False)
    return resample(signal, target_length, axis=1).astype(np.float32, copy=False)


def _preprocess_record(
    record_path: Path,
    lowcut: float,
    highcut: float,
    order: int,
    notch_hz: float,
    notch_q: float,
    target_length: int,
) -> tuple[np.ndarray, float, int]:
    signal, fs = _load_wfdb_signal(record_path)
    signal = _sanitize_and_clip(signal)
    filtered = _bandpass_filter(signal, fs=fs, lowcut=lowcut, highcut=highcut, order=order)
    filtered = _notch_filter(filtered, fs=fs, notch_hz=notch_hz, quality_factor=notch_q)
    filtered = _sanitize_and_clip(filtered)
    downsampled = _downsample_to_length(filtered, target_length=target_length)
    return downsampled, fs, signal.shape[1]


def _build_output_path(ecg_path: str, output_dir: Path) -> Path:
    return output_dir / f"{ecg_path}.npy"


def main():
    parser = argparse.ArgumentParser(description="Preprocess cached ECG signals with bandpass + downsampling")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--ecg-cache-dir", type=Path, default=DEFAULT_ECG_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--target-length", type=int, default=2500)
    parser.add_argument("--lowcut", type=float, default=0.5)
    parser.add_argument("--highcut", type=float, default=40.0)
    parser.add_argument("--filter-order", type=int, default=4)
    parser.add_argument("--notch-hz", type=float, default=0.0, help="Optional notch filter frequency, e.g. 50 or 60")
    parser.add_argument("--notch-q", type=float, default=30.0, help="Notch filter quality factor")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N ECGs")
    parser.add_argument("--overwrite", action="store_true", help="Recreate .npy files even if they already exist")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[DATA] Reading fusion dataset: {args.dataset_path}")
    df = pd.read_parquet(args.dataset_path, columns=["ecg_path"])
    ecg_paths = df["ecg_path"].dropna().astype(str).drop_duplicates().tolist()
    if args.limit is not None:
        ecg_paths = ecg_paths[: args.limit]

    print(f"[DATA] Unique ECG records to preprocess: {len(ecg_paths):,}")
    print(
        f"[CFG] Bandpass={args.lowcut:.2f}-{args.highcut:.2f} Hz | "
        f"order={args.filter_order} | notch={args.notch_hz:.2f} Hz | target_length={args.target_length}"
    )

    manifest_rows: list[dict] = []
    succeeded = 0
    failed = 0

    for ecg_path in tqdm(ecg_paths, desc="Preprocessing ECGs"):
        record_path = _get_local_record_path(ecg_path, args.ecg_cache_dir)
        output_path = _build_output_path(ecg_path, args.output_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not record_path.with_suffix(".hea").exists() or not record_path.with_suffix(".dat").exists():
            failed += 1
            manifest_rows.append(
                {
                    "ecg_path": ecg_path,
                    "source_record_path": str(record_path),
                    "preprocessed_npy_path": str(output_path),
                    "status": "missing_source",
                }
            )
            continue

        try:
            if output_path.exists() and not args.overwrite:
                array = np.load(output_path, mmap_mode="r")
                manifest_rows.append(
                    {
                        "ecg_path": ecg_path,
                        "source_record_path": str(record_path),
                        "preprocessed_npy_path": str(output_path),
                        "status": "reused_existing",
                        "n_leads": int(array.shape[0]),
                        "target_length": int(array.shape[1]),
                        "lowcut_hz": args.lowcut,
                        "highcut_hz": args.highcut,
                        "notch_hz": args.notch_hz,
                    }
                )
                succeeded += 1
                continue

            processed, fs, source_length = _preprocess_record(
                record_path=record_path,
                lowcut=args.lowcut,
                highcut=args.highcut,
                order=args.filter_order,
                notch_hz=args.notch_hz,
                notch_q=args.notch_q,
                target_length=args.target_length,
            )
            np.save(output_path, processed.astype(np.float32, copy=False))
            manifest_rows.append(
                {
                    "ecg_path": ecg_path,
                    "source_record_path": str(record_path),
                    "preprocessed_npy_path": str(output_path),
                    "status": "ok",
                    "sampling_rate_hz": fs,
                    "source_length": source_length,
                    "n_leads": int(processed.shape[0]),
                    "target_length": int(processed.shape[1]),
                    "lowcut_hz": args.lowcut,
                    "highcut_hz": args.highcut,
                    "notch_hz": args.notch_hz,
                }
            )
            succeeded += 1
        except Exception as exc:
            failed += 1
            manifest_rows.append(
                {
                    "ecg_path": ecg_path,
                    "source_record_path": str(record_path),
                    "preprocessed_npy_path": str(output_path),
                    "status": "error",
                    "error": str(exc),
                }
            )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(args.manifest_path, index=False)

    print(f"[SAVE] Manifest -> {args.manifest_path}")
    print(f"[DONE] Success={succeeded:,} | Failed={failed:,} | Output dir={args.output_dir}")


if __name__ == "__main__":
    main()
