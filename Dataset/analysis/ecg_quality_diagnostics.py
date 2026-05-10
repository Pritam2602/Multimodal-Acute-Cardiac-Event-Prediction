import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    package_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(package_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from late_fusion.config import PREPROCESSED_ECG_DIR, MEMMAP_DIR
from late_fusion.dataset import _get_preprocessed_ecg_path, _normalize_ecg


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "mimic_data" / "final_preprocessed_fusion_dataset.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "late_fusion" / "artifacts" / "ecg_diagnostics"
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _plot_ecg(signal: np.ndarray, title: str, save_path: Path):
    fig, axes = plt.subplots(6, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.plot(signal[i], linewidth=0.8)
        ax.set_title(LEAD_NAMES[i], fontsize=10)
        ax.grid(True, alpha=0.2)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _lead_identity_errors(signal: np.ndarray) -> dict[str, float]:
    i, ii, iii, avr, avl, avf = signal[0], signal[1], signal[2], signal[3], signal[4], signal[5]
    return {
        "lead_iii_vs_ii_minus_i": float(np.mean(np.abs(iii - (ii - i)))),
        "avr_vs_neg_half_i_plus_ii": float(np.mean(np.abs(avr - (-(i + ii) / 2.0)))),
        "avl_vs_i_minus_half_ii": float(np.mean(np.abs(avl - (i - ii / 2.0)))),
        "avf_vs_ii_minus_half_i": float(np.mean(np.abs(avf - (ii - i / 2.0)))),
    }


def _find_latest_preprocessed_memmap() -> Path | None:
    candidates = sorted(MEMMAP_DIR.glob("late_fusion_ecg_preprocessed_*.npy"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def main():
    parser = argparse.ArgumentParser(description="Audit preprocessed ECG signal quality for late fusion")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--preprocessed-dir", type=Path, default=PREPROCESSED_ECG_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=1000, help="Number of records to audit")
    parser.add_argument("--plot-count", type=int, default=3, help="Random positives and negatives to plot")
    parser.add_argument("--check-memmap", action="store_true", help="Compare source .npy records against latest preprocessed memmap")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.dataset_path, columns=["ecg_path", "AMI"]).dropna(subset=["ecg_path"]).drop_duplicates("ecg_path")
    if args.limit is not None:
        df = df.head(args.limit).copy()

    records = []
    lead_identity_rows = []
    positive_examples = []
    negative_examples = []

    memmap = None
    if args.check_memmap:
        memmap_path = _find_latest_preprocessed_memmap()
        if memmap_path is not None:
            memmap = np.load(memmap_path, mmap_mode="r")
        else:
            memmap_path = None
    else:
        memmap_path = None

    memmap_match_errors = []
    loaded_count = 0

    for row_idx, row in df.reset_index(drop=True).iterrows():
        ecg_path = str(row["ecg_path"])
        label = int(row["AMI"])
        npy_path = _get_preprocessed_ecg_path(ecg_path, args.preprocessed_dir)
        if not npy_path.exists():
            continue

        signal = np.load(npy_path).astype(np.float32, copy=False)
        normalized = _normalize_ecg(signal)
        loaded_count += 1

        lead_mean = signal.mean(axis=1)
        lead_std = signal.std(axis=1)
        lead_min = signal.min(axis=1)
        lead_max = signal.max(axis=1)
        lead_energy = np.mean(np.square(signal), axis=1)
        lead_noise_ratio = np.std(np.diff(signal, axis=1), axis=1) / np.maximum(lead_std, 1e-6)
        lead_flat = lead_std < 1e-4
        lead_nans = np.isnan(signal).sum(axis=1)

        for lead_idx, lead_name in enumerate(LEAD_NAMES):
            records.append(
                {
                    "row_index": row_idx,
                    "ecg_path": ecg_path,
                    "label": label,
                    "lead": lead_name,
                    "mean": float(lead_mean[lead_idx]),
                    "std": float(lead_std[lead_idx]),
                    "min": float(lead_min[lead_idx]),
                    "max": float(lead_max[lead_idx]),
                    "energy": float(lead_energy[lead_idx]),
                    "noise_ratio": float(lead_noise_ratio[lead_idx]),
                    "flat_flag": int(lead_flat[lead_idx]),
                    "nan_count": int(lead_nans[lead_idx]),
                }
            )

        lead_identity_rows.append({"ecg_path": ecg_path, "label": label, **_lead_identity_errors(signal)})

        if memmap is not None and row_idx < memmap.shape[0]:
            memmap_match_errors.append(float(np.max(np.abs(memmap[row_idx] - normalized))))

        example_item = (ecg_path, signal)
        if label == 1 and len(positive_examples) < args.plot_count:
            positive_examples.append(example_item)
        elif label == 0 and len(negative_examples) < args.plot_count:
            negative_examples.append(example_item)

    lead_df = pd.DataFrame(records)
    lead_df.to_csv(args.output_dir / "lead_statistics.csv", index=False)
    identity_df = pd.DataFrame(lead_identity_rows)
    identity_df.to_csv(args.output_dir / "lead_identity_checks.csv", index=False)

    if not lead_df.empty:
        summary_df = (
            lead_df.groupby("lead")[["mean", "std", "min", "max", "energy", "noise_ratio", "flat_flag", "nan_count"]]
            .agg(["mean", "median", "std"])
        )
        summary_df.to_csv(args.output_dir / "lead_summary.csv")

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        lead_df.boxplot(column="std", by="lead", ax=axes[0], rot=45)
        lead_df.boxplot(column="energy", by="lead", ax=axes[1], rot=45)
        lead_df.boxplot(column="noise_ratio", by="lead", ax=axes[2], rot=45)
        fig.suptitle("Per-lead ECG quality distributions", fontsize=14, fontweight="bold")
        for ax, title in zip(axes, ["Std", "Energy", "Noise Ratio"]):
            ax.set_title(title)
            ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(args.output_dir / "lead_quality_distributions.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    for idx, (ecg_path, signal) in enumerate(positive_examples, 1):
        _plot_ecg(signal, f"Positive ECG {idx}: {ecg_path}", args.output_dir / f"positive_ecg_{idx}.png")
    for idx, (ecg_path, signal) in enumerate(negative_examples, 1):
        _plot_ecg(signal, f"Negative ECG {idx}: {ecg_path}", args.output_dir / f"negative_ecg_{idx}.png")

    payload = {
        "dataset_path": str(args.dataset_path),
        "preprocessed_dir": str(args.preprocessed_dir),
        "records_requested": int(len(df)),
        "records_loaded": int(loaded_count),
        "memmap_path": str(memmap_path) if memmap_path is not None else None,
        "memmap_max_abs_diff_mean": float(np.mean(memmap_match_errors)) if memmap_match_errors else None,
        "memmap_max_abs_diff_max": float(np.max(memmap_match_errors)) if memmap_match_errors else None,
        "flat_lead_rate": float(lead_df["flat_flag"].mean()) if not lead_df.empty else None,
        "nan_rate": float((lead_df["nan_count"] > 0).mean()) if not lead_df.empty else None,
        "global_min": float(lead_df["min"].min()) if not lead_df.empty else None,
        "global_max": float(lead_df["max"].max()) if not lead_df.empty else None,
        "global_mean": float(lead_df["mean"].mean()) if not lead_df.empty else None,
        "global_std_mean": float(lead_df["std"].mean()) if not lead_df.empty else None,
    }
    (args.output_dir / "diagnostics_summary.json").write_text(json.dumps(payload, indent=2))

    print(f"[SAVE] Diagnostics -> {args.output_dir}")


if __name__ == "__main__":
    main()
