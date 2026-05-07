import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from early_fusion.config import (
    CLINICAL_FEATURES,
    DATASET_PATH,
    SEED,
    TARGET_COLUMN,
    TEST_SPLIT,
    VAL_SPLIT,
)
from early_fusion.dataset import infer_group_ids, stratified_group_train_test_split


DEFAULT_RUN_DIR = Path("early_fusion/artifacts/runs/baseline_es")
OUTCOME_ORDER = ["TP", "FP", "TN", "FN"]


def load_validation_frame() -> pd.DataFrame:
    df = pd.read_parquet(DATASET_PATH).reset_index(drop=True)
    labels = df[TARGET_COLUMN].astype(int).to_numpy()
    indices = np.arange(len(labels))
    _, groups = infer_group_ids(df)

    train_pool_idx, _test_idx = stratified_group_train_test_split(
        indices,
        labels,
        groups,
        test_size=TEST_SPLIT,
        seed=SEED,
    )
    _train_idx, val_idx = stratified_group_train_test_split(
        train_pool_idx,
        labels,
        groups,
        test_size=VAL_SPLIT,
        seed=SEED,
    )

    return df.iloc[val_idx].reset_index(drop=True)


def compute_threshold_table(labels: np.ndarray, probs: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in np.linspace(0.10, 0.90, 50):
        preds = (probs >= threshold).astype(int)
        rows.append(
            {
                "threshold": round(float(threshold), 6),
                "accuracy": accuracy_score(labels, preds),
                "precision": precision_score(labels, preds, zero_division=0),
                "recall": recall_score(labels, preds, zero_division=0),
                "f1": f1_score(labels, preds, zero_division=0),
                "predicted_positive_rate": float(preds.mean()),
                "tp": int(((labels == 1) & (preds == 1)).sum()),
                "fp": int(((labels == 0) & (preds == 1)).sum()),
                "tn": int(((labels == 0) & (preds == 0)).sum()),
                "fn": int(((labels == 1) & (preds == 0)).sum()),
            }
        )
    return pd.DataFrame(rows)


def add_prediction_columns(df: pd.DataFrame, labels: np.ndarray, probs: np.ndarray, threshold: float) -> pd.DataFrame:
    result = df.copy()
    preds = (probs >= threshold).astype(int)
    result["label"] = labels
    result["prob"] = probs
    result["pred"] = preds

    result["outcome"] = "TN"
    result.loc[(labels == 1) & (preds == 1), "outcome"] = "TP"
    result.loc[(labels == 0) & (preds == 1), "outcome"] = "FP"
    result.loc[(labels == 1) & (preds == 0), "outcome"] = "FN"
    return result


def summarize_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(df)
    for outcome in OUTCOME_ORDER:
        part = df[df["outcome"] == outcome]
        rows.append(
            {
                "outcome": outcome,
                "count": int(len(part)),
                "share": float(len(part) / max(total, 1)),
                "mean_probability": float(part["prob"].mean()) if len(part) else np.nan,
                "ami_rate": float(part["label"].mean()) if len(part) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_features(df: pd.DataFrame) -> pd.DataFrame:
    numeric_features = [col for col in CLINICAL_FEATURES if col in df.columns]
    grouped = df.groupby("outcome")[numeric_features].mean().reindex(OUTCOME_ORDER)
    rows = []
    for feature in numeric_features:
        values = grouped[feature]
        rows.append(
            {
                "feature": feature,
                "TP_mean": values.get("TP"),
                "FP_mean": values.get("FP"),
                "TN_mean": values.get("TN"),
                "FN_mean": values.get("FN"),
                "FP_minus_TN": values.get("FP") - values.get("TN"),
                "FN_minus_TP": values.get("FN") - values.get("TP"),
            }
        )
    out = pd.DataFrame(rows)
    out["abs_FP_minus_TN"] = out["FP_minus_TN"].abs()
    return out.sort_values("abs_FP_minus_TN", ascending=False)


def pick_case_columns(df: pd.DataFrame) -> list[str]:
    candidates = [
        "outcome",
        "label",
        "pred",
        "prob",
        "ecg_path",
        "anchor_age",
        "gender",
        "Troponin_T",
        "log1p_Troponin_T",
        "Troponin_T_high",
        "QRS_duration",
        "QTc",
        "QRS_axis",
        "T_axis",
        "T_axis_abnormal",
        "PR_interval_invalid",
        "P_axis_invalid",
        "num_diagnoses",
        "los",
    ]
    return [col for col in candidates if col in df.columns]


def main():
    parser = argparse.ArgumentParser(description="Analyze validation errors for a saved run")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--top-n", type=int, default=100)
    args = parser.parse_args()

    metrics_path = args.run_dir / "metrics" / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    outputs = metrics.get("validation_outputs")
    if not outputs:
        raise ValueError(f"No validation_outputs in {metrics_path}")

    labels = np.asarray(outputs["labels"], dtype=int)
    probs = np.asarray(outputs["probs"], dtype=float)
    threshold = float(metrics.get("best_threshold", metrics["val"]["threshold"]))

    val_df = load_validation_frame()
    if len(val_df) != len(labels):
        raise ValueError(
            f"Validation row mismatch: split has {len(val_df)} rows, metrics has {len(labels)} rows"
        )

    analysis_dir = args.run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    analyzed = add_prediction_columns(val_df, labels, probs, threshold)
    threshold_table = compute_threshold_table(labels, probs)
    outcome_summary = summarize_outcomes(analyzed)
    feature_summary = summarize_features(analyzed)
    case_cols = pick_case_columns(analyzed)

    false_positives = analyzed[analyzed["outcome"] == "FP"].sort_values("prob", ascending=False)
    false_negatives = analyzed[analyzed["outcome"] == "FN"].sort_values("prob", ascending=True)

    threshold_table.to_csv(analysis_dir / "threshold_curve.csv", index=False)
    outcome_summary.to_csv(analysis_dir / "outcome_summary.csv", index=False)
    feature_summary.to_csv(analysis_dir / "feature_outcome_means.csv", index=False)
    false_positives[case_cols].head(args.top_n).to_csv(
        analysis_dir / "top_false_positives.csv",
        index=False,
    )
    false_negatives[case_cols].head(args.top_n).to_csv(
        analysis_dir / "top_false_negatives.csv",
        index=False,
    )

    best_threshold_row = threshold_table.sort_values("f1", ascending=False).iloc[0]
    summary = {
        "run_dir": str(args.run_dir),
        "metrics_threshold": threshold,
        "best_curve_threshold": float(best_threshold_row["threshold"]),
        "best_curve_f1": float(best_threshold_row["f1"]),
        "counts": {
            row["outcome"]: int(row["count"])
            for _, row in outcome_summary.iterrows()
        },
        "top_fp_features_vs_tn": feature_summary[
            ["feature", "FP_mean", "TN_mean", "FP_minus_TN"]
        ].head(15).to_dict(orient="records"),
    }
    with open(analysis_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(" ERROR ANALYSIS")
    print("=" * 70)
    print(f"Run: {args.run_dir}")
    print(f"Threshold: {threshold:.4f}")
    print(outcome_summary.to_string(index=False))
    print("\nTop FP-vs-TN feature shifts:")
    print(
        feature_summary[["feature", "FP_mean", "TN_mean", "FP_minus_TN"]]
        .head(10)
        .to_string(index=False)
    )
    print(f"\n[SAVE] Analysis -> {analysis_dir}")


if __name__ == "__main__":
    main()
