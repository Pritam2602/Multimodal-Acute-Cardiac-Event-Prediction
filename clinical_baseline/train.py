import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from early_fusion.config import (
    CLINICAL_FEATURES,
    DATASET_PATH,
    PROJECT_ROOT,
    SEED,
    TARGET_COLUMN,
    TEST_SPLIT,
    VAL_SPLIT,
)


ARTIFACT_ROOT = PROJECT_ROOT / "clinical_baseline" / "artifacts"


def infer_group_ids(df: pd.DataFrame) -> tuple[str | None, np.ndarray | None]:
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
    local_labels = labels[indices]
    local_groups = groups[indices]
    train_local, test_local = next(
        splitter.split(np.zeros(len(indices)), local_labels, groups=local_groups)
    )
    return indices[train_local], indices[test_local]


def find_best_threshold(labels, probs, threshold_min=0.10, threshold_max=0.90, steps=50):
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)

    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(threshold_min, threshold_max, steps):
        preds = (probs >= threshold).astype(int)
        score = f1_score(labels, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold


def compute_metrics(labels, probs, threshold):
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    preds = (probs >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "auc": roc_auc_score(labels, probs),
        "average_precision": average_precision_score(labels, probs),
        "threshold": float(threshold),
    }


def build_models(seed: int):
    return {
        "logistic_balanced": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                solver="lbfgs",
                random_state=seed,
            ),
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            class_weight="balanced",
            early_stopping=True,
            random_state=seed,
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Clinical-only AMI baselines")
    parser.add_argument("--subset", type=int, default=None, help="Use only first N rows")
    parser.add_argument("--run-name", type=str, default="grouped_clinical_baseline")
    args = parser.parse_args()

    run_root = ARTIFACT_ROOT / "runs" / args.run_name
    metrics_dir = run_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print(f"[DATA] Loading: {DATASET_PATH}")
    df = pd.read_parquet(DATASET_PATH)
    if args.subset is not None:
        df = df.head(args.subset).copy()
        print(f"[DATA] Using subset: {len(df):,}")

    missing_features = [col for col in CLINICAL_FEATURES if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing clinical features: {missing_features}")

    labels = df[TARGET_COLUMN].astype(int).to_numpy()
    features = df[CLINICAL_FEATURES].replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median())
    x = features.to_numpy(dtype=np.float32)

    group_name, groups = infer_group_ids(df)
    if groups is not None:
        print(f"[DATA] Grouped split enabled: {group_name} ({len(np.unique(groups)):,} groups)")
    else:
        print("[DATA] Grouped split unavailable; falling back to row-level stratified split")

    indices = np.arange(len(labels))
    train_pool_idx, test_idx = stratified_group_train_test_split(
        indices, labels, groups, TEST_SPLIT, SEED
    )
    train_idx, val_idx = stratified_group_train_test_split(
        train_pool_idx, labels, groups, VAL_SPLIT, SEED
    )

    if groups is not None:
        print(f"[DATA] Stage 1 group overlap: {len(set(groups[train_pool_idx]) & set(groups[test_idx]))}")
        print(f"[DATA] Stage 2 group overlap: {len(set(groups[train_idx]) & set(groups[val_idx]))}")

    print(
        f"[DATA] Train={len(train_idx):,} Val={len(val_idx):,} Test={len(test_idx):,} | "
        f"train_prev={labels[train_idx].mean():.4%} val_prev={labels[val_idx].mean():.4%}"
    )

    rows = []
    best_name = None
    best_f1 = -1.0
    models = build_models(SEED)

    for name, model in models.items():
        print("\n" + "=" * 70)
        print(f" MODEL: {name}")
        print("=" * 70)

        model.fit(x[train_idx], labels[train_idx])

        val_probs = model.predict_proba(x[val_idx])[:, 1]
        threshold = find_best_threshold(labels[val_idx], val_probs)
        val_metrics = compute_metrics(labels[val_idx], val_probs, threshold)

        train_probs = model.predict_proba(x[train_idx])[:, 1]
        train_metrics = compute_metrics(labels[train_idx], train_probs, threshold)

        row = {
            "model": name,
            "train": {key: round(float(value), 6) for key, value in train_metrics.items()},
            "val": {key: round(float(value), 6) for key, value in val_metrics.items()},
        }
        rows.append(row)

        print(
            f"train_f1={train_metrics['f1']:.4f} val_f1={val_metrics['f1']:.4f} "
            f"val_auc={val_metrics['auc']:.4f} val_ap={val_metrics['average_precision']:.4f} "
            f"threshold={threshold:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_name = name

    summary = {
        "features": CLINICAL_FEATURES,
        "n_features": len(CLINICAL_FEATURES),
        "group_name": group_name,
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
        "best_model": best_name,
        "best_val_f1": round(float(best_f1), 6),
        "rows": rows,
    }

    metrics_path = metrics_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(" CLINICAL BASELINE SUMMARY")
    print("=" * 70)
    print(f"Best model: {best_name}")
    print(f"Best Val F1: {best_f1:.4f}")
    print(f"[SAVE] Metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
