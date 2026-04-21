# ==========================================
# STORE PREDICTIONS — Model Comparison + DB
# ==========================================
# Compares ALL available models on the same
# test set, picks the BEST one, and stores
# its predictions + reasoning into SQLite.
#
# Usage:
#   cd D:\MINI_PROJECT
#   python store_predictions.py
#   python store_predictions.py --subset 500
#
# Flow:
#   1. Load dataset → split test set
#   2. Load all available trained models
#   3. Evaluate each on the test set
#   4. Compare metrics → pick the best (by F1)
#   5. Run explainability with the best model
#   6. Store predictions + reasoning to SQLite
# ==========================================

import argparse
import importlib
import json
import sqlite3
import time
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)

from explain import (
    compute_clinical_attributions, generate_reasoning,
    CLINICAL_FEATURES,
)

# ── Project paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
SEED = 42

# ══════════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ══════════════════════════════════════════════════════════════════════════════
# Add new models here. Each entry defines:
#   - module    : Python import path for the model class
#   - class     : Model class name
#   - weights   : Path to saved .pth file (relative to project root)
#   - config    : Constructor kwargs for the model
#   - dataset   : Module that provides load_and_prepare_data()
# ══════════════════════════════════════════════════════════════════════════════

MODEL_REGISTRY = {
    "early_fusion": {
        "module":      "early_fusion.model",
        "class":       "EarlyFusionModel",
        "weights":     "early_fusion/artifacts/models/early_fusion_model.pth",
        "config": {
            "n_leads":    12,
            "ecg_length": 5000,
            "n_clinical": 24,
            "dropout":    0.4,
        },
        "metrics":      "early_fusion/artifacts/metrics/metrics.json",
        "dataset_module": "early_fusion.dataset",
    },
    # ── Add late_fusion here when ready ──────────────────────────────────
    # "late_fusion": {
    #     "module":      "late_fusion.model",
    #     "class":       "LateFusionModel",
    #     "weights":     "late_fusion/artifacts/models/late_fusion_model.pth",
    #     "config": {
    #         "n_leads":    12,
    #         "ecg_length": 5000,
    #         "n_clinical": 24,
    #         "dropout":    0.3,
    #     },
    #     "dataset_module": "late_fusion.dataset",
    # },
}

# Default DB location
DB_PATH = PROJECT_ROOT / "early_fusion" / "artifacts" / "predictions.db"


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE SETUP
# ══════════════════════════════════════════════════════════════════════════════

def create_database(db_path):
    """Create SQLite database with schema for the frontend."""
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            ecg_path              TEXT,
            anchor_age            REAL,
            gender                INTEGER,
            heart_rate            REAL,
            respiratory_rate      REAL,
            troponin_t            REAL,
            creatinine            REAL,
            sodium                REAL,
            potassium             REAL,
            num_diagnoses         REAL,
            los                   REAL,
            pr_interval           REAL,
            qrs_duration          REAL,
            qt_interval           REAL,
            qtc                   REAL,
            p_axis                REAL,
            qrs_axis              REAL,
            t_axis                REAL,
            rr_interval           REAL,
            ground_truth          INTEGER,
            predicted_label       INTEGER,
            predicted_probability REAL,
            risk_level            TEXT,
            reasoning_summary     TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS feature_importances (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id        INTEGER,
            feature_name      TEXT,
            display_name      TEXT,
            feature_value     REAL,
            normal_min        REAL,
            normal_max        REAL,
            unit              TEXT,
            is_abnormal       INTEGER,
            status            TEXT,
            attribution_score REAL,
            contribution_rank INTEGER,
            explanation       TEXT,
            FOREIGN KEY (patient_id) REFERENCES patients(id)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS model_metrics (
            metric_name  TEXT PRIMARY KEY,
            metric_value REAL,
            computed_at  TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS model_comparison (
            model_name  TEXT PRIMARY KEY,
            accuracy    REAL,
            precision_  REAL,
            recall      REAL,
            f1          REAL,
            auc         REAL,
            is_best     INTEGER,
            evaluated_at TEXT
        )
    """)

    conn.commit()
    return conn


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _load_model_threshold(info):
    metrics_path = PROJECT_ROOT / info.get("metrics", "")
    if not metrics_path.exists():
        return 0.5

    try:
        with open(metrics_path, "r") as f:
            payload = json.load(f)
        return float(payload.get("best_threshold", 0.5))
    except Exception:
        return 0.5


def load_model(model_name, device):
    """
    Dynamically load a model from the registry.

    Returns (model, model_info) or (None, None) if not available.
    """
    if model_name not in MODEL_REGISTRY:
        print(f"  [SKIP] '{model_name}' not in registry")
        return None, None

    info = MODEL_REGISTRY[model_name]
    weights_path = PROJECT_ROOT / info["weights"]

    if not weights_path.exists():
        print(f"  [SKIP] '{model_name}' — no trained weights at {weights_path}")
        return None, None

    try:
        # Dynamic import
        mod = importlib.import_module(info["module"])
        ModelClass = getattr(mod, info["class"])

        model = ModelClass(**info["config"]).to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()

        threshold = _load_model_threshold(info)
        print(f"  [OK]   '{model_name}' loaded from {weights_path} (threshold={threshold:.4f})")
        return model, {**info, "threshold": threshold}

    except Exception as e:
        print(f"  [FAIL] '{model_name}' — {e}")
        return None, None


# ══════════════════════════════════════════════════════════════════════════════
# MODEL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(model, test_local_paths, test_clinical, test_labels, device, threshold=0.5):
    """
    Run inference on the test set and compute metrics.

    Returns
    -------
    metrics   : dict — accuracy, precision, recall, f1, auc
    all_probs : np.ndarray — predicted probabilities
    all_preds : np.ndarray — binary predictions
    """
    from early_fusion.dataset import load_ecg_signal
    from early_fusion.config import ECG_LEADS, ECG_LENGTH

    model.eval()
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for i in range(len(test_labels)):
            # Load ECG
            try:
                ecg = load_ecg_signal(test_local_paths[i])
            except Exception:
                ecg = np.zeros((ECG_LEADS, ECG_LENGTH), dtype=np.float32)

            ecg = np.nan_to_num(ecg, nan=0.0, posinf=0.0, neginf=0.0)
            lead_mean = ecg.mean(axis=1, keepdims=True)
            lead_std  = ecg.std(axis=1, keepdims=True) + 1e-8
            ecg = (ecg - lead_mean) / lead_std

            ecg_t = torch.tensor(ecg, dtype=torch.float32).unsqueeze(0).to(device)
            cli_t = torch.tensor(test_clinical[i], dtype=torch.float32).unsqueeze(0).to(device)

            logits = model(ecg_t, cli_t)
            prob = torch.sigmoid(logits).item()
            all_probs.append(prob)
            all_preds.append(1 if prob >= threshold else 0)

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    labels    = test_labels.astype(int)

    metrics = {
        "accuracy":  accuracy_score(labels, all_preds),
        "precision": precision_score(labels, all_preds, zero_division=0),
        "recall":    recall_score(labels, all_preds, zero_division=0),
        "f1":        f1_score(labels, all_preds, zero_division=0),
    }
    try:
        metrics["auc"] = roc_auc_score(labels, all_probs)
    except ValueError:
        metrics["auc"] = 0.0

    return metrics, all_probs, all_preds


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Compare models, pick the best, store predictions + reasoning"
    )
    parser.add_argument("--subset", type=int, default=None,
                        help="Use only first N samples (quick testing)")
    args = parser.parse_args()

    # Seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {DEVICE}")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 1: Load data and split off test set
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(" STEP 1: LOADING DATA")
    print("=" * 70)

    from early_fusion.dataset import load_and_prepare_data
    _, _, _, test_metadata = load_and_prepare_data(subset=args.subset)

    test_df          = test_metadata["test_df"]
    test_local_paths = test_metadata["test_local_paths"]
    test_clinical    = test_metadata["test_clinical"]
    test_labels      = test_metadata["test_labels"]

    n_test = len(test_labels)
    print(f"\n[TEST] Test set: {n_test:,} patients  |  AMI: {int(test_labels.sum()):,}")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 2: Load and evaluate ALL available models
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(" STEP 2: LOADING & COMPARING MODELS")
    print("=" * 70)

    results = {}  # model_name → {model, metrics, probs, preds}

    for model_name in MODEL_REGISTRY:
        print(f"\n── {model_name} ──")
        model, info = load_model(model_name, DEVICE)
        if model is None:
            continue

        print(f"  Evaluating on {n_test:,} test patients ...")
        metrics, probs, preds = evaluate_model(
            model, test_local_paths, test_clinical, test_labels, DEVICE, threshold=info["threshold"]
        )

        results[model_name] = {
            "model": model,
            "threshold": info["threshold"],
            "metrics": metrics,
            "probs": probs,
            "preds": preds,
        }

        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall   : {metrics['recall']:.4f}")
        print(f"  F1       : {metrics['f1']:.4f}")
        print(f"  AUC      : {metrics['auc']:.4f}")

    if not results:
        print("\n[ERROR] No trained models found. Train at least one model first.")
        print("  → python -m early_fusion.train")
        return

    # ══════════════════════════════════════════════════════════════════════
    # STEP 3: Pick the BEST model (by F1 score)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(" STEP 3: SELECTING BEST MODEL")
    print("=" * 70)

    best_name = max(results, key=lambda k: results[k]["metrics"]["f1"])
    best = results[best_name]

    print(f"\n  ★ Best model: {best_name}")
    print(f"    F1 = {best['metrics']['f1']:.4f}  |  AUC = {best['metrics']['auc']:.4f}")

    if len(results) > 1:
        print(f"\n  Comparison:")
        print(f"  {'Model':<20} {'Accuracy':>8} {'Precision':>9} {'Recall':>7} {'F1':>7} {'AUC':>7}")
        print(f"  {'-'*60}")
        for name, res in sorted(results.items(), key=lambda x: x[1]['metrics']['f1'], reverse=True):
            m = res["metrics"]
            marker = " ★" if name == best_name else ""
            print(f"  {name:<20} {m['accuracy']:8.4f} {m['precision']:9.4f} "
                  f"{m['recall']:7.4f} {m['f1']:7.4f} {m['auc']:7.4f}{marker}")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 4: Run explainability with BEST model → store to SQLite
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(f" STEP 4: STORING PREDICTIONS ({best_name})")
    print("=" * 70)

    # Create / reset database
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = create_database(DB_PATH)
    cursor = conn.cursor()
    print(f"[DB] Created: {DB_PATH}")

    # Store model comparison results
    now = datetime.now().isoformat()
    for name, res in results.items():
        m = res["metrics"]
        cursor.execute("""
            INSERT OR REPLACE INTO model_comparison
            (model_name, accuracy, precision_, recall, f1, auc, is_best, evaluated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, m["accuracy"], m["precision"], m["recall"], m["f1"],
              m["auc"], 1 if name == best_name else 0, now))

    # Store aggregate metrics from best model
    for metric_name, value in best["metrics"].items():
        cursor.execute(
            "INSERT OR REPLACE INTO model_metrics (metric_name, metric_value, computed_at) VALUES (?, ?, ?)",
            (metric_name, round(value, 6), now)
        )
    cursor.execute(
        "INSERT OR REPLACE INTO model_metrics (metric_name, metric_value, computed_at) VALUES (?, ?, ?)",
        ("threshold", round(best.get("threshold", 0.5), 6), now)
    )
    # Also store which model was chosen
    cursor.execute(
        "INSERT OR REPLACE INTO model_metrics (metric_name, metric_value, computed_at) VALUES (?, ?, ?)",
        ("best_model", 0, f"{best_name}|{now}")
    )
    conn.commit()

    # Run explainability and store patient-by-patient
    best_model = best["model"]
    best_probs = best["probs"]
    best_preds = best["preds"]

    from early_fusion.dataset import load_ecg_signal
    from early_fusion.config import ECG_LEADS, ECG_LENGTH

    print(f"\n  Running explainability for {n_test:,} patients ...")

    for i in tqdm(range(n_test), desc="Storing predictions"):
        # Load ECG for attribution (needs gradients, can't use no_grad)
        try:
            ecg = load_ecg_signal(test_local_paths[i])
        except Exception:
            ecg = np.zeros((ECG_LEADS, ECG_LENGTH), dtype=np.float32)

        ecg = np.nan_to_num(ecg, nan=0.0, posinf=0.0, neginf=0.0)
        lead_mean = ecg.mean(axis=1, keepdims=True)
        lead_std  = ecg.std(axis=1, keepdims=True) + 1e-8
        ecg = (ecg - lead_mean) / lead_std

        ecg_tensor      = torch.tensor(ecg, dtype=torch.float32).unsqueeze(0)
        clinical_tensor = torch.tensor(test_clinical[i], dtype=torch.float32).unsqueeze(0)

        # Compute attributions
        attributions, _ = compute_clinical_attributions(
            best_model, ecg_tensor, clinical_tensor, DEVICE
        )

        probability     = best_probs[i]
        predicted_label = best_preds[i]
        risk_level      = "High Risk" if probability >= 0.5 else "Low Risk"
        if "threshold" in best:
            risk_level = "High Risk" if probability >= best["threshold"] else "Low Risk"
        ground_truth    = int(test_labels[i])

        # Raw clinical values for reasoning
        row = test_df.iloc[i]
        raw_values = {}
        for feat in CLINICAL_FEATURES:
            if feat in test_df.columns:
                val = row.get(feat, 0)
                raw_values[feat] = float(val) if not (isinstance(val, float) and np.isnan(val)) else 0.0

        reasoning, summary = generate_reasoning(attributions, raw_values, probability)

        # Insert patient
        cursor.execute("""
            INSERT INTO patients (
                ecg_path, anchor_age, gender, heart_rate, respiratory_rate,
                troponin_t, creatinine, sodium, potassium, num_diagnoses, los,
                pr_interval, qrs_duration, qt_interval, qtc,
                p_axis, qrs_axis, t_axis, rr_interval,
                ground_truth, predicted_label, predicted_probability,
                risk_level, reasoning_summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            test_local_paths[i],
            _sf(row.get("anchor_age")), int(row.get("gender", 0)),
            _sf(row.get("Heart_Rate")), _sf(row.get("Respiratory_Rate")),
            _sf(row.get("Troponin_T")), _sf(row.get("Creatinine")),
            _sf(row.get("Sodium")),     _sf(row.get("Potassium")),
            _sf(row.get("num_diagnoses")), _sf(row.get("los")),
            _sf(row.get("PR_interval")), _sf(row.get("QRS_duration")),
            _sf(row.get("QT_interval")), _sf(row.get("QTc")),
            _sf(row.get("P_axis")), _sf(row.get("QRS_axis")),
            _sf(row.get("T_axis")), _sf(row.get("RR_interval")),
            ground_truth, predicted_label, round(probability, 6),
            risk_level, summary,
        ))

        patient_id = cursor.lastrowid

        # Insert feature importances
        for r in reasoning:
            cursor.execute("""
                INSERT INTO feature_importances (
                    patient_id, feature_name, display_name, feature_value,
                    normal_min, normal_max, unit, is_abnormal, status,
                    attribution_score, contribution_rank, explanation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                patient_id, r["feature"], r["display_name"], r["value"],
                r["normal_min"], r["normal_max"], r["unit"],
                r["is_abnormal"], r["status"], r["attribution_score"],
                r["rank"], r["explanation"],
            ))

        if (i + 1) % 100 == 0:
            conn.commit()

    conn.commit()
    conn.close()

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(" ✓ DATABASE READY")
    print("=" * 70)
    print(f"  Best model : {best_name}")
    print(f"  DB path    : {DB_PATH}")
    print(f"  Patients   : {n_test:,}")
    print(f"  AMI cases  : {int(test_labels.sum()):,}")
    print(f"  Accuracy   : {best['metrics']['accuracy']:.4f}")
    print(f"  F1         : {best['metrics']['f1']:.4f}")
    print(f"  AUC        : {best['metrics']['auc']:.4f}")
    print("=" * 70)
    print(f"\n  Start the API:  python api.py")
    print(f"  Swagger docs:   http://localhost:5000/docs")


def _sf(val):
    """Safe float conversion."""
    if val is None:
        return 0.0
    try:
        f = float(val)
        return 0.0 if np.isnan(f) else f
    except (ValueError, TypeError):
        return 0.0


if __name__ == "__main__":
    main()
