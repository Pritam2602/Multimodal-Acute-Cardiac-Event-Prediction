# ==========================================
# API — FastAPI REST API for Frontend
# ==========================================
# Shared across ALL models (early_fusion, late_fusion, etc.)
# Reads from the SQLite database — model-agnostic.
#
# Usage:
#   cd D:\MINI_PROJECT
#   python api.py
#   # or: uvicorn api:app --reload --port 5000
#
# Docs: http://localhost:5000/docs  (Swagger UI)
# ==========================================

import sqlite3
import json
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import wfdb
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# ── Configuration ────────────────────────────────────────────────────────────
# DB path — shared across models. Each model's store_predictions.py
# writes to this same database.
PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "early_fusion" / "artifacts" / "predictions.db"

ECG_LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
                  "V1", "V2", "V3", "V4", "V5", "V6"]

# ── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AMI Prediction API",
    description="Clinical decision-support API for Acute Myocardial Infarction prediction. "
                "Serves test-set predictions, ECG waveforms, and model reasoning.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Database Helper ──────────────────────────────────────────────────────────

@contextmanager
def get_db():
    """Context manager for database connections."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# ── API Routes ───────────────────────────────────────────────────────────────

@app.get("/api/patients")
def get_patients(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    risk_level: str = Query(None, description="Filter: 'High Risk' or 'Low Risk'"),
    search: int = Query(None, description="Search by patient ID"),
):
    """
    List all test patients with pagination and optional filtering.
    Ordered by predicted probability (highest risk first).
    """
    with get_db() as db:
        offset = (page - 1) * per_page

        where_clauses = []
        params = []

        if risk_level:
            where_clauses.append("risk_level = ?")
            params.append(risk_level)
        if search is not None:
            where_clauses.append("id = ?")
            params.append(search)

        where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        total = db.execute(f"SELECT COUNT(*) FROM patients {where_sql}", params).fetchone()[0]

        rows = db.execute(f"""
            SELECT id, anchor_age, gender, heart_rate, troponin_t,
                   ground_truth, predicted_label, predicted_probability,
                   risk_level, reasoning_summary
            FROM patients {where_sql}
            ORDER BY predicted_probability DESC
            LIMIT ? OFFSET ?
        """, params + [per_page, offset]).fetchall()

        patients = []
        for row in rows:
            patients.append({
                "id": row["id"],
                "age": row["anchor_age"],
                "gender": "Male" if row["gender"] == 1 else "Female",
                "heart_rate": row["heart_rate"],
                "troponin_t": row["troponin_t"],
                "ground_truth": row["ground_truth"],
                "predicted_label": row["predicted_label"],
                "predicted_probability": round(row["predicted_probability"], 4),
                "risk_level": row["risk_level"],
                "reasoning_summary": row["reasoning_summary"],
            })

        return {
            "patients": patients,
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page,
        }


@app.get("/api/patients/{patient_id}")
def get_patient(patient_id: int):
    """Get full clinical detail and prediction for a single patient."""
    with get_db() as db:
        row = db.execute("SELECT * FROM patients WHERE id = ?", (patient_id,)).fetchone()

        if row is None:
            raise HTTPException(status_code=404, detail="Patient not found")

        patient = dict(row)
        patient["gender_display"] = "Male" if patient["gender"] == 1 else "Female"
        return patient


@app.get("/api/patients/{patient_id}/ecg")
def get_patient_ecg(patient_id: int):
    """
    Load 12-lead ECG waveform from local cache and return as JSON.

    The waveform is loaded on-demand from .hea/.dat files (~5ms).
    Returns an array of 12 leads, each with signal data ready for plotting.
    """
    with get_db() as db:
        row = db.execute(
            "SELECT ecg_path FROM patients WHERE id = ?", (patient_id,)
        ).fetchone()

        if row is None:
            raise HTTPException(status_code=404, detail="Patient not found")

        ecg_path = row["ecg_path"]

    try:
        record = wfdb.rdrecord(ecg_path)
        signal = record.p_signal  # (num_samples, num_leads)

        if signal is None:
            raise HTTPException(status_code=500, detail="No signal data in ECG record")

        leads = []
        n_leads = min(signal.shape[1], 12)
        for j in range(n_leads):
            lead_signal = signal[:, j].tolist()
            lead_signal = [
                0.0 if (x != x or abs(x) == float("inf")) else round(x, 4)
                for x in lead_signal
            ]
            leads.append({
                "name": ECG_LEAD_NAMES[j] if j < len(ECG_LEAD_NAMES) else f"Lead {j+1}",
                "signal": lead_signal,
            })

        # Pad if fewer than 12 leads
        while len(leads) < 12:
            leads.append({
                "name": ECG_LEAD_NAMES[len(leads)],
                "signal": [0.0] * signal.shape[0],
            })

        return {
            "patient_id": patient_id,
            "sampling_rate": record.fs if record.fs else 500,
            "num_samples": signal.shape[0],
            "duration_seconds": round(signal.shape[0] / (record.fs or 500), 2),
            "leads": leads,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load ECG: {str(e)}")


@app.get("/api/patients/{patient_id}/insights")
def get_patient_insights(patient_id: int):
    """
    Get model reasoning / explainability for a patient's prediction.

    Returns ranked feature attributions with human-readable explanations
    showing WHY the model predicted AMI (or not) for this patient.
    """
    with get_db() as db:
        patient = db.execute(
            "SELECT predicted_probability, risk_level, ground_truth FROM patients WHERE id = ?",
            (patient_id,)
        ).fetchone()

        if patient is None:
            raise HTTPException(status_code=404, detail="Patient not found")

        rows = db.execute("""
            SELECT feature_name, display_name, feature_value,
                   normal_min, normal_max, unit, is_abnormal, status,
                   attribution_score, contribution_rank, explanation
            FROM feature_importances
            WHERE patient_id = ?
            ORDER BY contribution_rank ASC
        """, (patient_id,)).fetchall()

        reasoning = []
        for row in rows:
            reasoning.append({
                "rank": row["contribution_rank"],
                "feature": row["feature_name"],
                "display_name": row["display_name"],
                "value": row["feature_value"],
                "unit": row["unit"],
                "normal_range": f"{row['normal_min']}–{row['normal_max']}",
                "is_abnormal": bool(row["is_abnormal"]),
                "status": row["status"],
                "attribution_score": row["attribution_score"],
                "explanation": row["explanation"],
            })

        return {
            "patient_id": patient_id,
            "risk_level": patient["risk_level"],
            "predicted_probability": round(patient["predicted_probability"], 4),
            "ground_truth": patient["ground_truth"],
            "reasoning": reasoning,
        }


@app.get("/api/metrics")
def get_metrics():
    """Get aggregate model performance metrics (accuracy, F1, AUC, etc.)."""
    with get_db() as db:
        rows = db.execute("SELECT * FROM model_metrics").fetchall()

        metrics = {}
        for row in rows:
            metrics[row["metric_name"]] = {
                "value": round(row["metric_value"], 4),
                "computed_at": row["computed_at"],
            }
        return metrics


@app.get("/api/stats")
def get_stats():
    """Get summary statistics for the dashboard overview."""
    with get_db() as db:
        total = db.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
        ami_count = db.execute(
            "SELECT COUNT(*) FROM patients WHERE ground_truth = 1"
        ).fetchone()[0]
        high_risk = db.execute(
            "SELECT COUNT(*) FROM patients WHERE risk_level = 'High Risk'"
        ).fetchone()[0]
        correct = db.execute(
            "SELECT COUNT(*) FROM patients WHERE predicted_label = ground_truth"
        ).fetchone()[0]

        metrics = {}
        for row in db.execute("SELECT * FROM model_metrics").fetchall():
            metrics[row["metric_name"]] = round(row["metric_value"], 4)

        return {
            "total_patients": total,
            "ami_cases": ami_count,
            "non_ami_cases": total - ami_count,
            "high_risk_predictions": high_risk,
            "low_risk_predictions": total - high_risk,
            "correct_predictions": correct,
            "accuracy": round(correct / max(total, 1), 4),
            "model_metrics": metrics,
        }


@app.get("/api/comparison")
def get_model_comparison():
    """
    Get model comparison results — shows all evaluated models
    and which one was selected as the best.
    """
    with get_db() as db:
        try:
            rows = db.execute(
                "SELECT * FROM model_comparison ORDER BY f1 DESC"
            ).fetchall()
        except Exception:
            return {"models": [], "best_model": None}

        models = []
        best_model = None
        for row in rows:
            entry = {
                "model_name": row["model_name"],
                "accuracy": round(row["accuracy"], 4),
                "precision": round(row["precision_"], 4),
                "recall": round(row["recall"], 4),
                "f1": round(row["f1"], 4),
                "auc": round(row["auc"], 4),
                "is_best": bool(row["is_best"]),
                "evaluated_at": row["evaluated_at"],
            }
            models.append(entry)
            if row["is_best"]:
                best_model = row["model_name"]

        return {
            "models": models,
            "best_model": best_model,
        }


# ── Run Server ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print(" AMI Prediction API (FastAPI)")
    print("=" * 70)
    print(f"  Database : {DB_PATH}")
    print(f"  Swagger  : http://localhost:5000/docs")
    print("=" * 70)

    uvicorn.run(app, host="0.0.0.0", port=5000)
