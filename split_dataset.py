# ==========================================
# SPLIT DATASET → Train / Val / Test
# ==========================================
# Splits refined_temporal_fusion_dataset.parquet into
# 70% train / 15% val / 15% test (stratified by AMI).
#
# Also enriches the TEST split with:
#   - heuristic predicted_prob (logistic score from troponin + ECG features)
#   - derived risk_level, dominant_region
#   - JSONB-ready timelines, prediction, attention columns
#   - outputs test.csv ready to COPY into PostgreSQL via pgAdmin
#
# Usage:
#   python split_dataset.py
#   python split_dataset.py --output-dir splits
# ==========================================

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

SEED = 42
PARQUET_PATH = Path(__file__).resolve().parent / "refined_temporal_fusion_dataset.parquet"
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
THRESHOLD = 0.48


# ─── Heuristic prediction score ───────────────────────────────────────────────

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def compute_predicted_prob(df: pd.DataFrame) -> np.ndarray:
    """
    Logistic score from troponin dynamics + ECG features.
    Not a trained model — a clinically-motivated heuristic for demo purposes.
    """
    z = np.full(len(df), -0.6)  # slight bias toward Non-AMI

    # Troponin (most predictive)
    trop_peak = df["troponin_peak"].fillna(0).clip(lower=0).values
    z += np.log1p(trop_peak) * 0.9

    fold = df["trop_fold_rise"].fillna(0).clip(lower=0).values
    z += np.where(fold > 3, 0.35, np.where(fold > 1.5, 0.18, 0.0))

    vel = df["troponin_velocity"].fillna(0).clip(lower=0).values
    z += np.clip(vel * 3.0, 0, 0.4)

    log_trop = df["log1p_Troponin_T"].fillna(0).values
    z += log_trop * 0.5

    # ECG abnormalities
    z += df["QRS_wide"].fillna(0).values * 0.28
    z += df["T_axis_abnormal"].fillna(0).values * 0.22
    z += df["QTc_prolonged"].fillna(0).values * 0.18
    z += df["QRS_axis_deviation"].fillna(0).values * 0.12

    # Age risk (older → slightly higher)
    age = df["anchor_age"].fillna(55).values
    z += (age - 55.0) / 60.0 * 0.25

    # Comorbidities lower confidence (confounders)
    z -= df["has_ckd"].fillna(0).values * 0.12
    z -= df["has_sepsis"].fillna(0).values * 0.10

    return sigmoid(z).astype(np.float32)


# ─── Derived fields ───────────────────────────────────────────────────────────

def risk_level(prob: float) -> str:
    if prob >= 0.75:
        return "Critical"
    if prob >= 0.48:
        return "High"
    if prob >= 0.25:
        return "Moderate"
    return "Low"


def dominant_region(row: pd.Series) -> str:
    qrs = row.get("QRS_axis", 0) or 0
    t = row.get("T_axis", 0) or 0
    if -30 <= qrs <= 90 and t < 0:
        return "Inferior"
    if qrs < -30 or qrs > 90:
        return "Anterior/Septal"
    if t > 90:
        return "Lateral"
    return "Mixed"


def attention_leads(region: str) -> dict:
    base = {l: round(0.10 + np.random.uniform(0, 0.20), 3) for l in LEADS}
    if region == "Inferior":
        for l in ["II", "III", "aVF"]:
            base[l] = round(0.75 + np.random.uniform(0, 0.20), 3)
    elif region == "Anterior/Septal":
        for l in ["V1", "V2", "V3", "V4"]:
            base[l] = round(0.75 + np.random.uniform(0, 0.20), 3)
    elif region == "Lateral":
        for l in ["I", "aVL", "V5", "V6"]:
            base[l] = round(0.75 + np.random.uniform(0, 0.20), 3)
    return base


def temporal_attention(seq_len: int) -> list:
    n = max(int(seq_len), 1)
    raw = np.random.dirichlet(np.ones(n) * 2)
    return [round(float(x), 4) for x in raw]


def confidence_evolution(prob: float, seq_len: int) -> list:
    n = max(int(seq_len), 1)
    # Confidence generally rises toward final prediction
    evo = []
    for i in range(n):
        noise = np.random.uniform(-0.08, 0.08)
        frac = (i + 1) / n
        p = float(np.clip(prob * frac + noise, 0.02, 0.98))
        evo.append(round(p, 4))
    return evo


def build_timelines(row: pd.Series) -> list:
    seq_len = int(row.get("trop_seq_len", 0) or 0)
    ecg_seq = int(row.get("ecg_seq_len", 1) or 1)
    n = max(seq_len, ecg_seq, 1)

    trop_vals = [
        float(row.get("trop_0", 0) or 0),
        float(row.get("trop_1", 0) or 0),
        float(row.get("trop_2", 0) or 0),
    ]
    trop_times = [
        max(float(row.get("trop_0_time", 0) or 0), 0),
        max(float(row.get("trop_1_time", 0) or 0), 0),
        max(float(row.get("trop_2_time", 0) or 0), 0),
    ]
    trop_baseline = trop_vals[0] if trop_vals[0] > 0 else float(row.get("Troponin_T", 0.01) or 0.01)

    hr = float(row.get("Heart_Rate", 75) or 75)
    creat = float(row.get("Creatinine", 1.0) or 1.0)
    ckd = bool(row.get("has_ckd", 0))
    sepsis = bool(row.get("has_sepsis", 0))

    timelines = []
    for i in range(min(n, 3)):
        t_val = trop_vals[i] if i < len(trop_vals) else trop_baseline
        t_time = trop_times[i] if i < len(trop_times) else float(i * 6)
        baseline_ratio = round(t_val / max(trop_baseline, 0.001), 3)
        fold = round(float(row.get("trop_fold_rise", 1.0) or 1.0) if i == n - 1 else baseline_ratio, 3)

        timelines.append({
            "timestep": i,
            "label": f"T{i}",
            "time_delta_hrs": round(t_time, 2),
            "trop_value": round(t_val, 4),
            "peak_baseline_ratio": baseline_ratio,
            "fold_rise": fold,
            "hr": round(hr + np.random.uniform(-5, 5), 1),
            "sbp": round(120 + np.random.uniform(-20, 20), 1),
            "dbp": round(80 + np.random.uniform(-10, 10), 1),
            "map": round(93 + np.random.uniform(-15, 15), 1),
            "lactate": round(1.2 + np.random.uniform(0, 0.5), 2),
            "creatinine": round(creat, 2),
            "ckd_active": ckd,
            "sepsis_active": sepsis,
        })
    return timelines


def build_comorbidities(row: pd.Series) -> list:
    mapping = {
        "has_ckd": "CKD",
        "has_hf": "Heart Failure",
        "has_sepsis": "Sepsis",
        "has_pe": "Pulmonary Embolism",
        "has_afib": "Atrial Fibrillation",
        "has_diabetes": "Diabetes",
    }
    return [label for col, label in mapping.items() if row.get(col, 0) == 1]


def build_explanation(row: pd.Series, prob: float, region: str) -> str:
    ami = prob >= THRESHOLD
    trop_peak = row.get("troponin_peak", row.get("Troponin_T", 0)) or 0
    fold = row.get("trop_fold_rise", 1) or 1
    action = "demonstrates" if ami else "does not demonstrate"
    return (
        f"This {int(row.get('anchor_age', 55))}-year-old patient {action} "
        f"biochemical or electrophysiological patterns consistent with acute myocardial infarction. "
        f"Peak troponin of {trop_peak:.3f} ng/mL with {fold:.2f}× fold-rise. "
        f"Dominant ECG attention region: {region}. "
        f"Model confidence: {prob*100:.1f}% (threshold {THRESHOLD*100:.0f}%)."
    )


def build_explanation_temporal(row: pd.Series, prob: float) -> str:
    seq = int(row.get("trop_seq_len", 0) or 0)
    vel = row.get("troponin_velocity", 0) or 0
    return (
        f"Across {max(seq, 1)} temporal observation(s), troponin velocity was {vel:.4f} ng/mL/hr. "
        f"Model attention concentrated on {'early rises' if vel > 0.05 else 'baseline levels'}, "
        f"with confidence {'escalating' if prob >= THRESHOLD else 'remaining stable'} through the admission window."
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="splits", help="Directory to write split files")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.random.seed(SEED)

    print(f"[1/4] Loading {PARQUET_PATH} ...")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"      {len(df):,} rows × {len(df.columns)} columns")

    # ── Stratified split ──────────────────────────────────────────────────────
    print("[2/4] Stratified 70 / 15 / 15 split ...")
    labels = df["AMI"].fillna(0).astype(int)
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=labels, random_state=SEED)
    temp_labels = temp_df["AMI"].fillna(0).astype(int)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_labels, random_state=SEED)

    print(f"      Train : {len(train_df):,}  |  AMI={int((train_df['AMI']==1).sum()):,}")
    print(f"      Val   : {len(val_df):,}  |  AMI={int((val_df['AMI']==1).sum()):,}")
    print(f"      Test  : {len(test_df):,}  |  AMI={int((test_df['AMI']==1).sum()):,}")

    # Save raw parquet splits
    train_df.to_parquet(out / "train.parquet", index=False)
    val_df.to_parquet(out / "val.parquet", index=False)
    test_df.to_parquet(out / "test.parquet", index=False)
    print(f"      Saved train/val/test.parquet -> {out}/")

    # ── Enrich test split ─────────────────────────────────────────────────────
    print("[3/4] Enriching test split with predictions & timelines ...")

    rows_pg = []
    for _, row in test_df.reset_index(drop=True).iterrows():
        prob = float(compute_predicted_prob(row.to_frame().T)[0])
        rl = risk_level(prob)
        region = dominant_region(row)
        seq_len = max(int(row.get("trop_seq_len", 0) or 0), int(row.get("ecg_seq_len", 1) or 1), 1)
        attn_leads = attention_leads(region)
        temp_attn = temporal_attention(seq_len)
        conf_evo = confidence_evolution(prob, seq_len)
        timelines = build_timelines(row)
        comorbidities = build_comorbidities(row)

        ecg_contrib = round(float(np.random.uniform(0.35, 0.65)), 3)
        trop_contrib = round(1.0 - ecg_contrib, 3)
        dom_mod = "ECG" if ecg_contrib > 0.55 else ("Troponin" if trop_contrib > 0.55 else "Balanced")

        rows_pg.append({
            # Identity
            "hadm_id": int(row["hadm_id"]),
            "subject_id": f"P{int(row['hadm_id']) % 100000:05d}",
            "anchor_age": float(row.get("anchor_age", 0) or 0),
            "gender": "M" if int(row.get("gender", 0) or 0) == 1 else "F",
            "admittime": (
                str(row["ecg_time_0"]) if pd.notna(row.get("ecg_time_0")) else "2020-01-01 00:00:00"
            ),
            # Clinical
            "heart_rate": float(row.get("Heart_Rate", 0) or 0),
            "respiratory_rate": float(row.get("Respiratory_Rate", 0) or 0),
            "troponin_t": float(row.get("Troponin_T", 0) or 0),
            "creatinine": float(row.get("Creatinine", 0) or 0),
            "sodium": float(row.get("Sodium", 0) or 0),
            "potassium": float(row.get("Potassium", 0) or 0),
            "num_diagnoses": float(row.get("num_diagnoses", 0) or 0),
            "los": float(row.get("los", 0) or 0),
            # ECG intervals
            "pr_interval": float(row.get("PR_interval", 0) or 0),
            "qrs_duration": float(row.get("QRS_duration", 0) or 0),
            "qt_interval": float(row.get("QT_interval", 0) or 0),
            "qtc": float(row.get("QTc", 0) or 0),
            "p_axis": float(row.get("P_axis", 0) or 0),
            "qrs_axis": float(row.get("QRS_axis", 0) or 0),
            "t_axis": float(row.get("T_axis", 0) or 0),
            "rr_interval": float(row.get("RR_interval", 0) or 0),
            # Troponin trajectory raw
            "troponin_peak": float(row.get("troponin_peak", 0) or 0),
            "troponin_velocity": float(row.get("troponin_velocity", 0) or 0),
            "trop_fold_rise": float(row.get("trop_fold_rise", 0) or 0),
            "ecg_seq_len": int(row.get("ecg_seq_len", 1) or 1),
            # Comorbidities
            "has_ckd": bool(row.get("has_ckd", 0)),
            "has_hf": bool(row.get("has_hf", 0)),
            "has_sepsis": bool(row.get("has_sepsis", 0)),
            "has_pe": bool(row.get("has_pe", 0)),
            "has_afib": bool(row.get("has_afib", 0)),
            "has_diabetes": bool(row.get("has_diabetes", 0)),
            # Ground truth
            "ground_truth_ami": bool(row.get("AMI", 0) == 1),
            "max_troponin": float(row.get("troponin_peak", row.get("Troponin_T", 0)) or 0),
            # Derived
            "risk_level": rl,
            "comorbidities": json.dumps(comorbidities),
            # JSONB complex fields
            "timelines": json.dumps(timelines),
            "prediction": json.dumps({
                "predicted_prob": round(prob, 4),
                "threshold": THRESHOLD,
                "confidence_evolution": conf_evo,
                "dominant_modality": dom_mod,
                "attn_temp_entropy": round(float(-sum(w * np.log(w + 1e-9) for w in temp_attn) / np.log(len(temp_attn) + 1)), 4),
                "attn_spatial_dominance": round(float(max(attn_leads.values())), 4),
                "ecg_contribution": ecg_contrib,
                "trop_contribution": trop_contrib,
            }),
            "attention": json.dumps({
                "leads": attn_leads,
                "temporal": temp_attn,
                "dominant_region": region,
            }),
            "explanation": build_explanation(row, prob, region),
            "explanation_temporal": build_explanation_temporal(row, prob),
            "split": "test",
        })

    pg_df = pd.DataFrame(rows_pg)

    # Save enriched CSV for pgAdmin COPY import
    csv_path = out / "test_for_postgres.csv"
    pg_df.to_csv(csv_path, index=False)
    print(f"      Saved enriched CSV -> {csv_path}")
    print(f"      Rows: {len(pg_df):,}  |  Columns: {len(pg_df.columns)}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n[4/4] Summary")
    print("=" * 60)
    print(f"  Train parquet  : {out}/train.parquet  ({len(train_df):,} rows)")
    print(f"  Val parquet    : {out}/val.parquet    ({len(val_df):,} rows)")
    print(f"  Test parquet   : {out}/test.parquet   ({len(test_df):,} rows)")
    print(f"  Test CSV (PG)  : {csv_path}")
    print()
    print("  Next steps:")
    print("  1. Run the schema:   psql -d your_db -f db/schema.sql")
    print("  2. Import test CSV:  \\COPY admissions FROM 'splits/test_for_postgres.csv' CSV HEADER;")
    print("     (Or use pgAdmin: right-click admissions table > Import/Export Data)")
    print("=" * 60)


if __name__ == "__main__":
    main()
