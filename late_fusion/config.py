# ==========================================
# CONFIGURATION — Late Fusion
# ==========================================

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "mimic_data" / "final_preprocessed_fusion_dataset.parquet"

ARTIFACT_ROOT = PROJECT_ROOT / "late_fusion" / "artifacts"
MODELS_DIR    = ARTIFACT_ROOT / "models"
PLOTS_DIR     = ARTIFACT_ROOT / "plots"
METRICS_DIR   = ARTIFACT_ROOT / "metrics"
DB_PATH       = ARTIFACT_ROOT / "predictions.db"

# ── Clinical feature columns (from preprocessed parquet) ─────────────────────
CLINICAL_FEATURES = [
    # Demographics
    "anchor_age",
    "gender",
    # Vital signs
    "Heart_Rate",
    "Respiratory_Rate",
    # Lab values
    "Troponin_T",
    "log1p_Troponin_T",
    "Troponin_T_positive",
    "Troponin_T_high",
    "age_x_log1p_Troponin_T",
    "Creatinine",
    "Creatinine_high",
    "Sodium",
    "Potassium",
    "Potassium_low",
    "Potassium_high",
    # Hospital stay info
    "num_diagnoses",
    "los",
    # ECG machine measurements (tabular, not raw signal)
    "PR_interval",
    "QRS_duration",
    "QT_interval",
    "QTc",
    "P_axis",
    "QRS_axis",
    "T_axis",
    "RR_interval",
    "HR_from_RR_interval",
    "HR_RR_disagreement",
    # ECG-machine abnormality flags
    "QRS_wide",
    "QTc_prolonged",
    "PR_prolonged",
    "QRS_axis_deviation",
    "T_axis_abnormal",
    # Missingness indicators
    "Troponin_T_missing",
    "Creatinine_missing",
    "Sodium_missing",
    "Potassium_missing",
    "Heart_Rate_missing",
    "Respiratory_Rate_missing",
    # Invalid ECG-machine indicators
    "PR_interval_invalid",
    "QRS_duration_invalid",
    "QT_interval_invalid",
    "QTc_invalid",
    "P_axis_invalid",
    "QRS_axis_invalid",
    "T_axis_invalid",
    "RR_interval_invalid",
]

TARGET_COLUMN = "AMI"

NUM_CLINICAL_FEATURES = len(CLINICAL_FEATURES)

# ── ECG signal settings ──────────────────────────────────────────────────────
ECG_LEADS    = 12
ECG_LENGTH   = 5000  # samples per lead

# ── Training hyper-parameters ─────────────────────────────────────────────────
BATCH_SIZE    = 64
LEARNING_RATE = 3e-4
NUM_EPOCHS    = 25
VAL_SPLIT     = 0.2
TEST_SPLIT    = 0.15   # held-out test set for frontend / evaluation
DROPOUT_RATE  = 0.4
DEFAULT_THRESHOLD = 0.35
THRESHOLD_SEARCH_MIN = 0.10
THRESHOLD_SEARCH_MAX = 0.90
THRESHOLD_SEARCH_STEPS = 50
FOCAL_LOSS_ALPHA = 1.0
FOCAL_LOSS_GAMMA = 2.0
# Windows DataLoader workers use shared file mappings for batch collation.
# Large ECG tensors can exhaust paging file / shared-memory limits there, so
# keep train loading modest and validation even lighter by default.
NUM_WORKERS   = 2 if os.name == "nt" else 12
VAL_NUM_WORKERS = 0 if os.name == "nt" else 4
PREFETCH_FACTOR = 1 if os.name == "nt" else 2

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42

# ── ECG lead names (standard 12-lead order) ──────────────────────────────────
ECG_LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# ── Clinical NORMAL ranges (for UI / explainability ONLY) ────────────────────
# ⚠️ These are NOT used for preprocessing or filtering!
#    Preprocessing uses VALID_RANGES in dataset.py (wide physiological ranges).
#    These narrow ranges are ONLY for the frontend to highlight abnormal values
#    and for the explainability module to generate reasoning text.
# Format: feature_name → (min, max, unit)
NORMAL_RANGES = {
    "anchor_age":        (18,  90,   "years"),
    "Heart_Rate":        (60,  100,  "bpm"),
    "Respiratory_Rate":  (12,  20,   "breaths/min"),
    "Troponin_T":        (0,   0.04, "ng/mL"),
    "Creatinine":        (0.6, 1.2,  "mg/dL"),
    "Sodium":            (136, 145,  "mEq/L"),
    "Potassium":         (3.5, 5.0,  "mEq/L"),
    "PR_interval":       (120, 200,  "ms"),
    "QRS_duration":      (60,  120,  "ms"),
    "QT_interval":       (350, 450,  "ms"),
    "QTc":               (350, 460,  "ms"),
    "P_axis":            (0,   75,   "°"),
    "QRS_axis":          (-30, 90,   "°"),
    "T_axis":            (15,  75,   "°"),
    "RR_interval":       (600, 1000, "ms"),
}
