# ==========================================
# CONFIGURATION — Early-Stage Feature Fusion
# ==========================================

import os
from pathlib import Path

from Dataset.feature_schema import CLINICAL_FEATURES, NORMAL_RANGES

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "mimic_data" / "final_preprocessed_fusion_dataset.parquet"

ARTIFACT_ROOT = PROJECT_ROOT / "early_fusion" / "artifacts"
MODELS_DIR    = ARTIFACT_ROOT / "models"
PLOTS_DIR     = ARTIFACT_ROOT / "plots"
METRICS_DIR   = ARTIFACT_ROOT / "metrics"
DB_PATH       = ARTIFACT_ROOT / "predictions.db"

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

