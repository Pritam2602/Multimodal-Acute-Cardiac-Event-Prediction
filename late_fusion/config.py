# ==========================================
# CONFIGURATION - Late Fusion
# ==========================================

import os
from pathlib import Path

from Dataset.feature_schema import CLINICAL_FEATURES, NORMAL_RANGES


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "mimic_data" / "final_preprocessed_fusion_dataset.parquet"
MEMMAP_DIR = PROJECT_ROOT / "mimic_data" / "memmap"

ARTIFACT_ROOT = PROJECT_ROOT / "late_fusion" / "artifacts"
MODELS_DIR = ARTIFACT_ROOT / "models"
PLOTS_DIR = ARTIFACT_ROOT / "plots"
METRICS_DIR = ARTIFACT_ROOT / "metrics"
DB_PATH = ARTIFACT_ROOT / "predictions.db"

TARGET_COLUMN = "AMI"
NUM_CLINICAL_FEATURES = len(CLINICAL_FEATURES)

ECG_LEADS = 12
ECG_LENGTH = 5000

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 25
VAL_SPLIT = 0.2
TEST_SPLIT = 0.15
DROPOUT_RATE = 0.4
DEFAULT_THRESHOLD = 0.35
THRESHOLD_SEARCH_MIN = 0.10
THRESHOLD_SEARCH_MAX = 0.90
THRESHOLD_SEARCH_STEPS = 50
FOCAL_LOSS_ALPHA = 1.0
FOCAL_LOSS_GAMMA = 2.0

CPU_COUNT = os.cpu_count() or 4
NUM_WORKERS = min(8, max(2, CPU_COUNT // 2))
VAL_NUM_WORKERS = min(4, max(1, CPU_COUNT // 4))
PREFETCH_FACTOR = 2

SEED = 42

ECG_LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
