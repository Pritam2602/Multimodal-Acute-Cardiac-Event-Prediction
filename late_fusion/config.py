# ==========================================
# CONFIGURATION - Late Fusion
# ==========================================

import os
from pathlib import Path

from Dataset.feature_schema import CLINICAL_FEATURES, NORMAL_RANGES


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "mimic_data" / "final_preprocessed_fusion_dataset.parquet"
MEMMAP_DIR = PROJECT_ROOT / "mimic_data" / "memmap"
PREPROCESSED_ECG_DIR = PROJECT_ROOT / "mimic_data" / "ecg_preprocessed_2500_bp"
USE_PREPROCESSED_ECG = False
FORCE_FRESH_TRAIN = True

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
LEARNING_RATE = 7e-4
NUM_EPOCHS = 25
VAL_SPLIT = 0.2
TEST_SPLIT = 0.15
DROPOUT_RATE = 0.3
DEFAULT_THRESHOLD = 0.35
THRESHOLD_SEARCH_MIN = 0.10
THRESHOLD_SEARCH_MAX = 0.90
THRESHOLD_SEARCH_STEPS = 50
FOCAL_LOSS_ALPHA = 1.0
FOCAL_LOSS_GAMMA = 2.0
ECG_AUX_LOSS_WEIGHT = 0.3
CLINICAL_AUX_LOSS_WEIGHT = 0.1
ECG_BRANCH_LR_MULTIPLIER = 0.5
EARLY_STOPPING_PATIENCE = 5
WEIGHTED_SAMPLING_DEFAULT = True
HARD_NEGATIVE_WEIGHT = 1.5
HARD_NEGATIVE_POWER = 2.0
ECG_BRANCH_TYPE = "lead_aware_dilated_cnn_bilstm_transformer_tpp"
FUSION_TYPE = "gated_late_embedding_fusion"
ECG_CNN_FILTERS = 32
ECG_LSTM_HIDDEN_SIZE = 64
FUSION_HIDDEN_DIM = 8
TRANSFORMER_NUM_HEADS = 4
TRANSFORMER_NUM_LAYERS = 2
TRANSFORMER_FF_DIM = 256

CPU_COUNT = os.cpu_count() or 4
NUM_WORKERS = min(8, max(2, CPU_COUNT // 2))
VAL_NUM_WORKERS = min(4, max(1, CPU_COUNT // 4))
PREFETCH_FACTOR = 2

SEED = 42

ECG_LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
