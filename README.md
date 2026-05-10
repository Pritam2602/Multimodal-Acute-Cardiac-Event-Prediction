# 🫀 Multimodal Acute Cardiac Event Prediction

> A deep learning system for predicting **Acute Myocardial Infarction (AMI / NSTEMI)** using 12-lead ECG waveforms and structured clinical data from the **MIMIC-IV** database.

---

## 🏆 Best Results

| Metric | Score | Architecture | Dataset |
|---|---|---|---|
| **F1 Score** | **0.7753** | Early Fusion (Curated) | Phase 10 |
| **AUC-ROC** | **0.9418** | Early Fusion (Curated) | Phase 10 |
| **Precision** | **0.814** | Early Fusion (Curated) | Phase 10 |
| **Recall** | **0.729** | Early Fusion (Curated) | Phase 10 |

---

## 📁 Repository Structure

```
.
├── Dataset_preproceesing/          # All data preprocessing scripts
│   ├── dataset_preprocessing.py    # Main preprocessing pipeline
│   ├── refine_temporal_cohort.py   # Label quality & trajectory filtering
│   ├── build_rich_trajectory.py    # Rich trajectory builder
│   ├── extract_serial_troponin.py  # Serial troponin extraction
│   ├── extract_temporal_ecg_seq.py # Temporal ECG sequence extraction
│   ├── pair_baseline_ecg.py        # Baseline ECG pairing
│   └── merge_*.py                  # Data merging utilities
│
├── early_fusion/                   # Early Fusion architecture
│   ├── config.py                   # Feature lists, hyperparameters, paths
│   ├── model.py                    # Base CNN-BiLSTM feature extractor
│   ├── lead_aware_model.py         # Anatomical Regional Attention module
│   ├── temporal_model.py           # Temporal GRU with attention
│   ├── temporal_dataset.py         # Temporal trajectory data loader
│   ├── temporal_engine.py          # Training/evaluation engine
│   ├── temporal_train.py           # Training entry point
│   ├── losses.py                   # Focal Loss, OHEM Focal Loss
│   ├── dataset.py                  # Static (single-snapshot) data loader
│   ├── engine.py                   # Static training engine
│   ├── train.py                    # Static training entry point
│   └── artifacts/runs/             # All experiment results
│       ├── phase10_early_fusion_curated/  # 🏆 Best model
│       ├── phase9_refined_cohort/
│       ├── phase8_late_fusion/
│       └── ... (41 total experiment runs)
│
├── late_fusion/                    # Late Fusion architecture
│   ├── temporal_model.py           # Hybrid Late Fusion (ECG + Clinical branches)
│   ├── temporal_engine.py          # Branch-disentangled training engine
│   ├── temporal_train.py           # Training entry point
│   ├── model.py                    # Base late fusion model
│   └── config.py, dataset.py, ...  # Shared utilities
│
├── mimic_data/                     # Data directory (not tracked)
│   ├── refined_temporal_fusion_dataset.parquet
│   └── ecg_cache/                  # Preprocessed ECG memmap
│
├── METHODOLOGY_REPORT.md           # Detailed methodology documentation
├── AMI_Prediction_Final_Report.docx # Comprehensive final report with plots
├── generate_report.py              # Script to regenerate the DOCX report
├── api.py                          # REST API for model serving
├── explain.py                      # Clinical feature attribution
└── requirements.txt                # Python dependencies
```

---

## 🚀 Quickstart: How to Use This Repo

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended, 8 GB+ VRAM)
- Access to MIMIC-IV and MIMIC-IV-ECG datasets (requires PhysioNet credentialed access)

### Step 1: Clone & Setup Environment

```bash
git clone https://github.com/Pritam2602/Multimodal-Acute-Cardiac-Event-Prediction.git
cd Multimodal-Acute-Cardiac-Event-Prediction

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### Step 2: Prepare the MIMIC-IV Data

You need these files from MIMIC-IV placed in `mimic_data/`:

| File | Source |
|---|---|
| `diagnoses_icd.csv.gz` | MIMIC-IV Hosp |
| `d_icd_diagnoses.csv.gz` | MIMIC-IV Hosp |
| `admissions.csv.gz` | MIMIC-IV Hosp |
| `labevents.csv.gz` | MIMIC-IV Hosp |
| `machine_measurements.csv` | MIMIC-IV-ECG |
| ECG waveform files (`.dat`, `.hea`) | MIMIC-IV-ECG |

### Step 3: Run the Preprocessing Pipeline

Execute the scripts **in order**:

```bash
# 1. Main preprocessing — creates the base fusion dataset
python Dataset_preproceesing/dataset_preprocessing.py

# 2. Extract serial troponin trajectories
python Dataset_preproceesing/extract_serial_troponin.py
python Dataset_preproceesing/merge_serial_troponin.py

# 3. Build temporal ECG + troponin sequences
python Dataset_preproceesing/extract_temporal_ecg_seq.py
python Dataset_preproceesing/extract_temporal_troponin_seq.py
python Dataset_preproceesing/merge_temporal_troponin.py

# 4. Pair baseline ECGs and build rich trajectories
python Dataset_preproceesing/pair_baseline_ecg.py
python Dataset_preproceesing/build_rich_trajectory.py

# 5. Refine the cohort (label quality + trajectory richness filtering)
python Dataset_preproceesing/refine_temporal_cohort.py
```

After this, you will have `mimic_data/refined_temporal_fusion_dataset.parquet`.

### Step 4: Train a Model

**Early Fusion (recommended — best results):**

```bash
python -m early_fusion.temporal_train \
    --run-name my_experiment \
    --batch-size 16 \
    --lr 5e-5 \
    --epochs 25
```

**Late Fusion:**

```bash
python -m late_fusion.temporal_train \
    --run-name my_late_fusion_experiment \
    --batch-size 16 \
    --lr 5e-5 \
    --epochs 25
```

### Step 5: View Results

Training artifacts are saved to `early_fusion/artifacts/runs/<run-name>/`:

```
<run-name>/
├── models/
│   ├── best_model.pth                  # Best F1 checkpoint
│   └── best_physiological_model.pth    # Best biologically-valid checkpoint
├── metrics/
│   ├── metrics.json                    # Best-epoch metrics
│   └── history.json                    # Epoch-by-epoch history
└── plots/
    └── best_val_confusion_matrix.png   # Confusion matrix
```

### Step 6: Generate the Final Report

```bash
python generate_report.py
```

This produces `AMI_Prediction_Final_Report.docx` with all metrics, plots, and analysis.

---

## 🧪 Experiment Roadmap (10 Phases)

The project evolved through 10 experimental phases. Each phase is fully documented in `METHODOLOGY_REPORT.md`.

```
Phase 1   Baseline CNN-BiLSTM + Focal Loss           → F1: 0.640
Phase 2   Cross-Attention & FiLM Conditioning         → F1: 0.665
Phase 3   Contrastive Learning & Entropy Reg.         → F1: 0.670
Phase 4   Siamese Temporal Architecture               → F1: 0.668
Phase 5   Temporal GRU + Lead-Delta Attention         → F1: 0.690
Phase 6   Entropy Band Regularization                 → F1: 0.695
Phase 7   Anatomical Regional Attention               → F1: 0.702, AUC: 0.929
Phase 8   Hybrid Late Fusion (Modality Disentangle)   → F1: 0.700, AUC: 0.933
─── Data Curation Pivot ───────────────────────────────────────────
Phase 9   Late Fusion + Curated Cohort                → F1: 0.773, AUC: 0.941
Phase 10  Early Fusion + Curated Cohort               → F1: 0.775, AUC: 0.942
```

### Key Scientific Finding

> The 0.70 F1 ceiling was **not** an architectural limitation — it was caused by **clinical label noise** in MIMIC-IV. By removing 1,841 ambiguous admissions (weak positives, weak negatives, temporally clustered ECGs, extreme confounders), both architectures immediately jumped to F1 > 0.77.

---

## 🏗️ Architecture Overview

### Early Fusion (Temporal)

```
12-Lead ECG Seq (B, T, 12, 5000)
    ↓
LeadAwareHybridExtractor (per-timestep)
    ↓
AnatomicalRegionalAttention (Inferior/Anterior/Lateral/aVR)
    ↓
Cross-Timestep Delta Embeddings
    ↓
GRU Sequence Model → Temporal Attention
    ↓                                    Clinical Features (B, 77)
    ↓                                         ↓
    ↓                                    Clinical Encoder
    ↓                                         ↓
    └──────── Concatenation ──────────────────┘
                    ↓
            Fusion Classifier → AMI Prediction
```

### Late Fusion (Temporal Hybrid)

```
ECG Branch                          Clinical Branch
    ↓                                    ↓
LeadAwareHybridExtractor            Clinical MLP
    ↓                                    ↓
AnatomicalRegionalAttention         Clinical GRU
    ↓                                    ↓
ECG GRU + Temporal Attention        Temporal Attention
    ↓                                    ↓
ECG Aux Logit                       Clinical Aux Logit
    ↓                                    ↓
    └──── Cross-Modal Sigmoid Gate ──────┘
                    ↓
            Late Fusion Classifier → AMI Prediction
```

---

## 📊 Clinical Features Used (77 total)

- **Demographics**: age, gender
- **Vitals**: heart rate, respiratory rate
- **Lab Values**: Troponin-T (raw, log, binary flags), Creatinine, Sodium, Potassium
- **Serial Troponin Dynamics**: initial, peak, delta, velocity, slope, acceleration, fold-rise
- **ECG Machine Measurements**: PR, QRS, QT, QTc, axes, RR interval
- **Comorbidity Flags**: CKD, heart failure, sepsis, PE, AFib, diabetes
- **Temporal Context**: hours to ECG, hours to troponin, ECG timing flags
- **Missing/Invalid Indicators**: 15 binary flags for data quality

---

## 🔧 Training Flags

| Flag | Description | Default |
|---|---|---|
| `--run-name` | Experiment name (required) | — |
| `--epochs` | Max training epochs | 25 |
| `--batch-size` | Batch size | 32 |
| `--lr` | Learning rate | 1e-4 |
| `--contrastive-lambda` | Contrastive loss weight | 0.1 |
| `--entropy-lambda` | Entropy regularization weight | 0.001 |

---

## 📄 Reports & Documentation

| Document | Description |
|---|---|
| `METHODOLOGY_REPORT.md` | Full methodology with all 10 phases |
| `AMI_Prediction_Final_Report.docx` | Comprehensive report with metrics & plots |
| `PHASE4_TEMPORAL_ROADMAP.md` | Temporal architecture design notes |
| `PHASE5_TEMPORAL_SEQUENCE_ROADMAP.md` | Sequence modeling roadmap |
| `architecture_summary.md` | High-level architecture overview |
| `walkthrough.md` | Code walkthrough |

---

## 📜 License

This project is licensed under the terms in the [LICENSE](LICENSE) file.

---

## 🙏 Acknowledgments

- **MIMIC-IV** and **MIMIC-IV-ECG** datasets from PhysioNet
- Built with PyTorch, NumPy, Pandas, scikit-learn
