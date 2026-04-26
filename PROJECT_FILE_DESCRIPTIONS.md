# Project File Descriptions

Updated on 2026-04-26.

This document gives a current, high-level description of the main files in the repository without trying to mirror every implementation detail line by line.

## Root-Level Files

### [README.md](/abs/path/d:/MINI_PROJECT/README.md:1)

High-level project introduction, training entrypoints, artifact layout, and example commands.

### [architecture_summary.md](/abs/path/d:/MINI_PROJECT/architecture_summary.md:1)

Main architecture reference for the early-fusion and late-fusion models. It explains where fusion happens and how the two designs differ.

### [walkthrough.md](/abs/path/d:/MINI_PROJECT/walkthrough.md:1)

Operational guide for training, cross-validation, artifact generation, prediction storage, and API serving.

### [api.py](/abs/path/d:/MINI_PROJECT/api.py:1)

FastAPI service layer for the dashboard and any other client consuming prediction results, waveform data, metrics, and explanations.

### [explain.py](/abs/path/d:/MINI_PROJECT/explain.py:1)

Shared explainability utilities focused on clinical-feature attribution and human-readable reasoning output.

### [store_predictions.py](/abs/path/d:/MINI_PROJECT/store_predictions.py:1)

Evaluates registered models on a held-out test set and writes patient-level outputs, metrics, and explanations into SQLite for serving.

### `requirements.txt`

Pinned Python dependencies for data processing, modeling, plotting, and API serving.

## Early Fusion Package

### [early_fusion/config.py](/abs/path/d:/MINI_PROJECT/early_fusion/config.py:1)

Defines paths, feature lists, model constants, default hyperparameters, threshold-search settings, and artifact directories for the early-fusion pipeline.

### [early_fusion/dataset.py](/abs/path/d:/MINI_PROJECT/early_fusion/dataset.py:1)

Prepares the multimodal dataset, handles ECG loading and caching, creates train/validation/test splits, standardizes clinical features, and builds DataLoaders. The current code also supports weighted sampling in the training loader.

### [early_fusion/model.py](/abs/path/d:/MINI_PROJECT/early_fusion/model.py:1)

Implements the early-fusion network, where projected clinical features are injected into the ECG stream before the shared Conv1D and BiLSTM backbone finishes feature extraction.

### [early_fusion/losses.py](/abs/path/d:/MINI_PROJECT/early_fusion/losses.py:1)

Loss-building utilities for focal loss and BCE-style training choices.

### [early_fusion/engine.py](/abs/path/d:/MINI_PROJECT/early_fusion/engine.py:1)

Reusable training and evaluation helpers, including threshold search and metric computation.

### [early_fusion/train.py](/abs/path/d:/MINI_PROJECT/early_fusion/train.py:1)

Single-run training entrypoint. Supports resumable checkpoints, named run directories, weighted sampling, configurable loss choice, dropout, epoch count, and final artifact export.

### [early_fusion/cross_validate.py](/abs/path/d:/MINI_PROJECT/early_fusion/cross_validate.py:1)

Cross-validation entrypoint for early fusion. Runs stratified folds over the train pool while keeping a fixed held-out test split outside the fold loop. Writes fold summaries to CSV and JSON.

### [early_fusion/plots.py](/abs/path/d:/MINI_PROJECT/early_fusion/plots.py:1)

Plotting utilities for loss curves, metric curves, confusion matrix, ROC curve, PR curve, and tabular comparison images.

### `early_fusion/artifacts/`

Stores training outputs such as model weights, resumable checkpoints, metrics files, plots, and named experiment runs.

## Late Fusion Package

### [late_fusion/config.py](/abs/path/d:/MINI_PROJECT/late_fusion/config.py:1)

Defines late-fusion paths, feature lists, model constants, default hyperparameters, and artifact directories.

### [late_fusion/dataset.py](/abs/path/d:/MINI_PROJECT/late_fusion/dataset.py:1)

Prepares the late-fusion training data and provides reusable split preparation plus fold DataLoader helpers for cross-validation. Also supports weighted sampling.

### [late_fusion/model.py](/abs/path/d:/MINI_PROJECT/late_fusion/model.py:1)

Implements the current late-fusion architecture. ECG and clinical branches each produce a learned embedding and a branch logit, and the final head fuses both embeddings plus both logits.

### [late_fusion/losses.py](/abs/path/d:/MINI_PROJECT/late_fusion/losses.py:1)

Loss-building utilities used by the late-fusion training and cross-validation paths.

### [late_fusion/engine.py](/abs/path/d:/MINI_PROJECT/late_fusion/engine.py:1)

Reusable training and evaluation logic for late fusion, including metric calculation and threshold tuning.

### [late_fusion/train.py](/abs/path/d:/MINI_PROJECT/late_fusion/train.py:1)

Single-run late-fusion training entrypoint. Supports resumable checkpoints plus user-configurable weighted sampling, loss choice, dropout, epoch count, and run naming.

### [late_fusion/cross_validate.py](/abs/path/d:/MINI_PROJECT/late_fusion/cross_validate.py:1)

Cross-validation entrypoint for late fusion, mirroring the early-fusion evaluation style for cleaner comparison.

### [late_fusion/plots.py](/abs/path/d:/MINI_PROJECT/late_fusion/plots.py:1)

Plotting utilities parallel to the early-fusion plotting layer.

### `late_fusion/artifacts/`

Stores late-fusion model weights, checkpoints, metrics, plots, and named experiment outputs.

## Data-Preparation Scripts

The `Dataset/` directory contains preprocessing scripts that:

- extract temporally valid clinical features before each ECG timestamp
- derive ECG machine-measurement features
- combine ECG metadata with clinical data
- create the final fused parquet dataset used by the model packages

These scripts prepare the data source consumed by both fusion pipelines.

## Generated Outputs Worth Knowing

Common artifact types across fusion packages:

- `models/*.pth`
- `metrics/metrics.json`
- `metrics/comparison_table.csv`
- `metrics/cross_validation.json`
- `metrics/cross_validation.csv`
- `plots/*.png`
- `artifacts/runs/<run-name>/...`

## Maintenance Notes

- The fusion-model docs should be updated together whenever the architecture or CLI surface changes.
- Artifact examples in the repo may represent earlier experiments rather than the latest code path.
- Supporting infrastructure such as `store_predictions.py` should be checked whenever model interfaces or artifact conventions change.
