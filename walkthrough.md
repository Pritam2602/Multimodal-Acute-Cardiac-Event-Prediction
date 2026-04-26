# Walkthrough: Training, Validation, Comparison, and Serving

This file explains the current end-to-end workflow in the repository.

For model design details, see [architecture_summary.md](/abs/path/d:/MINI_PROJECT/architecture_summary.md:1).

## What the Project Does

The system predicts Acute Myocardial Infarction from:

- raw 12-lead ECG signals
- structured clinical features derived from EHR data

There are now two active modeling paths:

- early fusion
- late fusion

## Typical Workflow

```text
Prepare fused dataset
-> train or cross-validate a fusion model
-> save metrics and artifacts
-> evaluate registered models with store_predictions.py
-> write patient-level results to SQLite
-> serve the results with api.py
```

## Single-Run Training

### Early fusion

```powershell
python -m early_fusion.train --auto-continue
```

Common options:

- `--weighted-sampling`
- `--loss-name bce`
- `--dropout 0.3`
- `--epochs 25`
- `--run-name my_run`

Artifacts are written to either:

- `early_fusion/artifacts/...`
- `early_fusion/artifacts/runs/<run-name>/...`

### Late fusion

```powershell
python -m late_fusion.train --auto-continue
```

Common options:

- `--weighted-sampling`
- `--loss-name bce`
- `--dropout 0.4`
- `--epochs 25`
- `--run-name my_run`

Artifacts are written to either:

- `late_fusion/artifacts/...`
- `late_fusion/artifacts/runs/<run-name>/...`

## Cross-Validation

### Early fusion CV

```powershell
python -m early_fusion.cross_validate --run-name ef_cv --weighted-sampling
```

### Late fusion CV

```powershell
python -m late_fusion.cross_validate --run-name lf_cv --weighted-sampling
```

Both CV scripts:

- keep a fixed held-out test split outside the fold loop
- run stratified folds on the remaining train pool
- save fold metrics to CSV and JSON under the run metrics directory

## Artifact Expectations

Single training runs save:

- best model weights
- latest checkpoint
- metrics JSON
- comparison CSV
- training plots

Cross-validation runs save:

- `cross_validation.csv`
- `cross_validation.json`

Named experiments are useful when comparing:

- weighted sampling on vs off
- focal loss vs BCE
- different dropout values
- different epoch budgets

## Model Comparison and Prediction Storage

[store_predictions.py](/abs/path/d:/MINI_PROJECT/store_predictions.py:1) evaluates registered models on the same held-out test set and writes outputs to SQLite.

Current responsibilities:

- load registered model weights
- evaluate them with their saved threshold when available
- store patient predictions
- store per-model metrics
- store feature-importance and explanation rows

Run it with:

```powershell
python store_predictions.py
```

Quick subset:

```powershell
python store_predictions.py --subset 500
```

## Explainability

[explain.py](/abs/path/d:/MINI_PROJECT/explain.py:1) computes clinical-feature attribution and converts it into dashboard-friendly reasoning text.

The explainability flow is currently centered on:

- clinical contribution ranking
- natural-language summaries
- patient-level explanation records

## API and Dashboard Flow

[api.py](/abs/path/d:/MINI_PROJECT/api.py:1) serves the SQLite outputs through FastAPI.

Start the API:

```powershell
python api.py
```

Swagger UI:

```text
http://localhost:5000/docs
```

Main endpoint groups:

- patients
- ECG waveform retrieval
- explainability insights
- aggregate metrics
- comparison outputs

## Practical Notes

- Single-run training is resumable because checkpoints are saved after every epoch.
- Cross-validation is intended for cleaner experiment comparison than one-off runs.
- Existing artifact files may belong to older checkpoints or earlier architectures.
- If a saved checkpoint no longer matches the current model definition, the training scripts are designed to fall back to a fresh run.
