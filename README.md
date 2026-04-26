# Multimodal Acute Myocardial Infarction Prediction

This repository predicts Acute Myocardial Infarction (AMI) from two modalities:

- raw 12-lead ECG waveforms
- structured clinical EHR features

The codebase now supports two training pipelines:

- `early_fusion/`: injects clinical context into the ECG stream before shared temporal modeling
- `late_fusion/`: learns ECG and clinical embeddings separately, then fuses the embeddings plus branch logits

## Repository Layout

```text
.
|-- README.md
|-- architecture_summary.md
|-- walkthrough.md
|-- PROJECT_FILE_DESCRIPTIONS.md
|-- api.py
|-- explain.py
|-- store_predictions.py
|-- early_fusion/
|   |-- config.py
|   |-- dataset.py
|   |-- model.py
|   |-- losses.py
|   |-- engine.py
|   |-- train.py
|   |-- cross_validate.py
|   |-- plots.py
|   `-- artifacts/
`-- late_fusion/
    |-- config.py
    |-- dataset.py
    |-- model.py
    |-- losses.py
    |-- engine.py
    |-- train.py
    |-- cross_validate.py
    |-- plots.py
    `-- artifacts/
```

## Data and Modeling Summary

The project uses:

- ECG input shape `(12, 5000)`
- 24 structured clinical features
- AMI as a binary target
- PyTorch for model training
- validation-threshold tuning for F1-aware classification

Shared modeling ideas:

- Conv1D layers for local ECG morphology
- BiLSTM sequence modeling for ECG time structure
- class-imbalance handling through weighted BCE or focal loss
- optional weighted sampling in the training loader

## Early Fusion

The early-fusion model is implemented in [early_fusion/model.py](/abs/path/d:/MINI_PROJECT/early_fusion/model.py:1).

High-level flow:

```text
Clinical features
-> projector
-> broadcast across ECG time axis

ECG waveform
-> concatenate with projected clinical channels
-> shared Conv1D stack
-> shared BiLSTM
-> classifier
-> AMI logit
```

Training entrypoint:

```powershell
python -m early_fusion.train
```

Cross-validation entrypoint:

```powershell
python -m early_fusion.cross_validate --run-name crossval
```

Useful training flags:

- `--weighted-sampling`
- `--loss-name {focal,bce}`
- `--dropout`
- `--epochs`
- `--learning-rate`
- `--weight-decay`
- `--run-name`
- `--auto-continue`

## Late Fusion

The late-fusion model is implemented in [late_fusion/model.py](/abs/path/d:/MINI_PROJECT/late_fusion/model.py:1).

Current high-level flow:

```text
ECG branch
-> Conv1D stack
-> BiLSTM
-> attention pooling
-> ECG embedding
-> ECG branch logit

Clinical branch
-> MLP
-> clinical embedding
-> clinical branch logit

Fusion head
-> concatenate ECG embedding, clinical embedding, ECG logit, clinical logit
-> MLP fusion head
-> final AMI logit
```

This is stronger than the older two-scalar-logit fusion design because the final head sees both learned embeddings and branch-level decision signals.

Training entrypoint:

```powershell
python -m late_fusion.train
```

Cross-validation entrypoint:

```powershell
python -m late_fusion.cross_validate --run-name crossval
```

Useful training flags:

- `--weighted-sampling`
- `--loss-name {focal,bce}`
- `--dropout`
- `--epochs`
- `--learning-rate`
- `--run-name`
- `--auto-continue`

## Artifacts

Single training runs save artifacts under each model package:

- `artifacts/models/`
- `artifacts/metrics/`
- `artifacts/plots/`

Named runs use:

- `artifacts/runs/<run-name>/models/`
- `artifacts/runs/<run-name>/metrics/`
- `artifacts/runs/<run-name>/plots/`

Typical outputs include:

- best-model weights
- latest checkpoint
- metrics JSON
- comparison CSV
- training and evaluation plots

## Example Commands

Train early fusion with weighted sampling:

```powershell
python -m early_fusion.train --run-name ef_ws --weighted-sampling --auto-continue
```

Train late fusion with BCE and custom dropout:

```powershell
python -m late_fusion.train --run-name lf_bce --loss-name bce --dropout 0.3 --auto-continue
```

Cross-validate both models:

```powershell
python -m early_fusion.cross_validate --run-name ef_cv --weighted-sampling
python -m late_fusion.cross_validate --run-name lf_cv --weighted-sampling
```

## Evaluation and Serving

The repo also contains:

- [store_predictions.py](/abs/path/d:/MINI_PROJECT/store_predictions.py:1) for evaluating registered models and writing patient-level outputs into SQLite
- [explain.py](/abs/path/d:/MINI_PROJECT/explain.py:1) for clinical-feature attribution and reasoning text
- [api.py](/abs/path/d:/MINI_PROJECT/api.py:1) for serving dashboard-friendly endpoints

Run the API with:

```powershell
python api.py
```

## Notes

- Full training was not automatically run as part of recent code edits.
- Existing artifact files may reflect earlier experiments rather than the latest code path.
- Some support scripts still assume early-fusion-centered artifact locations, so model-comparison and serving flows should be checked after major architecture changes.
