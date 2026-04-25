# Multimodal AMI Prediction - Architecture Summary

## Overview

This project predicts **Acute Myocardial Infarction (AMI)** using two complementary modalities:

- **12-lead ECG waveforms** with shape `(12, 5000)`
- **Clinical EHR features** with 24 structured inputs

The project is organized around a fair comparison between two multimodal strategies:

- **Early fusion**: merge modalities before shared temporal feature learning is complete
- **Late fusion**: keep modality branches separate and combine them only near the final decision

This document is the main architecture reference for the repository.

---

## 1. Early Fusion

File: `early_fusion/model.py`

### Current idea

The early-fusion model injects clinical information into the ECG processing stream **before** the shared CNN and BiLSTM finish learning the multimodal representation.

### Architecture

```text
Clinical Features (24,)
    -> Linear(24 -> 64) -> ReLU -> Dropout
    -> Linear(64 -> clinical_channels) -> ReLU
    -> Expand across ECG time axis

Raw ECG (12, 5000)
    -> Concatenate with projected clinical channels
    -> Shared Conv1D(12+clinical_channels -> 32) -> BN -> ReLU -> MaxPool
    -> Shared Conv1D(32 -> 64) -> BN -> ReLU -> MaxPool
    -> Shared Conv1D(64 -> 128) -> BN -> ReLU -> MaxPool
    -> BiLSTM(hidden=64, bidirectional=True)
    -> Dense(128 -> 256) -> ReLU -> Dropout
    -> Dense(256 -> 128) -> ReLU
    -> Classifier(128 -> 64 -> 1)
```

### Conceptual pipeline

```text
Clinical Features -> Projector -> Temporal Clinical Channels --\
                                                                +--> Shared CNN -> BiLSTM -> Classifier -> AMI
Raw ECG --------------------------------------------------------/
```

### Why this is early fusion

Because the model combines ECG and clinical information **before shared temporal modeling is finished**. The BiLSTM operates on a representation that already contains both modalities.

### Key design points

| Aspect | Current early-fusion choice |
|---|---|
| ECG input | Raw ECG plus projected clinical channels |
| Clinical handling | Projected early into temporal context channels |
| Shared encoder | Conv1D + BiLSTM |
| Fusion point | Before shared temporal feature extraction finishes |
| Main benefit | Learns cross-modal interactions early |

---

## 2. Late Fusion

File: `late_fusion/model.py`

### Current idea

The late-fusion model keeps ECG and clinical processing separate for most of the network. Each branch learns independently, and fusion happens only near the final prediction stage.

### Architecture

```text
ECG Branch
    Raw ECG (12, 5000)
    -> Conv1D(12 -> 32) -> BN -> ReLU -> MaxPool
    -> Conv1D(32 -> 64) -> BN -> ReLU -> MaxPool
    -> Conv1D(64 -> 128) -> BN -> ReLU -> MaxPool
    -> BiLSTM(hidden=64, bidirectional=True)
    -> Dense(128 -> 128) -> ReLU -> Dropout
    -> ECG head -> ECG logit

Clinical Branch
    Clinical Features (24,)
    -> Linear(24 -> 128) -> ReLU -> Dropout
    -> Linear(128 -> 64) -> ReLU
    -> Linear(64 -> 32) -> ReLU
    -> Clinical head -> Clinical logit

Decision Fusion
    [ECG logit, Clinical logit]
    -> Fusion head
    -> Final AMI logit
```

### Conceptual pipeline

```text
ECG -> ECG CNN -> BiLSTM -> ECG Logit --\
                                         +--> Fusion Head -> Final AMI Prediction
Clinical -> Clinical Encoder -> Logit --/
```

### Why this is late fusion

Because the ECG branch and clinical branch each learn and score the sample independently first. The model combines them only at the decision stage.

### Key design points

| Aspect | Proposed late-fusion choice |
|---|---|
| ECG input | Raw ECG only |
| Clinical handling | Independent tabular encoder |
| ECG encoder | Conv1D + BiLSTM |
| Fusion point | After branch-level logits |
| Main benefit | Preserves modality-specific reasoning |

---

## 3. Early Fusion vs Late Fusion

### Core difference

**Early fusion**

```text
Merge modalities first -> shared multimodal feature learning -> one classifier
```

**Late fusion**

```text
Learn each modality separately -> make branch-level decisions -> combine at the end
```

### Comparison table

| Aspect | Early Fusion | Late Fusion |
|---|---|---|
| Fusion point | Before shared feature extraction is complete | After branch-level prediction signals |
| ECG input | ECG plus projected clinical channels | ECG only |
| Clinical input | Injected early into sequence representation | Kept in its own MLP branch |
| ECG encoder | Shared Conv1D + BiLSTM on fused input | Conv1D + BiLSTM on ECG only |
| Clinical encoder | Small projector only | Full independent encoder |
| Joint learning | Happens early | Happens near the output |
| Interpretability | Harder to isolate modality effect | Easier to inspect branch contributions |
| Research question | Does joint multimodal temporal learning help? | Does independent reasoning plus decision fusion help? |

### Why BiLSTM is used in both

BiLSTM is useful because ECG is a temporal signal:

- Conv1D layers capture local waveform morphology
- BiLSTM captures longer-range sequential dependencies

Using BiLSTM in both architectures makes the experiment fairer. The main variable becomes the **fusion strategy**, not whether one model has a stronger ECG backbone.

---

## 4. Data Pipeline

```text
MIMIC-IV Parquet
    -> Stage 1 split: train_pool + test_set
    -> Stage 2 split: train + val
    -> Standardization fitted on train_pool only
    -> ECG cache on disk
    -> Lazy loading during training
    -> Held-out test set stored for evaluation and frontend serving
```

### Notes

- ECG waveforms are cached locally for efficient reuse
- Clinical features are standardized without leaking test information
- The same data split should be used for both fusion strategies

---

## 5. Training Setup

The current training pipeline includes:

- focal loss for class imbalance
- tuned learning rate
- threshold search on validation
- best-threshold persistence in metrics
- checkpoint resume handling

### Fair comparison rules

For a valid early-vs-late comparison, both models should use:

1. the same train/validation/test split
2. the same preprocessing
3. the same optimizer and learning-rate policy
4. the same threshold tuning strategy
5. the same evaluation metrics

### Recommended metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

---

## 6. Repository Structure

### Current early-fusion module

```text
early_fusion/
    config.py
    dataset.py
    model.py
    losses.py
    engine.py
    train.py
    plots.py
    artifacts/
```

### Proposed late-fusion module

```text
late_fusion/
    __init__.py
    config.py
    dataset.py
    model.py
    engine.py
    train.py
    artifacts/
```

Shared root-level infrastructure:

| File | Purpose |
|---|---|
| `store_predictions.py` | Evaluates models on the same test set and stores outputs |
| `explain.py` | Attribution and reasoning logic |
| `api.py` | FastAPI layer for the dashboard |

---

## 7. Frontend / API Context

The dashboard and API are model-agnostic. Once a model writes predictions to the database, the frontend reads them through shared endpoints.

### Main API endpoints

| Endpoint | Description |
|---|---|
| `GET /api/patients` | Paginated patient list |
| `GET /api/patients/{id}` | Full patient record and prediction |
| `GET /api/patients/{id}/ecg` | 12-lead waveform |
| `GET /api/patients/{id}/insights` | Explainability output |
| `GET /api/metrics` | Aggregate performance |
| `GET /api/stats` | Dashboard summary |
| `GET /api/comparison` | Model comparison output |

---

## 8. Recommended Late-Fusion Implementation

The cleanest late-fusion experiment for this project is:

- ECG branch: `Conv1D + BiLSTM + ECG head`
- Clinical branch: `MLP + clinical head`
- Final fusion: concatenate logits and pass through a small fusion head

That gives the clearest contrast with the current early-fusion model, where clinical information is injected before the shared CNN and BiLSTM.

---

## 9. Report-Ready Wording

You can use this explanation in the report:

> The early-fusion model combines ECG and clinical information before the shared temporal feature extractor completes multimodal representation learning. In contrast, the late-fusion model processes ECG and clinical data independently through separate branches and combines their prediction signals only at the final decision stage.

And for the sequence modeling:

> In both architectures, stacked convolutional layers followed by a BiLSTM are used to capture local waveform morphology and longer-range sequential dependencies. The difference lies in the fusion stage: in early fusion the BiLSTM operates on a jointly fused multimodal representation, whereas in late fusion it operates only on the ECG branch before decision-level fusion.
