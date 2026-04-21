# Walkthrough: Training, Comparison, and Serving

This file explains how the current AMI prediction pipeline works end to end: training, model comparison, prediction storage, explainability, and API serving.

For the architecture details, use [architecture_summary.md](/d:/MINI_PROJECT/architecture_summary.md:1) as the main reference.

## What the project does

The project predicts **Acute Myocardial Infarction (AMI)** from:

- raw 12-lead ECG waveforms
- structured clinical EHR features

The repository is set up to support a fair comparison between:

- a **current early-fusion model**
- a **planned late-fusion model**

After training, the system can evaluate all available models on the same test set, choose the best one, generate reasoning, and serve the results through a shared API.

---

## Current model status

### Early fusion

Implemented in [early_fusion/model.py](/d:/MINI_PROJECT/early_fusion/model.py:1).

The current early-fusion model works like this:

1. clinical features are projected into a small number of channels
2. those channels are repeated across the ECG timeline
3. they are concatenated with the raw ECG input
4. the fused input is passed through shared `Conv1D` layers
5. a shared `BiLSTM` models temporal dependencies
6. a classifier produces the AMI logit

So the model is genuinely early fusion, because the modalities are merged before the shared temporal modeling is complete.

### Late fusion

Not implemented yet as a separate module.

The intended late-fusion design is:

1. ECG branch processes only ECG using `Conv1D + BiLSTM`
2. clinical branch processes only tabular features using an MLP
3. each branch produces its own logit
4. a small fusion head combines the branch outputs at the end

That gives a clean research comparison where the main difference is the **fusion stage**.

---

## Repository flow

```text
Train model(s)
    -> Save best weights and metrics
    -> Run store_predictions.py
    -> Evaluate all registered models on the same held-out test set
    -> Pick the best model by metric
    -> Generate explainability outputs
    -> Store predictions and reasoning in SQLite
    -> Serve through api.py
```

---

## Project structure

```text
D:\MINI_PROJECT\
|-- architecture_summary.md
|-- walkthrough.md
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
|   |-- plots.py
|   `-- artifacts/
|       |-- models/
|       |-- metrics/
|       `-- plots/
`-- late_fusion/   # planned
```

---

## Training pipeline

### Early-fusion training

Run:

```bash
python -m early_fusion.train
```

The current training pipeline includes:

- focal loss for class imbalance
- tuned learning rate
- dropout
- validation threshold search
- checkpoint saving
- best-threshold persistence in `metrics.json`

### Important note

Because the early-fusion architecture was changed recently, old saved weights may not match the current model definition. Fresh training is the right move for the current version.

---

## Evaluation and model comparison

The comparison pipeline is handled by [store_predictions.py](/d:/MINI_PROJECT/store_predictions.py:1).

### What it does

1. loads the held-out test set
2. loads every registered trained model
3. evaluates each model on the same test patients
4. applies that model's saved threshold if available
5. compares metrics
6. picks the best model
7. stores predictions and reasoning in SQLite

### Current behavior

Right now the registry is set up for `early_fusion`.

Once `late_fusion` is implemented and trained, it can be added to the registry and compared automatically through the same script.

---

## Explainability pipeline

Explainability is handled by [explain.py](/d:/MINI_PROJECT/explain.py:1).

The pipeline computes feature attributions and generates readable reasoning text so the dashboard can show:

- predicted AMI risk
- top contributing clinical factors
- ranked explanation entries for each patient

This logic is shared and intended to stay model-agnostic as much as possible.

---

## Database and API

Predictions are stored in a shared SQLite database under the early-fusion artifacts area.

The API in [api.py](/d:/MINI_PROJECT/api.py:1) reads from that database and exposes endpoints for:

- patient list
- patient detail
- ECG waveform
- explainability insights
- model metrics
- comparison results

Start the API with:

```bash
python api.py
```

Swagger UI:

```text
http://localhost:5000/docs
```

---

## Typical workflow

### Train the current early-fusion model

```bash
python -m early_fusion.train
```

### Store predictions and build the database

```bash
python store_predictions.py
```

Quick subset run:

```bash
python store_predictions.py --subset 500
```

### Start the API

```bash
python api.py
```

---

## Planned late-fusion workflow

After `late_fusion/` is created:

```bash
python -m late_fusion.train
python store_predictions.py
python api.py
```

At that point, `store_predictions.py` should compare both fusion strategies on the same test set and pick the stronger one automatically.

---

## How this fits the research project

The repo is now organized around a defensible comparison:

- **Early fusion** asks whether joint multimodal temporal learning helps when clinical context is injected before shared CNN and BiLSTM processing finishes.
- **Late fusion** asks whether independent modality reasoning followed by final decision fusion works better.

Using `BiLSTM` in both models keeps the comparison fairer, because the main experimental variable becomes the fusion strategy instead of one model simply having a stronger ECG sequence encoder.

---

## Where to read next

- Architecture details: [architecture_summary.md](/d:/MINI_PROJECT/architecture_summary.md:1)
- Early-fusion model: [early_fusion/model.py](/d:/MINI_PROJECT/early_fusion/model.py:1)
- Training loop: [early_fusion/train.py](/d:/MINI_PROJECT/early_fusion/train.py:1)
- Model comparison and DB storage: [store_predictions.py](/d:/MINI_PROJECT/store_predictions.py:1)
- API: [api.py](/d:/MINI_PROJECT/api.py:1)
