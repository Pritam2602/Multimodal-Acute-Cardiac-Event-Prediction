import argparse
import json
import random
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":
    package_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(package_root))
    __package__ = "late_fusion"

import numpy as np
import optuna
import pandas as pd
import torch
import torch.optim as optim
from optuna.importance import get_param_importances

from .config import (
    ARTIFACT_ROOT,
    DEFAULT_THRESHOLD,
    ECG_LENGTH,
    ECG_LEADS,
    FOCAL_LOSS_ALPHA,
    FOCAL_LOSS_GAMMA,
    NUM_CLINICAL_FEATURES,
    SEED,
)
from .dataset import load_and_prepare_data
from .engine import evaluate, train_one_epoch
from .losses import BCEWithLogitsLossWrapper, FocalLoss, HybridBCELoss
from .model import LateFusionModel
from .plots import (
    save_optuna_history_plot,
    save_optuna_param_importance_plot,
    save_optuna_param_slices_plot,
)


TUNING_ROOT = ARTIFACT_ROOT / "tuning"
BEST_MODEL_PATH = TUNING_ROOT / "best_trial_model.pth"
BEST_TRIAL_PATH = TUNING_ROOT / "best_trial.json"
STUDY_PATH = TUNING_ROOT / "study.db"
TRIALS_CSV_PATH = TUNING_ROOT / "trials.csv"
HISTORY_PLOT_PATH = TUNING_ROOT / "optimization_history.png"
IMPORTANCE_PLOT_PATH = TUNING_ROOT / "param_importances.png"
SLICE_PLOT_PATH = TUNING_ROOT / "param_slices.png"


def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _build_criterion(loss_mode: str, pos_weight: torch.Tensor, gamma: float):
    if loss_mode == "bce":
        return BCEWithLogitsLossWrapper(pos_weight=pos_weight), "BCEWithLogitsLoss"
    if loss_mode == "hybrid":
        return HybridBCELoss(
            alpha=FOCAL_LOSS_ALPHA,
            gamma=gamma,
            pos_weight=pos_weight,
            bce_weight=0.5,
            focal_weight=0.5,
        ), f"HybridBCELoss(gamma={gamma:.3f})"
    return FocalLoss(
        alpha=FOCAL_LOSS_ALPHA,
        gamma=gamma,
        pos_weight=pos_weight,
    ), f"FocalLoss(gamma={gamma:.3f})"


def _make_grad_scaler(amp_enabled: bool):
    amp_module = getattr(torch, "amp", None)
    if amp_module is not None and hasattr(amp_module, "GradScaler"):
        return amp_module.GradScaler(enabled=amp_enabled)
    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
        return torch.cuda.amp.GradScaler(enabled=amp_enabled)
    return None


def _suggest_hparams(trial: optuna.Trial) -> dict:
    batch_size = trial.suggest_categorical("batch_size", [32, 48, 64, 96])
    cnn_filters = trial.suggest_categorical("cnn_filters", [16, 24, 32, 48, 64])
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
        "batch_size": batch_size,
        "dropout": trial.suggest_float("dropout", 0.2, 0.6),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "cnn_filters": cnn_filters,
        "lstm_hidden_size": trial.suggest_categorical("lstm_hidden_size", [32, 48, 64, 96, 128]),
        "fusion_hidden_dim": trial.suggest_categorical("fusion_hidden_dim", [8, 16, 32, 64]),
        "focal_loss_gamma": trial.suggest_float("focal_loss_gamma", 1.0, 4.0),
    }
    return params


def _save_best_trial_if_needed(payload: dict):
    current_best = None
    if BEST_TRIAL_PATH.exists():
        current_best = json.loads(BEST_TRIAL_PATH.read_text())

    if current_best and current_best.get("best_val_f1", float("-inf")) >= payload["best_val_f1"]:
        return

    torch.save(payload["model_payload"], BEST_MODEL_PATH)
    serializable = {k: v for k, v in payload.items() if k != "model_payload"}
    BEST_TRIAL_PATH.write_text(json.dumps(serializable, indent=2))
    print(
        f"[SAVE] New best trial {payload['trial_number']} | "
        f"Val F1={payload['best_val_f1']:.4f} -> {BEST_TRIAL_PATH}"
    )


def _finalize_study_outputs(study: optuna.Study, tuned_params: list[str]):
    trials_df = study.trials_dataframe()
    if trials_df.empty:
        return

    trials_df.to_csv(TRIALS_CSV_PATH, index=False)
    save_optuna_history_plot(trials_df, str(HISTORY_PLOT_PATH))
    save_optuna_param_slices_plot(trials_df, tuned_params, str(SLICE_PLOT_PATH))

    try:
        importances = get_param_importances(study)
    except Exception:
        importances = {}
    save_optuna_param_importance_plot(importances, str(IMPORTANCE_PLOT_PATH))


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Optuna tuning for late-fusion AMI model")
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials to run")
    parser.add_argument("--epochs", type=int, default=20, help="Max epochs per trial")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience per trial")
    parser.add_argument("--subset", type=int, default=None, help="Use only first N samples")
    parser.add_argument("--num-workers", type=int, default=None, help="Override training DataLoader worker count")
    parser.add_argument("--val-workers", type=int, default=None, help="Override validation DataLoader worker count")
    parser.add_argument("--disable-amp", action="store_true", help="Disable CUDA automatic mixed precision")
    parser.add_argument("--weighted-sampling", action="store_true", help="Use weighted sampling on the training split")
    parser.add_argument("--loss-mode", choices=["bce", "focal", "hybrid"], default="hybrid")
    parser.add_argument("--study-name", type=str, default="late_fusion_f1_tuning")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser


def main():
    args = _build_arg_parser().parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda" and not args.disable_amp
    TUNING_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Device: {device}")
    print(f"[INFO] AMP: {'enabled' if amp_enabled else 'disabled'}")
    print(f"[DIR]  {TUNING_ROOT}")

    pruner = optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=3)
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=f"sqlite:///{STUDY_PATH.as_posix()}",
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    tuned_params = [
        "learning_rate",
        "batch_size",
        "dropout",
        "weight_decay",
        "cnn_filters",
        "lstm_hidden_size",
        "fusion_hidden_dim",
        "focal_loss_gamma",
    ]

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_hparams(trial)
        seed_everything(args.seed)

        train_loader, val_loader, pos_weight, _ = load_and_prepare_data(
            subset=args.subset,
            train_num_workers=args.num_workers,
            val_num_workers=args.val_workers,
            weighted_sampling=args.weighted_sampling,
            batch_size=params["batch_size"],
        )

        model = LateFusionModel(
            n_leads=ECG_LEADS,
            ecg_length=ECG_LENGTH,
            n_clinical=NUM_CLINICAL_FEATURES,
            dropout=params["dropout"],
            cnn_filters=params["cnn_filters"],
            lstm_hidden_size=params["lstm_hidden_size"],
            fusion_hidden_dim=params["fusion_hidden_dim"],
        ).to(device)

        pos_weight = pos_weight.to(device)
        criterion, loss_name = _build_criterion(
            args.loss_mode,
            pos_weight=pos_weight,
            gamma=params["focal_loss_gamma"],
        )
        optimizer = optim.AdamW(
            model.parameters(),
            lr=params["learning_rate"],
            weight_decay=params["weight_decay"],
        )
        scaler = _make_grad_scaler(amp_enabled)

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_f1": [],
            "val_f1": [],
        }
        active_threshold = DEFAULT_THRESHOLD
        best_val_f1 = float("-inf")
        best_threshold = DEFAULT_THRESHOLD
        best_epoch = 0
        epochs_without_improvement = 0
        best_model_state = None

        start_time = time.time()
        for epoch in range(1, args.epochs + 1):
            train_loss, train_metrics = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                threshold=active_threshold,
                amp_enabled=amp_enabled,
                scaler=scaler,
            )
            val_loss, val_metrics = evaluate(
                model,
                val_loader,
                criterion,
                device,
                auto_threshold=True,
                amp_enabled=amp_enabled,
            )
            active_threshold = val_metrics["threshold"]

            history["train_loss"].append(float(train_loss))
            history["val_loss"].append(float(val_loss))
            history["train_f1"].append(float(train_metrics["f1"]))
            history["val_f1"].append(float(val_metrics["f1"]))

            trial.report(val_metrics["f1"], step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned(f"Pruned at epoch {epoch} with val_f1={val_metrics['f1']:.4f}")

            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = float(val_metrics["f1"])
                best_threshold = float(active_threshold)
                best_epoch = epoch
                epochs_without_improvement = 0
                best_model_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.patience:
                    break

        elapsed = time.time() - start_time
        trial.set_user_attr("best_threshold", best_threshold)
        trial.set_user_attr("best_epoch", best_epoch)
        trial.set_user_attr("elapsed_sec", round(elapsed, 2))
        trial.set_user_attr("loss_name", loss_name)

        if best_model_state is not None:
            payload = {
                "trial_number": trial.number,
                "best_val_f1": best_val_f1,
                "best_threshold": best_threshold,
                "best_epoch": best_epoch,
                "elapsed_sec": round(elapsed, 2),
                "loss_mode": args.loss_mode,
                "loss_name": loss_name,
                "params": params,
                "model_kwargs": {
                    "n_leads": ECG_LEADS,
                    "ecg_length": ECG_LENGTH,
                    "n_clinical": NUM_CLINICAL_FEATURES,
                    "dropout": params["dropout"],
                    "cnn_filters": params["cnn_filters"],
                    "lstm_hidden_size": params["lstm_hidden_size"],
                    "fusion_hidden_dim": params["fusion_hidden_dim"],
                },
                "history": {key: [round(v, 6) for v in values] for key, values in history.items()},
                "model_payload": {
                    "trial_number": trial.number,
                    "best_val_f1": best_val_f1,
                    "best_threshold": best_threshold,
                    "best_epoch": best_epoch,
                    "params": params,
                    "model_kwargs": {
                        "n_leads": ECG_LEADS,
                        "ecg_length": ECG_LENGTH,
                        "n_clinical": NUM_CLINICAL_FEATURES,
                        "dropout": params["dropout"],
                        "cnn_filters": params["cnn_filters"],
                        "lstm_hidden_size": params["lstm_hidden_size"],
                        "fusion_hidden_dim": params["fusion_hidden_dim"],
                    },
                    "state_dict": best_model_state,
                },
            }
            _save_best_trial_if_needed(payload)

        return best_val_f1

    study.optimize(objective, n_trials=args.trials)
    _finalize_study_outputs(study, tuned_params)

    best = study.best_trial
    summary = {
        "study_name": args.study_name,
        "best_trial": best.number,
        "best_value": best.value,
        "best_params": best.params,
        "best_user_attrs": best.user_attrs,
        "n_trials": len(study.trials),
    }
    (TUNING_ROOT / "study_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 70)
    print(" TUNING COMPLETE")
    print("=" * 70)
    print(f"Best trial    : {best.number}")
    print(f"Best val F1   : {best.value:.4f}")
    print(f"Best params   : {best.params}")
    print(f"Study DB      : {STUDY_PATH}")
    print(f"Trials CSV    : {TRIALS_CSV_PATH}")
    print(f"Best trial    : {BEST_TRIAL_PATH}")
    print(f"Best model    : {BEST_MODEL_PATH}")
    print(f"Plots         : {HISTORY_PLOT_PATH}")
    print(f"                {IMPORTANCE_PLOT_PATH}")
    print(f"                {SLICE_PLOT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
