"""
objective.py
------------
Optuna multi-objective trial function for Fashion-MNIST CNN search.

Objectives returned (in order):
    1. -val_accuracy         (minimise → maximise accuracy)
    2.  inference_time_ms    (minimise)
    3.  n_parameters         (minimise)
"""

from __future__ import annotations

import gc
from typing import Tuple

import optuna
import torch
import torch.optim as optim

from data import get_dataloaders
from model import ConfigurableCNN, count_parameters
from utils import measure_inference_time, set_seed, train_model

# Fixed constants (keep search efficient)
N_CLASSES = 10
IN_CHANNELS = 1
VAL_FRACTION = 0.1
DATA_DIR = "./data"
NUM_WORKERS = 0          # 0 workers avoids multiprocessing issues inside Optuna


def build_search_space(trial: optuna.Trial) -> dict:
    """Sample all hyperparameters from the Optuna search space."""

    # ── Architecture ────────────────────────────────────────────────────────
    n_conv_layers = trial.suggest_int("n_conv_layers", 1, 3)
    conv_channels = [
        trial.suggest_categorical(f"conv_ch_{i}", [16, 32, 64])
        for i in range(n_conv_layers)
    ]
    n_fc_layers = trial.suggest_int("n_fc_layers", 1, 2)
    fc_units = trial.suggest_categorical("fc_units", [64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)

    # ── Training ─────────────────────────────────────────────────────────────
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    epochs = trial.suggest_int("epochs", 3, 8)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])

    # ── Input resolution ──────────────────────────────────────────────────────
    input_resolution = trial.suggest_categorical("input_resolution", [14, 20, 28])

    return dict(
        n_conv_layers=n_conv_layers,
        conv_channels=conv_channels,
        n_fc_layers=n_fc_layers,
        fc_units=fc_units,
        dropout=dropout,
        lr=lr,
        batch_size=batch_size,
        epochs=epochs,
        optimizer_name=optimizer_name,
        input_resolution=input_resolution,
    )


def make_objective(seed: int = 42, data_dir: str = DATA_DIR):
    """
    Factory that captures fixed settings and returns the Optuna objective.

    The returned function has signature  f(trial) → Tuple[float, float, float]
    matching what Optuna expects for a 3-objective study.
    """

    def objective(trial: optuna.Trial) -> Tuple[float, float, float]:
        # Per-trial seed: deterministic but unique across trials
        trial_seed = seed + trial.number
        set_seed(trial_seed)

        hp = build_search_space(trial)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Data loaders ─────────────────────────────────────────────────────
        train_loader, val_loader, _ = get_dataloaders(
            batch_size=hp["batch_size"],
            input_resolution=hp["input_resolution"],
            val_fraction=VAL_FRACTION,
            data_dir=data_dir,
            num_workers=NUM_WORKERS,
            seed=trial_seed,
        )

        # ── Model ─────────────────────────────────────────────────────────────
        model = ConfigurableCNN(
            in_channels=IN_CHANNELS,
            n_classes=N_CLASSES,
            n_conv_layers=hp["n_conv_layers"],
            conv_channels=hp["conv_channels"],
            n_fc_layers=hp["n_fc_layers"],
            fc_units=hp["fc_units"],
            dropout=hp["dropout"],
            input_resolution=hp["input_resolution"],
        ).to(device)

        n_params = count_parameters(model)

        # ── Optimizer ─────────────────────────────────────────────────────────
        if hp["optimizer_name"] == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=hp["lr"])
        else:
            optimizer = optim.SGD(
                model.parameters(), lr=hp["lr"], momentum=0.9, weight_decay=1e-4
            )

        # ── Training ──────────────────────────────────────────────────────────
        val_accuracy = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            epochs=hp["epochs"],
            optuna_trial=trial,
        )

        # ── Inference timing ──────────────────────────────────────────────────
        model.eval()
        inference_ms = measure_inference_time(
            model=model,
            loader=val_loader,
            device=device,
            n_warmup=5,
            n_measure=50,
        )

        # ── Cleanup (avoid memory leaks across trials) ────────────────────────
        del model, optimizer, train_loader, val_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Return:   obj1 = -accuracy (minimise),  obj2 = ms,  obj3 = #params
        return (-val_accuracy, inference_ms, float(n_params))

    return objective
