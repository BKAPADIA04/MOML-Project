"""
utils.py
--------
Training loop, validation, inference timing, and seed utilities.
"""

import time
from typing import Optional

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader


# ── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Set seeds for Python, NumPy, and PyTorch (CPU + CUDA)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Training ─────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Train the model for one epoch.

    Returns
    -------
    float  Average training loss over the epoch.
    """
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)  # type: ignore[arg-type]


# ── Validation ───────────────────────────────────────────────────────────────

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Evaluate classification accuracy.

    Returns
    -------
    float  Accuracy in [0, 1].
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total


# ── Inference timing ─────────────────────────────────────────────────────────

def measure_inference_time(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_warmup: int = 10,
    n_measure: int = 100,
) -> float:
    """
    Measure mean inference time **per sample** in milliseconds.

    The function uses a single sample (batch_size=1) approach by iterating
    over the loader and breaking after `n_warmup + n_measure` images.
    The model must already be in eval mode.

    Args:
        model:     Trained model (must be on `device`).
        loader:    DataLoader (any batch size; we time individual images).
        device:    Target device.
        n_warmup:  Warm-up inferences (excluded from timing).
        n_measure: Inferences to measure.

    Returns:
        float  Mean per-sample inference latency in milliseconds.
    """
    model.eval()
    times: list[float] = []
    count = 0

    with torch.no_grad():
        for images, _ in loader:
            for i in range(images.size(0)):
                single = images[i : i + 1].to(device, non_blocking=True)

                # --- Device-specific timing ---
                if device.type == "cuda":
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    _ = model(single)
                    torch.cuda.synchronize()
                else:
                    start = time.perf_counter()
                    _ = model(single)

                elapsed_ms = (time.perf_counter() - start) * 1_000

                if count >= n_warmup:
                    times.append(elapsed_ms)

                count += 1
                if count >= n_warmup + n_measure:
                    return float(np.mean(times))

    # Fallback if loader exhausted before n_measure samples
    return float(np.mean(times)) if times else float("inf")


# ── Full training pipeline ───────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 5,
    optuna_trial=None,
) -> float:
    """
    Train the model and return the best validation accuracy observed.

    Args:
        model:         Model to train (moved to `device` internally).
        train_loader:  Training DataLoader.
        val_loader:    Validation DataLoader.
        optimizer:     Configured optimizer.
        device:        CPU or CUDA device.
        epochs:        Number of training epochs.
        optuna_trial:  Optional Optuna trial for pruning support.

    Returns:
        float  Best validation accuracy in [0, 1].
    """
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    for epoch in range(epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate(model, val_loader, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        # Optuna pruning (optional but good practice)
        if optuna_trial is not None:
            optuna_trial.report(val_acc, epoch)
            if optuna_trial.should_prune():
                import optuna
                raise optuna.exceptions.TrialPruned()

    return best_val_acc
