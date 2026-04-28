"""
visualize.py
------------
Pareto front extraction and 2-D / 3-D visualisation of the study results.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import optuna


# ── Pareto utilities ──────────────────────────────────────────────────────────

def is_dominated(costs: np.ndarray, i: int) -> bool:
    """
    Return True if solution i is dominated by any other solution.
    All objectives are assumed to be *minimised*.
    """
    for j, c in enumerate(costs):
        if j == i:
            continue
        if np.all(c <= costs[i]) and np.any(c < costs[i]):
            return True
    return False


def extract_pareto_front(
    trials: list[optuna.trial.FrozenTrial],
) -> list[optuna.trial.FrozenTrial]:
    """Return only the non-dominated (Pareto-optimal) trials."""
    costs = np.array([t.values for t in trials])
    pareto = [t for i, t in enumerate(trials) if not is_dominated(costs, i)]
    return pareto


# ── Result table helpers ──────────────────────────────────────────────────────

def trials_to_dataframe(trials: list[optuna.trial.FrozenTrial]) -> pd.DataFrame:
    """Convert a list of Optuna trials to a tidy pandas DataFrame."""
    rows = []
    for t in trials:
        row = {
            "trial": t.number,
            "val_accuracy": -t.values[0],      # flip sign back
            "inference_ms": t.values[1],
            "n_params": t.values[2],
        }
        row.update(t.params)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.sort_values("val_accuracy", ascending=False).reset_index(drop=True)


# ── 2-D scatter plots ─────────────────────────────────────────────────────────

def _scatter_2d(
    ax: plt.Axes,
    all_df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
) -> None:
    ax.scatter(
        all_df[x_col], all_df[y_col],
        c="steelblue", alpha=0.4, s=25, label="All trials", zorder=2,
    )
    ax.scatter(
        pareto_df[x_col], pareto_df[y_col],
        c="crimson", s=70, edgecolors="black", linewidths=0.6,
        label="Pareto front", zorder=3,
    )
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_2d_pareto(
    all_trials_df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    save_dir: str = "results",
) -> None:
    """Generate and save three 2-D Pareto scatter plots."""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Multi-Objective Pareto Front — Fashion-MNIST CNN Search", fontsize=14)

    _scatter_2d(
        axes[0], all_trials_df, pareto_df,
        "inference_ms", "val_accuracy",
        "Inference Time (ms / sample)", "Validation Accuracy",
        "Accuracy vs Latency",
    )
    _scatter_2d(
        axes[1], all_trials_df, pareto_df,
        "n_params", "val_accuracy",
        "# Trainable Parameters", "Validation Accuracy",
        "Accuracy vs Model Size",
    )
    _scatter_2d(
        axes[2], all_trials_df, pareto_df,
        "n_params", "inference_ms",
        "# Trainable Parameters", "Inference Time (ms / sample)",
        "Latency vs Model Size",
    )

    plt.tight_layout()
    path = os.path.join(save_dir, "pareto_2d.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Saved 2-D Pareto plots → {path}")


# ── 3-D scatter plot ──────────────────────────────────────────────────────────

def plot_3d_pareto(
    all_trials_df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    save_dir: str = "results",
) -> None:
    """Generate and save a 3-D Pareto scatter plot."""
    os.makedirs(save_dir, exist_ok=True)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    # All trials
    ax.scatter(
        all_trials_df["inference_ms"],
        all_trials_df["n_params"],
        all_trials_df["val_accuracy"],
        c="steelblue", alpha=0.3, s=20, label="All trials",
    )
    # Pareto front
    sc = ax.scatter(
        pareto_df["inference_ms"],
        pareto_df["n_params"],
        pareto_df["val_accuracy"],
        c=pareto_df["val_accuracy"],
        cmap="RdYlGn", s=80, edgecolors="black", linewidths=0.5,
        label="Pareto front", zorder=5,
    )
    fig.colorbar(sc, ax=ax, shrink=0.5, label="Validation Accuracy")

    ax.set_xlabel("Inference (ms)", fontsize=10, labelpad=8)
    ax.set_ylabel("# Parameters", fontsize=10, labelpad=8)
    ax.set_zlabel("Val Accuracy", fontsize=10, labelpad=8)  # type: ignore[attr-defined]
    ax.set_title("3-D Pareto Front — Fashion-MNIST MOO", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    path = os.path.join(save_dir, "pareto_3d.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Saved 3-D Pareto plot  → {path}")


# ── Optuna built-in visualisation ─────────────────────────────────────────────

def plot_optuna_charts(study: optuna.Study, save_dir: str = "results") -> None:
    """Save Optuna's built-in interactive plots as static images."""
    try:
        from optuna.visualization.matplotlib import (
            plot_pareto_front,
            plot_param_importances,
        )

        os.makedirs(save_dir, exist_ok=True)

        # --- Pareto front (objectives 0 & 1) ---
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_pareto_front(study, targets=lambda t: (t.values[0], t.values[1]), ax=ax)
        ax.set_xlabel("-Validation Accuracy")
        ax.set_ylabel("Inference Time (ms)")
        ax.set_title("Optuna Pareto Front: accuracy vs latency")
        fig.savefig(os.path.join(save_dir, "optuna_pareto_front.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    except Exception as exc:
        print(f"[visualize] Optuna built-in chart skipped: {exc}")
