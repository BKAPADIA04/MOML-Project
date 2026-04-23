"""
main.py
-------
Entry point for the Multi-Objective Optimisation study on Fashion-MNIST.

Usage
-----
    python main.py [--n-trials N] [--seed S] [--data-dir DIR] [--results-dir DIR]

Objectives
----------
    1. -val_accuracy   (minimise → maximise accuracy)
    2. inference_ms    (minimise latency)
    3. n_params        (minimise model size)
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure the package directory is on the path
sys.path.insert(0, os.path.dirname(__file__))

import optuna
import pandas as pd

from objective import make_objective
from utils import set_seed
from visualize import (
    extract_pareto_front,
    plot_2d_pareto,
    plot_3d_pareto,
    plot_optuna_charts,
    trials_to_dataframe,
)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-Objective Optimisation of Fashion-MNIST CNN using Optuna"
    )
    parser.add_argument(
        "--n-trials", type=int, default=50,
        help="Number of Optuna trials (default: 50, fits in ~2h on CPU)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Global random seed (default: 42)",
    )
    parser.add_argument(
        "--data-dir", type=str, default="./data",
        help="Root directory for Fashion-MNIST download (default: ./data)",
    )
    parser.add_argument(
        "--results-dir", type=str, default="./results",
        help="Directory to save CSV and plot outputs (default: ./results)",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Parallel Optuna jobs (default: 1; set >1 only with CUDA or many CPUs)",
    )
    return parser.parse_args()


# ── Study creation ────────────────────────────────────────────────────────────

def create_study(seed: int) -> optuna.Study:
    sampler = optuna.samplers.NSGAIISampler(seed=seed)
    study = optuna.create_study(
        directions=["minimize", "minimize", "minimize"],
        sampler=sampler,
        study_name="moo_fashion_mnist",
    )
    return study


# ── Results printing ──────────────────────────────────────────────────────────

def print_pareto_solutions(pareto_df: pd.DataFrame, n: int = 4) -> None:
    """Pretty-print at least `n` Pareto-optimal solutions."""
    print("\n" + "=" * 70)
    print("  PARETO-OPTIMAL SOLUTIONS (sorted by Validation Accuracy ↓)")
    print("=" * 70)
    display = pareto_df.head(max(n, len(pareto_df)))
    for _, row in display.iterrows():
        print(
            f"  Trial #{int(row['trial']):>3d} │ "
            f"Val Acc: {row['val_accuracy']:.4f} │ "
            f"Inference: {row['inference_ms']:>7.3f} ms │ "
            f"Params: {int(row['n_params']):>8,d}"
        )
        # Print key hyperparameters
        hp_keys = [
            "n_conv_layers", "n_fc_layers", "fc_units",
            "lr", "batch_size", "epochs", "optimizer",
            "dropout", "input_resolution",
        ]
        hp_str = "  " + " | ".join(
            f"{k}={row[k]}" for k in hp_keys if k in row
        )
        print(hp_str)
        print()
    print("=" * 70)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("  Multi-Objective CNN Search on Fashion-MNIST")
    print(f"  Trials: {args.n_trials}  |  Seed: {args.seed}")
    print(f"{'='*60}\n")

    # Silence Optuna INFO logs (keep WARNING+ only)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    set_seed(args.seed)

    # ── Optuna study ──────────────────────────────────────────────────────────
    study = create_study(seed=args.seed)
    objective_fn = make_objective(seed=args.seed, data_dir=args.data_dir)

    print(f"[main] Starting optimisation ({args.n_trials} trials) …")
    study.optimize(
        objective_fn,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True,   # requires tqdm
        gc_after_trial=True,
    )
    print("[main] Optimisation complete.\n")

    # ── Collect completed trials ──────────────────────────────────────────────
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    print(f"[main] Completed trials : {len(completed)}")
    print(f"[main] Pruned   trials  : "
          f"{sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)}")

    # ── Pareto front ──────────────────────────────────────────────────────────
    pareto_trials = extract_pareto_front(completed)
    print(f"[main] Pareto-optimal   : {len(pareto_trials)}\n")

    all_df    = trials_to_dataframe(completed)
    pareto_df = trials_to_dataframe(pareto_trials)

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    all_csv     = os.path.join(args.results_dir, "all_trials.csv")
    pareto_csv  = os.path.join(args.results_dir, "pareto_trials.csv")
    all_df.to_csv(all_csv,    index=False)
    pareto_df.to_csv(pareto_csv, index=False)
    print(f"[main] Saved all-trial results  → {all_csv}")
    print(f"[main] Saved Pareto results     → {pareto_csv}\n")

    # ── Print solutions ───────────────────────────────────────────────────────
    print_pareto_solutions(pareto_df, n=4)

    # ── Visualise ─────────────────────────────────────────────────────────────
    print("[main] Generating plots …")
    plot_2d_pareto(all_df, pareto_df, save_dir=args.results_dir)
    plot_3d_pareto(all_df, pareto_df, save_dir=args.results_dir)
    plot_optuna_charts(study, save_dir=args.results_dir)
    print("[main] All done!  Results saved to:", args.results_dir)


if __name__ == "__main__":
    main()
