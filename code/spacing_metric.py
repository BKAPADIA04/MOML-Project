"""
spacing_metric.py
-----------------
Spacing Metric (SP) computation for the multi-objective CNN NAS study.

Default data source: csv/pareto_trials.csv  (saved by the Kaggle notebook).

Definition (from Schott 1995)
-----------------------------

    SP = sqrt( (1 / (|P| - 1))  *  Σ_i ( d̄ - d_i )² )

where
    P   = the set of Pareto-optimal solutions
    d_i = min_{j ≠ i} || f(x_i) - f(x_j) ||₂   (nearest-neighbour distance
          in normalised objective space)
    d̄   = (1/|P|) Σ_i d_i                        (mean of d_i)

A lower SP indicates a more uniformly spread front.
SP = 0 means all solutions are equidistant from their nearest neighbours.

Three objectives (normalised to [0, 1] before distance computation):
    O1: -Accuracy          (minimise  ⟺  maximise accuracy)
    O2:  Latency  (ms)     (minimise)
    O3:  Parameters        (minimise)

Usage
-----
    python spacing_metric.py                        # uses csv/pareto_trials.csv by default
    python spacing_metric.py --csv path/to/file.csv # any compatible CSV

CSV format accepted:
  • Kaggle output format (preferred):
        trial, val_accuracy, inference_ms, n_params, ...   (val_accuracy is auto-negated)
  • Raw minimisation format:
        neg_accuracy, latency_ms, n_params
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Sequence

import numpy as np


# ── Paths ────────────────────────────────────────────────────────────────────

DEFAULT_CSV = "csv/pareto_trials.csv"   # real Kaggle output

# (Hard-coded fallback removed — real data is loaded from DEFAULT_CSV)


# ── Normalisation ─────────────────────────────────────────────────────────────

def normalise(points: np.ndarray) -> np.ndarray:
    """
    Min-max normalise each objective column to [0, 1].

    Parameters
    ----------
    points : (n, d) array

    Returns
    -------
    (n, d) array with each column in [0, 1].
    """
    col_min = points.min(axis=0)
    col_max = points.max(axis=0)
    scale = col_max - col_min
    # Avoid division by zero for degenerate objectives
    scale = np.where(scale == 0, 1.0, scale)
    return (points - col_min) / scale


# ── Core metric ───────────────────────────────────────────────────────────────

def _nearest_neighbour_distances(points: np.ndarray) -> np.ndarray:
    """
    Compute d_i = min_{j ≠ i} ||points[i] - points[j]||₂  for each i.

    Parameters
    ----------
    points : (n, d) normalised array

    Returns
    -------
    d : (n,) array of nearest-neighbour distances
    """
    n = len(points)
    d = np.empty(n, dtype=float)
    for i in range(n):
        diffs = points - points[i]               # (n, d) differences
        dists = np.linalg.norm(diffs, axis=1)    # (n,) Euclidean distances
        dists[i] = np.inf                         # exclude self
        d[i] = dists.min()
    return d


def compute_spacing(
    pareto_points: Sequence[Sequence[float]],
    *,
    normalise_first: bool = True,
    verbose: bool = True,
) -> dict[str, float]:
    """
    Compute the Spacing Metric (SP) for a Pareto front.

    Parameters
    ----------
    pareto_points  : List of objective vectors (all in minimisation form).
    normalise_first: If True, normalise objectives to [0, 1] before computing
                     distances (recommended for objectives on different scales).
    verbose        : Print intermediate values.

    Returns
    -------
    dict with keys:
        "sp"        — Spacing metric value
        "d_mean"    — Mean nearest-neighbour distance (d̄)
        "d_min"     — Minimum d_i
        "d_max"     — Maximum d_i
        "d_values"  — Individual d_i values as a list
    """
    pts = np.array(pareto_points, dtype=float)

    if pts.ndim != 2:
        raise ValueError(f"Expected a 2-D array of points, got shape {pts.shape}.")

    n = len(pts)
    if n < 2:
        raise ValueError(
            f"Need at least 2 Pareto points to compute spacing; got {n}."
        )

    if normalise_first:
        pts = normalise(pts)
        if verbose:
            print("[SP] Objectives normalised to [0, 1].")

    d = _nearest_neighbour_distances(pts)
    d_mean = float(d.mean())
    sp = float(np.sqrt(np.mean((d_mean - d) ** 2)))

    if verbose:
        _print_details(d, d_mean, sp)

    return {
        "sp": sp,
        "d_mean": d_mean,
        "d_min": float(d.min()),
        "d_max": float(d.max()),
        "d_values": d.tolist(),
    }


# ── Pretty-printing helpers ───────────────────────────────────────────────────

def _print_details(d: np.ndarray, d_mean: float, sp: float) -> None:
    """Print a formatted breakdown of nearest-neighbour distances and SP."""
    n = len(d)
    labels = [f"P{i+1:02d}" for i in range(n)]

    col_w = 10
    print(f"\n{'─'*50}")
    print(f"  {'Solution':<10}  {'d_i (NN dist)':>{col_w}}")
    print(f"{'─'*50}")
    for label, di in zip(labels, d):
        print(f"  {label:<10}  {di:>{col_w}.6f}")
    print(f"{'─'*50}")
    print(f"  {'d̄  (mean)':^20}  {d_mean:>{col_w}.6f}")
    print(f"  {'d_min':^20}  {d.min():>{col_w}.6f}")
    print(f"  {'d_max':^20}  {d.max():>{col_w}.6f}")
    print(f"{'─'*50}\n")


# ── CLI helpers ───────────────────────────────────────────────────────────────

def _load_csv(path: str) -> list[list[float]]:
    """
    Load a Pareto front from a CSV file.

    Accepts two formats automatically:

    1. Kaggle notebook output (detected by a 'val_accuracy' column header):
           trial, val_accuracy, inference_ms, n_params, ...
       val_accuracy is negated → -val_accuracy for minimisation.

    2. Raw minimisation format (3 numeric columns, no header or unknown header):
           neg_accuracy, latency_ms, n_params
    """
    rows: list[list[float]] = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header and "val_accuracy" in header:
            # Kaggle format — extract by column name
            acc_i = header.index("val_accuracy")
            lat_i = header.index("inference_ms")
            par_i = header.index("n_params")
            for row in reader:
                if not any(row):          # skip blank lines
                    continue
                rows.append([
                    -float(row[acc_i]),   # negate: maximise → minimise
                     float(row[lat_i]),
                     float(row[par_i]),
                ])
        else:
            # Raw 3-column minimisation format
            if header:
                try:
                    rows.append([float(v) for v in header])  # header is data
                except ValueError:
                    pass  # genuine header, skip
            for row in reader:
                if any(row):
                    rows.append([float(v) for v in row[:3]])
    return rows


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute the Spacing Metric (SP) for a 3-objective Pareto front.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--csv",
        metavar="FILE",
        default=DEFAULT_CSV,
        help=f"Path to a CSV file (Kaggle format or raw 3-col format). "
             f"Default: '{DEFAULT_CSV}'",
    )
    p.add_argument(
        "--no-normalise",
        action="store_true",
        help="Skip per-objective normalisation (compute distances in raw units).",
    )
    return p


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Load data
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[SP] Error: file '{args.csv}' not found.", file=sys.stderr)
        sys.exit(1)
    points = _load_csv(str(csv_path))
    print(f"[SP] Loaded {len(points)} point(s) from '{args.csv}'.")

    result = compute_spacing(
        points,
        normalise_first=not args.no_normalise,
        verbose=True,
    )

    print(f"{'='*45}")
    print(f"  Spacing Metric  SP  :  {result['sp']:.6f}")
    print(f"  Mean NN distance d̄ :  {result['d_mean']:.6f}")
    print(f"{'='*45}")

    # Interpretation hint
    if result["sp"] < 0.05:
        quality = "Excellent — very uniform spread."
    elif result["sp"] < 0.10:
        quality = "Good spread with minor clustering."
    elif result["sp"] < 0.20:
        quality = "Moderate spread; some gaps in the front."
    else:
        quality = "Poor spread; front is highly irregular."

    print(f"\n  Interpretation: {quality}")


if __name__ == "__main__":
    main()
