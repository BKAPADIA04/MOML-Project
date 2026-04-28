"""
hypervolume.py
--------------
Hypervolume Indicator (HV) computation for the multi-objective CNN NAS study.

Default data source: csv/pareto_trials.csv  (saved by the Kaggle notebook).

Three objectives (all cast as minimisation internally):
    O1: -Accuracy          (minimise  ⟺  maximise accuracy)
    O2:  Latency  (ms)     (minimise)
    O3:  Parameters        (minimise)

Reference point (worst acceptable values, used as the HV upper boundary):
    r = [-0.30,  4.00,  700_000]

Algorithm
---------
WFG (Walking Fish Group) exact hypervolume algorithm — O(n · 2^d) worst-case,
perfectly tractable for the small fronts produced by this study (≤100 points,
3 objectives).  No external libraries required beyond NumPy.

Usage
-----
    python hypervolume.py                        # uses csv/pareto_trials.csv by default
    python hypervolume.py --csv path/to/file.csv # any compatible CSV

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

# ── Reference point ──────────────────────────────────────────────────────────

# O1 = -accuracy  → worst acceptable = -0.30  (accuracy of 0.30 or lower)
# O2 = latency_ms → worst acceptable =  4.00 ms
# O3 = n_params   → worst acceptable =  700 000
REFERENCE_POINT: list[float] = [-0.30, 4.00, 700_000.0]


# (Hard-coded fallback removed — real data is loaded from DEFAULT_CSV)


# ── Normalisation ─────────────────────────────────────────────────────────────

def normalise(
    points: np.ndarray,
    ref: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Min-max normalise each objective to [0, 1] using the ideal point (column
    minima of the front) and the reference point as the upper bound.

    Parameters
    ----------
    points : (n, d) array  — Pareto front in minimisation form.
    ref    : (d,)   array  — Reference point.

    Returns
    -------
    normalised_points : (n, d) array in [0, 1]
    normalised_ref    : (d,)   array  (will be all-ones after normalisation)
    """
    ideal = points.min(axis=0)           # best value per objective
    scale = ref - ideal                  # range per objective
    # Guard against degenerate objectives (all solutions identical)
    scale = np.where(scale == 0, 1.0, scale)
    return (points - ideal) / scale, np.ones(ref.shape)


# ── WFG Hypervolume ───────────────────────────────────────────────────────────

def _exclusive_hv(point: np.ndarray, others: np.ndarray, ref: np.ndarray) -> float:
    """
    Exclusive hypervolume contribution of `point` with respect to `others`
    and `ref`.  Used internally by wfg_hypervolume.
    """
    dominated = _limit(others, point)
    return _wfg(np.array([point]), ref) - _wfg(dominated, ref) if dominated.size else _wfg(np.array([point]), ref)


def _limit(points: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Restrict `points` to those dominated by `ref` (component-wise ≤),
    clipping each component to max(p_i, ref_i).
    """
    if points.size == 0:
        return points
    clipped = np.maximum(points, ref)
    # Keep only points that are still ≤ reference after clipping
    dominated_mask = np.all(clipped <= ref, axis=1)
    return clipped[dominated_mask]


def _wfg(points: np.ndarray, ref: np.ndarray) -> float:
    """
    WFG recursive hypervolume computation (minimisation, all objectives).

    Parameters
    ----------
    points : (n, d) array — non-dominated points (already filtered vs. ref).
    ref    : (d,)   array — reference point.

    Returns
    -------
    float  Hypervolume value.
    """
    if points.size == 0:
        return 0.0

    n, d = points.shape

    # Base case: 1-D hypervolume
    if d == 1:
        return float(ref[0] - points[:, 0].min())

    # Sort ascending on the last objective
    points = points[np.argsort(points[:, -1])]

    hv = 0.0
    for i, p in enumerate(points):
        # Slice between current point and next (or reference) in last dimension
        if i + 1 < n:
            depth = points[i + 1, -1] - p[-1]
        else:
            depth = ref[-1] - p[-1]

        if depth <= 0:
            continue

        # Limit remaining points to slice and recurse on d-1 objectives
        dominated = _limit(points[i + 1 :], p)
        sub_ref = ref[:-1]
        sub_points = np.vstack([p[:-1], dominated[:, :-1]]) if dominated.size else p[:-1][np.newaxis]
        hv += depth * _wfg(_non_dominated(sub_points), sub_ref)

    return hv


def _non_dominated(points: np.ndarray) -> np.ndarray:
    """
    Return the non-dominated subset of `points` (minimisation).
    Simple O(n²) filter — sufficient for the small fronts in this study.
    """
    if points.shape[0] <= 1:
        return points
    mask = np.ones(len(points), dtype=bool)
    for i, p in enumerate(points):
        if not mask[i]:
            continue
        # p is dominated if there exists q ≤ p component-wise with q ≠ p
        others = points[mask]
        dominated_by_others = np.all(others <= p, axis=1) & np.any(others < p, axis=1)
        if dominated_by_others.any():
            mask[i] = False
    return points[mask]


# ── Public API ────────────────────────────────────────────────────────────────

def compute_hypervolume(
    pareto_points: Sequence[Sequence[float]],
    reference_point: Sequence[float],
    *,
    normalise_first: bool = True,
    verbose: bool = True,
) -> float:
    """
    Compute the Hypervolume Indicator for a given Pareto front.

    All objectives must be in **minimisation** form.
    For accuracy (maximisation), pass **-accuracy**.

    Parameters
    ----------
    pareto_points    : List of [obj1, obj2, ..., objD] rows.
    reference_point  : Worst-acceptable point [r1, r2, ..., rD].
    normalise_first  : If True, normalise objectives to [0, 1] before
                       computing HV (makes HV comparable across studies).
    verbose          : Print per-step information.

    Returns
    -------
    float  Hypervolume indicator value.
    """
    pts = np.array(pareto_points, dtype=float)
    ref = np.array(reference_point, dtype=float)

    if pts.ndim != 2 or pts.shape[1] != ref.shape[0]:
        raise ValueError(
            f"Shape mismatch: points are {pts.shape}, "
            f"reference point has {ref.shape[0]} objectives."
        )

    # Remove points that do not dominate the reference point
    feasible_mask = np.all(pts < ref, axis=1)
    n_infeasible = (~feasible_mask).sum()
    if n_infeasible:
        print(f"[HV] Warning: {n_infeasible} point(s) do not dominate the "
              f"reference point and will be excluded.")
    pts = pts[feasible_mask]

    if pts.size == 0:
        print("[HV] No feasible points remain. Hypervolume = 0.")
        return 0.0

    # Keep only the non-dominated subset
    pts = _non_dominated(pts)

    if verbose:
        print(f"[HV] {len(pts)} non-dominated point(s) after filtering.")

    if normalise_first:
        pts, ref = normalise(pts, ref)
        if verbose:
            print("[HV] Objectives normalised to [0, 1].")

    hv = _wfg(pts, ref)

    if verbose:
        label = "(normalised)" if normalise_first else "(raw units)"
        print(f"[HV] Hypervolume {label}: {hv:.6f}")

    return float(hv)


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
    import csv as _csv
    rows: list[list[float]] = []
    with open(path, newline="") as f:
        reader = _csv.reader(f)
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
        description="Compute the Hypervolume Indicator for a 3-objective Pareto front.",
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
        "--ref",
        nargs=3,
        metavar=("R1", "R2", "R3"),
        type=float,
        default=REFERENCE_POINT,
        help="Reference point (minimisation form). "
             f"Default: {REFERENCE_POINT}",
    )
    p.add_argument(
        "--no-normalise",
        action="store_true",
        help="Skip per-objective normalisation (returns HV in raw objective units).",
    )
    return p


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Load data
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[HV] Error: file '{args.csv}' not found.", file=sys.stderr)
        sys.exit(1)
    points = _load_csv(str(csv_path))
    print(f"[HV] Loaded {len(points)} point(s) from '{args.csv}'.")

    ref = args.ref
    print(f"[HV] Reference point : {ref}")

    hv = compute_hypervolume(
        points,
        ref,
        normalise_first=not args.no_normalise,
        verbose=True,
    )

    print(f"\n{'='*45}")
    print(f"  Hypervolume Indicator  :  {hv:.6f}")
    print(f"{'='*45}")


if __name__ == "__main__":
    main()
