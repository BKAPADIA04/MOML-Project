# %% [markdown]
# ## Cell 1 — Install Optuna

# %%
import subprocess
subprocess.run(["pip", "install", "optuna", "-q"], check=True)
print("✓ Optuna installed")


# %% [markdown]
# ## Cell 2 — Imports & GPU Setup

# %%
import os
import gc
import time
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend for Kaggle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import optuna
from optuna.samplers import NSGAIISampler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Device setup ──────────────────────────────────────────────────────────────
N_GPUS             = torch.cuda.device_count()
DEVICE             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_DATA_PARALLEL  = N_GPUS > 1
USE_AMP            = torch.cuda.is_available()   # Automatic Mixed Precision

print(f"PyTorch       : {torch.__version__}")
print(f"Optuna        : {optuna.__version__}")
print(f"Device        : {DEVICE}")
print(f"# GPUs        : {N_GPUS}")
for i in range(N_GPUS):
    print(f"  GPU {i}      : {torch.cuda.get_device_name(i)}")
print(f"DataParallel  : {USE_DATA_PARALLEL}")
print(f"AMP enabled   : {USE_AMP}")


# %% [markdown]
# ## Cell 3 — Global Config

# %%
SEED         = 42
N_TRIALS     = 60           # ~25-35 min on T4 x2 with AMP
DATA_DIR     = "/kaggle/working/data"
RESULTS_DIR  = "/kaggle/working/results"
VAL_FRACTION = 0.1
N_CLASSES    = 10
IN_CHANNELS  = 1
NUM_WORKERS  = 4            # more workers to keep GPU fed

os.makedirs(DATA_DIR,    exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"Config  : {N_TRIALS} trials | seed={SEED} | AMP={USE_AMP} | workers={NUM_WORKERS}")
print(f"Output  : {RESULTS_DIR}")


# %% [markdown]
# ## Cell 4 — Reproducibility

# %%
def set_seed(seed: int) -> None:
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False   # must be False for determinism

set_seed(SEED)
print("✓ Seeds set")


# %% [markdown]
# ## Cell 5 — Dataset Loader

# %%
def get_dataloaders(
    batch_size: int       = 256,
    input_resolution: int = 28,
    val_fraction: float   = VAL_FRACTION,
    data_dir: str         = DATA_DIR,
    num_workers: int      = NUM_WORKERS,
    seed: int             = SEED,
):
    """
    Return (train_loader, val_loader, test_loader) for Fashion-MNIST.
    Images are optionally resized to input_resolution × input_resolution.
    prefetch_factor=4 keeps the GPU data pipeline fully loaded.
    """
    tf_list = []
    if input_resolution != 28:
        tf_list.append(transforms.Resize((input_resolution, input_resolution)))
    tf_list += [
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),   # Fashion-MNIST channel stats
    ]
    transform = transforms.Compose(tf_list)

    full_train = datasets.FashionMNIST(data_dir, train=True,  download=True, transform=transform)
    test_ds    = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)

    val_size   = int(len(full_train) * val_fraction)
    train_size = len(full_train) - val_size
    gen        = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_train, [train_size, val_size], generator=gen)

    pin = torch.cuda.is_available()
    # prefetch_factor: each worker pre-loads 4 batches ahead → GPU never starves
    kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kwargs)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader

# Sanity check
_tl, _vl, _tel = get_dataloaders(batch_size=256)
print(f"✓ Loaders ready  |  train={len(_tl.dataset):,}  val={len(_vl.dataset):,}  test={len(_tel.dataset):,}")
del _tl, _vl, _tel


# %% [markdown]
# ## Cell 6 — Configurable CNN

# %%
class ConfigurableCNN(nn.Module):
    """
    Flexible CNN for Fashion-MNIST.

    Architecture:
        [Conv2d → BN → ReLU → MaxPool] × (n_conv - 1)
        [Conv2d → BN → ReLU]           × 1
        → AdaptiveAvgPool2d(1)           (resolution-agnostic — works with any input size)
        → [Linear → ReLU → Dropout]    × n_fc
        → Linear(n_classes)
    """

    def __init__(
        self,
        in_channels: int      = 1,
        n_classes: int        = 10,
        n_conv_layers: int    = 2,
        conv_channels: list   = None,
        n_fc_layers: int      = 1,
        fc_units: int         = 256,
        dropout: float        = 0.3,
        input_resolution: int = 28,
    ):
        super().__init__()
        if conv_channels is None:
            conv_channels = [64] * n_conv_layers

        # ── Conv backbone ────────────────────────────────────────────────────
        blocks = []
        ch_in = in_channels
        for i, ch_out in enumerate(conv_channels):
            blocks += [
                nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
            ]
            if i < n_conv_layers - 1:
                blocks.append(nn.MaxPool2d(2))
            ch_in = ch_out

        self.conv_backbone   = nn.Sequential(*blocks)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # (B, C, 1, 1)

        # ── FC head ──────────────────────────────────────────────────────────
        fc_in  = conv_channels[-1]
        fc_blk = []
        for _ in range(n_fc_layers):
            fc_blk += [nn.Linear(fc_in, fc_units), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            fc_in = fc_units
        fc_blk.append(nn.Linear(fc_in, n_classes))
        self.fc_head = nn.Sequential(*fc_blk)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_backbone(x)       # (B, C, H', W')
        x = self.global_avg_pool(x)     # (B, C, 1, 1)
        x = x.flatten(start_dim=1)      # (B, C)
        return self.fc_head(x)          # (B, n_classes)


def count_parameters(model: nn.Module) -> int:
    """Total trainable parameters (unwraps DataParallel if needed)."""
    m = model.module if isinstance(model, nn.DataParallel) else model
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# Quick forward pass test
_m = ConfigurableCNN()
_x = torch.randn(4, 1, 28, 28)
assert _m(_x).shape == (4, 10), "Shape mismatch!"
print(f"✓ ConfigurableCNN OK | params={count_parameters(_m):,}")
del _m, _x


# %% [markdown]
# ## Cell 7 — Training & Evaluation Utilities (AMP enabled)

# %%
# One GradScaler per process — shared across all trials
_scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    One full training epoch with Automatic Mixed Precision (AMP).
    AMP uses float16 for forward/backward, float32 for weight updates.
    This roughly doubles GPU throughput on Tensor Core GPUs (T4, V100, A100).
    """
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            loss = criterion(model(images), labels)

        _scaler.scale(loss).backward()
        _scaler.step(optimizer)
        _scaler.update()

        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    """Classification accuracy in [0, 1] with AMP inference."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / total


def measure_inference_time(model, loader, device, n_warmup=10, n_measure=100):
    """
    Mean per-sample inference latency in milliseconds.
    Uses CUDA Events for accurate GPU timing (not time.perf_counter).
    Times the base model only (not DataParallel wrapper) — single GPU latency.
    """
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    base_model.eval()
    times = []
    count = 0

    with torch.no_grad():
        for images, _ in loader:
            for i in range(images.size(0)):
                single = images[i : i + 1].to(device, non_blocking=True)

                if device.type == "cuda":
                    start_ev = torch.cuda.Event(enable_timing=True)
                    end_ev   = torch.cuda.Event(enable_timing=True)
                    start_ev.record()
                    with torch.cuda.amp.autocast(enabled=USE_AMP):
                        _ = base_model(single)
                    end_ev.record()
                    torch.cuda.synchronize()
                    elapsed_ms = start_ev.elapsed_time(end_ev)
                else:
                    t0 = time.perf_counter()
                    _ = base_model(single)
                    elapsed_ms = (time.perf_counter() - t0) * 1_000

                if count >= n_warmup:
                    times.append(elapsed_ms)
                count += 1
                if count >= n_warmup + n_measure:
                    return float(np.mean(times))

    return float(np.mean(times)) if times else float("inf")


def train_model(model, train_loader, val_loader, optimizer, device, epochs):
    """
    Train for `epochs` epochs. Returns best validation accuracy.
    NOTE: trial.report / trial.should_prune are NOT used — Optuna does not
    support pruning in multi-objective (NSGA-II) studies.
    """
    criterion = nn.CrossEntropyLoss()
    best_acc  = 0.0
    for epoch in range(epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        acc = evaluate(model, val_loader, device)
        if acc > best_acc:
            best_acc = acc
    return best_acc


print("✓ Training utilities defined  (AMP={})".format(USE_AMP))


# %% [markdown]
# ## Cell 8 — Pareto Front Utilities

# %%
def is_dominated(costs: np.ndarray, i: int) -> bool:
    """True if point i is dominated by any other point (all objectives minimised)."""
    for j in range(len(costs)):
        if j == i:
            continue
        if np.all(costs[j] <= costs[i]) and np.any(costs[j] < costs[i]):
            return True
    return False


def extract_pareto_front(trials):
    """Return the list of non-dominated Optuna trials."""
    costs = np.array([t.values for t in trials])
    return [t for i, t in enumerate(trials) if not is_dominated(costs, i)]


def trials_to_df(trials):
    """Convert a list of Optuna FrozenTrials → tidy pandas DataFrame."""
    rows = []
    for t in trials:
        row = {
            "trial":        t.number,
            "val_accuracy": -t.values[0],   # flip sign: stored as -acc
            "inference_ms": t.values[1],
            "n_params":     int(t.values[2]),
        }
        row.update(t.params)
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values("val_accuracy", ascending=False)
        .reset_index(drop=True)
    )


print("✓ Pareto utilities defined")


# %% [markdown]
# ## Cell 9 — Optuna Objective

# %%
def objective(trial: optuna.Trial):
    """
    Optuna 3-objective trial function.

    Search space is sized for T4 GPU (larger batches, bigger channels).
    Returns: (-val_accuracy, inference_ms, n_params)  — all minimised.
    """
    trial_seed = SEED + trial.number
    set_seed(trial_seed)

    # ── Search space ──────────────────────────────────────────────────────────
    n_conv     = trial.suggest_int("n_conv_layers", 1, 3)
    conv_chs   = [
        trial.suggest_categorical(f"conv_ch_{i}", [32, 64, 128])   # bigger channels
        for i in range(n_conv)
    ]
    n_fc       = trial.suggest_int("n_fc_layers", 1, 2)
    fc_units   = trial.suggest_categorical("fc_units", [128, 256, 512])   # bigger FC
    dropout    = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
    lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024]) # large batches for GPU
    epochs     = trial.suggest_int("epochs", 3, 8)
    opt_name   = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    resolution = trial.suggest_categorical("input_resolution", [14, 20, 28])

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=batch_size,
        input_resolution=resolution,
        data_dir=DATA_DIR,
        num_workers=NUM_WORKERS,
        seed=trial_seed,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ConfigurableCNN(
        in_channels=IN_CHANNELS,
        n_classes=N_CLASSES,
        n_conv_layers=n_conv,
        conv_channels=conv_chs,
        n_fc_layers=n_fc,
        fc_units=fc_units,
        dropout=dropout,
        input_resolution=resolution,
    )

    # Wrap in DataParallel to use both T4s for each trial
    if USE_DATA_PARALLEL:
        model = nn.DataParallel(model)

    model = model.to(DEVICE)
    n_params = count_parameters(model)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    if opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    # ── Train ─────────────────────────────────────────────────────────────────
    val_acc = train_model(model, train_loader, val_loader, optimizer, DEVICE, epochs)

    # ── Inference timing ──────────────────────────────────────────────────────
    model.eval()
    inf_ms = measure_inference_time(model, val_loader, DEVICE, n_warmup=10, n_measure=50)

    # ── Cleanup — prevents memory leaks across 60 trials ─────────────────────
    del model, optimizer, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()

    # All three objectives are minimised internally:
    #   obj1 = -accuracy  → minimising -acc  ≡  maximising accuracy
    #   obj2 = latency ms → minimise
    #   obj3 = # params   → minimise
    return (-val_acc, inf_ms, float(n_params))


print("✓ Objective defined")


# %% [markdown]
# ## Cell 10 — Run the Optuna Study

# %%
set_seed(SEED)

study = optuna.create_study(
    directions=["minimize", "minimize", "minimize"],
    sampler=NSGAIISampler(seed=SEED),
    study_name="moo_fashion_mnist_kaggle",
)

print(f"Starting optimisation: {N_TRIALS} trials on {DEVICE} ({N_GPUS} GPU(s)) …")
print(f"AMP={USE_AMP}  |  DataParallel={USE_DATA_PARALLEL}\n")

study.optimize(
    objective,
    n_trials=N_TRIALS,
    n_jobs=1,               # Keep 1: DataParallel already uses BOTH T4s per trial
    show_progress_bar=True,
    gc_after_trial=True,
)

completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
failed    = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

print(f"\nCompleted : {len(completed)}")
print(f"Pruned    : {len(pruned)}")
print(f"Failed    : {len(failed)}")


# %% [markdown]
# ## Cell 11 — Pareto Front & Print Results

# %%
pareto_trials = extract_pareto_front(completed)
all_df        = trials_to_df(completed)
pareto_df     = trials_to_df(pareto_trials)

print(f"Pareto-optimal solutions: {len(pareto_trials)}\n")
print("=" * 78)
print("  PARETO-OPTIMAL SOLUTIONS  (sorted by Validation Accuracy ↓)")
print("=" * 78)

HP_KEYS = [
    "n_conv_layers", "n_fc_layers", "fc_units",
    "lr", "batch_size", "epochs", "optimizer", "dropout", "input_resolution",
]

for _, row in pareto_df.iterrows():
    print(
        f"  Trial #{int(row['trial']):>3d}  |"
        f"  Acc: {row['val_accuracy']:.4f}  |"
        f"  Latency: {row['inference_ms']:>7.3f} ms  |"
        f"  Params: {int(row['n_params']):>9,}"
    )
    hp_str = "    " + "  |  ".join(f"{k}={row[k]}" for k in HP_KEYS if k in row)
    print(hp_str)
    print()

print("=" * 78)


# %% [markdown]
# ## Cell 12 — Save CSVs

# %%
all_csv    = os.path.join(RESULTS_DIR, "all_trials.csv")
pareto_csv = os.path.join(RESULTS_DIR, "pareto_trials.csv")

all_df.to_csv(all_csv,    index=False)
pareto_df.to_csv(pareto_csv, index=False)

print(f"✓ Saved → {all_csv}    ({len(all_df)} rows)")
print(f"✓ Saved → {pareto_csv}  ({len(pareto_df)} rows)")
all_df.head(5)


# %% [markdown]
# ## Cell 13 — 2-D Pareto Scatter Plots

# %%
def plot_2d(all_df, pareto_df, save_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Multi-Objective Pareto Front — Fashion-MNIST CNN (Kaggle T4×2 + AMP)",
        fontsize=14, fontweight="bold",
    )

    pairs = [
        ("inference_ms", "val_accuracy", "Inference Time (ms / sample)", "Val Accuracy",   "Accuracy vs Latency"),
        ("n_params",     "val_accuracy", "# Trainable Parameters",       "Val Accuracy",   "Accuracy vs Model Size"),
        ("n_params",     "inference_ms", "# Trainable Parameters",       "Inference (ms)", "Latency vs Model Size"),
    ]
    for ax, (xcol, ycol, xl, yl, title) in zip(axes, pairs):
        ax.scatter(
            all_df[xcol],    all_df[ycol],
            c="steelblue", alpha=0.35, s=25, label="All trials", zorder=2,
        )
        ax.scatter(
            pareto_df[xcol], pareto_df[ycol],
            c="crimson", s=70, edgecolors="k", linewidths=0.6,
            label="Pareto front", zorder=3,
        )
        ax.set_xlabel(xl, fontsize=11)
        ax.set_ylabel(yl, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "pareto_2d.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✓ Saved → {path}")

plot_2d(all_df, pareto_df, RESULTS_DIR)


# %% [markdown]
# ## Cell 14 — 3-D Pareto Scatter Plot

# %%
def plot_3d(all_df, pareto_df, save_dir):
    fig = plt.figure(figsize=(11, 8))
    ax  = fig.add_subplot(111, projection="3d")

    ax.scatter(
        all_df["inference_ms"], all_df["n_params"], all_df["val_accuracy"],
        c="steelblue", alpha=0.25, s=20, label="All trials",
    )
    sc = ax.scatter(
        pareto_df["inference_ms"], pareto_df["n_params"], pareto_df["val_accuracy"],
        c=pareto_df["val_accuracy"], cmap="RdYlGn",
        s=90, edgecolors="black", linewidths=0.5, label="Pareto front", zorder=5,
    )
    fig.colorbar(sc, ax=ax, shrink=0.5, label="Val Accuracy")

    ax.set_xlabel("Inference (ms)",  fontsize=10, labelpad=8)
    ax.set_ylabel("# Parameters",   fontsize=10, labelpad=8)
    ax.set_zlabel("Val Accuracy",   fontsize=10, labelpad=8)
    ax.set_title("3-D Pareto Front — Fashion-MNIST MOO", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    path = os.path.join(save_dir, "pareto_3d.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✓ Saved → {path}")

plot_3d(all_df, pareto_df, RESULTS_DIR)


# %% [markdown]
# ## Cell 15 — Distributions

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Trial Distributions", fontsize=13, fontweight="bold")

axes[0].hist(all_df["val_accuracy"], bins=20, color="steelblue", edgecolor="white", alpha=0.85)
axes[0].set_xlabel("Validation Accuracy", fontsize=11)
axes[0].set_ylabel("Count", fontsize=11)
axes[0].set_title("Accuracy Distribution", fontsize=12)
axes[0].grid(True, alpha=0.3)

axes[1].hist(
    np.log10(all_df["n_params"].astype(float)), bins=20,
    color="darkorchid", edgecolor="white", alpha=0.85,
)
axes[1].set_xlabel("log₁₀(# Parameters)", fontsize=11)
axes[1].set_ylabel("Count", fontsize=11)
axes[1].set_title("Model Size Distribution", fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
dist_path = os.path.join(RESULTS_DIR, "distributions.png")
fig.savefig(dist_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"✓ Saved → {dist_path}")


# %% [markdown]
# ## Cell 16 — List & Download Results

# %%
from IPython.display import FileLink, display

print("📁 Files in results directory:\n")
for f in sorted(os.listdir(RESULTS_DIR)):
    fp   = os.path.join(RESULTS_DIR, f)
    size = os.path.getsize(fp)
    print(f"  {f:40s}  {size / 1024:.1f} KB")
    display(FileLink(fp))

print("\n✅ All done! Download above or find files in /kaggle/working/results/")


# %% [markdown]
# ## Cell 17 — Pareto Quality Metrics: Hypervolume & Spacing Metric

# %%
# ─────────────────────────────────────────────────────────────────────────────
#  Helper: Non-dominated filter  (minimisation, all objectives)
# ─────────────────────────────────────────────────────────────────────────────
def _nd_filter(pts: np.ndarray) -> np.ndarray:
    """Return non-dominated subset of pts (all objectives minimised)."""
    mask = np.ones(len(pts), dtype=bool)
    for i, p in enumerate(pts):
        if not mask[i]:
            continue
        others = pts[mask]
        if (np.all(others <= p, axis=1) & np.any(others < p, axis=1)).any():
            mask[i] = False
    return pts[mask]


def _minmax_normalise(pts: np.ndarray, ref: np.ndarray) -> tuple:
    """Min-max normalise pts and ref to [0,1] using column min & ref as max."""
    ideal = pts.min(axis=0)
    scale = ref - ideal
    scale = np.where(scale == 0, 1.0, scale)
    return (pts - ideal) / scale, np.ones_like(ref)


# ─────────────────────────────────────────────────────────────────────────────
#  Hypervolume — WFG recursive algorithm (exact, no external libs)
# ─────────────────────────────────────────────────────────────────────────────
def _limit_pts(pts: np.ndarray, ref: np.ndarray) -> np.ndarray:
    if pts.size == 0:
        return pts
    clipped = np.maximum(pts, ref)
    return clipped[np.all(clipped <= ref, axis=1)]


def _wfg(pts: np.ndarray, ref: np.ndarray) -> float:
    if pts.size == 0:
        return 0.0
    n, d = pts.shape
    if d == 1:
        return float(ref[0] - pts[:, 0].min())
    pts = pts[np.argsort(pts[:, -1])]
    hv = 0.0
    for i, p in enumerate(pts):
        depth = (pts[i + 1, -1] if i + 1 < n else ref[-1]) - p[-1]
        if depth <= 0:
            continue
        dominated = _limit_pts(pts[i + 1:], p)
        sub_pts = np.vstack([p[:-1], dominated[:, :-1]]) if dominated.size else p[:-1][np.newaxis]
        hv += depth * _wfg(_nd_filter(sub_pts), ref[:-1])
    return hv


def compute_hypervolume(pareto_pts: np.ndarray, ref: np.ndarray) -> float:
    """
    Hypervolume Indicator (normalised).

    Parameters
    ----------
    pareto_pts : (n, 3) array  — objectives in minimisation form
                 [−accuracy, latency_ms, n_params]
    ref        : (3,)   array  — reference (worst acceptable) point

    Returns
    -------
    float  Normalised hypervolume in [0, 1].
    """
    feasible = pareto_pts[np.all(pareto_pts < ref, axis=1)]
    nd_pts   = _nd_filter(feasible)
    norm_pts, norm_ref = _minmax_normalise(nd_pts, ref)
    return _wfg(norm_pts, norm_ref)


# ─────────────────────────────────────────────────────────────────────────────
#  Spacing Metric — Schott (1995)
# ─────────────────────────────────────────────────────────────────────────────
def compute_spacing(pareto_pts: np.ndarray) -> dict:
    """
    Spacing Metric SP — uniformity of Pareto front distribution.

    SP = sqrt( mean_i( (d̄ - d_i)² ) )

    where d_i = min_{j≠i} ||pts_i - pts_j||₂  in normalised objective space.

    Parameters
    ----------
    pareto_pts : (n, 3) array — objectives in minimisation form (raw values).
                 Normalised internally before distance computation.

    Returns
    -------
    dict  with keys sp, d_mean, d_min, d_max, d_values.
    """
    # Normalise each objective to [0, 1]
    col_min   = pareto_pts.min(axis=0)
    col_range = pareto_pts.max(axis=0) - col_min
    col_range = np.where(col_range == 0, 1.0, col_range)
    pts = (pareto_pts - col_min) / col_range

    n = len(pts)
    d = np.empty(n)
    for i in range(n):
        dists    = np.linalg.norm(pts - pts[i], axis=1)
        dists[i] = np.inf
        d[i]     = dists.min()

    d_mean = d.mean()
    sp     = float(np.sqrt(np.mean((d_mean - d) ** 2)))
    return {"sp": sp, "d_mean": float(d_mean),
            "d_min": float(d.min()), "d_max": float(d.max()),
            "d_values": d.tolist()}


# ─────────────────────────────────────────────────────────────────────────────
#  Run on the live Pareto front
# ─────────────────────────────────────────────────────────────────────────────

# Reference point: worst acceptable values (all objectives in minimisation form)
#   O1 = −accuracy  →  −0.30  (accuracy ≤ 0.30 is unacceptable)
#   O2 = latency_ms →   4.00 ms
#   O3 = n_params   → 700,000
REF_POINT = np.array([-0.30, 4.00, 700_000.0])

# Build (n, 3) minimisation array from the live pareto_df
pareto_min = np.column_stack([
    -pareto_df["val_accuracy"].values,   # negate: maximise → minimise
     pareto_df["inference_ms"].values,
     pareto_df["n_params"].values.astype(float),
])

n_pareto = len(pareto_min)

# ── Hypervolume ───────────────────────────────────────────────────────────────
hv = compute_hypervolume(pareto_min, REF_POINT)

# ── Spacing Metric ────────────────────────────────────────────────────────────
sp_result = compute_spacing(pareto_min)

# ── Per-solution nearest-neighbour table ──────────────────────────────────────
print("=" * 60)
print(f"  PARETO QUALITY METRICS  ({n_pareto} solutions)")
print("=" * 60)

print(f"\n{'Solution':<10}  {'Val Acc':>8}  {'Latency (ms)':>13}  {'Params':>10}  {'d_i (NN)':>10}")
print("-" * 60)
d_vals = sp_result["d_values"]
for idx, (_, row) in enumerate(pareto_df.iterrows()):
    print(
        f"  P{idx+1:02d}      "
        f"  {row['val_accuracy']:>8.4f}"
        f"  {row['inference_ms']:>13.4f}"
        f"  {int(row['n_params']):>10,}"
        f"  {d_vals[idx]:>10.6f}"
    )

print("-" * 60)
print(f"\n  {'Hypervolume Indicator  HV':<35}:  {hv:.6f}")
print(f"  {'Spacing Metric         SP':<35}:  {sp_result['sp']:.6f}")
print(f"  {'Mean NN distance       d̄':<35}:  {sp_result['d_mean']:.6f}")
print(f"  {'Min  NN distance       d_min':<35}:  {sp_result['d_min']:.6f}")
print(f"  {'Max  NN distance       d_max':<35}:  {sp_result['d_max']:.6f}")
print("=" * 60)

# ── Qualitative interpretation ────────────────────────────────────────────────
sp_val = sp_result["sp"]
if   sp_val < 0.05:  sp_quality = "Excellent — very uniform spread."
elif sp_val < 0.10:  sp_quality = "Good spread with minor clustering."
elif sp_val < 0.20:  sp_quality = "Moderate spread; some gaps in the front."
else:                sp_quality = "Poor spread; front is highly irregular."

print(f"\n  HV  interpretation: {hv*100:.1f}% of normalised objective space dominated.")
print(f"  SP  interpretation: {sp_quality}")

# ── Save metrics to CSV ───────────────────────────────────────────────────────
metrics_dict = {
    "n_pareto_solutions": [n_pareto],
    "hypervolume_HV":     [round(hv, 6)],
    "spacing_SP":         [round(sp_result["sp"], 6)],
    "mean_nn_distance":   [round(sp_result["d_mean"], 6)],
    "min_nn_distance":    [round(sp_result["d_min"], 6)],
    "max_nn_distance":    [round(sp_result["d_max"], 6)],
    "ref_neg_accuracy":   [REF_POINT[0]],
    "ref_latency_ms":     [REF_POINT[1]],
    "ref_n_params":       [REF_POINT[2]],
}
metrics_csv = os.path.join(RESULTS_DIR, "pareto_metrics.csv")
pd.DataFrame(metrics_dict).to_csv(metrics_csv, index=False)
print(f"\n✓ Metrics saved → {metrics_csv}")

