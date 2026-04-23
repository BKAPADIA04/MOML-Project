# Multi-Objective Optimisation of Fashion-MNIST CNN

A complete, reproducible project for **multi-objective neural architecture search** on the Fashion-MNIST dataset using [Optuna](https://optuna.org/) with the NSGA-II evolutionary sampler and [PyTorch](https://pytorch.org/).

---

## Objectives

| # | Objective | Direction |
|---|-----------|-----------|
| 1 | Validation Accuracy | **Maximise** |
| 2 | Inference Time (ms / sample) | **Minimise** |
| 3 | # Trainable Parameters | **Minimise** |

---

## Project Structure

```
moo_fashion_mnist/
├── main.py          # Entry point: creates study, runs optimisation, saves results
├── objective.py     # Optuna trial objective (search space + training pipeline)
├── model.py         # Configurable CNN architecture
├── data.py          # Fashion-MNIST DataLoader factory
├── utils.py         # Training loop, validation, inference timing, seeding
├── visualize.py     # Pareto extraction, 2D/3D scatter plots
├── requirements.txt # Python dependencies
└── README.md        # This file
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the optimisation

```bash
# From inside moo_fashion_mnist/
python main.py --n-trials 50 --seed 42
```

The dataset is downloaded automatically to `./data/` on first run.

### 3. Command-line options

| Flag | Default | Description |
|------|---------|-------------|
| `--n-trials` | 50 | Number of Optuna trials |
| `--seed` | 42 | Global random seed |
| `--data-dir` | `./data` | Directory for Fashion-MNIST |
| `--results-dir` | `./results` | Output directory |
| `--n-jobs` | 1 | Parallel trial workers |

**Example (quick 20-trial smoke test):**
```bash
python main.py --n-trials 20 --seed 0
```

---

## Design Summary

### Search Space

| Hyperparameter | Values / Range |
|----------------|----------------|
| `n_conv_layers` | 1, 2, 3 |
| `conv_channels` (per layer) | 16, 32, 64 |
| `n_fc_layers` | 1, 2 |
| `fc_units` | 64, 128, 256 |
| `dropout` | 0.0 – 0.5 (step 0.1) |
| `lr` | 1e-4 – 1e-2 (log scale) |
| `batch_size` | 32, 64, 128 |
| `epochs` | 3 – 8 |
| `optimizer` | Adam, SGD |
| `input_resolution` | 14, 20, 28 |

### Architecture

```
Input (1 × H × H)
   ↓
[Conv2d → BN → ReLU → MaxPool(2)] × (n_conv_layers - 1)
[Conv2d → BN → ReLU]              × 1
   ↓
AdaptiveAvgPool2d(1)              → flattens to (B, C)
   ↓
[Linear → ReLU → Dropout] × n_fc_layers
   ↓
Linear(n_classes=10)
```

`AdaptiveAvgPool2d(1)` decouples the FC head from the spatial dimensions, making the model compatible with any input resolution without size-dependent reshaping.

### Sampler

**NSGA-II** (`optuna.samplers.NSGAIISampler`) — a genetic algorithm designed for multi-objective problems. It maintains a population of diverse, non-dominated solutions and improves them iteratively using crossover and mutation.

### Outputs

After a run, the `results/` directory contains:

| File | Description |
|------|-------------|
| `all_trials.csv` | Full results for every completed trial |
| `pareto_trials.csv` | Non-dominated (Pareto-optimal) solutions only |
| `pareto_2d.png` | Three 2-D scatter plots of the Pareto front |
| `pareto_3d.png` | 3-D scatter plot (accuracy × latency × size) |
| `optuna_pareto_front.png` | Optuna's built-in front (accuracy vs latency) |

---

## Reproducibility

- Global + per-trial seeds are set via `utils.set_seed()`.
- `torch.backends.cudnn.deterministic = True` is enforced.
- Dataset splits use a fixed `torch.Generator` seed.

---

## Runtime Estimate

| Setting | Estimated Time |
|---------|---------------|
| 50 trials, CPU, small models | 60–120 min |
| 50 trials, GPU (e.g. A100) | 10–20 min |
| 20 trials, CPU (smoke test) | 25–50 min |

---

## Dependencies

| Package | Minimum |
|---------|---------|
| torch | 2.0.0 |
| torchvision | 0.15.0 |
| optuna | 3.4.0 |
| matplotlib | 3.7.0 |
| pandas | 2.0.0 |
| numpy | 1.24.0 |
| tqdm | 4.65.0 |
