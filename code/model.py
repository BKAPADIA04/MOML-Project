"""
model.py
--------
Configurable CNN for Fashion-MNIST classification.
Supports variable numbers of convolutional and fully-connected layers,
dropout, and arbitrary input resolutions.
"""

import math
from typing import List

import torch
import torch.nn as nn


class ConfigurableCNN(nn.Module):
    """
    A flexible CNN whose architecture is driven entirely by hyperparameters.

    Architecture
    ------------
    [Conv Block] * n_conv_layers  →  [Global Avg Pool]  →  [FC Block] * n_fc_layers  →  [Output]

    Each Conv Block:
        Conv2d  →  BatchNorm2d  →  ReLU  →  MaxPool2d(2)   (only first n_conv_layers-1 blocks pool)

    Each FC Block (except last):
        Linear  →  ReLU  →  Dropout

    Args:
        in_channels:      Input channels (1 for Fashion-MNIST).
        n_classes:        Number of output classes (10).
        n_conv_layers:    Number of convolutional blocks.
        conv_channels:    List of output channel sizes for each conv block.
        n_fc_layers:      Number of hidden FC layers (before output).
        fc_units:         Hidden units in each FC layer.
        dropout:          Dropout probability applied after each hidden FC layer.
        input_resolution: Spatial side length of the input image.
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 10,
        n_conv_layers: int = 2,
        conv_channels: List[int] = None,
        n_fc_layers: int = 1,
        fc_units: int = 128,
        dropout: float = 0.3,
        input_resolution: int = 28,
    ) -> None:
        super().__init__()

        if conv_channels is None:
            conv_channels = [32] * n_conv_layers
        if len(conv_channels) != n_conv_layers:
            raise ValueError(
                f"len(conv_channels)={len(conv_channels)} must equal "
                f"n_conv_layers={n_conv_layers}"
            )

        # ── Convolutional backbone ──────────────────────────────────────────
        conv_blocks: List[nn.Module] = []
        in_ch = in_channels
        for i, out_ch in enumerate(conv_channels):
            conv_blocks += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            # Add spatial pooling for all layers except the last conv block
            # to avoid reducing feature maps to <1×1
            if i < n_conv_layers - 1:
                conv_blocks.append(nn.MaxPool2d(2))
            in_ch = out_ch

        self.conv_backbone = nn.Sequential(*conv_blocks)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # → (B, C, 1, 1)

        # ── Fully-connected head ────────────────────────────────────────────
        fc_in = conv_channels[-1]  # flattened size after GAP
        fc_layers: List[nn.Module] = []
        for _ in range(n_fc_layers):
            fc_layers += [
                nn.Linear(fc_in, fc_units),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ]
            fc_in = fc_units

        # Output projection
        fc_layers.append(nn.Linear(fc_in, n_classes))
        self.fc_head = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 1, H, W)
        x = self.conv_backbone(x)          # (B, C, H', W')
        x = self.global_avg_pool(x)        # (B, C, 1, 1)
        x = x.flatten(start_dim=1)         # (B, C)
        x = self.fc_head(x)                # (B, n_classes)
        return x


def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
