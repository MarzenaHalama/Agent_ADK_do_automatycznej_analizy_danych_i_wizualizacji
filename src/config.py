"""
Global configuration for plotting theme, model defaults, and paths.

All visual and model hyperparameters are centralised here so that
every module draws from a single source of truth.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")

# ---------------------------------------------------------------------------
# Plotting theme
# ---------------------------------------------------------------------------
BG_COLOR: str = "#555555"
GRID_COLOR: str = "#d9d9d9"
TEXT_COLOR: str = "#f3f3f3"
PALETTE: list[str] = ["#77e6e2", "#00be9f", "#00bccc"]
FONT_SIZE: int = 12


@dataclass(frozen=True)
class MLPConfig:
    """Default hyper-parameters for MLP models."""

    hidden_layer_sizes: Tuple[int, ...] = (64, 32)
    max_iter: int = 500
    early_stopping: bool = True
    n_iter_no_change: int = 20
    random_state: int = 42


@dataclass(frozen=True)
class AnomalyConfig:
    """Default hyper-parameters for anomaly detection."""

    contamination: float = 0.05
    zscore_threshold: float = 3.0
    random_state: int = 42


# Convenient singletons
MLP_DEFAULTS = MLPConfig()
ANOMALY_DEFAULTS = AnomalyConfig()
