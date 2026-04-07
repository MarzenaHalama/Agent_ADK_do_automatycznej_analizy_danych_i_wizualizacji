"""
Publication-ready visualizations with a consistent dark theme.

All plots share the same colour palette, background, grid style, and
font settings so that outputs are visually cohesive regardless of
which chart type is produced.

Supported chart types
---------------------
- ``histogram`` -- frequency distribution of a single column.
- ``boxplot`` -- five-number summary with outlier markers.
- ``scatter`` -- two-variable scatter plot.
- ``reg_line`` -- scatter with a linear-fit overlay.
- ``confusion_matrix`` -- classification confusion matrix heatmap.
- ``pred_vs_true`` -- predicted vs actual scatter (regression quality).
- ``pred_sequence`` -- predicted and actual as line series.
- ``residuals`` -- histogram of prediction residuals.
- ``anomaly_scatter`` -- 2-D scatter with anomaly highlights.
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import ConfusionMatrixDisplay

from src.config import BG_COLOR, FONT_SIZE, GRID_COLOR, PALETTE, TEXT_COLOR

# Convenience aliases for the three palette colours
C0, C1, C2 = PALETTE

# Custom colourmap for heatmaps
CMAP = LinearSegmentedColormap.from_list("brand", [C1, C2, C0], N=256)


def apply_theme() -> None:
    """Apply the global Matplotlib theme.

    Called automatically on module import so that every figure
    produced after ``import src.visualization`` uses the project theme.
    """
    plt.rcParams.update({
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": BG_COLOR,
        "axes.grid": True,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.4,
        "grid.linestyle": "-",
        "grid.linewidth": 0.8,
        "axes.edgecolor": TEXT_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "text.color": TEXT_COLOR,
        "font.size": FONT_SIZE,
        "legend.facecolor": BG_COLOR,
        "legend.edgecolor": GRID_COLOR,
        "legend.framealpha": 0.6,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
    })


# Apply theme on import
apply_theme()


# -------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------

def _finalise(fig: plt.Figure, save_path: Optional[str]) -> None:
    """Tight-layout the figure and optionally save to disk."""
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=140, facecolor=BG_COLOR)
        plt.close(fig)


# -------------------------------------------------------------------
# Public plotting functions
# -------------------------------------------------------------------

def plot_histogram(
    df: pd.DataFrame,
    column: str,
    *,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a histogram of a single column."""
    data = df[column].dropna()
    fig, ax = plt.subplots()
    ax.hist(data, bins=30, color=C0, edgecolor=BG_COLOR)
    ax.set_title(f"Histogram: {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    _finalise(fig, save_path)
    return fig


def plot_boxplot(
    df: pd.DataFrame,
    column: str,
    *,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a box-and-whisker chart for a single column."""
    data = df[column].dropna().values
    fig, ax = plt.subplots()
    ax.boxplot(
        data,
        vert=True,
        labels=[column],
        patch_artist=True,
        boxprops=dict(facecolor=C0, color=GRID_COLOR),
        medianprops=dict(color=TEXT_COLOR, linewidth=2),
        whiskerprops=dict(color=GRID_COLOR),
        capprops=dict(color=GRID_COLOR),
        flierprops=dict(
            marker="o",
            markerfacecolor=C2,
            markeredgecolor=BG_COLOR,
            alpha=0.9,
        ),
    )
    ax.set_title(f"Boxplot: {column}")
    _finalise(fig, save_path)
    return fig


def plot_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a scatter chart for two columns."""
    fig, ax = plt.subplots()
    ax.scatter(
        df[x], df[y],
        s=45, alpha=0.9, color=C0, edgecolors=BG_COLOR, linewidths=0.6,
    )
    ax.set_title(f"Scatter: {x} vs {y}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    _finalise(fig, save_path)
    return fig


def plot_regression_line(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot data points with a linear-fit line."""
    xv, yv = df[x].values, df[y].values
    m, b = np.polyfit(xv, yv, 1)
    xs = np.linspace(xv.min(), xv.max(), 200)

    fig, ax = plt.subplots()
    ax.scatter(
        xv, yv,
        s=45, alpha=0.9, color=C1, edgecolors=BG_COLOR, linewidths=0.6,
        label="Data",
    )
    ax.plot(xs, m * xs + b, color=C0, linewidth=3, label="Linear fit")
    ax.set_title(f"Regression: {x} -> {y}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend()
    _finalise(fig, save_path)
    return fig


def plot_confusion_matrix(
    y_true,
    y_pred,
    *,
    labels=None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a confusion matrix heatmap."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=labels, cmap=CMAP, ax=ax, colorbar=False,
    )
    plt.setp(disp.text_, color="#000000")
    for spine in ax.spines.values():
        spine.set_color(TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    ax.set_title("Confusion Matrix", color=TEXT_COLOR)
    ax.grid(False)
    _finalise(fig, save_path)
    return fig


def plot_pred_vs_true(
    y_true,
    y_pred,
    *,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Scatter of predicted vs actual values (regression quality check)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())

    fig, ax = plt.subplots()
    ax.scatter(
        y_true, y_pred,
        s=45, alpha=0.9, color=C0, edgecolors=BG_COLOR, linewidths=0.6,
        label="Samples",
    )
    ax.plot([lo, hi], [lo, hi], color=C2, linewidth=3, label="Ideal")
    ax.set_title("Predicted vs Actual")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.legend()
    _finalise(fig, save_path)
    return fig


def plot_pred_sequence(
    y_true,
    y_pred,
    *,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Line chart comparing predicted and actual by test-set index."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    fig, ax = plt.subplots()
    ax.plot(y_true, linewidth=3, color=C1, label="Actual")
    ax.plot(y_pred, linewidth=3, color=C0, label="Predicted")
    ax.set_title("Prediction Sequence (Test Set)")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.legend()
    _finalise(fig, save_path)
    return fig


def plot_residuals(
    residuals,
    *,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Histogram of prediction residuals."""
    residuals = np.asarray(residuals)

    fig, ax = plt.subplots()
    ax.hist(residuals, bins=30, color=C0, edgecolor=BG_COLOR)
    ax.set_title("Residuals Distribution")
    ax.set_xlabel("Residual (actual - predicted)")
    ax.set_ylabel("Count")
    _finalise(fig, save_path)
    return fig


def plot_anomaly_scatter(
    df: pd.DataFrame,
    f1: str,
    f2: str,
    labels,
    *,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """2-D scatter highlighting anomalies.

    Parameters
    ----------
    labels : array-like
        1 = normal, -1 = anomaly (Isolation Forest convention).
    """
    labels = np.asarray(labels)
    normal = labels == 1
    anomaly = labels == -1

    fig, ax = plt.subplots()
    ax.scatter(
        df.loc[normal, f1], df.loc[normal, f2],
        s=45, alpha=0.85, color=C0, edgecolors=BG_COLOR, linewidths=0.6,
        label="Normal",
    )
    ax.scatter(
        df.loc[anomaly, f1], df.loc[anomaly, f2],
        s=80, alpha=1.0, color=C2, marker="x", linewidths=2.4,
        label="Anomaly",
    )
    ax.set_title(f"Anomaly Detection: {f1} vs {f2}")
    ax.set_xlabel(f1)
    ax.set_ylabel(f2)
    ax.legend()
    _finalise(fig, save_path)
    return fig
