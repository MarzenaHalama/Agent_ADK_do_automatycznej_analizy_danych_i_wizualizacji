"""
Main pipeline -- run all analyses and generate plots.

Usage::

    python -m src.pipeline
"""

from __future__ import annotations

import os
import sys

import pandas as pd

from src.analysis import run_analysis
from src.anomaly_detection import detect_anomalies
from src.config import DATA_DIR, PLOTS_DIR
from src.data_loader import load_csv
from src.visualization import (
    plot_anomaly_scatter,
    plot_confusion_matrix,
    plot_histogram,
    plot_pred_sequence,
    plot_pred_vs_true,
    plot_regression_line,
    plot_residuals,
    plot_scatter,
)


def _ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def run_regression_pipeline() -> None:
    """Train regression models on the synthetic dataset and plot results."""
    csv_path = os.path.join(DATA_DIR, "data_regression.csv")
    if not os.path.exists(csv_path):
        print(f"[SKIP] Regression data not found: {csv_path}")
        return

    out_dir = os.path.join(PLOTS_DIR, "regression")
    _ensure_dirs(out_dir)

    df = load_csv(csv_path)
    results = run_analysis(df, target="y", task="regression")
    mlp = results["mlp"]

    print(
        f"[REG] MLP  R2={mlp['r2']:.3f}  MAE={mlp['mae']:.3f}  MSE={mlp['mse']:.3f}"
    )

    plot_pred_vs_true(
        mlp["y_true"], mlp["y_pred"],
        save_path=os.path.join(out_dir, "mlp_pred_vs_true.png"),
    )
    plot_pred_sequence(
        mlp["y_true"], mlp["y_pred"],
        save_path=os.path.join(out_dir, "mlp_pred_sequence.png"),
    )
    plot_residuals(
        mlp["residuals"],
        save_path=os.path.join(out_dir, "mlp_residuals.png"),
    )

    # EDA scatter for first numeric feature
    num_cols = [c for c in df.select_dtypes("number").columns if c != "y"]
    if num_cols:
        x_col = num_cols[0]
        plot_scatter(
            df, x_col, "y",
            save_path=os.path.join(out_dir, f"scatter_{x_col}_y.png"),
        )
        plot_regression_line(
            df, x_col, "y",
            save_path=os.path.join(out_dir, f"regline_{x_col}_y.png"),
        )


def run_classification_pipeline() -> None:
    """Train classification models on the Iris dataset and plot results."""
    csv_path = os.path.join(DATA_DIR, "data_classification_iris.csv")
    if not os.path.exists(csv_path):
        print(f"[SKIP] Classification data not found: {csv_path}")
        return

    out_dir = os.path.join(PLOTS_DIR, "classification")
    _ensure_dirs(out_dir)

    df = load_csv(csv_path)
    results = run_analysis(df, target="species", task="classification")
    mlp = results["mlp"]

    print(f"[CLS] MLP  Accuracy={mlp['accuracy']:.3f}")

    plot_confusion_matrix(
        mlp["y_true"], mlp["y_pred"],
        labels=mlp.get("classes"),
        save_path=os.path.join(out_dir, "confusion_matrix_iris.png"),
    )

    if "petal_width" in df.columns:
        plot_histogram(
            df, "petal_width",
            save_path=os.path.join(out_dir, "hist_petal_width.png"),
        )
    if {"sepal_length", "petal_length"}.issubset(df.columns):
        plot_scatter(
            df, "sepal_length", "petal_length",
            save_path=os.path.join(out_dir, "scatter_sepal_petal.png"),
        )


def run_anomaly_pipeline() -> None:
    """Detect anomalies in the water-usage time series and plot results."""
    csv_path = os.path.join(DATA_DIR, "data_timeseries_water.csv")
    if not os.path.exists(csv_path):
        print(f"[SKIP] Anomaly data not found: {csv_path}")
        return

    out_dir = os.path.join(PLOTS_DIR, "water_anomaly")
    _ensure_dirs(out_dir)

    df = load_csv(csv_path, date_columns=["date"])
    anomalies = detect_anomalies(df, features=["liters"], contamination=0.14)

    n_anomalies = len(anomalies["anomaly_indices"])
    print(f"[ANO] Detected {n_anomalies} anomalous records")

    # Build a plot-friendly DataFrame with a numeric time axis
    dfp = df.copy()
    dfp["t"] = range(len(dfp))
    label_map = {i: lab for i, lab in zip(anomalies["index"], anomalies["isoforest_labels"])}
    labels = [label_map.get(i, 1) for i in dfp.index]

    plot_anomaly_scatter(
        dfp, "t", "liters", labels,
        save_path=os.path.join(out_dir, "anomaly_water.png"),
    )


def main() -> None:
    """Entry point -- execute all three pipelines."""
    print("=" * 60)
    print("  Data Analysis & Visualization Pipeline")
    print("=" * 60)

    run_regression_pipeline()
    run_classification_pipeline()
    run_anomaly_pipeline()

    print("\nDone. Plots saved to:", PLOTS_DIR)


if __name__ == "__main__":
    main()
