"""Tests for the visualization module.

These tests verify that each plotting function:
  1. Returns a matplotlib Figure object.
  2. Saves to disk when ``save_path`` is provided.
  3. Does not raise exceptions on valid input.

Visual correctness is not programmatically verified -- the focus is on
structural integrity and I/O.
"""

from __future__ import annotations

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.visualization import (
    apply_theme,
    plot_anomaly_scatter,
    plot_boxplot,
    plot_confusion_matrix,
    plot_histogram,
    plot_pred_sequence,
    plot_pred_vs_true,
    plot_regression_line,
    plot_residuals,
    plot_scatter,
)

# Use non-interactive backend for CI
matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _close_figures():
    """Ensure all figures are closed after each test."""
    yield
    plt.close("all")


class TestHistogram:
    def test_returns_figure(self, regression_df):
        fig = plot_histogram(regression_df, "x1")
        assert isinstance(fig, plt.Figure)

    def test_saves_to_disk(self, regression_df, tmp_path):
        path = str(tmp_path / "hist.png")
        plot_histogram(regression_df, "x1", save_path=path)
        assert os.path.isfile(path)


class TestBoxplot:
    def test_returns_figure(self, regression_df):
        fig = plot_boxplot(regression_df, "x1")
        assert isinstance(fig, plt.Figure)

    def test_saves_to_disk(self, regression_df, tmp_path):
        path = str(tmp_path / "box.png")
        plot_boxplot(regression_df, "x1", save_path=path)
        assert os.path.isfile(path)


class TestScatter:
    def test_returns_figure(self, regression_df):
        fig = plot_scatter(regression_df, "x1", "y")
        assert isinstance(fig, plt.Figure)

    def test_saves_to_disk(self, regression_df, tmp_path):
        path = str(tmp_path / "scatter.png")
        plot_scatter(regression_df, "x1", "y", save_path=path)
        assert os.path.isfile(path)


class TestRegressionLine:
    def test_returns_figure(self, regression_df):
        fig = plot_regression_line(regression_df, "x1", "y")
        assert isinstance(fig, plt.Figure)

    def test_saves_to_disk(self, regression_df, tmp_path):
        path = str(tmp_path / "regline.png")
        plot_regression_line(regression_df, "x1", "y", save_path=path)
        assert os.path.isfile(path)


class TestConfusionMatrix:
    def test_returns_figure(self):
        y_true = ["a", "b", "a", "b", "a"]
        y_pred = ["a", "b", "b", "b", "a"]
        fig = plot_confusion_matrix(y_true, y_pred)
        assert isinstance(fig, plt.Figure)

    def test_saves_to_disk(self, tmp_path):
        path = str(tmp_path / "cm.png")
        y_true = ["a", "b", "a"]
        y_pred = ["a", "a", "a"]
        plot_confusion_matrix(y_true, y_pred, save_path=path)
        assert os.path.isfile(path)


class TestPredVsTrue:
    def test_returns_figure(self):
        fig = plot_pred_vs_true([1, 2, 3], [1.1, 2.2, 2.8])
        assert isinstance(fig, plt.Figure)

    def test_saves_to_disk(self, tmp_path):
        path = str(tmp_path / "pvt.png")
        plot_pred_vs_true([1, 2, 3], [1, 2, 3], save_path=path)
        assert os.path.isfile(path)


class TestPredSequence:
    def test_returns_figure(self):
        fig = plot_pred_sequence([1, 2, 3, 4], [1.1, 1.9, 3.2, 3.8])
        assert isinstance(fig, plt.Figure)

    def test_saves_to_disk(self, tmp_path):
        path = str(tmp_path / "seq.png")
        plot_pred_sequence([1, 2], [1, 2], save_path=path)
        assert os.path.isfile(path)


class TestResiduals:
    def test_returns_figure(self):
        fig = plot_residuals([0.1, -0.2, 0.05, -0.3])
        assert isinstance(fig, plt.Figure)

    def test_saves_to_disk(self, tmp_path):
        path = str(tmp_path / "resid.png")
        plot_residuals([0.1, -0.1], save_path=path)
        assert os.path.isfile(path)


class TestAnomalyScatter:
    def test_returns_figure(self):
        df = pd.DataFrame({"t": range(10), "val": range(10)})
        labels = np.array([1] * 8 + [-1] * 2)
        fig = plot_anomaly_scatter(df, "t", "val", labels)
        assert isinstance(fig, plt.Figure)

    def test_saves_to_disk(self, tmp_path):
        path = str(tmp_path / "anomaly.png")
        df = pd.DataFrame({"t": range(5), "val": range(5)})
        labels = np.array([1, 1, 1, -1, 1])
        plot_anomaly_scatter(df, "t", "val", labels, save_path=path)
        assert os.path.isfile(path)


class TestTheme:
    def test_apply_theme_sets_background(self):
        apply_theme()
        assert plt.rcParams["figure.facecolor"] == "#555555"
        assert plt.rcParams["axes.facecolor"] == "#555555"
