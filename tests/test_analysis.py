"""Tests for the analysis module -- regression and classification pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import compute_statistics, infer_task, run_analysis


# -------------------------------------------------------------------
# Statistics
# -------------------------------------------------------------------

class TestComputeStatistics:
    """Verify basic descriptive statistics."""

    def test_returns_shape(self, regression_df):
        stats = compute_statistics(regression_df)
        assert stats["shape"] == (100, 3)

    def test_returns_dtypes(self, regression_df):
        stats = compute_statistics(regression_df)
        assert "x1" in stats["dtypes"]

    def test_describe_numeric_keys(self, regression_df):
        stats = compute_statistics(regression_df)
        desc = stats["describe_numeric"]
        assert "x1" in desc
        assert "mean" in desc["x1"]

    def test_missing_values_counted(self):
        df = pd.DataFrame({"a": [1, np.nan, 3], "b": [4, 5, 6]})
        stats = compute_statistics(df)
        assert stats["missing_per_column"]["a"] == 1
        assert stats["missing_per_column"]["b"] == 0


# -------------------------------------------------------------------
# Task inference
# -------------------------------------------------------------------

class TestInferTask:
    """Verify automatic task detection."""

    def test_numeric_many_unique_is_regression(self):
        y = pd.Series(np.random.randn(100))
        assert infer_task(y) == "regression"

    def test_numeric_few_unique_is_classification(self):
        y = pd.Series([0, 1, 2] * 10)
        assert infer_task(y) == "classification"

    def test_string_labels_is_classification(self):
        y = pd.Series(["cat", "dog", "cat", "dog"])
        assert infer_task(y) == "classification"


# -------------------------------------------------------------------
# Regression pipeline
# -------------------------------------------------------------------

class TestRegressionPipeline:
    """End-to-end regression analysis tests."""

    def test_returns_required_keys(self, regression_df):
        res = run_analysis(regression_df, target="y", task="regression")
        assert "basic_stats" in res
        assert res["task"] == "regression"
        assert "linear_regression" in res
        assert "mlp" in res

    def test_linear_regression_metrics(self, regression_df):
        res = run_analysis(regression_df, target="y", task="regression")
        lr = res["linear_regression"]
        assert 0.0 <= lr["r2"] <= 1.0
        assert lr["mse"] >= 0
        assert lr["mae"] >= 0

    def test_mlp_regression_metrics(self, regression_df):
        res = run_analysis(regression_df, target="y", task="regression")
        mlp = res["mlp"]
        assert mlp["type"] == "MLPRegressor"
        assert "r2" in mlp
        assert "mse" in mlp
        assert "mae" in mlp
        assert len(mlp["y_true"]) == len(mlp["y_pred"])

    def test_residuals_length_matches(self, regression_df):
        res = run_analysis(regression_df, target="y", task="regression")
        mlp = res["mlp"]
        assert len(mlp["residuals"]) == len(mlp["y_true"])

    def test_high_r2_on_linear_data(self):
        """A perfectly linear dataset should yield R2 close to 1."""
        x = np.linspace(0, 10, 200)
        df = pd.DataFrame({"x": x, "y": 2.5 * x + 1.0})
        res = run_analysis(df, target="y", task="regression")
        assert res["linear_regression"]["r2"] > 0.99

    def test_no_target_returns_stats_only(self, regression_df):
        res = run_analysis(regression_df, target=None)
        assert "basic_stats" in res
        assert "note" in res
        assert "mlp" not in res


# -------------------------------------------------------------------
# Classification pipeline
# -------------------------------------------------------------------

class TestClassificationPipeline:
    """End-to-end classification analysis tests."""

    def test_returns_required_keys(self, classification_df):
        res = run_analysis(classification_df, target="species", task="classification")
        assert res["task"] == "classification"
        assert "mlp" in res
        assert res["linear_regression"] is None

    def test_accuracy_range(self, classification_df):
        res = run_analysis(classification_df, target="species", task="classification")
        acc = res["mlp"]["accuracy"]
        assert 0.0 <= acc <= 1.0

    def test_confusion_matrix_shape(self, classification_df):
        res = run_analysis(classification_df, target="species", task="classification")
        cm = res["mlp"]["confusion_matrix"]
        n_classes = len(res["mlp"]["classes"])
        assert len(cm) == n_classes
        assert all(len(row) == n_classes for row in cm)

    def test_classification_report_keys(self, classification_df):
        res = run_analysis(classification_df, target="species", task="classification")
        report = res["mlp"]["classification_report"]
        assert "accuracy" in report
        assert "weighted avg" in report

    def test_predictions_length(self, classification_df):
        res = run_analysis(classification_df, target="species", task="classification")
        mlp = res["mlp"]
        assert len(mlp["y_true"]) == len(mlp["y_pred"])

    def test_high_accuracy_on_separable_data(self, classification_df):
        """Well-separated classes should yield high accuracy."""
        res = run_analysis(classification_df, target="species", task="classification")
        assert res["mlp"]["accuracy"] > 0.80

    def test_classes_list(self, classification_df):
        res = run_analysis(classification_df, target="species", task="classification")
        classes = res["mlp"]["classes"]
        assert "setosa" in classes
        assert "versicolor" in classes
        assert "virginica" in classes
