"""Tests for the data_loader module."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    _clean_column_names,
    _coerce_numeric_columns,
    load_csv,
    prepare_features,
)


class TestCleanColumnNames:
    """Verify BOM and whitespace stripping from column names."""

    def test_strips_bom(self):
        df = pd.DataFrame({"\ufeffname": [1]})
        result = _clean_column_names(df)
        assert "name" in result.columns

    def test_strips_whitespace(self):
        df = pd.DataFrame({"  x  ": [1], " y": [2]})
        result = _clean_column_names(df)
        assert list(result.columns) == ["x", "y"]


class TestCoerceNumericColumns:
    """Verify comma-decimal and whitespace in numbers are handled."""

    def test_comma_decimal(self):
        df = pd.DataFrame({"val": ["1,5", "2,3", "3,7"]})
        result = _coerce_numeric_columns(df)
        assert np.isclose(result["val"].iloc[0], 1.5)

    def test_leaves_text_columns(self):
        df = pd.DataFrame({"name": ["alice", "bob", "charlie"]})
        result = _coerce_numeric_columns(df)
        assert result["name"].dtype == object


class TestLoadCsv:
    """Integration tests for CSV loading."""

    def test_loads_semicolon_csv(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("a;b;c\n1;2;3\n4;5;6\n", encoding="utf-8")
        df = load_csv(str(csv_path))
        assert df.shape == (2, 3)
        assert list(df.columns) == ["a", "b", "c"]

    def test_date_parsing(self, tmp_path):
        csv_path = tmp_path / "dates.csv"
        csv_path.write_text("date;val\n01.09.2025;100\n02.09.2025;200\n")
        df = load_csv(str(csv_path), date_columns=["date"])
        assert pd.api.types.is_datetime64_any_dtype(df["date"])


class TestPrepareFeatures:
    """Verify feature/target preparation logic."""

    def test_auto_selects_numeric_features(self, regression_df):
        X, y = prepare_features(regression_df, target="y")
        assert list(X.columns) == ["x1", "x2"]
        assert len(X) == len(y)

    def test_explicit_features(self, regression_df):
        X, y = prepare_features(regression_df, target="y", features=["x1"])
        assert list(X.columns) == ["x1"]

    def test_missing_target_raises(self, regression_df):
        with pytest.raises(KeyError, match="not found"):
            prepare_features(regression_df, target="nonexistent")

    def test_drops_nan_rows(self):
        df = pd.DataFrame({"x": [1, 2, np.nan, 4], "y": [10, 20, 30, 40]})
        X, y = prepare_features(df, target="y")
        assert len(X) == 3
