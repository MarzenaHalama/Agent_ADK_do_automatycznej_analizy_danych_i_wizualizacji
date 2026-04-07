"""
Robust CSV / data loading utilities.

Handles common real-world problems:
  - BOM characters in column names
  - Mixed decimal separators (comma vs dot)
  - Automatic separator detection
  - Date parsing with day-first format
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


def load_csv(
    path: str,
    *,
    date_columns: Optional[List[str]] = None,
    dayfirst: bool = True,
) -> pd.DataFrame:
    """Read a CSV file and return a cleaned DataFrame.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    date_columns : list[str] | None
        Column names to parse as dates.
    dayfirst : bool
        Whether dates use day-first format (e.g. ``01.09.2025``).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with normalised column names and numeric coercion.
    """
    df = pd.read_csv(path, sep=None, engine="python")
    df = _clean_column_names(df)
    df = _coerce_numeric_columns(df)

    if date_columns:
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], dayfirst=dayfirst, errors="coerce")

    return df


def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip BOM markers and whitespace from column names."""
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )
    return df


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to convert text columns to numeric values.

    Handles European-style decimal comma and whitespace in numbers.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            series = (
                df[col]
                .astype(str)
                .str.strip()
                .str.replace(" ", "", regex=False)
                .str.replace(",", ".", regex=False)
            )
            converted = pd.to_numeric(series, errors="coerce")
            # Only replace if at least half of non-null values converted
            if converted.notna().sum() >= series.notna().sum() * 0.5:
                df[col] = converted
    return df


def prepare_features(
    df: pd.DataFrame,
    target: str,
    features: Optional[List[str]] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Select feature columns and target, dropping rows with missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Input data (already cleaned).
    target : str
        Name of the target column.
    features : list[str] | None
        Explicit list of feature columns.  If *None*, all numeric columns
        except the target are used.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        ``(X, y)`` with no missing values.

    Raises
    ------
    KeyError
        If the target column is not found.
    ValueError
        If no usable features remain after cleaning.
    """
    df = _clean_column_names(df)

    if target not in df.columns:
        raise KeyError(
            f"Target column '{target}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    if features is not None:
        norm_map = {c.lower(): c for c in df.columns}
        resolved = [norm_map[f.lower()] for f in features if f.lower() in norm_map]
        if not resolved:
            features = None  # fall back to auto
        else:
            features = resolved

    if features is None:
        features = [
            c
            for c in df.select_dtypes(include=np.number).columns
            if c != target
        ]

    if not features:
        raise ValueError(
            "No numeric feature columns available after cleaning. "
            f"Columns in data: {list(df.columns)}"
        )

    combined = pd.concat([df[features], df[target]], axis=1).dropna()
    return combined[features], combined[target]
