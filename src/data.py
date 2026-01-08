from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class DataConfig:
    target_column: str
    id_column: Optional[str]
    problem_type: Optional[str]


def load_config() -> DataConfig:
    target_column = os.environ.get("TARGET_COLUMN")
    if not target_column:
        raise ValueError("TARGET_COLUMN environment variable is required.")

    id_column = os.environ.get("ID_COLUMN")
    problem_type = os.environ.get("PROBLEM_TYPE")

    if problem_type is not None:
        problem_type = problem_type.lower().strip()
        if problem_type not in {"classification", "regression"}:
            raise ValueError("PROBLEM_TYPE must be 'classification' or 'regression'.")

    return DataConfig(
        target_column=target_column,
        id_column=id_column,
        problem_type=problem_type,
    )


def load_dataframe(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def infer_problem_type(target: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(target):
        unique_values = target.nunique(dropna=True)
        if unique_values <= 20:
            return "classification"
        return "regression"
    return "classification"


def split_features_target(df: pd.DataFrame, config: DataConfig) -> tuple[pd.DataFrame, pd.Series]:
    if config.target_column not in df.columns:
        raise KeyError(f"Target column '{config.target_column}' not found in training data.")

    target = df[config.target_column]
    features = df.drop(columns=[config.target_column])
    if config.id_column and config.id_column in features.columns:
        features = features.drop(columns=[config.id_column])
    return features, target


def align_features(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    aligned_df = df.copy()
    missing_columns = [col for col in feature_columns if col not in aligned_df.columns]
    for col in missing_columns:
        aligned_df[col] = pd.NA
    aligned = aligned_df[feature_columns].copy()
    return aligned
