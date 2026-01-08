from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass(frozen=True)
class ModelBundle:
    pipeline: Pipeline
    problem_type: str
    feature_columns: list[str]
    model_type: str


def build_pipeline(features: pd.DataFrame, problem_type: str, model_type: str) -> Pipeline:
    numeric_features = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [
        col
        for col in features.columns
        if col not in numeric_features
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    if model_type == "random_forest":
        if problem_type == "classification":
            estimator = RandomForestClassifier(n_estimators=300, random_state=42)
        else:
            estimator = RandomForestRegressor(n_estimators=300, random_state=42)
    else:
        if problem_type == "classification":
            estimator = HistGradientBoostingClassifier()
        else:
            estimator = HistGradientBoostingRegressor()

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )


def build_model_bundle(features: pd.DataFrame, problem_type: str, model_type: str) -> ModelBundle:
    pipeline = build_pipeline(features, problem_type, model_type)
    return ModelBundle(
        pipeline=pipeline,
        problem_type=problem_type,
        feature_columns=features.columns.tolist(),
        model_type=model_type,
    )
