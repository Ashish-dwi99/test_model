from __future__ import annotations

import argparse
import pathlib

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from src.data import infer_problem_type, load_config, load_dataframe, split_features_target
from src.model import build_model_bundle


def evaluate_classification(y_true: pd.Series, y_pred, y_proba) -> dict[str, float]:
    metrics: dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
    }
    if y_proba is not None:
        metrics["log_loss"] = log_loss(y_true, y_proba)
    return metrics


def evaluate_regression(y_true: pd.Series, y_pred) -> dict[str, float]:
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return {
        "rmse": rmse,
        "r2": r2_score(y_true, y_pred),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline model.")
    parser.add_argument("--train-path", required=True, help="Path to training CSV.")
    parser.add_argument(
        "--output-path",
        default="artifacts/model.joblib",
        help="Where to store the trained model.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Validation split size.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/validation split.",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=0,
        help="Number of CV folds to report (0 disables CV).",
    )
    parser.add_argument(
        "--model-type",
        choices=["hgb", "random_forest"],
        default="hgb",
        help="Model family to use for the baseline.",
    )
    return parser.parse_args()


def run_cross_validation(
    features: pd.DataFrame,
    target: pd.Series,
    problem_type: str,
    model_type: str,
    folds: int,
    random_state: int,
) -> dict[str, float]:
    if problem_type == "classification":
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    else:
        splitter = KFold(n_splits=folds, shuffle=True, random_state=random_state)

    scores: list[float] = []
    for train_idx, val_idx in splitter.split(features, target):
        X_train, X_val = features.iloc[train_idx], features.iloc[val_idx]
        y_train, y_val = target.iloc[train_idx], target.iloc[val_idx]
        bundle = build_model_bundle(features, problem_type, model_type)
        bundle.pipeline.fit(X_train, y_train)
        y_pred = bundle.pipeline.predict(X_val)
        if problem_type == "classification":
            scores.append(accuracy_score(y_val, y_pred))
        else:
            scores.append(mean_squared_error(y_val, y_pred, squared=False))

    mean_score = sum(scores) / len(scores)
    return {
        "cv_mean_score": mean_score,
        "cv_folds": folds,
    }


def main() -> None:
    args = parse_args()
    config = load_config()

    df = load_dataframe(args.train_path)
    features, target = split_features_target(df, config)

    problem_type = config.problem_type or infer_problem_type(target)

    X_train, X_val, y_train, y_val = train_test_split(
        features,
        target,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=target if problem_type == "classification" else None,
    )

    bundle = build_model_bundle(features, problem_type, args.model_type)
    bundle.pipeline.fit(X_train, y_train)

    y_pred = bundle.pipeline.predict(X_val)
    y_proba = None
    if problem_type == "classification":
        try:
            y_proba = bundle.pipeline.predict_proba(X_val)
        except AttributeError:
            y_proba = None

    if problem_type == "classification":
        metrics = evaluate_classification(y_val, y_pred, y_proba)
    else:
        metrics = evaluate_regression(y_val, y_pred)

    print("Validation metrics:")
    for name, value in metrics.items():
        print(f"- {name}: {value:.5f}")

    if args.cv and args.cv > 1:
        cv_metrics = run_cross_validation(
            features,
            target,
            problem_type,
            args.model_type,
            args.cv,
            args.random_state,
        )
        print("Cross-validation:")
        print(f"- folds: {cv_metrics['cv_folds']}")
        print(f"- mean_score: {cv_metrics['cv_mean_score']:.5f}")

    output_path = pathlib.Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)
    print(f"Saved model to {output_path}")


if __name__ == "__main__":
    main()
