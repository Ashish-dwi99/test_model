from __future__ import annotations

import argparse
import pathlib

import joblib
import numpy as np
import pandas as pd

from src.data import align_features, load_config, load_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Kaggle submission file.")
    parser.add_argument("--test-path", required=True, help="Path to test CSV.")
    parser.add_argument(
        "--model-path",
        default="artifacts/model.joblib",
        help="Path to a trained model.",
    )
    parser.add_argument(
        "--output-path",
        default="submissions/submission.csv",
        help="Where to write the submission file.",
    )
    parser.add_argument(
        "--use-proba",
        action="store_true",
        help="Use predict_proba for classification submissions when available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()

    test_df = load_dataframe(args.test_path)
    bundle = joblib.load(args.model_path)

    id_series = None
    if config.id_column and config.id_column in test_df.columns:
        id_series = test_df[config.id_column]
        test_df = test_df.drop(columns=[config.id_column])

    aligned_features = align_features(test_df, bundle.feature_columns)

    proba_used = False
    if args.use_proba and bundle.problem_type == "classification":
        try:
            proba = bundle.pipeline.predict_proba(aligned_features)
            if proba.ndim == 2 and proba.shape[1] == 2:
                predictions = proba[:, 1]
            else:
                predictions = proba
            proba_used = True
        except AttributeError:
            predictions = bundle.pipeline.predict(aligned_features)
    else:
        predictions = bundle.pipeline.predict(aligned_features)

    submission = build_submission(config, id_series, predictions, bundle, proba_used)

    output_path = pathlib.Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Saved submission to {output_path}")


if __name__ == "__main__":
    main()


def build_submission(
    config,
    id_series: pd.Series | None,
    predictions: np.ndarray,
    bundle,
    proba_used: bool,
) -> pd.DataFrame:
    if proba_used and predictions.ndim == 2 and predictions.shape[1] > 1:
        classes = None
        model = bundle.pipeline.named_steps.get("model")
        if model is not None and hasattr(model, "classes_"):
            classes = model.classes_
        if classes is not None and len(classes) == predictions.shape[1]:
            column_names = [f"{config.target_column}_{label}" for label in classes]
        else:
            column_names = [
                f"{config.target_column}_{idx}" for idx in range(predictions.shape[1])
            ]
        submission = pd.DataFrame(predictions, columns=column_names)
    else:
        submission = pd.DataFrame({config.target_column: predictions})

    if id_series is not None:
        submission.insert(0, config.id_column, id_series)

    return submission
