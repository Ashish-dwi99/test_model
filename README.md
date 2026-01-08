# Kaggle Starter Toolkit (Google Tunix Hackathon)

This repo is a lightweight, competition-agnostic Kaggle starter kit. It includes a baseline training script, a prediction script for generating submissions, and a simple data pipeline that handles numeric + categorical columns.

## Quickstart

### 1) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Download data

Use the Kaggle CLI (install separately with `pip install kaggle` and set up your Kaggle API token) to download the competition data into `data/`:

```bash
mkdir -p data
kaggle competitions download -c google-tunix-hackathon -p data
unzip -o data/google-tunix-hackathon.zip -d data
```

> If the download name differs, list files with `ls data` and adjust the paths below.

### 3) Train a baseline model

Set the target column (and optionally ID column) based on the competition data schema. The ID column is excluded from training features automatically.

```bash
export TARGET_COLUMN=target
export ID_COLUMN=id
python -m src.train --train-path data/train.csv
```

This will create `artifacts/model.joblib` and print a local validation score.
You can also add `--cv 5` to report a cross-validation score or `--model-type random_forest` for a stronger baseline.

### 4) Generate a submission

```bash
python -m src.predict \
  --test-path data/test.csv \
  --model-path artifacts/model.joblib \
  --output-path submissions/submission.csv
```

If the competition expects probabilities instead of hard class labels, add `--use-proba` to the prediction command. For multi-class classification, this will emit one column per class (e.g. `target_classA`, `target_classB`).

## Configuration

Environment variables:

- `TARGET_COLUMN` (required): name of the target column in `train.csv`.
- `ID_COLUMN` (optional): name of the ID column used for submissions.
- `PROBLEM_TYPE` (optional): `classification` or `regression`. If omitted, the script infers it.

## Project Layout

```
.
├── data/                  # Kaggle data (not tracked)
├── src/
│   ├── data.py             # I/O and schema utilities
│   ├── model.py            # Model + preprocessing pipeline
│   ├── train.py            # Training entry point
│   └── predict.py          # Submission generation
├── artifacts/              # Trained models (not tracked)
└── submissions/            # Submission files (not tracked)
```

## Notes

This is intentionally minimal. Once you know the metric and data quirks, you can customize:

- Feature engineering in `src/model.py`
- Validation logic in `src/train.py`
- Custom metrics in `src/train.py`
