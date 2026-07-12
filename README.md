# Customer Interaction Classification

A multi-label classification system that categorises customer support messages
across three label dimensions (`y2`, `y3`, `y4`). Refactored from an early
prototype into a modular, reproducible ML pipeline covering training, model
comparison, evaluation, and batch inference on new messages.

## Requirements

- Python 3.9+
- Dependencies in `requirements.txt`

Install:

```bash
pip install -r requirements.txt
```

## Project structure

```
.
├── data/                          # input CSVs (you provide these)
│   ├── AppGallery.csv
│   ├── Purchasing.csv
│   └── new_messages.csv           # example batch-inference input
├── src/
│   ├── config.py                  # central config: paths, columns, labels, hyperparams
│   ├── data_loader.py             # loading, validation, multi-label targets
│   ├── preprocessing.py           # de-duplication + text cleaning
│   ├── features.py                # TF-IDF (fit on train only, persisted)
│   ├── train.py                   # training + model comparison entry point
│   ├── predict.py                 # batch inference entry point
│   ├── evaluation.py              # metrics, reports, error analysis
│   └── models/
│       ├── base_model.py          # shared model interface
│       ├── random_forest_model.py
│       └── logistic_regression_model.py
├── artifacts/                     # generated: model + vectoriser (.joblib)
├── outputs/                       # generated: metrics, reports, predictions
├── requirements.txt
└── README.md
```

`artifacts/` and `outputs/` are created automatically on first run.

## Data setup

Place the three CSVs in `data/` at the repo root. `AppGallery.csv` and
`Purchasing.csv` are used for training; `new_messages.csv` is a sample input for
batch inference.

## Usage

All commands are run **from the project root** using module syntax
(`python -m src.<module>`), so the `src` package imports resolve correctly.

### 1. Train

```bash
python -m src.train
```

This loads and validates the data, preprocesses the text, splits into train/test
(stratified on `y2`), fits the TF-IDF vectoriser on the **training set only**,
then trains and compares a Random Forest and a Logistic Regression model.

Artefacts written to `artifacts/`:

- `model.joblib` — best model by mean macro-F1 (used by `predict.py`)
- `model_random_forest.joblib`, `model_logistic_regression.joblib` — each model
- `vectorizer.joblib` — the fitted TF-IDF feature extractor

Outputs written to `outputs/`:

- `metrics.json` — model comparison summary
- `metrics_<model>.json` — full per-label metrics for each model
- `classification_report_<model>.csv` — per-class precision/recall/F1
- `errors_<model>.csv` — misclassified examples for error analysis

Optionally point training at different input files:

```bash
python -m src.train --input data/AppGallery.csv data/Purchasing.csv
```

### 2. Predict (batch inference)

```bash
python -m src.predict --input data/new_messages.csv --output outputs/predictions.csv
```

Loads the saved model and vectoriser, applies the **same** preprocessing and
feature extraction used in training, and writes `predictions.csv` containing, per
message: the id, predicted label and confidence for each of `y2`/`y3`/`y4`, a
text reference, the model name/version, and a prediction timestamp.

## Notes

- **Multi-label:** the system predicts `y2`, `y3`, and `y4` jointly. Rows missing
  `y3`/`y4` are filled with an `"unknown"` sentinel so every message is
  predictable across all three labels.
- **No leakage:** the vectoriser is fit on training data only and persisted, so
  the test set and new messages are mapped into an unchanged feature space.
- **Training-serving consistency:** `predict.py` reuses the same `preprocess()`
  function and the same fitted vectoriser as training — there is no separate
  inference-only data path.
- **Configuration** is centralised in `src/config.py` (paths, column names,
  label definitions, TF-IDF and split settings).