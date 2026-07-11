# Customer Interaction Classification - Starter Prototype

This folder contains an early prototype for classifying customer interaction messages.
It is intentionally not production-ready. Your task is to analyse, refactor, extend,
evaluate, and design a more production-ready AI system around it.

## What the current prototype does

The current prototype:

- loads two CSV files from the `data/` folder
- performs basic preprocessing
- creates TF-IDF features
- trains a Random Forest classifier
- prints a classification report

The core target label for the assessment is currently `y2`, configured in `Config.py`.
The other labels are available in the dataset but are not required for the core task.

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run the current prototype

```bash
python main.py
```

## Notes

- You may see warnings from the inherited regular expressions in the preprocessing file.
  These warnings do not necessarily stop the prototype from running.
- The current prototype is deliberately limited. Do not treat it as a final system.
- You are expected to improve the structure, evaluation, and production-readiness as part of the assessment.
