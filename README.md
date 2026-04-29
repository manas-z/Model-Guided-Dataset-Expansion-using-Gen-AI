# Model-Guided Dataset Expansion using Gen AI

This repository contains a deep learning mini-project on model-guided dataset expansion for imbalanced chest X-ray classification. The project trains a baseline classifier, identifies the weakest minority class from validation performance, expands that class with generative augmentation, and compares the result against the original baseline.

The current codebase is notebook-first, with additional utility code for publication-ready result aggregation.

## Project Summary

Medical imaging datasets are often imbalanced: common classes have many examples, while clinically important disease classes may have fewer. This project explores a data-centric strategy:

1. Train a baseline chest X-ray classifier.
2. Use validation metrics to find weak minority classes.
3. Generate or add targeted samples for those weak classes.
4. Retrain/fine-tune the classifier.
5. Compare macro metrics, not only top-line accuracy.

The key research claim should be framed carefully:

> Model-guided dataset expansion can improve class-balanced performance in this experimental setting, but external validation and repeated trials are still needed before making broader claims.

## Repository Structure

| Path | Purpose |
| --- | --- |
| `main_eda.ipynb` | Exploratory data analysis for the COVID-19 radiography dataset. |
| `main.ipynb` | Main classifier, augmentation, and evaluation workflow. |
| `src/losses.py` | Focal loss implementation for stronger imbalance baselines. |
| `src/publication_metrics.py` | Aggregates repeated experiment runs into mean/std result tables. |
| `tests/` | Lightweight tests for reusable publication utilities. |
| `results/repeated_runs.json` | Current cached-run result file; add more seeds here after reruns. |
| `PUBLISHING_ROADMAP.md` | Practical checklist for improving the paper before submission. |

## Dataset

The notebooks use the COVID-19 Radiography Dataset with these classes:

- COVID
- Lung_Opacity
- Normal
- Viral Pneumonia

Expected dataset layout:

```text
data/covid_xray/COVID-19_Radiography_Dataset/
  COVID/images/
  Lung_Opacity/images/
  Normal/images/
  Viral Pneumonia/images/
```

The notebook currently points to:

```python
DATA_DIR = r"D:\dcgan\data\covid_xray\COVID-19_Radiography_Dataset"
```

## Current Cached Result

The cached result in `cache_covid_project/results_summary.json` reports:

| Metric | Baseline | Retrained / expanded |
| --- | ---: | ---: |
| Accuracy | 0.8932 | 0.8914 |
| Macro recall | 0.8820 | 0.9044 |
| Macro F1 | 0.8965 | 0.8989 |

This is useful for the paper because macro recall and macro F1 are more relevant than accuracy for imbalanced medical classification. The paper should not claim that every metric improves.

## Running Tests

```bash
python -m unittest tests.test_losses tests.test_publication_metrics
```

## Aggregating Publication Results

After adding multiple seed runs to `results/repeated_runs.json`, generate a summary table:

```bash
python -m src.publication_metrics
```

This writes:

```text
results/repeated_runs_summary.csv
```

## Recommended Next Experiments

For a stronger paper submission, run at least:

- baseline classifier
- continued training
- real oversampling
- focal loss baseline
- model-guided generative expansion

Repeat the full experiment for at least three seeds and report mean plus standard deviation.

## Disclaimer

This project is for research and education only. It is not intended for clinical diagnosis or medical decision-making.
