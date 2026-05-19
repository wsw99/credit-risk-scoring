# Explainable & Fair Credit Risk Scoring

End-to-end personal loan default prediction system with **XAI**, **fairness audit**, and **LLM-generated approval reports**.

## Overview

- **Stage 1**: EDA + data cleaning (LendingClub 2007-2018)
- **Stage 2**: Classical bank scorecard (WOE + Logistic Regression)
- **Stage 3**: Tree ensemble models (XGBoost, LightGBM, CatBoost, Stacking)
- **Stage 4**: Deep learning comparison (TabNet, FT-Transformer, EBM)
- **Stage 5**: XAI + Fairness (SHAP, LIME, DiCE, Fairlearn)
- **Stage 6**: LLM approval report + Gradio demo

## Benchmark

| Model | Test AUC | Split Method |
|---|---|---|
| Logistic Regression | 0.714 | Temporal (2007-2014 / 2015 / 2016) |

## Setup

```bash
conda env create -f environment.yml
conda activate credit-risk-scoring
```

## Data

[LendingClub Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club) — 2.2M loans, 151 features.

## Reports

See `results/analysis_reports/` for EDA charts and [data analysis report](results/analysis_reports/data_analysis_report.md).
