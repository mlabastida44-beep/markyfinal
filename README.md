# Student Performance Prediction System

A classroom prototype built for a Data Mining laboratory. It estimates whether a
student is at risk of failing the final grade using historical student records.

## Problem
Academic risk is often noticed only after poor results. This prototype explores
early identification of students who may need additional academic support.

## Dataset
UCI Machine Learning Repository, *Student Performance* (ID 320):
https://archive.ics.uci.edu/dataset/320/student+performance
It is downloaded automatically by `train_model.py` via `ucimlrepo` and saved as
`student_data.csv`.

## Prediction Target
- `G3 < 10` -> **AT RISK** (1)
- `G3 >= 10` -> **NOT AT RISK** (0)

`G3` is the target only and is never used as an input feature (no target leakage).

## Technologies
Python 3.10+, pandas, scikit-learn (Random Forest), joblib, Streamlit, GitHub.

## Installation and Run
```bash
git clone https://github.com/YOUR-USERNAME/student-performance-prediction.git
cd student-performance-prediction
pip install -r requirements.txt
python train_model.py      # downloads data, trains model, saves student_model.pkl
streamlit run app.py       # opens http://localhost:8501
```
In the app, download the sample input CSV, upload it back, and view or download
the predictions.

## Model Evaluation
Run `python train_model.py` to see accuracy, precision, recall and F1-score.
Results from my run:

| Metric (At risk class) | Value |
|---|---|
| Accuracy | _fill in_ |
| Precision | _fill in_ |
| Recall | _fill in_ |
| F1-score | _fill in_ |

## Limitations
- Trained on one Portuguese secondary-school dataset; it may not generalize to
  other institutions or populations.
- G1 and G2 (earlier grades) are strong predictors, so performance is high partly
  because they are closely related to G3.
- Predictions are not diagnoses and must not be used as the sole basis for real
  student decisions.
