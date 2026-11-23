# Sri Lanka Rainfall Prediction: Ensemble Learning Dashboard

A Streamlit dashboard that compares three ensemble regressors (Bagging, Gradient Boosting and Stacking) trained to predict daily rainfall (mm) for Sri Lankan cities from weather inputs.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard |
| `ensemble_sri_lanka.ipynb` | Notebook that trains the models (written for Google Colab) |
| `trained_models.joblib` | Trained scikit-learn pipelines (3 models) |
| `cities.joblib` | City list for the dropdown |
| `metrics.csv` | Test-set RMSE / MAE / R2 for each model |
| `requirements.txt` | Python dependencies |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Python 3.11 or newer is recommended. The models were pickled with **scikit-learn 1.6.1**, so keep that version pinned. Loading them with a different version can fail or give wrong results.

## Deploy on Streamlit Community Cloud

Choose Python 3.11 or 3.12 under Advanced settings when deploying. The Python version can only be set at deploy time, so delete and redeploy the app to change it.

## Models and results

Trained on a 25,000-row random sample of the Sri Lanka weather dataset (80/20 split).

| Model | RMSE (mm) | MAE (mm) | R2 |
|---|---|---|---|
| Boosting (Gradient Boosting) | 6.590 | 3.238 | 0.5212 |
| Stacking (RF + GBR + Ridge) | 6.648 | 3.230 | 0.5127 |
| Bagging (Decision Trees) | 6.693 | 3.234 | 0.5061 |

Input features: max/min temperature, shortwave radiation, max wind speed, ET0, elevation, month, day of year and city.

## Retraining

The notebook expects the dataset `SriLanka_Weather_Dataset_V1.csv`, which is not included in this repository. Add it at the path set in the notebook's first code cell, run all cells, and replace the three generated artifacts.
