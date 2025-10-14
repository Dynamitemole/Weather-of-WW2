# ITDS Final Project — Regression & Time-Series Forecasting (Historical Weather)


A course project for *Introduction to Data Science (ITDS)* that explores:
- **Univariate regression** on analytical functions (with/without noise)
- **Multivariate regression** on synthetic data (`make_regression`)
- **Time-series forecasting** of daily temperature using WWII-era weather records

---

## Table of Contents
- [Overview](#overview)
- [Data](#data)
- [Repository Structure](#repository-structure)
- [Environment & Installation](#environment--installation)
- [How to Run](#how-to-run)
- [Notebook Map (What’s Inside)](#notebook-map-whats-inside)
- [Methods & Models](#methods--models)
- [Evaluation & Visualization](#evaluation--visualization)
- [Reproducibility](#reproducibility)
- [Results (Typical Observations)](#results-typical-observations)
- [Troubleshooting](#troubleshooting)
- [Roadmap / Future Work](#roadmap--future-work)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [How to Cite](#how-to-cite)
- [Contact](#contact)

---

## Overview
This project benchmarks classic and modern regression approaches for both **synthetic** and **real** data:

1) **Univariate regression**  
   Fit several models to hand-crafted functions, inspect bias/variance under noise, and visualize predicted curves vs. ground truth.

2) **Multivariate regression**  
   Use `sklearn.datasets.make_regression` to probe robustness as noise rises and as non-informative features are added.

3) **Time-series forecasting**  
   Predict next-day mean temperature from daily historical observations of a single weather station. Create **lag (rolling-window) features**, split the series temporally, and compare a linear baseline to a tuned gradient-boosting model.

An assignment brief providing the original task context is included as `ITDS_Final_Project.pdf`.

---

## Data
- **`SummaryofWeather.csv`** — Daily weather observations (WWII-era dataset used in the course)
- **`WeatherStationLocations.csv`** — Station metadata (lat/lon, etc.; optional for the notebook)
- The notebook typically selects **one station** (example: `STA=22508`) and builds a univariate temperature series.

> If you prefer a cleaner repo, you can place CSVs under `data/` and adjust file paths in the notebook.

---

## Repository Structure
```
.
├─ H2BGRS_ITDS_Final_2025.ipynb    # Main notebook with all experiments
├─ ITDS_Final_Project.pdf          # Assignment / project brief (read-only)
├─ SummaryofWeather.csv            # Historical daily weather observations
├─ WeatherStationLocations.csv     # Station metadata (optional)
└─ README.md
```

(*Optional structure if you move data:*)
```
.
├─ H2BGRS_ITDS_Final_2025.ipynb
├─ docs/
│  └─ ITDS_Final_Project.pdf
├─ data/
│  ├─ SummaryofWeather.csv
│  └─ WeatherStationLocations.csv
└─ README.md
```

---

## Environment & Installation
**Python:** 3.9+ (tested with 3.10/3.11)

**Key libraries**
- Core: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `xgboost`, `jupyter`
- Optional: `seaborn`, `tqdm`, `statsmodels`

**Option A — pip**
```bash
pip install -U numpy pandas matplotlib seaborn scikit-learn xgboost jupyterlab tqdm statsmodels
```

**Option B — requirements.txt**
Create `requirements.txt` with:
```txt
numpy>=1.23
pandas>=1.5
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
xgboost>=1.7
jupyterlab>=4.0
tqdm>=4.65
statsmodels>=0.14
```
Then:
```bash
pip install -r requirements.txt
```

**(Optional) Conda**
```bash
conda create -n itds python=3.11 -y
conda activate itds
pip install -r requirements.txt
```

---

## How to Run
1. Ensure `SummaryofWeather.csv` (and optionally `WeatherStationLocations.csv`) are present in the repo root **or** update paths in the notebook (e.g., `data/SummaryofWeather.csv`).
2. Launch Jupyter:
```bash
jupyter lab       # or: jupyter notebook
```
3. Open **`H2BGRS_ITDS_Final_2025.ipynb`** and run cells top-to-bottom.

> The notebook sets `random_state=42` in several places for reproducibility.

---

## Notebook Map (What’s Inside)
### 1) Univariate Regression on Analytical Functions
- Three custom functions (`f1`, `f2`, `f3`) with noiseless and noisy variants
- Compare: `LinearRegression`, `PolynomialFeatures+LinearRegression`, `RandomForestRegressor`, `MLPRegressor`
- Visualize predicted curves and compute MSE / R²

### 2) Multivariate Regression on Synthetic Data
- Use `make_regression` to control `noise`, number of **informative** vs **non-informative** features
- Track how performance degrades with noise and irrelevant features
- Compare linear, tree-based, and neural baselines

### 3) Time-Series Forecasting (Daily Temperature)
- Load daily mean temperature for a chosen station from `SummaryofWeather.csv`
- Construct **lag features** with rolling windows (e.g., previous 7–10 days)
- **Temporal split** (e.g., train on earlier years, test on later year)
- Baseline: `LinearRegression`
- Tuned model: `XGBRegressor` with `TimeSeriesSplit` + `GridSearchCV`
- Plots: Actual vs. Predicted (full test span, and seasonal slices)

---

## Methods & Models
- **Feature engineering**
  - Polynomial feature expansion (univariate)
  - Lagged features from rolling windows (time-series)
- **Models**
  - Linear Regression, Polynomial Regression
  - Random Forest Regressor
  - MLP Regressor (feed-forward neural network)
  - XGBoost Regressor (time-series setup with forward-chaining CV)
- **Validation**
  - Standard train/test splits for i.i.d. setups
  - **TimeSeriesSplit** for temporal data (no leakage)

---

## Evaluation & Visualization
- **Metrics:** `R²` (coefficient of determination), `MSE` (mean squared error)  
  *(You can easily add MAE/RMSE if preferred.)*
- **Plots:** 
  - Predicted curve vs. ground truth (univariate)
  - Error vs. noise level / feature count (multivariate)
  - Actual vs. Predicted through time; seasonal zoom-ins (time-series)

---

## Reproducibility
- Random seeds fixed where applicable: `random_state=42`
- Deterministic splits for time-series
- If CSVs are large, consider **Git LFS**:
```bash
git lfs install
git lfs track "*.csv"
```

---

## Results (Typical Observations)
- **Univariate:** On non-linear targets, Polynomial/Tree/MLP often beat plain Linear Regression; higher noise favors tree ensembles.
- **Multivariate:** Performance degrades with more noise and non-informative features; linear models expose irrelevance via small coefficients.
- **Time-series:** Lag features help capture seasonality/trend; short-horizon (next-day) predictions are reasonable, while longer horizons become harder.

> Exact numbers depend on the chosen station, lag window, and hyperparameters. See final cells/plots in the notebook output.

---

## Troubleshooting
- **FileNotFoundError:** Ensure CSV paths match the notebook (`SummaryofWeather.csv` in repo root by default).
- **`xgboost` import error:** Install `xgboost>=1.7` (see requirements).
- **Plots not showing:** Use `jupyter lab` or ensure `%matplotlib inline` is active.

---

## License
This repository is shared for educational purposes.  
If you plan to reuse/extend, consider adding a formal license (e.g., **MIT**). Create a `LICENSE` file at the repo root.

---

## Acknowledgements
- ITDS course staff for the assignment outline (`ITDS_Final_Project.pdf`)
- Historical weather data provided with course materials (WWII-era dataset)
