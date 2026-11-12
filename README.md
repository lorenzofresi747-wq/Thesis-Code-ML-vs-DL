# Machine Learning vs. Deep Learning for Stock Return Forecasting

### Master's Thesis — Quantitative Finance

**Author:** Lorenzo Salvatore Fresi  
**University:** University of Milan  
**Academic Year:** 2025/2026

**Thesis:** *Market Conditions and the Relative Forecasting Performance of Machine Learning and Deep Learning: Evidence from One-Day-Ahead Stock Returns*

---

## 1. Project Overview

This repository contains the R and Python scripts developed for my Master's thesis in Quantitative Finance at the University of Milan.

The research investigates whether the additional complexity of Deep Learning (DL) leads to improved forecasting performance compared with traditional Machine Learning (ML), and whether these differences depend on specific market conditions.

The empirical analysis examines **33 U.S. equities across 11 economic sectors**, using daily financial data from January 2010 to December 2025.

The study compares six traditional Machine Learning methods with three Deep Learning architectures, evaluating their forecasting accuracy, robustness, generalization, and computational requirements.

### Models considered

**Traditional Machine Learning:**
- Ordinary Least Squares (OLS)
- Ridge Regression
- LASSO Regression
- Bagging
- Random Forest
- Gradient Boosting

**Deep Learning:**
- Multilayer Perceptron (MLP)
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)

The objective is to predict one-day-ahead logarithmic stock returns using historical market information.

## 2. Dataset and Experimental Design

The financial data were obtained from Yahoo Finance using the `quantmod` package in R.

The dataset includes historical prices, returns, trading volume, and technical indicators.

The final experiment uses **28 predictors**, including lagged returns, moving averages, momentum indicators, volatility measures, and trading activity variables.

The observations are divided chronologically into three periods:

| Dataset | Period | Purpose |
|---|---|---|
| Training | 2010–2017 | Model estimation |
| Validation | 2018–2019 | Hyperparameter tuning and model selection |
| Testing | 2020–2025 | Final out-of-sample evaluation |

All models are selected using validation performance, without using test-period observations for model selection.

The main forecasting benchmark is the **zero-return forecast**.

### Important data availability note

The repository contains the source code and small supporting input files.

The complete prepared dataset (`model_data.csv`), approximately 103 MiB, is not included because of standard GitHub upload limits.

The original dataset must be supplied separately to reproduce the full analysis. Historical data downloaded again from Yahoo Finance may also differ from the original downloaded observations.

## 3. Software Requirements

The analysis was developed using:

- **R / RStudio** for data preparation, traditional Machine Learning, statistical comparisons, and visualization.
- **Python 3.11 / PyTorch** for Deep Learning implementation and training.
- **Windows** as the original operating environment.

### R packages

The main packages include:

`quantmod`, `xts`, `zoo`, `TTR`, `dplyr`, `tidyr`, `purrr`, `tibble`, `glmnet`, `ranger`, `gbm`, `forecast`, `lmtest`, `sandwich`, `ggplot2`, `callr`, and `ps`.

Additional dependencies may be required by individual scripts.

### Python packages

The main Python dependencies are:

`torch`, `numpy`, `pandas`, and `psutil`.

They are also listed in `requirements.txt`.

The original R launchers expect a Python environment at:

`%LOCALAPPDATA%\thesis-dl-torch\Scripts\python.exe`

If Python is installed elsewhere, this path must be updated in the corresponding R scripts.

## 4. Repository Structure

The principal files are organized according to the different stages of the analysis.

| File | Description |
|---|---|
| `11_STAGE_A_rerun_ML_28_predictors.R` | Final traditional ML estimation using 28 predictors |
| `prepare_and_run_dl.R` | Preparation and execution of the DL experiment |
| `train_dl_33.py` | Main PyTorch training for all 33 stocks |
| `03_DL_vs_ML_analysis.R` | Direct ML–DL forecasting comparisons |
| `04_regime_and_influence.R` | Volatility regimes and influential observations |
| `05_run_seed_stability.R` | Launcher for repeated DL initializations |
| `05_seed_stability.py` | Python implementation of the multi-seed experiment |
| `06_analyse_seed_stability.R` | Analysis of results across random initializations |
| `07_MULTI_SEED_PRIMARY_RESULTS.R` | Final multi-seed comparison |
| `08_run_overfitting_diagnostics.R` | Launcher for the DL generalization analysis |
| `08_overfitting_diagnostics.py` | Training–validation diagnostic calculations |
| `09_Thesis_Empirical_Figures_recovered.R` | Generation of empirical figures |
| `ONE_SCRIPT_ML_DL_BENCHMARK.R` | Computational resource comparison |
| `train_dl_33_with_RAM.py` | PyTorch training with memory and execution-time monitoring |
| `Machine Learning Code.R` | Original ML development and data preparation |
| `archive/` | Preliminary experiments and alternative scripts |
| `supplementary_analyses/` | Additional diagnostics and supporting analyses |

**Important:** `Machine Learning Code.R` includes an earlier stage using 27 predictors. The final comparison is based on the updated 28-predictor ML implementation.

The scripts are preserved for transparency, including preliminary and exploratory stages that are not necessarily part of the final computational workflow.

## 5. Execution Guide

The scripts were developed with the R working directory set to the project root.

Before running the analysis, ensure the required input files are available and that the working directory points to the main project folder.

The following sequence describes the intended execution order.

### Step 1 — Prepare the data

Load the original prepared dataset (`model_data.csv`) into R.

The supporting files `stock_info.csv` and `volatility_thresholds.csv` should also be available in the working directory.

The original data preparation and feature-engineering code is retained in `Machine Learning Code.R`.

### Step 2 — Run traditional Machine Learning

Execute:

`11_STAGE_A_rerun_ML_28_predictors.R`

This script estimates the final traditional ML models using the 28-predictor specification.

The results are saved in:

`ML_28features_STAGE_A/`

These outputs must be used for the final ML–DL comparison rather than those from the earlier 27-predictor experiment.

### Step 3 — Run Deep Learning

Execute:

`prepare_and_run_dl.R`

This script prepares the input dataset and launches the Python trainer:

`train_dl_33.py`

The Python script trains the six candidate neural-network configurations across the 33 securities and selects the preferred model using validation RMSE.

The final DL results are stored in:

`dl_output/`

The outputs include selected models, forecasts, validation results, and saved PyTorch checkpoints.

### Step 4 — Compare ML and DL

Execute the following R scripts in order:

1. `03_DL_vs_ML_analysis.R`
2. `04_regime_and_influence.R`

These scripts perform the direct forecasting comparison and examine differences across market conditions.

The analysis includes RMSE and MAE comparisons, Diebold–Mariano tests, false discovery rate adjustments, and investigations of sector, volatility, and liquidity characteristics.

Results are stored primarily in:

`FINAL_DL_vs_ML/`

### Step 5 — Multi-seed robustness analysis

Execute:

1. `05_run_seed_stability.R`
2. `06_analyse_seed_stability.R`
3. `07_MULTI_SEED_PRIMARY_RESULTS.R`

The experiment evaluates the sensitivity of DL performance to different random initializations.

The original run is compared with three additional random seeds: **101, 202, and 303**.

The selected neural-network architectures and hyperparameters remain fixed during this robustness exercise.

### Step 6 — Generalization analysis

Execute:

`08_run_overfitting_diagnostics.R`

This script launches:

`08_overfitting_diagnostics.py`

The procedure evaluates the differences between training and validation forecasting performance.

Positive generalization gaps may indicate overfitting, but they can also reflect changes in market conditions and should not be interpreted as definitive proof of overfitting.

The repository also includes exploratory ML–DL generalization comparisons. Some of these require manual adjustment before they can be run independently.

### Step 7 — Generate figures

Execute:

`09_Thesis_Empirical_Figures_recovered.R`

This script uses previously calculated results to produce the empirical visualizations, without retraining the forecasting models.

### Step 8 — Computational requirements

The scripts:

`ONE_SCRIPT_ML_DL_BENCHMARK.R`

and

`train_dl_33_with_RAM.py`

are used to examine execution time and incremental memory consumption.

The benchmarking workflow requires additional preparation of the R environment and verification of the final ML specification.

The original computational measurements should be retained when attempting to reproduce the results reported in the thesis.

## 6. Manual File Selection

Some R scripts use `file.choose()` to request input files interactively.

If the requested file is already available in its expected directory, the script may locate it automatically.

Otherwise, select the corresponding file indicated below.

| Script | File to select |
|---|---|
| `11_STAGE_A_rerun_ML_28_predictors.R` | `model_data.csv` or the relevant dataset export |
| `03_DL_vs_ML_analysis.R` | Final selected ML/DL predictions and models, as requested |
| `05_run_seed_stability.R` | `05_seed_stability.py`, if requested |
| `05_run_seed_stability.R` | `train_dl_33.py`, if requested |
| `05_run_seed_stability.R` | `dl_input_33.csv` and `dl_selected_models.csv`, if requested |
| `07_MULTI_SEED_PRIMARY_RESULTS.R` | `seed_05_all_seed_scores.csv` |
| `08_run_overfitting_diagnostics.R` | `08_overfitting_diagnostics.py`, if requested |
| `08_run_overfitting_diagnostics.R` | `train_dl_33.py`, if requested |
| `08_run_overfitting_diagnostics.R` | Original PyTorch checkpoint (`.pt`) when prompted |
| `ONE_SCRIPT_ML_DL_BENCHMARK.R` | `train_dl_33_with_RAM.py` |

For the final ML–DL analysis, use the ML outputs from `ML_28features_STAGE_A/`, the DL outputs from `dl_output/`, and their corresponding input datasets.

Do not select files from preliminary experiments merely because they have similar filenames.

## 7. Forecast Evaluation and Robustness

The main performance measures include:

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Out-of-sample R-squared
- Directional Accuracy
- Benchmark-relative forecasting improvements

The statistical analysis includes Diebold–Mariano and Pesaran–Timmermann tests, together with Benjamini–Hochberg multiple-testing adjustments.

Additional analyses examine:

- Structural volatility
- Time-varying volatility states
- Trading liquidity
- Economic sectors
- Cross-sectional relationships
- Influential observations
- Random initialization sensitivity
- Training–validation generalization gaps
- Computational time and memory consumption


---

**Master's Thesis — University of Milan, Academic Year 2025/2026**

**Lorenzo Salvatore Fresi**
