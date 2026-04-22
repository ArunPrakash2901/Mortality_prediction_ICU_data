# GEMINI.md - ICU Mortality Prediction Project Context

This document provides instructional context and an architectural overview of the MIMIC-III ICU Mortality Prediction project.

## Project Overview
This is a high-performance Machine Learning project developed for a Kaggle-style competition. The objective is to predict hospital mortality (`HOSPITAL_EXPIRE_FLAG`) for ICU patients using the MIMIC-III dataset.

### Core Technologies
- **Language:** R (v4.x+)
- **Data Science:** `tidyverse`, `tidymodels`, `recipes`
- **Machine Learning:** `xgboost`, `lightgbm`, `keras` (TensorFlow), `ranger` (Random Forest)
- **Clinical Intelligence:** `comorbidity` (Charlson/Elixhauser indices)
- **Explainability:** `shapviz`, `fastshap` (SHAP analysis)
- **NLP:** `text2vec` (GloVe embeddings for ICD-9 codes)

## Directory Structure
- `data/`: Raw clinical CSV files (Training/Test splits).
- `RDS/`: Serialized R objects containing pre-processed data and final trained models.
- `SHAP/`: Scripts and output visualizations for model interpretability.
- `Submissions/`: Generated CSV files for Kaggle submissions.
- `tables/`: Patient-level case study tables generated from SHAP values.
- `Presentation.qmd`: Quarto-based presentation of results and methodology.

## Building and Running

### 1. Environment Setup
Ensure you have R and the required libraries installed. You can install missing packages using:
```r
install.packages(c("tidyverse", "tidymodels", "xgboost", "lightgbm", "keras", "comorbidity", "shapviz", "text2vec", "doParallel"))
```

### 2. Data Pre-processing
The primary entry point for data preparation is `pre-processing.R` (or the integrated pipeline in `updated.R`). This script:
- Merges clinical features with mortality labels.
- Calculates Charlson and Elixhauser comorbidity indices.
- Generates 50-dimensional GloVe embeddings for ICD-9 diagnostic sequences.
- Prepares a `tidymodels` recipe for normalization and imputation.

### 3. Model Training
Various model scripts are available:
- **XGBoost:** Primary model found in `updated.R` or `new.R`.
- **LightGBM:** Specialized implementation in `LightGBM.R`.
- **Neural Networks:** Keras/TensorFlow implementation in `NN_MMICIII.R`.
- **Ensemble:** Stacking and blending logic in `Ensemble model(NN and XGB).R` and `Stack(Logit, RF, and XGB).R`.

### 4. Explainability & Analysis
To generate SHAP plots and importance tables:
- Run `SHAP/SHAP generation(XGB).R` for global beeswarm plots.
- Run `tables/SHAP and ID table.R` to generate individual patient risk driver tables.

## Development Conventions
- **Feature Engineering:** Clinical knowledge is prioritized. ICD-9 codes are mapped to chapters and transformed into embeddings.
- **Class Imbalance:** Mortality is rare (~11%). All tree-based models use `scale_pos_weight` (approx. 7.91) to balance the positive class.
- **Model Evaluation:** The primary metric is **ROC-AUC**. Cross-validation (5-fold) is standard across all training scripts.
- **Persistence:** Models and processed datasets are saved as `.rds` files in the `RDS/` directory for reproducibility.
- **Parallelism:** The project utilizes `doParallel` and `future` to accelerate hyperparameter grid searches.

## Key Files
- `main_model_suite.R`: The most comprehensive "all-in-one" pipeline script.
- `integrated_pipeline_v6.R`: A robust, clinical-knowledge-integrated modeling pipeline.
- `pre-processing.R`: Dedicated data engineering logic.
- `xgb_final.rds`: The production-ready XGBoost model (stored in `RDS/`).
- `ML report.pdf`: Detailed technical documentation of the methodology and results.
