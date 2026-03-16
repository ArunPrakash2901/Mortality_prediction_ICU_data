# MIMIC III Mortality Prediction

## Project Description
This project predicts if a patient will die during their hospital stay. It uses data from the MIMIC III dataset. The project starts with simple models and moves to complex machine learning methods.

## Repository Structure
Here is an overview of the main folders and files in this project:

* `data/`: Contains the raw and processed MIMIC III datasets.
* `RDS/`: Stores saved model objects, such as the final XGBoost model.
* `SHAP/`: Contains the generated SHAP visualization images.
* `Submissions/`: Stores the final prediction CSV files.
* `pre-processing.R`: Script for data cleaning and feature engineering.
* `MMIC.R`: Main pipeline script covering logistic regression, Random Forest, and XGBoost.
* `NN_MMICIII.R` and `LightGBM.R`: Scripts for training the Neural Network and LightGBM models.
* `Stack(Logit, RF, and XGB.R` and `Ensemble model(NN and XGB).R`: Scripts for building stacked ensemble models.
* `shap generation_2(XGB).R` and `Feature importance(XGB).R`: Scripts to generate SHAP values and feature importance plots.
* `Severity__score.R`: Script for calculating clinical severity scores.
* `ML report.pdf`: Detailed project report explaining the methodology.
* `Presentation.qmd` / `Presentation.html`: Project presentation files.
  
## Feature Engineering
The original data has hundreds of different ICD 9 diagnosis codes. To help the model learn, I created new features:
* I grouped the ICD 9 codes into broad medical chapters, like "Infectious" and "Neoplasms".
* I calculated Charlson and Elixhauser comorbidity scores to measure disease burden.
* I counted the unique ICD codes for each patient to measure complexity.
* I used GloVe embeddings to change the diagnosis codes into 50 numerical columns.

## Preprocessing
I used the recipes package to clean and prepare the data. The steps include:
* Calculating the patient age.
* Removing date fields and ID columns.
* Using one hot encoding for categorical variables.
* Imputing missing numeric values and normalizing all numeric predictors.

## Models Used
I trained and compared several models:
* Logistic Regression as a baseline model.
* Random Forest with tuned hyperparameters.
* XGBoost with tuned learning rate and tree depth.
* Neural Network using keras.
* A stacked ensemble model that combines the XGBoost and Neural Network predictions.

## Results and Interpretability
I evaluated the models using 5 fold cross validation and the ROC AUC metric. The stacked ensemble model is the best performer. It achieves a final ROC AUC of around 0.95.
[Leaderboard link](https://www.kaggle.com/competitions/predicting-the-probability-of-death-in-an-icu/leaderboard)

To explain how the XGBoost model makes decisions, the code generates SHAP values. It creates waterfall and beeswarm plots to show feature importance.

## Acknowledgement
I used AI to research methods and improve my feature engineering code.
