# ==== LOAD PACKAGES ====
library(tidyverse)
library(tidymodels)
library(keras3)
library(xgboost)
library(yardstick)

# ==== SET SEED ====
set.seed(7)

# ==== LOAD & CLEAN DATA ====
data_dir <- "data/"
x_train <- read_csv(file.path(data_dir, "mimic_train_X.csv")) %>% select(-starts_with("...1"))
y_train <- read_csv(file.path(data_dir, "mimic_train_y.csv")) %>% select(-starts_with("...1"))
x_test  <- read_csv(file.path(data_dir, "mimic_test_X.csv"))  %>% select(-starts_with("...1"))

train  <- x_train %>% inner_join(y_train, by = "icustay_id")

add_engineered <- function(df) {
  df %>%
    # ── Age features ───────────────────────────────────────────────
    mutate(
      age_yrs = as.numeric(difftime(ADMITTIME, DOB, units = "days"))/365.25,
      age_bin = cut(age_yrs,
                    breaks = c(-Inf, 40, 60, 80, Inf),
                    labels = c("<=40","40_60","60_80",">80"))
    ) %>%

    # ── Vital spreads (explicit) ──────────────────────────────────
    mutate(
      HeartRate_spread = HeartRate_Max - HeartRate_Min,
      SysBP_spread     = SysBP_Max     - SysBP_Min,
      DiasBP_spread    = DiasBP_Max    - DiasBP_Min,
      MeanBP_spread    = MeanBP_Max    - MeanBP_Min,
      RespRate_spread  = RespRate_Max  - RespRate_Min,
      TempC_spread     = TempC_Max     - TempC_Min,
      SpO2_spread      = SpO2_Max      - SpO2_Min,
      Glucose_spread   = Glucose_Max   - Glucose_Min
    ) %>%

    # ── Derived indices & flags ───────────────────────────────────
    mutate(
      shock_idx      = HeartRate_Mean / pmax(SysBP_Mean, 1),
      low_spo2       = as.integer(SpO2_Min  < 90),
      tachy          = as.integer(HeartRate_Max > 120),
      fever          = as.integer(TempC_Max > 38.3),
      hypotension    = as.integer(SysBP_Min < 90),
      hypotension_flag = hypotension,
      log_glucose    = log1p(Glucose_Mean)
    )
}


# apply feature engineering
x_train <- add_engineered(x_train)
x_test  <- add_engineered(x_test)

# rebuild the modelling frame – **now** it contains age_bin & friends
train   <- inner_join(x_train, y_train, by = "icustay_id")



# ==== PREPROCESSING RECIPE (as before) ====
small_cat <- c("GENDER","ADMISSION_TYPE","INSURANCE","RELIGION",
               "MARITAL_STATUS","ETHNICITY","FIRST_CAREUNIT")
large_cat <- c("ICD9_diagnosis","DIAGNOSIS")
id_vars <- c("icustay_id")

rec <- recipe(HOSPITAL_EXPIRE_FLAG ~ ., data = train) %>%
  # Set ID variables with a special role so they're retained but not used as predictors
  update_role(all_of(id_vars), new_role = "id") %>%

  # Remove unneeded datetime and diff columns
  step_rm(DOB, ADMITTIME, Diff) %>%

  # Ensure age_bin is handled properly
  step_unknown(age_bin) %>%
  step_dummy(age_bin) %>%

  # Collapse rare categories in DIAGNOSIS
  step_other(DIAGNOSIS, threshold = 0.01, other = "rare") %>%

  # Handle novel levels at prediction time
  step_novel(all_nominal_predictors()) %>%

  # One-hot encode all nominal predictors
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%

  # Impute missing values in numeric predictors
  step_impute_median(all_numeric_predictors()) %>%

  # Remove predictors with zero variance
  step_zv(all_predictors()) %>%

  # Normalize all numeric predictors
  step_normalize(all_numeric_predictors())


prep_rec <- prep(rec_final)
train_processed <- bake(prep_rec, NULL)
test_processed  <- bake(prep_rec, new_data = test_X)

# ==== SPLIT TRAIN/VAL ====
y_all <- train_processed$HOSPITAL_EXPIRE_FLAG
x_all <- train_processed %>% select(-HOSPITAL_EXPIRE_FLAG)

split  <- initial_split(tibble(y = y_all), prop = .8, strata = y)
i_trn  <- split$in_id
x_trn  <- x_all[i_trn, ]
x_val  <- x_all[-i_trn, ]
y_trn  <- y_all[i_trn]
y_val  <- y_all[-i_trn]

# ==== PREPARE KERAS INPUTS & TRAIN NN (as before) ====
numeric_inds <- which(! colnames(x_trn) %in% large_cat)

# Numeric + embed splits
x_trn_num <- as.matrix(x_trn[, numeric_inds])
x_val_num <- as.matrix(x_val[, numeric_inds])
x_trn_icd <- as.integer(x_trn$ICD9_diagnosis)
x_val_icd <- as.integer(x_val$ICD9_diagnosis)
x_trn_dia <- as.integer(x_trn$DIAGNOSIS)
x_val_dia <- as.integer(x_val$DIAGNOSIS)

# -- build, compile, and fit your NN (omitted here for brevity) --
# assume after fitting you have:
#   pred_nn_val  — validation predictions (vector of length n_val)
#   pred_nn_test — test set predictions (vector of length n_test)

# For demonstration, let's quickly re-fit a minimal NN and get preds:
model <- keras_model_sequential() %>%
  layer_dense(32, activation = "relu", input_shape = ncol(x_trn_num)) %>%
  layer_dropout(.4) %>%
  layer_dense(16, activation = "relu") %>%
  layer_dropout(.3) %>%
  layer_dense(1, activation = "sigmoid")

model %>% compile(
  optimizer = optimizer_adam(1e-3),
  loss      = loss_binary_focal_crossentropy(),
  metrics   = metric_auc()
)
es_cb <- callback_early_stopping("val_auc", patience = 5, restore_best_weights = TRUE)

history <- model %>% fit(
  x = x_trn_num, y = y_trn,
  validation_data = list(x_val_num, y_val),
  epochs = 30, batch_size = 512,
  callbacks = list(es_cb), verbose = 0
)

pred_nn_val  <- as.vector(model %>% predict(x_val_num))
pred_nn_test <- as.vector(model %>% predict(as.matrix(test_processed[, numeric_inds])))

# ==== TRAIN XGBOOST ON SAME FEATURES ====
dtrain <- xgb.DMatrix(data = as.matrix(x_trn), label = y_trn)
dval   <- xgb.DMatrix(data = as.matrix(x_val), label = y_val)
dtest  <- xgb.DMatrix(data = as.matrix(test_processed))

params <- list(
  objective   = "binary:logistic",
  eval_metric = "auc",
  eta         = 0.1,
  max_depth   = 6,
  subsample   = 0.8,
  colsample_bytree = 0.8
)

watchlist <- list(train = dtrain, eval = dval)
xgb_mod <- xgb.train(
  params,
  dtrain,
  nrounds = 200,
  watchlist,
  early_stopping_rounds = 10,
  verbose = 0
)

pred_xgb_val  <- predict(xgb_mod, dval)
pred_xgb_test <- predict(xgb_mod, dtest)

# ==== ENSEMBLE & EVALUATE ON VALIDATION ====
pred_ens_val  <- (pred_nn_val + pred_xgb_val) / 2

roc_auc_vec <- roc_auc(
  tibble(truth = factor(y_val), .pred = pred_ens_val),
  truth, .pred, event_level = "second"
)
print(roc_auc_vec)   # should be > max(individual)

# ==== FINAL SUBMISSION ====
pred_ens_test <- (pred_nn_test + pred_xgb_test) / 2

submission <- tibble(
  ID = x_test$icustay_id,
  HOSPITAL_EXPIRE_FLAG = pred_ens_test
)

write_csv(submission, "ensemble_submission.csv")
