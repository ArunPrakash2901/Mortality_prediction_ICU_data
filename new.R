###############################################################################
#  MIMIC-III  –  Hospital_Expire_Flag pipeline (CPU only + progress bars)
#  • engineered features
#  • 5-fold CV hyper-parameter tuning (xgboost + keras3)
#  • ridge-logistic stacking
#  • writes ensemble_submission.csv
#  18-May-2025
###############################################################################

suppressPackageStartupMessages({
  library(tidyverse)
  library(tidymodels)    # 1.3.0
  library(keras3)        # 1.4.0
  library(xgboost)       # 1.7+
  library(glmnet)        # for ridge-stack
  library(doParallel)    # parallel backend
  library(progress)      # progress bars
  library(glue)
})

## ── 0. GLOBAL OPTIONS & PARALLEL BACKEND ------------------------------------
set.seed(7)
n_cores <- parallel::detectCores(logical = FALSE)
registerDoParallel(n_cores)    # foreach/doParallel for tune_grid()

data_dir <- "data"
id_vars  <- "icustay_id"

###############################################################################
## 1. HELPERS -----------------------------------------------------------------
###############################################################################
icd9_to_chap <- function(code_chr) {
  code_int <- as.integer(substr(code_chr, 1, 3))
  dplyr::case_when(
    code_int >=   1 & code_int <= 139 ~ "Infectious",
    code_int >= 140 & code_int <= 239 ~ "Neoplasms",
    code_int >= 240 & code_int <= 279 ~ "Endocrine",
    code_int >= 280 & code_int <= 289 ~ "Blood",
    code_int >= 290 & code_int <= 319 ~ "Mental",
    code_int >= 320 & code_int <= 389 ~ "Nervous",
    code_int >= 390 & code_int <= 459 ~ "Circulatory",
    code_int >= 460 & code_int <= 519 ~ "Respiratory",
    code_int >= 520 & code_int <= 579 ~ "Digestive",
    code_int >= 580 & code_int <= 629 ~ "Genitourinary",
    code_int >= 630 & code_int <= 679 ~ "Pregnancy",
    code_int >= 680 & code_int <= 709 ~ "Skin",
    code_int >= 710 & code_int <= 739 ~ "Musculoskeletal",
    code_int >= 740 & code_int <= 759 ~ "Congenital",
    code_int >= 760 & code_int <= 779 ~ "Perinatal",
    code_int >= 780 & code_int <= 799 ~ "Symptoms",
    code_int >= 800 & code_int <= 999 ~ "Injury",
    TRUE                                ~ "Unknown"
  )
}

add_engineered <- function(df) {
  df %>%
    # ICD-9 → chapter
    mutate(ICD9_chap = icd9_to_chap(ICD9_diagnosis)) %>%
    # age & bins
    mutate(
      age_yrs = as.numeric(difftime(ADMITTIME, DOB, units = "days")) / 365.25,
      age_bin = cut(age_yrs,
                    breaks = c(-Inf, 40, 60, 80, Inf),
                    labels = c("<=40","40_60","60_80",">80"))
    ) %>%
    # vital spreads
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
    # flags & indices
    mutate(
      shock_idx   = HeartRate_Mean / pmax(SysBP_Mean, 1),
      low_spo2    = as.integer(SpO2_Min  < 90),
      tachy       = as.integer(HeartRate_Max > 120),
      fever       = as.integer(TempC_Max > 38.3),
      hypotension = as.integer(SysBP_Min < 90),
      log_glucose = log1p(Glucose_Mean)
    )
}

###############################################################################
## 2. LOAD & ENGINEER DATA ----------------------------------------------------
###############################################################################
message("📂 Loading data…")
train_X <- read_csv(file.path(data_dir, "mimic_train_X.csv"), show_col_types = FALSE) %>%
  select(-starts_with("..."))
train_y <- read_csv(file.path(data_dir, "mimic_train_y.csv"), show_col_types = FALSE) %>%
  select(-starts_with("..."))
test_X  <- read_csv(file.path(data_dir, "mimic_test_X.csv"),  show_col_types = FALSE) %>%
  select(-starts_with("..."))

train <- inner_join(train_X, train_y, by = "icustay_id") %>% add_engineered()
test_X <- add_engineered(test_X)

train <- train %>% mutate(HOSPITAL_EXPIRE_FLAG = factor(HOSPITAL_EXPIRE_FLAG, levels = c(0,1)))

###############################################################################
## 3. PREPROCESSING RECIPE ----------------------------------------------------
###############################################################################
rec <- recipe(HOSPITAL_EXPIRE_FLAG ~ ., data = train) %>%
  update_role(all_of(id_vars), new_role = "id") %>%
  step_rm(DOB, ADMITTIME) %>%
  step_other(DIAGNOSIS, threshold = 0.01) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

mat_prep   <- prep(rec, verbose = FALSE)
train_proc <- bake(mat_prep, NULL)
test_proc  <- bake(mat_prep, test_X)

###############################################################################
## 4. CREATE 5-FOLDS ----------------------------------------------------------
###############################################################################
set.seed(7)
folds <- vfold_cv(train_proc, v = 5, strata = HOSPITAL_EXPIRE_FLAG)

###############################################################################
## 5-A/B PARAMETER SET & GRID ------------------------------------------------
###############################################################################
xgb_params <- parameters(
  trees(range = c(200, 1500)),
  tree_depth(range = c(3L, 10L)),
  learn_rate(range = c(-4, -0.5)),
  loss_reduction(range = c(0, 5)),
  sample_prop(range = c(0.5, 1)),
  mtry() %>% finalize(train_proc)
)

set.seed(7)
xgb_grid <- grid_space_filling(xgb_params, size = 20, orig = TRUE)

###############################################################################
## 5-C XGBOOST SPEC & WORKFLOW -----------------------------------------------
###############################################################################
xgb_spec <- boost_tree(
  trees          = tune(),
  tree_depth     = tune(),
  learn_rate     = tune(),
  loss_reduction = tune(),
  sample_size    = tune(),
  mtry           = tune()
) %>%
  set_mode("classification") %>%
  set_engine("xgboost", eval_metric = "auc", nthread = parallel::detectCores(logical = FALSE))

xgb_wf <- workflow() %>%
  add_model(xgb_spec) %>%
  add_formula(HOSPITAL_EXPIRE_FLAG ~ .)

###############################################################################
## 5-D TUNE XGBOOST WITH PROGRESS BAR -----------------------------------------
###############################################################################
total_xgb <- nrow(xgb_grid) * length(folds$splits)
pb_xgb <- progress_bar$new(
  format = "XGB tuning [:bar] :current/:total (:percent) eta: :eta",
  total  = total_xgb,
  clear  = FALSE, width = 60
)

xgb_res <- tune_grid(
  xgb_wf,
  resamples = folds,
  grid      = xgb_grid,
  metrics   = metric_set(roc_auc),
  control   = control_grid(
    save_pred     = TRUE,
    verbose       = FALSE,
    parallel_over = "resamples",
    extract       = function(x) { pb_xgb$tick(); NULL }
  )
)

best_xgb     <- select_best(xgb_res, metric = "roc_auc")
xgb_final    <- finalize_workflow(xgb_wf, best_xgb) %>% fit(train_proc)
pred_xgb_val <- collect_predictions(
  xgb_res,
  parameters = best_xgb
) %>%
  arrange(.row) %>%
  pull(.pred_1)
pred_xgb_test<- predict(xgb_final, test_proc, type = "prob")$.pred_1

###############################################################################
## 6. NEURAL-NET CV + FINAL FIT WITH PROGRESS BAR -----------------------------
###############################################################################

# helper to select numeric columns as a matrix
num_mat <- function(df) df %>% select(where(is.numeric)) %>% data.matrix()

# model-builder with a fixed “auc” name
build_nn <- function(un1, un2, d1, d2, lr, n_in) {
  keras_model_sequential() %>%
    layer_dense(units = un1, activation = "relu", input_shape = n_in) %>%
    layer_batch_normalization() %>%
    layer_dropout(rate = d1) %>%
    layer_dense(units = un2, activation = "relu") %>%
    layer_batch_normalization() %>%
    layer_dropout(rate = d2) %>%
    layer_dense(units = 1, activation = "sigmoid") %>%
    compile(
      optimizer = optimizer_adam(lr),
      loss      = "binary_crossentropy",
      metrics   = list(metric_auc(name = "auc"))  # yields "loss","auc","val_loss","val_auc"
    )
}

# tuning function
tune_nn <- function(train_proc, nfold = 5, grid = NULL, epochs = 60, batch = 512) {
  if (is.null(grid)) {
    grid <- expand_grid(
      un1 = c(256,128,64),
      un2 = c(128,64,32),
      d1  = c(0.4,0.3,0.2),
      d2  = c(0.3,0.2,0.1),
      lr  = c(1e-4,3e-4,1e-3)
    )
  }

  folds_nn <- vfold_cv(train_proc, v = nfold, strata = HOSPITAL_EXPIRE_FLAG)
  n_in      <- ncol(train_proc) - 1
  total_nn  <- nrow(grid) * length(folds_nn$splits)

  pb_nn <- progress_bar$new(
    format = "NN tuning  [:bar] :current/:total (:percent) eta: :eta",
    total  = total_nn, clear = FALSE, width = 60
  )

  grid_log <- grid %>% mutate(cv_auc = NA_real_)
  best_auc <- -Inf
  best_oof <- numeric(nrow(train_proc))

  for (i in seq_len(nrow(grid))) {
    params <- grid[i, ]
    fold_auc <- numeric(nfold)

    for (k in seq_along(folds_nn$splits)) {
      ids_tr <- folds_nn$splits[[k]]$in_id
      ids_va <- folds_nn$splits[[k]]$out_id

      x_tr <- num_mat(train_proc[ids_tr, ]); y_tr <- as.numeric(train_proc$HOSPITAL_EXPIRE_FLAG[ids_tr])
      x_va <- num_mat(train_proc[ids_va, ]); y_va <- as.numeric(train_proc$HOSPITAL_EXPIRE_FLAG[ids_va])

      nn <- build_nn(params$un1, params$un2, params$d1, params$d2, params$lr, n_in)
      nn %>% fit(
        x_tr, y_tr,
        validation_data = list(x_va, y_va),
        epochs     = epochs,
        batch_size = batch,
        verbose    = 0,
        callbacks  = callback_early_stopping(
          monitor              = "val_auc",
          patience             = 8,
          mode                 = "max",
          restore_best_weights = TRUE
        )
      )

      # evaluate + collect
      eval_res     <- nn %>% evaluate(x_va, y_va, verbose = 0)
      fold_auc[k]  <- eval_res[["auc"]]
      preds        <- as.vector(nn %>% predict(x_va))

      # guard against NA in indices
      valid        <- !is.na(ids_va)
      best_oof[ ids_va[valid] ] <- preds[valid]

      pb_nn$tick()
    }

    grid_log$cv_auc[i] <- mean(fold_auc)
    if (grid_log$cv_auc[i] > best_auc) {
      best_auc <- grid_log$cv_auc[i]
      best_cfg <- params
    }
  }

  message(glue::glue(
    "🏆 best NN config: {best_cfg$un1}-{best_cfg$un2} | ",
    "drop {best_cfg$d1}/{best_cfg$d2} | lr {best_cfg$lr} | ",
    "CV-AUC {round(best_auc,3)}"
  ))

  # final fit on full training data
  x_all   <- num_mat(train_proc); y_all <- as.numeric(train_proc$HOSPITAL_EXPIRE_FLAG)
  nn_best <- build_nn(best_cfg$un1, best_cfg$un2, best_cfg$d1, best_cfg$d2, best_cfg$lr, ncol(train_proc)-1)
  nn_best %>% fit(
    x_all, y_all,
    epochs     = epochs,
    batch_size = batch,
    verbose    = 2,
    callbacks  = callback_early_stopping(
      monitor              = "auc",
      patience             = 8,
      mode                 = "max",
      restore_best_weights = TRUE
    )
  )

  list(
    grid = arrange(grid_log, desc(cv_auc)),
    best = list(cfg     = best_cfg,
                model   = nn_best,
                cv_auc  = best_auc,
                oof_pred= best_oof)
  )
}

# === Usage ===
tune_out    <- tune_nn(train_proc)
pred_nn_val <- tune_out$best$oof_pred
nn_final    <- tune_out$best$model
pred_nn_test<- as.vector(nn_final %>% predict(num_mat(test_proc)))


###############################################################################
## 7. BLEND + RIDGE-STACK ------------------------------------------------------
###############################################################################
w_grid    <- seq(0, 1, 0.05)
blend_auc <- map_dbl(w_grid, ~ roc_auc(
  tibble(truth = train_proc$HOSPITAL_EXPIRE_FLAG,
         .pred  = .x * pred_xgb_val + (1 - .x) * pred_nn_val),
  truth, .pred
)$.estimate)
w_best    <- w_grid[which.max(blend_auc)]

stack_fit <- cv.glmnet(
  x           = cbind(xgb = pred_xgb_val, nn = pred_nn_val),
  y           = as.numeric(train_proc$HOSPITAL_EXPIRE_FLAG),
  family      = "binomial",
  alpha       = 0,
  nfolds      = 5,
  type.measure= "auc"
)

auc_tbl <- tibble(
  model = c("XGB-cv", "NN-cv", glue("Blend w={w_best}"), "Ridge stack"),
  auc   = c(
    max(collect_metrics(xgb_res)$mean),
    tune_out$best$cv_auc,
    max(blend_auc),
    max(stack_fit$cvm)
  )
)
print(auc_tbl)

###############################################################################
## 8. WRITE SUBMISSION --------------------------------------------------------
###############################################################################
pred_stack_test <- as.vector(
  predict(stack_fit,
          newx = cbind(xgb = pred_xgb_test, nn = pred_nn_test),
          s    = "lambda.min",
          type = "response")
)

write_csv(
  tibble(ID = test_X$icustay_id,
         HOSPITAL_EXPIRE_FLAG  = pred_stack_test),
  "ensemble_submission.csv"
)

message("✅ Finished — ridge-stack val-AUC = ", sprintf("%.3f", max(stack_fit$cvm)))
