library(tidymodels)
library(stacks)
library(doParallel)
library(future)
library(xgboost)
library(ranger)

# -------------------------------------------------------------------------
# 0. Parallel backend ------------------------------------------------------
core_cnt <- parallel::detectCores(logical = FALSE)
cl       <- makePSOCKcluster(core_cnt)
plan(cluster, workers = cl)

# -------------------------------------------------------------------------
ctrl_rs  <- control_resamples(save_pred = TRUE, save_workflow = TRUE,
                              verbose = TRUE)

ctrl_grid <- control_grid(save_pred = TRUE, save_workflow = TRUE,
                          parallel_over = "everything", verbose = TRUE)

# -------------------------------------------------------------------------
# 2. CV folds --------------------------------------------------------------
cv5 <- vfold_cv(train, v = 5, strata = HOSPITAL_EXPIRE_FLAG)

# -------------------------------------------------------------------------
# 3. Logistic regression ---------------------------------------------------
log_wf <- workflow() %>%
  add_recipe(rec_final) %>%
  add_model(logistic_reg() %>% set_engine("glm"))

log_res <- fit_resamples(
  log_wf,
  resamples = cv5,
  metrics   = metric_set(roc_auc),
  control   = ctrl_rs
)

# -------------------------------------------------------------------------
# 4. Random forest (tuned) -------------------------------------------------
rf_spec <- rand_forest(mtry = tune(), trees = 1000, min_n = 5) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

grid_rf <- grid_regular(mtry(range = c(2, 40)), levels = 5)

rf_wf <- workflow() %>%
  add_recipe(rec_final) %>%
  add_model(rf_spec)

rf_res <- tune_grid(
  rf_wf,
  resamples = cv5,
  grid      = grid_rf,
  metrics   = metric_set(roc_auc),
  control   = ctrl_grid
)

# -------------------------------------------------------------------------
# 5. XGBoost (tuned) -------------------------------------------------------
prepped   <- prep(rec_final)     # <-- now mtry() can be finalised
pos       <- sum(train$HOSPITAL_EXPIRE_FLAG == "1")
neg       <- sum(train$HOSPITAL_EXPIRE_FLAG == "0")
spw_val   <- neg / pos

xgb_spec <- boost_tree(
  trees          = 2000,
  stop_iter      = 100,
  tree_depth     = tune(),
  learn_rate     = tune(),
  loss_reduction = tune(),
  sample_size    = tune(),
  mtry           = tune()
) %>%
  set_engine("xgboost",
             tree_method      = "hist",
             nthread          = !!core_cnt,
             eval_metric      = "auc",
             scale_pos_weight = !!spw_val) %>%
  set_mode("classification")

wf_xgb <- workflow() %>% add_recipe(rec_final) %>% add_model(xgb_spec)

grid_xgb <- grid_space_filling(
  tree_depth(),
  learn_rate(range = c(-4, -1)),
  loss_reduction(),
  sample_prop(),
  finalize(mtry(), juice(prepped)),
  size = 20
)

xgb_res <- tune_grid(
  wf_xgb,
  resamples = cv5,
  grid      = grid_xgb,
  metrics   = metric_set(roc_auc),
  control   = ctrl_grid
)

# -------------------------------------------------------------------------
# 6. Stack, blend, fit -----------------------------------------------------
model_stack <- stacks() %>%
  add_candidates(log_res) %>%
  add_candidates(rf_res)  %>%
  add_candidates(xgb_res) %>%
  blend_predictions(metric = metric_set(roc_auc)) %>%
  fit_members(model_stack)

valid_preds <- predict(model_stack, new_data = test_X, type = "prob")

valid_results <- bind_cols(test_X, valid_preds)


auc_valid <- roc_auc(
  valid_results,
  truth       = HOSPITAL_EXPIRE_FLAG,
  .pred_1,
  event_level = "second"
)

print(auc_valid)


test_probs <- predict(
  model_stack,          # the fitted <linear_stack>
  new_data = test_X,
  type     = "prob"
)

###############################################################################
## 2.  Build the submission data-frame ----------------------------------------
##     - assumes test_X has an `icustay_id` column
##     - renames the probability column to the target header your competition /
##       instructor expects (often "HOSPITAL_EXPIRE_FLAG" or "prob").
###############################################################################
submission <- test_X %>%
  dplyr::select(icustay_id) %>%              # keep only the row ID
  dplyr::bind_cols(test_probs %>%            # add predicted P(death = 1)
                     dplyr::select(.pred_1)) %>%
  dplyr::rename(HOSPITAL_EXPIRE_FLAG = .pred_1)  # adjust header if needed

###############################################################################
## 3.  Save to CSV ------------------------------------------------------------
###############################################################################
readr::write_csv(submission, "submission.csv")
