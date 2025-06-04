###############################################################################
#  MIMIC-III – HOSPITAL_EXPIRE_FLAG PIPELINE  (v-5, 17-May-2025)
#  • High-cardinality fix (ICD-9 chapter + DIAGNOSIS lump)
#  • Logistic baseline, tuned Random-Forest, tuned XGBoost
#  • Parallel via {future} on every physical CPU core
#  • Writes submission_xgb.csv ready for Kaggle
###############################################################################

# ── 0.  Libraries ────────────────────────────────────────────────────────────
library(tidyverse)       # readr, dplyr, tidyr, ggplot2, forcats …
library(tidymodels)      # recipes, workflows, tune, rsample, yardstick
library(xgboost)         # XGBoost engine
library(doParallel)      # make PSOCK clusters
library(future)          # modern backend used by tune

# ── 1.  Load & merge data ────────────────────────────────────────────────────
set.seed(7)
data_dir <- "data/"                         # <— adjust to your folder

train_X <- read_csv(file.path(data_dir, "mimic_train_X.csv"))
train_y <- read_csv(file.path(data_dir, "mimic_train_y.csv"))
test_X  <- read_csv(file.path(data_dir, "mimic_test_X.csv"))


train <- inner_join(train_X, train_y, by = "icustay_id") %>%
  select(-starts_with("...1")) %>%                         # drop row indices
  mutate(
    HOSPITAL_EXPIRE_FLAG = factor(HOSPITAL_EXPIRE_FLAG, levels = c(0, 1))
  )

# ── 2.  ICD-9 helper  – applied once (not in recipe) ────────────────────────
icd9_to_chapter <- function(code_chr) {
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

train <- train  %>% mutate(ICD9_chapter = icd9_to_chapter(ICD9_diagnosis)) %>%
  select(-ICD9_diagnosis)
test_X <- test_X %>% mutate(ICD9_chapter = icd9_to_chapter(ICD9_diagnosis)) %>%
  select(-ICD9_diagnosis)


# ── 3.  Recipe – collapse DIAGNOSIS, dummy, impute, scale ───────────────────
id_vars <- c("subject_id", "hadm_id", "icustay_id")

rec_final <- recipe(HOSPITAL_EXPIRE_FLAG ~ ., data = train) %>%
  update_role(all_of(id_vars), new_role = "id") %>%
  step_mutate(
    AGE_YRS = as.numeric(difftime(ADMITTIME, DOB, units = "days")) / 365.25
  ) %>%
  step_rm(DOB, ADMITTIME, Diff) %>%
  step_other(DIAGNOSIS, threshold = 0.01, other = "rare") %>%
  step_novel(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

prepped <- prep(rec_final, verbose = FALSE)
cat("Predictor columns after preprocessing:", ncol(juice(prepped)), "\n\n")

# ── 4.  Parallel backend (PSOCK + future) ───────────────────────────────────
core_cnt <- parallel::detectCores(logical = FALSE)   # numeric literal
cl        <- makePSOCKcluster(core_cnt)
plan(cluster, workers = cl)

ctrl_grid <- control_grid(save_pred = TRUE,
                          parallel_over = "everything",
                          verbose = TRUE)
ctrl_rs   <- control_resamples(parallel_over = "everything",
                               verbose = TRUE)

# ── 5.  Baseline models ─────────────────────────────────────────────────────
cv5 <- vfold_cv(train, v = 5, strata = HOSPITAL_EXPIRE_FLAG)

## logistic
log_res <- workflow() %>%
  add_recipe(rec_final) %>% add_model(logistic_reg() %>% set_engine("glm")) %>%
  fit_resamples(cv5, metric_set(roc_auc), control = ctrl_rs)
print(collect_metrics(log_res))

## random-forest (tune mtry)
rf_spec <- rand_forest(mtry = tune(), trees = 1000, min_n = 5) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")
grid_rf <- grid_regular(mtry(range = c(2, 40)), levels = 5)

rf_res <- workflow() %>% add_recipe(rec_final) %>% add_model(rf_spec) %>%
  tune_grid(cv5, grid = grid_rf, metrics = metric_set(roc_auc),
            control = ctrl_grid)
print(collect_metrics(rf_res) %>% arrange(desc(mean)))

# ── 6.  XGBoost tuning (all literals) ----------------------------------------
pos      <- sum(train$HOSPITAL_EXPIRE_FLAG == "1")
neg      <- sum(train$HOSPITAL_EXPIRE_FLAG == "0")
spw_val  <- neg / pos                    # numeric literal

xgb_spec <- boost_tree(
  trees          = 2000,
  stop_iter      = 100,
  tree_depth     = tune(),
  learn_rate     = tune(),
  loss_reduction = tune(),
  sample_size    = tune(),
  mtry           = tune()
) %>%
  set_engine(
    "xgboost",
    tree_method      = "hist",
    nthread          = !!core_cnt,   # bang-bang embeds the integer
    eval_metric      = "auc",
    scale_pos_weight = !!spw_val
  ) %>%
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

xgb_res <- tune_grid(wf_xgb, resamples = cv5, grid = grid_xgb,
                     metrics = metric_set(roc_auc), control = ctrl_grid)

collect_metrics(xgb_res) %>% arrange(desc(mean)) %>% print()

best_xgb  <- select_best(xgb_res, metric = "roc_auc")
final_xgb <- finalize_workflow(wf_xgb, best_xgb) %>% fit(train)

# ── 7.  submission -----------------------------------------------------------
prob_df <- predict(final_xgb,
                   new_data = test_X,
                   type     = "prob")

# Extract the class-1 probabilities
prob <- prob_df %>% pull(.pred_1)


write_csv(tibble(ID = test_X$icustay_id,
                 HOSPITAL_EXPIRE_FLAG = prob),
          "submission_xgb.csv")
cat("\n✔ submission_xgb.csv written\n")

# ── 8.  cleanup --------------------------------------------------------------
plan(sequential)
stopCluster(cl)
