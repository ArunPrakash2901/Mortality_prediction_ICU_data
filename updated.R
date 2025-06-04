###############################################################################
#  MIMIC-III – HOSPITAL_EXPIRE_FLAG PIPELINE (v-6, with Comorbidity Features)
#  • Adds Charlson & Elixhauser indices and flags
#  • Diagnosis count per admission
#  • Logistic, Random Forest, XGBoost with parallel tuning
#  • Separate Kaggle-ready submissions for each model
###############################################################################

# ── 0.  Libraries ────────────────────────────────────────────────────────────
library(tidyverse)
library(dplyr)
library(tidymodels)
library(xgboost)
library(doParallel)
library(future)
library(comorbidity)
library(keras)
library(parsnip)
library(tensorflow)
library(rsample)
library(text2vec)
library(lubridate)
library(yardstick)
library(stacks)
library(embed)

# ── 1.  Load & Merge Data ────────────────────────────────────────────────────
set.seed(7)
data_dir <- "data/"

train_X <- read_csv(file.path(data_dir, "mimic_train_X.csv"))
train_y <- read_csv(file.path(data_dir, "mimic_train_y.csv"))
test_X  <- read_csv(file.path(data_dir, "mimic_test_X.csv"))

train <- inner_join(train_X, train_y, by = "icustay_id") %>%
  select(-starts_with("...1")) %>%
  mutate(
    HOSPITAL_EXPIRE_FLAG = factor(HOSPITAL_EXPIRE_FLAG, levels = c(0, 1))
  )

# ── 2.  ICD-9 Helper: Chapter ────────────────────────────────────────────────
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

train <- train %>%
  mutate(ICD9_chapter = icd9_to_chapter(ICD9_diagnosis)) %>%
  select(-ICD9_diagnosis)

test_X <- test_X %>%
  mutate(ICD9_chapter = icd9_to_chapter(ICD9_diagnosis)) %>%
  select(-ICD9_diagnosis)

# ── 4.  Comorbidity Features ─────────────────────────────────────────────────
diag_df <- read_csv(file.path(data_dir, "MIMIC_diagnoses.csv"))

# Charlson
charlson <- comorbidity(x = diag_df,
                        id = "HADM_ID",
                        code = "ICD9_CODE",
                        map = "charlson_icd9_quan",
                        assign0 = TRUE)
charlson$CharlsonIndex <- score(charlson, weights = "charlson", assign0 = TRUE)

# Elixhauser
elix <- comorbidity(x = diag_df,
                    id = "HADM_ID",
                    code = "ICD9_CODE",
                    map = "elixhauser_icd9_quan",
                    assign0 = TRUE)
elix$ElixhauserIndex <- score(elix, weights = "vw", assign0 = TRUE)

diag_counts <- diag_df %>%
  group_by(HADM_ID) %>%
  summarise(icd9_ncodes = n_distinct(ICD9_CODE), .groups = "drop")

charlson    <- charlson    %>% rename(hadm_id = HADM_ID)
elix        <- elix        %>% rename(hadm_id = HADM_ID)
diag_counts <- diag_counts %>% rename(hadm_id = HADM_ID)

train <- train %>%
  left_join(charlson,    by = "hadm_id") %>%
  left_join(elix,        by = "hadm_id", suffix = c("", "_elix")) %>%
  left_join(diag_counts, by = "hadm_id")

test_X <- test_X %>%
  left_join(charlson,    by = "hadm_id") %>%
  left_join(elix,        by = "hadm_id", suffix = c("", "_elix")) %>%
  left_join(diag_counts, by = "hadm_id")

# ── 5.  Diagnosis Embeddings (GloVe) ──────────────────────────────────────────
diag_df <- diag_df %>% arrange(HADM_ID, SEQ_NUM)
sentences <- diag_df %>%
  group_by(HADM_ID) %>%
  summarise(icd_seq = list(as.character(ICD9_CODE)), .groups = "drop")

tokens     <- itoken(sentences$icd_seq, progressbar = TRUE)
vocab      <- create_vocabulary(tokens)
vectorizer <- vocab_vectorizer(vocab)
tcm        <- create_tcm(tokens, vectorizer, skip_grams_window = 5L)

set.seed(7)
glove      <- GlobalVectors$new(rank = 50, x_max = 10)
wv_main    <- glove$fit_transform(tcm, n_iter = 20, convergence_tol = 0.005)
wv_context <- glove$components
word_vecs  <- wv_main + t(wv_context)

emb_tbl <- as_tibble(word_vecs, .name_repair = "unique") %>%
  mutate(ICD9_CODE = rownames(word_vecs))
names(emb_tbl)[1:50] <- str_glue("icd_emb_{1:50}")

admit_emb <- diag_df %>%
  inner_join(emb_tbl, by = "ICD9_CODE") %>%
  group_by(HADM_ID) %>%
  summarise(across(starts_with("icd_emb_"), mean, na.rm = TRUE), .groups = "drop") %>%
  rename(hadm_id = HADM_ID)

train  <- train  %>% left_join(admit_emb, by = "hadm_id")
test_X <- test_X %>% left_join(admit_emb, by = "hadm_id")

cat("Added", sum(startsWith(names(train), "icd_emb_")), "embedding features\n")

# ── 6.  Preprocessing Recipe ─────────────────────────────────────────────────
id_vars <- c("subject_id", "hadm_id", "icustay_id")

rec_final <- recipe(HOSPITAL_EXPIRE_FLAG ~ ., data = train) %>%
  update_role(all_of(id_vars), new_role = "id") %>%

  # 1) age + nonlinear age
  step_mutate(
    AGE_YRS = as.numeric(difftime(ADMITTIME, DOB, units = "days")) / 365.25,
  ) %>%
  step_dummy(starts_with("any_")) %>%
  step_rm(DOB, ADMITTIME, Diff) %>%
  step_other(DIAGNOSIS, threshold = 0.01, other = "rare") %>%
  step_novel(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_normalize(all_numeric_predictors())

prepped <- prep(rec_final, verbose = FALSE)
cat("Post-prep predictors:", ncol(juice(prepped)), "\n\n")

# ── 7.  Parallel Backend ─────────────────────────────────────────────────────
core_cnt <- parallel::detectCores(logical = FALSE)
cl       <- makePSOCKcluster(core_cnt)
plan(cluster, workers = cl)

ctrl_grid <- control_grid()
ctrl_rs   <- control_resamples(parallel_over = "everything", verbose = TRUE)

# ── 8.  CV Folds ─────────────────────────────────────────────────────────────
cv5 <- vfold_cv(train, v = 5, strata = HOSPITAL_EXPIRE_FLAG)
# ── 7.  Logistic Regression ──────────────────────────────────────────────────
log_res <- workflow() %>%
  add_recipe(rec_final) %>%
  add_model(logistic_reg() %>% set_engine("glm")) %>%
  fit_resamples(cv5, metric_set(roc_auc), control = ctrl_rs)
print(collect_metrics(log_res))

final_logit <- workflow() %>%
  add_recipe(rec_final) %>%
  add_model(logistic_reg() %>% set_engine("glm")) %>%
  fit(data = train)

prob_logit <- predict(final_logit, new_data = test_X, type = "prob")$.pred_1

write_csv(tibble(ID = test_X$icustay_id, HOSPITAL_EXPIRE_FLAG = prob_logit), "submission_logit.csv")
cat("✔ submission_logit.csv written\n")

# ── 8.  Random Forest ────────────────────────────────────────────────────────
rf_spec <- rand_forest(mtry = tune(), trees = 1000, min_n = 5) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")
grid_rf <- grid_regular(mtry(range = c(2, 40)), levels = 5)

rf_res <- workflow() %>% add_recipe(rec_final) %>% add_model(rf_spec) %>%
  tune_grid(cv5, grid = grid_rf, metrics = metric_set(roc_auc), control = ctrl_grid)
print(collect_metrics(rf_res) %>% arrange(desc(mean)))

best_rf  <- select_best(rf_res, metric = "roc_auc")
final_rf <- finalize_workflow(workflow() %>% add_recipe(rec_final) %>% add_model(rf_spec), best_rf) %>%
  fit(train)

prob_rf <- predict(final_rf, new_data = test_X, type = "prob")$.pred_1

write_csv(tibble(ID = test_X$icustay_id, HOSPITAL_EXPIRE_FLAG = prob_rf), "submission_rf.csv")
cat("✔ submission_rf.csv written\n")

# ── 9.  XGBoost ──────────────────────────────────────────────────────────────
pos      <- sum(train$HOSPITAL_EXPIRE_FLAG == "1")
neg      <- sum(train$HOSPITAL_EXPIRE_FLAG == "0")
spw_val  <- neg / pos

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
    nthread          = !!core_cnt,
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
XGB <- collect_metrics(xgb_res) %>% arrange(desc(mean)) %>% print()

best_xgb  <- select_best(xgb_res, metric = "roc_auc")
final_xgb <- finalize_workflow(wf_xgb, best_xgb) %>% workflows::fit(train)

prob_xgb <- predict(final_xgb, new_data = test_X, type = "prob")$.pred_1

write_csv(tibble(ID = test_X$icustay_id, HOSPITAL_EXPIRE_FLAG = prob_xgb), "submission_xgb.csv")
cat("✔ submission_xgb.csv written\n")

nn_spec <- mlp(
  hidden_units = tune(),
  dropout = tune(),
  penalty = tune(),
  epochs = tune()
) %>%
  set_mode("classification") %>%
  set_engine("keras", optimizer = "adam")

# More expressive grid with max entropy sampling
grid_nn <- grid_max_entropy(
  hidden_units(range = c(32, 256)),
  dropout(range = c(0, 0.5)),
  penalty(range = c(-6, -2), trans = scales::exp_trans()), # 1e-6 to 1e-2
  epochs(range = c(20, 80)),
  size = 20
)

nn_res <- tune_grid(
  nn_wf,
  resamples = cv5,
  grid = grid_nn,
  metrics = metric_set(roc_auc),
  control = ctrl_grid
)
collect_metrics(nn_res) %>% arrange(desc(mean)) %>% print()

best_nn <- select_best(nn_res, metric = "roc_auc")
final_nn <- finalize_workflow(nn_wf, best_nn) %>% workflows::fit(train)

prob_nn <- predict(final_nn, new_data = test_X, type = "prob")$.pred_1

write_csv(tibble(ID = test_X$icustay_id, HOSPITAL_EXPIRE_FLAG = prob_nn), "submission_nn.csv")
cat("✔ submission_nn.csv written\n")

# ── 10.  Cleanup ─────────────────────────────────────────────────────────────
plan(sequential)
stopCluster(cl)
