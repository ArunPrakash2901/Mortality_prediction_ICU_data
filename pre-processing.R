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

# ── 1.  Load & Merge Data ────────────────────────────────────────────────────
set.seed(7)
data_dir <- "data/"

train_X <- read_csv(file.path(data_dir, "mimic_train_X.csv"))
train_y <- read_csv(file.path(data_dir, "mimic_train_y.csv"))
test_X  <- read_csv(file.path(data_dir, "mimic_test_X.csv"))

train <- inner_join(train_X, train_y, by = "icustay_id") %>%
  select(-starts_with("...1")) %>%
  mutate(HOSPITAL_EXPIRE_FLAG = factor(HOSPITAL_EXPIRE_FLAG, levels = c(0, 1)))

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

train <- train  %>%
  mutate(ICD9_chapter = icd9_to_chapter(ICD9_diagnosis)) %>%
  select(-ICD9_diagnosis)
test_X <- test_X %>%
  mutate(ICD9_chapter = icd9_to_chapter(ICD9_diagnosis)) %>%
  select(-ICD9_diagnosis)

# ── 3.  Comorbidity Features ─────────────────────────────────────────────────
diag_df <- read_csv(file.path(data_dir, "MIMIC_diagnoses.csv"))   # HADM_ID, ICD9_CODE

# -- Charlson
charlson <- comorbidity(x = diag_df, id = "HADM_ID", code = "ICD9_CODE", map = "charlson_icd9_quan", assign0 = TRUE)
charlson$CharlsonIndex <- score(charlson, weights = "charlson", assign0 = TRUE)

# -- Elixhauser
elix <- comorbidity(x = diag_df, id = "HADM_ID", code = "ICD9_CODE", map = "elixhauser_icd9_quan", assign0 = TRUE)
elix$ElixhauserIndex <- score(elix, weights = "vw", assign0 = TRUE)

# -- Diagnosis count per admission
diag_counts <- diag_df %>%
  group_by(HADM_ID) %>%
  summarise(icd9_ncodes = n_distinct(ICD9_CODE), .groups = "drop")

# Standardize ID column names to hadm_id for clean joins
charlson    <- charlson    %>% rename(hadm_id = HADM_ID)
elix        <- elix        %>% rename(hadm_id = HADM_ID)
diag_counts <- diag_counts %>% rename(hadm_id = HADM_ID)

# --- Merge all features into train/test (after all renaming!) ---

train <- train %>%
  left_join(charlson,    by = "hadm_id") %>%
  left_join(elix,        by = "hadm_id", suffix = c("", "_elix")) %>%
  left_join(diag_counts, by = "hadm_id")

test_X <- test_X %>%
  left_join(charlson,    by = "hadm_id") %>%
  left_join(elix,        by = "hadm_id", suffix = c("", "_elix")) %>%
  left_join(diag_counts, by = "hadm_id")

# ── 4.  Preprocessing Recipe ─────────────────────────────────────────────────
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
