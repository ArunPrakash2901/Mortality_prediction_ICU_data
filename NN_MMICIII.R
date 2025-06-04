# ── Packages ──────────────────────────────────────────────────────────
library(tidyverse)
library(tidymodels)        # recipes, rsample, yardstick, etc.
library(keras3)
library(tfruns)            # optional: helps track runs

# ── Reproducibility ───────────────────────────────────────────────────
set.seed(7)

data_dir   <- "data/"

x_train <- read_csv(file.path(data_dir, "mimic_train_X.csv"))
y_train <- read_csv(file.path(data_dir, "mimic_train_y.csv"))
x_test  <- read_csv(file.path(data_dir, "mimic_test_X.csv"))

x_train <- x_train %>% select(-starts_with("...1"))
y_train <- y_train %>% select(-starts_with("...1"))
x_test  <- x_test  %>% select(-starts_with("...1"))

train <- x_train %>% inner_join(y_train, by = "icustay_id")

# ── Column groups ─────────────────────────────────────────────────────
id_cols    <- c("icustay_id")
drop_cols  <- c("subject_id", "hadm_id", "DOB", "ADMITTIME")
small_cat  <- c("GENDER","ADMISSION_TYPE","INSURANCE",
                "RELIGION","MARITAL_STATUS","ETHNICITY","FIRST_CAREUNIT")
large_cat  <- c("ICD9_diagnosis","DIAGNOSIS")

# ── Recipe ────────────────────────────────────────────────────────────
rec <- recipe(HOSPITAL_EXPIRE_FLAG ~ ., data = train) %>%
  # Drop unneeded
  step_rm(all_of(drop_cols))                                     %>%
  # Numeric: clip then z-score
  step_mutate_at(HeartRate_Min:Glucose_Mean, Diff,
                 fn = ~ pmin(pmax(., quantile(., .01, na.rm = TRUE)),
                             quantile(., .99, na.rm = TRUE)))    %>%
  step_normalize(HeartRate_Min:Glucose_Mean, Diff)                                %>%
  # Small categoricals: one-hot
  step_unknown(all_of(small_cat)) %>%
  step_other(all_of(small_cat), threshold = 0.01) %>%
  step_dummy(all_of(small_cat))                                    %>%
  # Large categoricals: keep as integers (factor → int)
  step_string2factor(all_of(large_cat))                            %>%
  step_integer(all_of(large_cat), zero_based = FALSE)              %>%
  # Final: put icustay_id aside so it isn’t fed to the network
  step_rm(all_of(id_cols))

prep_rec <- prep(rec)
train_processed <- bake(prep_rec, new_data = NULL)
test_processed  <- bake(prep_rec, new_data = x_test)


# Split back into parts
y   <- train_processed$HOSPITAL_EXPIRE_FLAG
x   <- train_processed %>% select(-HOSPITAL_EXPIRE_FLAG)

# Identify column positions
numeric_inds <- which(! names(x) %in% large_cat)
embed_inds   <- match(large_cat, names(x))           # integer positions

# Train-val split (80-20 stratified)
val_ix <- initial_split(tibble(y = y), prop = 0.8, strata = y)
x_train <- x[val_ix$in_id,  ]
x_val   <- x[-val_ix$in_id, ]
y_train <- y[val_ix$in_id]
y_val   <- y[-val_ix$in_id]


# ── Hyper-params you can tune ─────────────────────────────────────────
embed_dim   <- list(ICD9_diagnosis = 32,         # √(n_cat) rule of thumb
                    DIAGNOSIS       = 64)
dense_units <- c(128, 64)
dropouts    <- c(0.3, 0.2)

# ── Inputs ────────────────────────────────────────────────────────────
numeric_input <- layer_input(shape = length(numeric_inds), name = "numeric")

embed_inputs  <- map2(large_cat, embed_dim, ~
                        layer_input(shape = 1, dtype = "int32", name = .x)
)

# ── Embeddings ────────────────────────────────────────────────────────
embed_layers <- map2(embed_inputs, large_cat, ~
                       .x %>%
                       layer_embedding(
                         input_dim  = max(x[[.y]]) + 1,     # +1 because 0 is reserved for NA/UNK
                         output_dim = embed_dim[[.y]],
                         name       = paste0(.y, "_emb")
                       ) %>% layer_flatten()
)

# ── Concatenate numeric + embeddings ─────────────────────────────────
conc <- layer_concatenate(c(list(numeric_input), embed_layers))

# ── Dense trunk ──────────────────────────────────────────────────────
dense <- conc
for (i in seq_along(dense_units)) {
  dense <- dense %>%
    layer_dense(units = dense_units[i], activation = "relu") %>%
    layer_dropout(rate = dropouts[i])
}

output <- dense %>% layer_dense(units = 1, activation = "sigmoid")

model <- keras_model(
  inputs  = c(list(numeric_input), embed_inputs),
  outputs = output
)


# Class weights: inverse frequency
tbl <- table(y_train)
cw  <- list("0" = as.numeric(tbl[2]/tbl[1]),
            "1" = 1)

model %>% compile(
  optimizer = optimizer_adam(learning_rate = 1e-3),
  loss      = "binary_crossentropy",
  metrics   = metric_auc(name = "auc")
)

early_stop <- callback_early_stopping(
  monitor = "val_auc", mode = "max",
  patience = 5, restore_best_weights = TRUE
)

history <- model %>% fit(
  x = as.list(x_train), y = y_train,
  validation_data = list(as.list(x_val), y_val),
  epochs = 50, batch_size = 512,
  class_weight = cw,
  callbacks = list(early_stop),
  verbose = 2
)

