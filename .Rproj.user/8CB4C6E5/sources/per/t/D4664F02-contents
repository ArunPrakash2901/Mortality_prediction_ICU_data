# ===============================================================
#  SHAP top-driver tables with Patient ID  –  error-free version
#  Inputs already in workspace:
#      bst  ... xgboost booster
#      sv   ... shapviz object
# ===============================================================

library(xgboost)
library(shapviz)

# Load your model
bst <- readRDS("RDS/xgb_final.rds")
if (inherits(bst, "workflow")) {
  bst <- extract_fit_parsnip(bst)$fit
} else if (inherits(bst, "model_fit")) {
  bst <- bst$fit
}

# Load your processed test or train data
X_test <- readRDS("RDS/test_processed.rds") # or "train_processed.rds"
feat_names <- bst$feature_names
X_test <- X_test[, feat_names, drop = FALSE]
X_test <- as.matrix(X_test)

# Create the shapviz object (sv)
sv <- shapviz(bst, X_pred = X_test, X = X_test, exact = FALSE)

# ---- 1.  Access SHAP & feature data ---------------------------
S <- sv$S            # SHAP matrix (n × p)
X <- sv$X            # original feature data.frame
preds <- predict(bst, as.matrix(X), type = "prob")

feat_names <- colnames(S)        # model’s feature set (p cols)

# ---- 2.  Identify an ID column or create ROW_ID ---------------
id_col <- "ICUSTAY_ID"
if (!id_col %in% colnames(X)) {
  X <- X %>% mutate(ROW_ID = row_number())
  id_col <- "ROW_ID"
}

# ---- 3.  Pick exemplar indices -------------------------------
high_id <- which.max(preds)
low_id  <- which.min(preds)
mid_id  <- which.min(abs(preds - 0.50))

cases <- list(high = high_id, mid = mid_id, low = low_id)

# ---- 4.  Build tidy table -------------------------------------
build_tbl <- function(row_id, keep = 8) {
  tibble(
    Feature = feat_names,
    SHAP    = S[row_id, ]
  ) %>%
    arrange(desc(abs(SHAP))) %>%
    slice_head(n = keep) %>%
    mutate(Patient = X[[id_col]][row_id], .before = Feature)
}

# ---- 5.  Save table as PNG ------------------------------------
save_tbl_png <- function(tbl, fname) {
  tbl_round <- tbl %>% mutate(across(where(is.numeric), round, 2))
  png(fname, width = 1800, height = 500, res = 200)
  grid.table(tbl_round)
  dev.off()
}

dir.create("tables", showWarnings = FALSE)

for (nm in names(cases)) {
  tbl   <- build_tbl(cases[[nm]])
  fpath <- file.path("tables", paste0("case_", nm, "_table.png"))
  save_tbl_png(tbl, fpath)
  cat("✅  Saved", fpath, "\n")
}
