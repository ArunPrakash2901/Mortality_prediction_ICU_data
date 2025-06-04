# =======================================================================
#  ETC5250  –  Generate SHAP visualisations for final XGBoost model
#  Outputs: shap_beeswarm.png | shap_high.png | shap_low.png  (in working dir)
# =======================================================================

# ---- 0.  Install/upgrade shapviz (once) -------------------------------
if (!requireNamespace("shapviz", quietly = TRUE) ||
    packageVersion("shapviz") < "0.9.7") {
  install.packages("shapviz")          # CRAN latest
}

# ---- 1.  Attach packages in safe order --------------------------------
library(xgboost)      # load first so S3 class exists
library(shapviz)      # registers shapviz.xgb.Booster (≥0.9.7)
library(tidymodels)
library(data.table)
library(ggplot2)

# ---- 2.  Check & (if needed) register the S3 method -------------------
if (is.null(getS3method("shapviz", "xgb.Booster", optional = TRUE))) {
  utils::registerS3method("shapviz", "xgb.Booster",
                          shapviz:::shapviz.xgb.Booster,
                          envir = asNamespace("shapviz"))
}

# --- 2. Load model and processed test data
bst <- readRDS("RDS/xgb_final.rds")
if (inherits(bst, "workflow")) {
  bst <- extract_fit_parsnip(bst)$fit
} else if (inherits(bst, "model_fit")) {
  bst <- bst$fit
}

X_test <- readRDS("RDS/test_processed.rds") %>% as.matrix()
# Ensure same feature order as in the booster
feat_names <- bst$feature_names
X_test <- X_test[, feat_names, drop = FALSE]

# --- 3. Compute SHAP values on test set
sv_test <- shapviz(bst, X_pred = X_test, X = X_test, exact = FALSE)

# --- 4. Model predictions (probability of positive class)
preds_test <- predict(bst, X_test, type = "prob")

# If predict(type="prob") returns a matrix/dataframe, use the correct column
# For xgboost, usually this is just a vector, but if not, use: preds_test[, "1"]

# --- 5. Identify high/low risk patients (row index)
high_id_test <- which.max(preds_test)
low_id_test  <- which.min(preds_test)

# --- 6. Beeswarm plot (global SHAP on test set)
p_bee_test <- sv_importance(sv_test, kind = "beeswarm", top_n = 20)
ggsave("shap_beeswarm_test.png", plot = p_bee_test, width = 6, height = 4, dpi = 300)

# --- 7. Waterfall plots for individual patients (test set)
p_high_test <- sv_waterfall(sv_test, high_id_test, max_display = 12)
p_low_test  <- sv_waterfall(sv_test, low_id_test, max_display = 12)
ggsave("shap_high_test.png", plot = p_high_test, width = 6, height = 4, dpi = 300)
ggsave("shap_low_test.png",  plot = p_low_test, width = 6, height = 4, dpi = 300)

cat("\n✅  SHAP visualisations for TEST set saved as:\n",
    "  shap_beeswarm_test.png\n  shap_high_test.png\n  shap_low_test.png\n")
