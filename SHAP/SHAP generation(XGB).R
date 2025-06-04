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
  # Force-register if shapviz forgot
  utils::registerS3method("shapviz", "xgb.Booster",
                          shapviz:::shapviz.xgb.Booster,
                          envir = asNamespace("shapviz"))
}

# ---- 3.  Load model and extract booster -------------------------------
wf_or_fit <- readRDS("RDS/xgb_final.rds")
bst <- if (inherits(wf_or_fit, "workflow")) {
  extract_fit_parsnip(wf_or_fit)$fit
} else if (inherits(wf_or_fit, "model_fit")) {
  wf_or_fit$fit
} else if (inherits(wf_or_fit, "xgb.Booster")) {
  wf_or_fit
} else stop("xgb_final.rds isn’t a workflow, model_fit, or xgb.Booster")

# ---- 4.  Build design matrix in booster’s feature order ---------------
X_train <- readRDS("RDS/train_processed.rds") |>
  dplyr::select(-HOSPITAL_EXPIRE_FLAG) |>
  as.matrix()

feat_names <- bst$feature_names
X_train    <- X_train[, feat_names, drop = FALSE]   # reorder & drop extras

# ---- 5.  Construct SHAP object (works for shapviz ≥0.9) ---------------
sv <- shapviz(bst, X_pred = X_train, X = X_train, exact = FALSE)

# ---- 6.  Global beeswarm (top 20) -------------------------------------
p_bee <- sv_importance(sv, kind = "beeswarm", top_n = 20)  # beeswarm plot
ggsave("shap_beeswarm.png", plot = p_bee,
       width = 6, height = 4, dpi = 300)

# ---- 7.  Waterfall plots: highest- vs lowest-risk patient -------------
preds   <- predict(bst, X_train, type = "prob")

high_id <- which.max(preds)
low_id  <- which.min(preds)

p_high <- sv_waterfall(sv, high_id, max_display = 12)
p_low  <- sv_waterfall(sv, low_id,  max_display = 12)

ggsave("shap_high.png", plot = p_high,
       width = 6, height = 4, dpi = 300)
ggsave("shap_low.png",  plot = p_low,
       width = 6, height = 4, dpi = 300)

cat("\n✅  SHAP visualisations saved:\n",
    "  shap_beeswarm.png\n  shap_high.png\n  shap_low.png\n",
    "Embed these PNGs into your Quarto slides (Slide 2 & 3).\n")
