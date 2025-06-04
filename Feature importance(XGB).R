library(xgboost)
library(tidymodels)
library(dplyr)

# 1. Pull the booster from whatever wrapper you saved --------------------
get_booster <- function(obj) {
  if (inherits(obj, "workflow"))         obj %>% extract_fit_parsnip() %>% pluck("fit")
  else if (inherits(obj, "model_fit"))   pluck(obj, "fit")
  else if (inherits(obj, "xgb.Booster")) obj
  else stop("Cannot locate an xgboost booster.")
}

bst <- get_booster(readRDS("RDS/xgb_final.rds"))

# 2. Feature importance table (Gain) ------------------------------------
imp <- xgb.importance(model = bst)  # columns: Feature, Gain, Cover, Frequency

# 3. View / save the top 20 --------------------------------------------
top_imp <- imp %>% slice_head(n = 20)

print(top_imp[, c("Feature", "Gain")], row.names = FALSE)

# Optional: save to CSV if you want to paste easily
# write.csv(top_imp, "xgb_top20_importance.csv", row.names = FALSE)
