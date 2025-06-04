library(bonsai)
library(glue)

# Define LightGBM spec using mtry
lgb_spec <- boost_tree(
  trees      = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  mtry       = tune(),
  min_n      = tune()
) %>%
  set_engine("lightgbm",
             objective        = "binary",
             metric           = "auc",
             scale_pos_weight = !!spw_val,
             num_threads      = !!core_cnt) %>%
  set_mode("classification")

wf_lgb <- workflow() %>% add_recipe(rec_final) %>% add_model(lgb_spec)

# Compute predictor count after recipe
processed_train <- juice(prepped) %>% select(-all_of(id_vars), -HOSPITAL_EXPIRE_FLAG)
n_feats <- ncol(processed_train)

# Create tuning grid
grid_lgb <- grid_space_filling(
  trees(c(500, 2500)),
  tree_depth(c(3, 10)),
  learn_rate(range = c(-4, -1)),
  finalize(mtry(), processed_train),
  min_n(c(5, 50)),
  size = 20
)

# Tune and print CV AUC
lgb_res <- tune_grid(
  wf_lgb, resamples = cv5, grid = grid_lgb,
  metrics = metric_set(roc_auc), control = ctrl_grid
)
print_auc(lgb_res, "LightGBM")

# Finalize and fit
final_lgb <- finalize_workflow(wf_lgb, select_best(lgb_res, metric = "roc_auc")) %>%
  workflows::fit(train)

# Predict and save
prob_lgb <- predict(final_lgb, new_data = test_X, type = "prob")$.pred_1
write_csv(
  tibble(ID = test_X$icustay_id,
         HOSPITAL_EXPIRE_FLAG = prob_lgb),
  "submission_lgb.csv")

print_auc <- function(res_obj, name){
  auc <- collect_metrics(res_obj) %>%
    filter(.metric == "roc_auc") %>%
    slice_max(order_by = mean, n = 1) %>%
    pull(mean)
  cat(glue("✔ {name} CV roc_auc = {round(auc, 4)}
"))
}
