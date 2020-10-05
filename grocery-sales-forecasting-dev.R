#options(java.parameters = "-Xmx16g")
library(dplyr)
library(tidyverse)
library(data.table)
library(lubridate)
library(lightgbm)
#library(h2o)
printTimestamp <- function(stepName) {
  print(paste(rep('=', 50), collapse = ""));print(stepName);print(Sys.time());print(paste(rep('=', 50), collapse = ""))
}

convert2log <- function(x) {
  if (x<0) {0}
  else {base::log1p(x)}
}

convertFromLog <- function(x) {
  y <- base::expm1(x)
  if (y<0) {0}
  else if (y>1000) {1000}
  else {y}
}

#-------------------------- load data --------------------------
train <- fread('data/train.csv', drop = c("id"), colClasses=c(date="character",
                                              store_nbr="integer",item_nbr="integer",
                                              unit_sales="numeric",onpromotion="character"))
train$onpromotion <- if_else(train$onpromotion=='True', 1, if_else(train$onpromotion=='False', 0, -1))
store <- fread('data/stores.csv'); store$city <- as.numeric(as.factor(store$city)); store$state <- as.numeric(as.factor(store$state)); store$type <- as.numeric(as.factor(store$type))
item <- fread('data/items.csv'); item$family <- as.numeric(as.factor(item$family))
#transaction <- fread('data/transactions.csv')
#oil <- fread('data/oil.csv')
#holiday <- fread('data/holidays_events.csv')
test <- fread('data/test.csv', colClasses=c(id="integer",date="character",
                                            store_nbr="integer",item_nbr="integer",
                                            onpromotion="character"))
test$onpromotion <- if_else(test$onpromotion=='True', 1, if_else(test$onpromotion=='False', 0, -1))
test$date <- ymd(test$date)
test$unit_sales <- sapply(test$unit_sales, FUN = convert2log)
sample_sub <- fread('data/sample_submission.csv')

#---- select data period ----
dd <- train %>% filter(date>='2017-01-01'); rm(train); gc()
dd$unit_sales <- sapply(dd$unit_sales, FUN = convert2log); gc()
dd <- dd %>% complete(date, nesting(store_nbr, item_nbr), fill = list(unit_sales=0, onpromotion=0)); gc()
#dd <- dd %>% sample_frac(size = 0.1); gc() ## for debug !!!!
dg <- dd %>% left_join(item, by = 'item_nbr') %>% left_join(store, by = 'store_nbr');rm(dd); gc()
dg$date <- ymd(dg$date); gc()


#train_dates <- seq(ymd('2017-06-14'),ymd('2017-07-19'), by = '1 day'); train_dates
#valid_dates <- seq(ymd('2017-07-26'),ymd('2017-08-10'), by = '1 day'); valid_dates
test_dates <- seq(ymd('2017-08-16'),ymd('2017-08-31'), by = '1 day'); ntest_dates <- length(test_dates); test_dates

#----Helper functions ----
buildFeat<- function(ds, dh, d0, grp.cols) {
  #---- doc ----
  #time windows:
  #   nearest days: [1,3,5,7,14,30,60,140]
  #key：store x item, item, store x class
  #target: promotion, unit_sales, zeros
  #method/stats: 
  #   mean, median, max, min, std
  #   days since last appearance
  #   difference of mean value between adjacent time windows(only for equal time windows)
  #----
  for (agg_window in c(1,3,7,14,30,60,140)) {
    dhw <- dh %>% filter(date>=d0-days(agg_window))
    #1. sales all
    dhf <- dhw %>%
      group_by(.dots = grp.cols) %>% 
      arrange(date, .by_group = TRUE) %>%
      summarise(
        sales_min=round(min(unit_sales), 2),
        sales_max=round(max(unit_sales), 2),
        sales_mean=round(mean(unit_sales, na.rm = T), 2),
        sales_median=round(median(unit_sales, na.rm = T), 2),
        sales_std=round(sd(unit_sales, na.rm = T), 2),
        sales_last_date=max(date),
        sales_zero_count=sum(unit_sales==0),
        sales_mean_decay=round(sum(unit_sales*0.95**(agg_window:1)), 2),
        sales_mean_diff=round(sum(diff(unit_sales), na.rm = T), 2),
        #sales_mean_dow=round(, 2),
        onpromo_count=sum(onpromotion==1),
        onpromo_mean=round(mean(onpromotion==1), 2)
      )
    if (length(grp.cols)==2) {colnames(dhf) <- c(grp.cols, paste(paste(grp.cols, collapse = "_"), colnames(dhf)[-1:-2], agg_window, sep = "_")); datecol <- colnames(dhf)[8]}
    else if (length(grp.cols)==1) {colnames(dhf) <- c(grp.cols, paste(grp.cols, colnames(dhf)[-1], agg_window, sep = '_')); datecol <- colnames(dhf)[7]}
    ds <- ds %>% left_join(dhf, by = grp.cols); gc()
    ds[, datecol] <- as.integer(ds$date - ds%>%pull(datecol))
    
    #2. sales only on promo
    dhf <- dhw %>% 
      filter(onpromotion==1) %>%
      group_by(.dots = grp.cols) %>% 
      summarise(
        #sales_onpromo_min=round(min(unit_sales), 2),
        #sales_onpromo_max=round(max(unit_sales), 2),
        sales_onpromo_mean=round(mean(unit_sales, na.rm = T), 2),
        #sales_onpromo_median=round(median(unit_sales, na.rm = T), 2),
        #sales_onpromo_std=round(sd(unit_sales, na.rm = T), 2),
        sales_onpromo_last_date=max(date)
      )
    if (length(grp.cols)==2) {colnames(dhf) <- c(grp.cols, paste(paste(grp.cols, collapse = "_"), colnames(dhf)[-1:-2], agg_window, sep = "_")); datecol <- colnames(dhf)[4]}
    else if (length(grp.cols)==1) {colnames(dhf) <- c(grp.cols, paste(grp.cols, colnames(dhf)[-1], agg_window, sep = '_')); datecol <- colnames(dhf)[3]}
    ds <- ds %>% left_join(dhf, by = grp.cols); gc()
    ds[, datecol] <- as.integer(ds$date - ds%>%pull(datecol))
    
    rm(dhw); gc()
  }
  ds
}

#----step 1: Build 6 partitions of feature matrix, and bind rows (train features) -16 days x 6 ----
train.dates <- seq(ymd('2017-06-14'),ymd('2017-07-19'), by = '7 day'); train.dates
for (date.idx in 1:length(train.dates)) {
  date.partition <- train.dates[date.idx]
  print(date.partition)
  #----
  dtr <- dg %>% filter(date>=date.partition, date<date.partition+days(16))
  dh <- dg %>% filter(date<date.partition)
  grp.cols <- c('store_nbr', 'item_nbr'); dtr <- buildFeat(dtr, dh, date.partition, grp.cols)
  grp.cols <- c('item_nbr'); dtr <- buildFeat(dtr, dh, date.partition, grp.cols)
  grp.cols <- c('store_nbr', 'class'); dtr <- buildFeat(dtr, dh, date.partition, grp.cols)
  dtr$date.lag <- as.integer(dtr$date-(date.partition-days(1)))
  if (date.idx==1) {dtr.all <- dtr} else {dtr.all <- bind_rows(dtr.all, dtr)}
  rm(dtr, dh); gc()
  #----
}
#----

#----step 2: Build 1 validset feature matrix -16 days x 1 ----
dval <- dg %>% filter(date>=ymd('2017-07-26'), date<=ymd('2017-08-10'))
dh <- dg %>% filter(date<ymd('2017-07-26'))
d0 <- ymd('2017-07-26')
grp.cols <- c('store_nbr', 'item_nbr'); dval <- buildFeat(dval, dh, d0, grp.cols)
grp.cols <- c('item_nbr'); dval <- buildFeat(dval, dh, d0, grp.cols)
grp.cols <- c('store_nbr', 'class'); dval <- buildFeat(dval, dh, d0, grp.cols)
dval$date.lag <- as.integer(dval$date-(d0-days(1)))
rm(dh); gc()
#----

#----step 3: Build testset feature matrix ----
dt <- test %>% left_join(item, by = 'item_nbr') %>% left_join(store, by = 'store_nbr')
dh <- dg %>% filter(date<=ymd('2017-08-15'))
d0 <- ymd('2017-08-16')
grp.cols <- c('store_nbr', 'item_nbr'); dt <- buildFeat(dt, dh, d0, grp.cols)
grp.cols <- c('item_nbr'); dt <- buildFeat(dt, dh, d0, grp.cols)
grp.cols <- c('store_nbr', 'class'); dt <- buildFeat(dt, dh, d0, grp.cols)
dt$date.lag <- as.integer(dt$date-(d0-days(1)))
rm(dh); gc()
#----

#---- Train 16 separate lgbs for each test date ----
for (d in 1:ntest_dates) {
  #if (d!=1) {next}
  dtest <- test_dates[d]
  print(dtest)

  dtr.day <- dtr.all %>% filter(date.lag==d) %>% select(-date.lag)
  dval.day <- dval %>% filter(date.lag==d) %>% select(-date.lag)
  
  #---- h2o model training ----
  # h2o.init(ip = "127.0.0.1", port = 54321, max_mem_size = "128G", nthreads = 1)#enable_assertions = FALSE
  # #col data types
  # all_cols <- colnames(dtr); remove_cols <- c('store_nbr', 'item_nbr', 'date'); all_cols <- setdiff(all_cols, remove_cols)
  # pred_cols <- all_cols[-1]; response_col <- "unit_sales"
  # h2o_train <- dtr %>% select(all_of(all_cols)) %>% as.h2o(); h2o_valid <- dval %>% select(all_of(all_cols)) %>% as.h2o()
  # categorical_features <- c("onpromotion", "family", "class", "perishable", "city", "state", "type", "cluster");print(categorical_features)
  # for (name in categorical_features) {h2o_train[, name] <- as.factor(h2o_train[, name]); h2o_valid[, name] <- as.factor(h2o_valid[, name])}
  # 
  # printTimestamp('Start AutoML training...')
  # sales_model <- h2o.automl(x = pred_cols, y = response_col, 
  #                           training_frame = h2o_train, validation_frame = h2o_valid, 
  #                           nfolds = 0, max_models = 8, max_runtime_secs = 600,
  #                           stopping_metric = "RMSE", sort_metric = "RMSE", 
  #                           exploitation_ratio = 0.1, include_algos = c("XGBoost"))
  # printTimestamp('AutoML training completed')
  # 
  # lb <- h2o.get_leaderboard(sales_model)
  # head(lb)
  # best_model <- sales_model@leader
  # View(best_model@model$variable_importances)
  # perf_train <- h2o.performance(best_model, train = TRUE); perf_valid <- h2o.performance(best_model, valid = TRUE)
  # print(perf_train);print(perf_valid)
  # h2o.saveModel(object = best_model, path = "model/autoMLv1")
  # rm(h2o_train, h2o_valid); gc()
  # rm(dtr, dval); gc()
  #----
  
  #---- LightGBM model training ----
  #col data types
  all_cols <- colnames(dtr.day); remove_cols <- c('store_nbr', 'item_nbr', 'date'); all_cols <- setdiff(all_cols, remove_cols)
  pred_cols <- all_cols[-1]; response_col <- "unit_sales"
  categorical_features <- c("onpromotion", "family", "class", "perishable", "city", "state", "type", "cluster");print(categorical_features)
  lgb.dtr <- lgb.Dataset(data = as.matrix(dtr.day %>% select(all_of(pred_cols))), label=dtr.day%>%pull(response_col), categorical_feature=categorical_features, weight = dtr.day$perishable*0.25+1)
  lgb.dval <- lgb.Dataset(data = as.matrix(dval.day %>% select(all_of(pred_cols))), label=dval.day%>%pull(response_col), categorical_feature=categorical_features, weight = dval.day$perishable*0.25+1)
  
  printTimestamp('Start LightGBM training...')
  params <- list(objective='regression',
                 metric='l2',
                 boosting='gbdt',
                 #max_bin=255,
                 #max_depth=9,
                 num_leaves=2**8,
                 min_data_in_leaf=2**8-1,
                 bagging_fraction=0.7,
                 bagging_freq=1,
                 feature_fraction=0.7,
                 seed=42)
  sales_model <- lgb.train(params, 
                           lgb.dtr, valids = list(train=lgb.dtr,valid=lgb.dval),
                           nrounds=2000, learning_rate=0.02, early_stopping_rounds=100,
                           num_threads=8, verbose = 1, eval_freq=10)
  printTimestamp('LightGBM training completed')
  
  #tree_imp <- lgb.importance(sales_model, percentage = TRUE)
  #lgb.plot.importance(tree_imp, top_n = 20L, measure = "Gain")
  lgb.save(sales_model, paste("model/v1-date-lag-", d, '.txt', sep = ''))
  
  rm(lgb.dtr, lgb.dval); gc()
  rm(dtr.day, dval.day); gc()
  #----
  
  #---- testset ----
  dt.day <- dt %>% filter(date.lag==d) %>% select(-date.lag)
  #----
  
  #---- h2o predict ----
  # h2o.init(ip = "127.0.0.1", port = 54321)#enable_assertions = FALSE
  # h2o_test <- dt %>% select(pred_cols) %>% as.h2o()
  # for (name in categorical_features) {h2o_test[, name] <- as.factor(h2o_test[, name])}
  # preds <- h2o.predict(sales_model, h2o_test); preds <- as.data.frame(preds); preds <- preds$predict
  # preds <- sapply(preds, FUN = convertFromLog)
  # print(quantile(preds, probs = c(0,0.25,0.5,0.75,1)))
  #----
  
  #---- LightGBM predict ----
  lgb.dt <- as.matrix(dt.day %>% select(all_of(pred_cols)))
  preds <- predict(sales_model, lgb.dt)
  preds <- sapply(preds, FUN = convertFromLog)
  print(quantile(preds, probs = c(0,0.25,0.5,0.75,1)))
  
  sub <- dt.day %>% select(id) %>% mutate(unit_sales=preds)
  sub$unit_sales <- round(sub$unit_sales, 1)
  
  if (d==1) {sub.all <- sub} else {sub.all <- bind_rows(sub.all, sub)}
  rm(dt.day, lgb.dt); gc()
  #----
}
#----

#----
#analysis
# grp <- train %>% filter(date>='2017-08-01') %>% group_by(store_nbr, item_nbr) %>% summarise(median_sales=median(unit_sales, na.rm = T))
# sub <- test %>% left_join(grp, by = c('store_nbr', 'item_nbr')) %>% mutate(unit_sales=round(median_sales, 2)) %>% select(id, unit_sales)
# sub[is.na(sub$unit_sales), 'unit_sales'] <- 0
# sub[sub$unit_sales<0, 'unit_sales'] <- 0; sub[sub$unit_sales>1000, 'unit_sales'] <- 1000
# write.csv(sub, file = "submission/1004-test.csv", row.names = F)
#----

#-------------------------- submit --------------------------
write.csv(sub.all, file = "submission/1005-01-perish-new-feat.csv", row.names = F)

grp <- dg %>% group_by(store_nbr, item_nbr) %>% summarise(exist=1)
sub.all2 <- test %>% select(-unit_sales) %>% left_join(sub.all, by = 'id') %>% left_join(grp, by = c('store_nbr', 'item_nbr'))
sub.all2[is.na(sub.all2$exist), 'unit_sales'] <- 0
sub.all2 <- sub.all2 %>% select(id, unit_sales)
write.csv(sub.all2, file = "submission/1005-02-perish-new-feat.csv", row.names = F)

#-------------------------- End --------------------------