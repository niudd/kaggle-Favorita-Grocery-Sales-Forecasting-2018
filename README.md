# kaggle-Favorita-Grocery-Sales-Forecasting-2018

kaggle link: [favorita-grocery-sales-forecasting](https://www.kaggle.com/c/favorita-grocery-sales-forecasting/overview)

[first place python code as reference](https://www.kaggle.com/shixw125/1st-place-lgb-model-public-0-506-private-0-511)

[the writeup of above 1st place solution by winner team](https://www.kaggle.com/c/favorita-grocery-sales-forecasting/discussion/47582)

[this reference has run on kaggle and training logs](https://www.kaggle.com/vrtjso/lgbm-one-step-ahead)



### This can be an R template for time series forecasting task in terms of

1. to forecast 16 days (i.e. steps) ahead
   - trained 16 separate lgb models for each day ahead, i.e. 1d ahead, 2d ahead, ..., 16d ahead
   - build 6 times feature matrix (by 6 independant days), and combine together as training set, while use 1 single feature matrix (on a single day) for validation. Note the testset is also on a single day

2. feature engineering key take-aways
   - a correct time-lagged feature should be that time should be aligned among all sample points
   - use time period window for aggregation, don't do last time sales, latest three times sales, ... because time are not aligned among samples
   - the task is to predict every possible (or even not seen before) combination of store x item sales for each of 16 days ahead. obviously lots of combinations did not exist for some of the days so that we will fill those sales with 0, this is important because otherwise the model will be trained on all non zeros dataset and after that being applied to predict a 80% nonzeros testset.