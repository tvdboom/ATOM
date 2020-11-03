# Early stopping
---------------------------------

This example shows how we can use early stopping to reduce the time it takes to run the pipeline. This option is only available for models that allow in-training evaluation (XGBoost, LightGBM and CatBoost).

Import the breast cancer dataset from [sklearn.datasets](https://scikit-learn.org/stable/datasets/index.html#wine-dataset). This is a small and easy to train dataset whose goal is to predict whether a patient has breast cancer or not.

## Load the data


```python
# Import packages
from sklearn.datasets import load_breast_cancer
from atom import ATOMClassifier
```


```python
# Get the dataset's features and targets
X, y = load_breast_cancer(return_X_y=True)
```

## Run the pipeline


```python
# Start ATOM and fit the models using early stopping
# An early stopping of 0.1 means that the model will stop if it
# didn't improve in the last 10% of it's iterations.
atom = ATOMClassifier(X, y, n_jobs=2, verbose=2, warnings=False, random_state=1)
atom.run('LGB', metric='ap', n_calls=7, n_initial_points=3, bo_params={'early_stopping': 0.1, 'cv': 1})
```

    << ================== ATOM ================== >>
    Algorithm task: binary classification.
    Parallel processing with 2 cores.
    
    Dataset stats ================== >>
    Shape: (569, 31)
    Scaled: False
    -----------------------------------
    Train set size: 456
    Test set size: 113
    -----------------------------------
    Train set balance: 0:1 <==> 1.0:1.7
    Test set balance: 0:1 <==> 1.0:1.5
    -----------------------------------
    Distribution of classes:
    |    |   dataset |   train |   test |
    |---:|----------:|--------:|-------:|
    |  0 |       212 |     167 |     45 |
    |  1 |       357 |     289 |     68 |
    
    
    Training ===================================== >>
    Models: LGB
    Metric: average_precision
    
    
    Running BO for LightGBM...
    Initial point 1 ---------------------------------
    Parameters --> {'n_estimators': 499, 'learning_rate': 0.73, 'max_depth': 1, 'num_leaves': 40, 'min_child_weight': 5, 'min_child_samples': 18, 'subsample': 0.7, 'colsample_bytree': 0.8, 'reg_alpha': 100.0, 'reg_lambda': 10.0}
    Early stop at iteration 50 of 499.
    Evaluation --> average_precision: 0.6304  Best average_precision: 0.6304
    Time iteration: 0.024s   Total time: 0.045s
    Initial point 2 ---------------------------------
    Parameters --> {'n_estimators': 170, 'learning_rate': 0.11, 'max_depth': 4, 'num_leaves': 25, 'min_child_weight': 11, 'min_child_samples': 28, 'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 100.0, 'reg_lambda': 10.0}
    Early stop at iteration 18 of 170.
    Evaluation --> average_precision: 0.6304  Best average_precision: 0.6304
    Time iteration: 0.020s   Total time: 0.069s
    Initial point 3 ---------------------------------
    Parameters --> {'n_estimators': 364, 'learning_rate': 0.4, 'max_depth': 1, 'num_leaves': 30, 'min_child_weight': 17, 'min_child_samples': 27, 'subsample': 0.9, 'colsample_bytree': 0.5, 'reg_alpha': 0.0, 'reg_lambda': 1.0}
    Early stop at iteration 42 of 364.
    Evaluation --> average_precision: 0.9774  Best average_precision: 0.9774
    Time iteration: 0.020s   Total time: 0.094s
    Iteration 4 -------------------------------------
    Parameters --> {'n_estimators': 238, 'learning_rate': 0.49, 'max_depth': 2, 'num_leaves': 29, 'min_child_weight': 18, 'min_child_samples': 25, 'subsample': 0.9, 'colsample_bytree': 0.4, 'reg_alpha': 0.0, 'reg_lambda': 10.0}
    Early stop at iteration 30 of 238.
    Evaluation --> average_precision: 0.9911  Best average_precision: 0.9911
    Time iteration: 0.021s   Total time: 1.420s
    Iteration 5 -------------------------------------
    Parameters --> {'n_estimators': 31, 'learning_rate': 0.07, 'max_depth': 5, 'num_leaves': 21, 'min_child_weight': 18, 'min_child_samples': 28, 'subsample': 0.8, 'colsample_bytree': 0.5, 'reg_alpha': 0.0, 'reg_lambda': 100.0}
    Evaluation --> average_precision: 0.9920  Best average_precision: 0.9920
    Time iteration: 0.021s   Total time: 1.785s
    Iteration 6 -------------------------------------
    Parameters --> {'n_estimators': 42, 'learning_rate': 0.55, 'max_depth': 3, 'num_leaves': 39, 'min_child_weight': 11, 'min_child_samples': 12, 'subsample': 0.8, 'colsample_bytree': 0.4, 'reg_alpha': 0.01, 'reg_lambda': 100.0}
    Evaluation --> average_precision: 0.9991  Best average_precision: 0.9991
    Time iteration: 0.023s   Total time: 2.158s
    Iteration 7 -------------------------------------
    Parameters --> {'n_estimators': 109, 'learning_rate': 1.0, 'max_depth': -1, 'num_leaves': 40, 'min_child_weight': 1, 'min_child_samples': 10, 'subsample': 0.8, 'colsample_bytree': 0.3, 'reg_alpha': 100.0, 'reg_lambda': 100.0}
    Early stop at iteration 11 of 109.
    Evaluation --> average_precision: 0.6304  Best average_precision: 0.9991
    Time iteration: 0.020s   Total time: 2.628s
    
    Results for LightGBM:         
    Bayesian Optimization ---------------------------
    Best parameters --> {'n_estimators': 42, 'learning_rate': 0.55, 'max_depth': 3, 'num_leaves': 39, 'min_child_weight': 11, 'min_child_samples': 12, 'subsample': 0.8, 'colsample_bytree': 0.4, 'reg_alpha': 0.01, 'reg_lambda': 100.0}
    Best evaluation --> average_precision: 0.9991
    Time elapsed: 3.118s
    Fit ---------------------------------------------
    Train evaluation --> average_precision: 0.9975
    Test evaluation --> average_precision: 0.9885
    Time elapsed: 0.026s
    -------------------------------------------------
    Total time: 3.147s
    
    
    Final results ========================= >>
    Duration: 3.149s
    ------------------------------------------
    LightGBM --> average_precision: 0.988
    

## Analyze the results


```python
# For these models, we can plot the evaluation on the train and test set during training
# Note that the metric is provided by the estimator's package, not ATOM!
atom.lgb.plot_evals(title="LightGBM's evaluation curve", figsize=(11, 9))
```


![png](output_7_0.png)

