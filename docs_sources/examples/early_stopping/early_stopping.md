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
# didn"t improve in the last 10% of it's iterations.
atom = ATOMClassifier(X, y, n_jobs=2, verbose=2, warnings=False, random_state=1)
atom.run("LGB", metric="ap", n_calls=7, n_initial_points=3, bo_params={"early_stopping": 0.1, "cv": 1})
```

    << ================== ATOM ================== >>
    Algorithm task: binary classification.
    Parallel processing with 2 cores.
    Applying data cleaning...
    
    Dataset stats ================= >>
    Shape: (569, 31)
    Scaled: False
    ----------------------------------
    Train set size: 456
    Test set size: 113
    ----------------------------------
    Train set balance: 0:1 <==> 0.6:1.0
    Test set balance: 0:1 <==> 0.7:1.0
    ----------------------------------
    Instances in target per class:
    |    |    total |    train_set |    test_set |
    |---:|---------:|-------------:|------------:|
    |  0 |      212 |          167 |          45 |
    |  1 |      357 |          289 |          68 |
    
    
    Running pipeline ============================= >>
    Models in pipeline: LGB
    Metric: average_precision
    
    
    Running BO for LightGBM...
    Random start 1 ----------------------------------
    Parameters --> {"n_estimators": 499, "learning_rate": 0.73, "max_depth": 2, "num_leaves": 40, "min_child_weight": 5, "min_child_samples": 18, "subsample": 0.7, "colsample_bytree": 0.8, "reg_alpha": 100.0, "reg_lambda": 10.0}
    Early stop at iteration 50 of 499.
    Evaluation --> average_precision: 0.6304  Best average_precision: 0.6304
    Time iteration: 0.031s   Total time: 0.047s
    Random start 2 ----------------------------------
    Parameters --> {"n_estimators": 170, "learning_rate": 0.11, "max_depth": 5, "num_leaves": 25, "min_child_weight": 11, "min_child_samples": 28, "subsample": 0.7, "colsample_bytree": 0.6, "reg_alpha": 100.0, "reg_lambda": 10.0}
    Early stop at iteration 18 of 170.
    Evaluation --> average_precision: 0.6304  Best average_precision: 0.6304
    Time iteration: 0.028s   Total time: 0.075s
    Random start 3 ----------------------------------
    Parameters --> {"n_estimators": 364, "learning_rate": 0.4, "max_depth": 2, "num_leaves": 30, "min_child_weight": 17, "min_child_samples": 27, "subsample": 0.9, "colsample_bytree": 0.5, "reg_alpha": 0.0, "reg_lambda": 1.0}
    Early stop at iteration 42 of 364.
    Evaluation --> average_precision: 0.9819  Best average_precision: 0.9819
    Time iteration: 0.020s   Total time: 0.099s
    Iteration 4 -------------------------------------
    Parameters --> {"n_estimators": 238, "learning_rate": 0.49, "max_depth": 3, "num_leaves": 29, "min_child_weight": 18, "min_child_samples": 25, "subsample": 0.9, "colsample_bytree": 0.4, "reg_alpha": 0.0, "reg_lambda": 10.0}
    Early stop at iteration 30 of 238.
    Evaluation --> average_precision: 0.9911  Best average_precision: 0.9911
    Time iteration: 0.016s   Total time: 1.343s
    Iteration 5 -------------------------------------
    Parameters --> {"n_estimators": 31, "learning_rate": 0.07, "max_depth": 6, "num_leaves": 21, "min_child_weight": 18, "min_child_samples": 28, "subsample": 0.9, "colsample_bytree": 0.5, "reg_alpha": 0.0, "reg_lambda": 100.0}
    Evaluation --> average_precision: 0.9920  Best average_precision: 0.9920
    Time iteration: 0.016s   Total time: 1.762s
    Iteration 6 -------------------------------------
    Parameters --> {"n_estimators": 20, "learning_rate": 1.0, "max_depth": 3, "num_leaves": 40, "min_child_weight": 20, "min_child_samples": 10, "subsample": 0.8, "colsample_bytree": 0.3, "reg_alpha": 0.0, "reg_lambda": 100.0}
    Early stop at iteration 12 of 20.
    Evaluation --> average_precision: 0.9953  Best average_precision: 0.9953
    Time iteration: 0.016s   Total time: 2.178s
    Iteration 7 -------------------------------------
    Parameters --> {"n_estimators": 69, "learning_rate": 0.17, "max_depth": 7, "num_leaves": 26, "min_child_weight": 17, "min_child_samples": 14, "subsample": 0.5, "colsample_bytree": 0.8, "reg_alpha": 0.01, "reg_lambda": 1.0}
    Early stop at iteration 22 of 69.
    Evaluation --> average_precision: 0.9978  Best average_precision: 0.9978
    Time iteration: 0.016s   Total time: 2.499s
    
    Results for LightGBM:         
    Bayesian Optimization ---------------------------
    Best parameters --> {"n_estimators": 69, "learning_rate": 0.17, "max_depth": 7, "num_leaves": 26, "min_child_weight": 17, "min_child_samples": 14, "subsample": 0.5, "colsample_bytree": 0.8, "reg_alpha": 0.01, "reg_lambda": 1.0}
    Best evaluation --> average_precision: 0.9978
    Time elapsed: 2.912s
    Fitting -----------------------------------------
    Early stop at iteration 27 of 69.
    Score on the train set --> average_precision: 0.9962
    Score on the test set  --> average_precision: 0.9712
    Time elapsed: 0.016s
    -------------------------------------------------
    Total time: 2.928s
    
    
    Final results ========================= >>
    Duration: 2.928s
    ------------------------------------------
    LightGBM --> average_precision: 0.971
    

## Analyze the results


```python
# For these models, we can plot the evaluation on the train and test set during training
# Note that the metric is provided by the estimator's package, not ATOM!
atom.lgb.plot_evals(title="LightGBM's evaluation curve", figsize=(11, 9))
```


![png](output_7_0.png)

