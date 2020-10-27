# Multiclass classification
---------------------------------

This example shows how to compare the performance of three models on a multiclass classification task.

Import the wine dataset from [sklearn.datasets](https://scikit-learn.org/stable/datasets/index.html#breast-cancer-wisconsin-diagnostic-dataset). This is a small and easy to train dataset whose goal is to predict wines into three groups (which cultivator it's from) using features based on the results of chemical analysis.

## Load the data


```python
# Import packages
from sklearn.datasets import load_wine
from atom import ATOMClassifier
```


```python
# Load the dataset's features and targets
X, y = load_wine(return_X_y=True, as_frame=True)

# Let's have a look at a subsample of the data
X.sample(frac=1).iloc[:5, :8]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>alcohol</th>
      <th>malic_acid</th>
      <th>ash</th>
      <th>alcalinity_of_ash</th>
      <th>magnesium</th>
      <th>total_phenols</th>
      <th>flavanoids</th>
      <th>nonflavanoid_phenols</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>57</th>
      <td>13.29</td>
      <td>1.97</td>
      <td>2.68</td>
      <td>16.8</td>
      <td>102.0</td>
      <td>3.00</td>
      <td>3.23</td>
      <td>0.31</td>
    </tr>
    <tr>
      <th>65</th>
      <td>12.37</td>
      <td>1.21</td>
      <td>2.56</td>
      <td>18.1</td>
      <td>98.0</td>
      <td>2.42</td>
      <td>2.65</td>
      <td>0.37</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14.38</td>
      <td>1.87</td>
      <td>2.38</td>
      <td>12.0</td>
      <td>102.0</td>
      <td>3.30</td>
      <td>3.64</td>
      <td>0.29</td>
    </tr>
    <tr>
      <th>102</th>
      <td>12.34</td>
      <td>2.45</td>
      <td>2.46</td>
      <td>21.0</td>
      <td>98.0</td>
      <td>2.56</td>
      <td>2.11</td>
      <td>0.34</td>
    </tr>
    <tr>
      <th>89</th>
      <td>12.08</td>
      <td>1.33</td>
      <td>2.30</td>
      <td>23.6</td>
      <td>70.0</td>
      <td>2.20</td>
      <td>1.59</td>
      <td>0.42</td>
    </tr>
  </tbody>
</table>
</div>



## Run the pipeline


```python
atom = ATOMClassifier(X, y, n_jobs=-1, warnings="ignore", verbose=2, random_state=1)

# Fit the pipeline with the selected models
atom.run(
    models=["LR","LDA", "RF"],
    metric="roc_auc_ovr",
    n_calls=4,
    n_initial_points=3,
    bo_params={"base_estimator": "rf", "max_time": 100},
    bagging=5
)
```

    << ================== ATOM ================== >>
    Algorithm task: multiclass classification.
    Parallel processing with 16 cores.
    Applying data cleaning...
    
    Dataset stats ================= >>
    Shape: (178, 14)
    Scaled: False
    ----------------------------------
    Train set size: 143
    Test set size: 35
    ----------------------------------
    Train set balance: 0:1:2 <==> 1.4:1.7:1.0
    Test set balance: 0:1:2 <==> 0.7:1.0:1.0
    ----------------------------------
    Instances in target per class:
    |    |    total |    train_set |    test_set |
    |---:|---------:|-------------:|------------:|
    |  0 |       59 |           50 |           9 |
    |  1 |       71 |           58 |          13 |
    |  2 |       48 |           35 |          13 |
    
    
    Running pipeline ============================= >>
    Models in pipeline: LR, LDA, RF
    Metric: roc_auc_ovr
    
    
    Running BO for Logistic Regression...
    Initial point 1 ---------------------------------
    Parameters --> {"penalty": "l2", "C": 46.003, "solver": "lbfgs", "max_iter": 745}
    Evaluation --> roc_auc_ovr: 1.0000  Best roc_auc_ovr: 1.0000
    Time iteration: 3.672s   Total time: 3.676s
    Initial point 2 ---------------------------------
    Parameters --> {"penalty": "none", "solver": "newton-cg", "max_iter": 490}
    Evaluation --> roc_auc_ovr: 1.0000  Best roc_auc_ovr: 1.0000
    Time iteration: 3.177s   Total time: 6.859s
    Initial point 3 ---------------------------------
    Parameters --> {"penalty": "l2", "C": 0.037, "solver": "liblinear", "max_iter": 352}
    Evaluation --> roc_auc_ovr: 0.9993  Best roc_auc_ovr: 1.0000
    Time iteration: 3.195s   Total time: 10.059s
    Iteration 4 -------------------------------------
    Parameters --> {"penalty": "none", "solver": "newton-cg", "max_iter": 378}
    Evaluation --> roc_auc_ovr: 1.0000  Best roc_auc_ovr: 1.0000
    Time iteration: 2.641s   Total time: 12.912s
    
    Results for Logistic Regression:         
    Bayesian Optimization ---------------------------
    Best parameters --> {"penalty": "l2", "C": 46.003, "solver": "lbfgs", "max_iter": 745}
    Best evaluation --> roc_auc_ovr: 1.0000
    Time elapsed: 13.115s
    Fit ---------------------------------------------
    Score on the train set --> roc_auc_ovr: 1.0000
    Score on the test set  --> roc_auc_ovr: 0.9965
    Time elapsed: 0.024s
    Bagging -----------------------------------------
    Score --> roc_auc_ovr: 0.9942 ± 0.0026
    Time elapsed: 0.084s
    -------------------------------------------------
    Total time: 13.229s
    
    
    Running BO for Linear Discriminant Analysis...
    Initial point 1 ---------------------------------
    Parameters --> {"solver": "eigen", "shrinkage": 1.0}
    Evaluation --> roc_auc_ovr: 0.8975  Best roc_auc_ovr: 0.8975
    Time iteration: 0.040s   Total time: 0.042s
    Initial point 2 ---------------------------------
    Parameters --> {"solver": "svd"}
    Evaluation --> roc_auc_ovr: 1.0000  Best roc_auc_ovr: 1.0000
    Time iteration: 0.026s   Total time: 0.072s
    Initial point 3 ---------------------------------
    Parameters --> {"solver": "svd"}
    Evaluation --> roc_auc_ovr: 1.0000  Best roc_auc_ovr: 1.0000
    Time iteration: 0.023s   Total time: 0.098s
    Iteration 4 -------------------------------------
    Parameters --> {"solver": "lsqr", "shrinkage": 0.7}
    Evaluation --> roc_auc_ovr: 0.8996  Best roc_auc_ovr: 1.0000
    Time iteration: 0.021s   Total time: 0.298s
    
    Results for Linear Discriminant Analysis:         
    Bayesian Optimization ---------------------------
    Best parameters --> {"solver": "svd"}
    Best evaluation --> roc_auc_ovr: 1.0000
    Time elapsed: 0.477s
    Fit ---------------------------------------------
    Score on the train set --> roc_auc_ovr: 1.0000
    Score on the test set  --> roc_auc_ovr: 1.0000
    Time elapsed: 0.015s
    Bagging -----------------------------------------
    Score --> roc_auc_ovr: 0.9998 ± 0.0005
    Time elapsed: 0.026s
    -------------------------------------------------
    Total time: 0.523s
    
    
    Running BO for Random Forest...
    Initial point 1 ---------------------------------
    Parameters --> {"n_estimators": 245, "criterion": "entropy", "max_depth": None, "min_samples_split": 13, "min_samples_leaf": 6, "max_features": 0.6, "bootstrap": True, "ccp_alpha": 0.007, "max_samples": 0.6}
    Evaluation --> roc_auc_ovr: 0.9950  Best roc_auc_ovr: 0.9950
    Time iteration: 0.388s   Total time: 0.393s
    Initial point 2 ---------------------------------
    Parameters --> {"n_estimators": 400, "criterion": "entropy", "max_depth": 8, "min_samples_split": 7, "min_samples_leaf": 19, "max_features": 0.9, "bootstrap": True, "ccp_alpha": 0.008, "max_samples": 0.7}
    Evaluation --> roc_auc_ovr: 0.9914  Best roc_auc_ovr: 0.9950
    Time iteration: 0.526s   Total time: 0.925s
    Initial point 3 ---------------------------------
    Parameters --> {"n_estimators": 78, "criterion": "gini", "max_depth": 5, "min_samples_split": 2, "min_samples_leaf": 14, "max_features": None, "bootstrap": False, "ccp_alpha": 0.003}
    Evaluation --> roc_auc_ovr: 0.9671  Best roc_auc_ovr: 0.9950
    Time iteration: 0.117s   Total time: 1.046s
    Iteration 4 -------------------------------------
    Parameters --> {"n_estimators": 394, "criterion": "entropy", "max_depth": 3, "min_samples_split": 19, "min_samples_leaf": 14, "max_features": None, "bootstrap": False, "ccp_alpha": 0.015}
    Evaluation --> roc_auc_ovr: 0.9477  Best roc_auc_ovr: 0.9950
    Time iteration: 0.460s   Total time: 1.810s
    
    Results for Random Forest:         
    Bayesian Optimization ---------------------------
    Best parameters --> {"n_estimators": 245, "criterion": "entropy", "max_depth": None, "min_samples_split": 13, "min_samples_leaf": 6, "max_features": 0.6, "bootstrap": True, "ccp_alpha": 0.007, "max_samples": 0.6}
    Best evaluation --> roc_auc_ovr: 0.9950
    Time elapsed: 2.124s
    Fit ---------------------------------------------
    Score on the train set --> roc_auc_ovr: 0.9999
    Score on the test set  --> roc_auc_ovr: 0.9767
    Time elapsed: 0.354s
    Bagging -----------------------------------------
    Score --> roc_auc_ovr: 0.9751 ± 0.0127
    Time elapsed: 1.568s
    -------------------------------------------------
    Total time: 4.050s
    
    
    Final results ========================= >>
    Duration: 17.804s
    ------------------------------------------
    Logistic Regression          --> roc_auc_ovr: 0.994 ± 0.003
    Linear Discriminant Analysis --> roc_auc_ovr: 1.000 ± 0.000 !
    Random Forest                --> roc_auc_ovr: 0.975 ± 0.013
    

## Analyze the results


```python
# We can access the pipeline's results via the results attribute
atom.results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>metric_bo</th>
      <th>time_bo</th>
      <th>metric_train</th>
      <th>metric_test</th>
      <th>time_fit</th>
      <th>mean_bagging</th>
      <th>std_bagging</th>
      <th>time_bagging</th>
      <th>time</th>
    </tr>
    <tr>
      <th>model</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LR</th>
      <td>Logistic Regression</td>
      <td>1.00000</td>
      <td>13.115s</td>
      <td>1.000000</td>
      <td>0.996503</td>
      <td>0.024s</td>
      <td>0.994172</td>
      <td>0.002553</td>
      <td>0.084s</td>
      <td>13.229s</td>
    </tr>
    <tr>
      <th>LDA</th>
      <td>Linear Discriminant Analysis</td>
      <td>1.00000</td>
      <td>0.477s</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.015s</td>
      <td>0.999767</td>
      <td>0.000466</td>
      <td>0.026s</td>
      <td>0.523s</td>
    </tr>
    <tr>
      <th>RF</th>
      <td>Random Forest</td>
      <td>0.99499</td>
      <td>2.124s</td>
      <td>0.999928</td>
      <td>0.976690</td>
      <td>0.354s</td>
      <td>0.975058</td>
      <td>0.012652</td>
      <td>1.568s</td>
      <td>4.050s</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Show the scoring for a different metric than the one we trained on
atom.scoring("precision_macro")
```

    Results ===================== >>
    Logistic Regression          --> precision_macro: 0.956
    Linear Discriminant Analysis --> precision_macro: 0.976
    Random Forest                --> precision_macro: 0.9
    

**Let's have a closer look at the Random Forest**


```python
# Get the results on some other metrics
print("Jaccard score:", atom.rf.scoring("jaccard_weighted"))
print("Recall score:", atom.rf.scoring("recall_macro"))
```

    Jaccard score: 0.7957142857142857
    Recall score: 0.8974358974358975
    


```python
# Plot the confusion matrix
atom.RF.plot_confusion_matrix(figsize=(9, 9))
```


![png](output_11_0.png)



```python
# Save the estimator as a pickle file
atom.RF.save_estimator("Random_Forest_model")
```

    Random Forest estimator saved successfully!
    
