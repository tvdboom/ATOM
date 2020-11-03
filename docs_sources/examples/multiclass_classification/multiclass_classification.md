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
      <th>134</th>
      <td>12.51</td>
      <td>1.24</td>
      <td>2.25</td>
      <td>17.5</td>
      <td>85.0</td>
      <td>2.00</td>
      <td>0.58</td>
      <td>0.60</td>
    </tr>
    <tr>
      <th>70</th>
      <td>12.29</td>
      <td>1.61</td>
      <td>2.21</td>
      <td>20.4</td>
      <td>103.0</td>
      <td>1.10</td>
      <td>1.02</td>
      <td>0.37</td>
    </tr>
    <tr>
      <th>95</th>
      <td>12.47</td>
      <td>1.52</td>
      <td>2.20</td>
      <td>19.0</td>
      <td>162.0</td>
      <td>2.50</td>
      <td>2.27</td>
      <td>0.32</td>
    </tr>
    <tr>
      <th>59</th>
      <td>12.37</td>
      <td>0.94</td>
      <td>1.36</td>
      <td>10.6</td>
      <td>88.0</td>
      <td>1.98</td>
      <td>0.57</td>
      <td>0.28</td>
    </tr>
    <tr>
      <th>115</th>
      <td>11.03</td>
      <td>1.51</td>
      <td>2.20</td>
      <td>21.5</td>
      <td>85.0</td>
      <td>2.46</td>
      <td>2.17</td>
      <td>0.52</td>
    </tr>
  </tbody>
</table>
</div>



## Run the pipeline


```python
atom = ATOMClassifier(X, y, n_jobs=-1, warnings='ignore', verbose=2, random_state=1)

# Fit the pipeline with the selected models
atom.run(
    models=['LR','LDA', 'RF'],
    metric='roc_auc_ovr',
    n_calls=4,
    n_initial_points=3,
    bo_params={'base_estimator': 'rf', 'max_time': 100},
    bagging=5
)
```

    << ================== ATOM ================== >>
    Algorithm task: multiclass classification.
    Parallel processing with 16 cores.
    
    Dataset stats ================== >>
    Shape: (178, 14)
    Scaled: False
    -----------------------------------
    Train set size: 143
    Test set size: 35
    -----------------------------------
    Train set balance: 0:1:2 <==> 1.4:1.7:1.0
    Test set balance: 0:1:2 <==> 1.0:1.4:1.4
    -----------------------------------
    Distribution of classes:
    |    |   dataset |   train |   test |
    |---:|----------:|--------:|-------:|
    |  0 |        59 |      50 |      9 |
    |  1 |        71 |      58 |     13 |
    |  2 |        48 |      35 |     13 |
    
    
    Training ===================================== >>
    Models: LR, LDA, RF
    Metric: roc_auc_ovr
    
    
    Running BO for Logistic Regression...
    Initial point 1 ---------------------------------
    Parameters --> {'penalty': 'l2', 'C': 46.003, 'solver': 'lbfgs', 'max_iter': 745}
    Evaluation --> roc_auc_ovr: 1.0000  Best roc_auc_ovr: 1.0000
    Time iteration: 3.985s   Total time: 4.078s
    Initial point 2 ---------------------------------
    Parameters --> {'penalty': 'none', 'solver': 'newton-cg', 'max_iter': 490}
    Evaluation --> roc_auc_ovr: 1.0000  Best roc_auc_ovr: 1.0000
    Time iteration: 3.537s   Total time: 7.620s
    Initial point 3 ---------------------------------
    Parameters --> {'penalty': 'l2', 'C': 0.037, 'solver': 'liblinear', 'max_iter': 352}
    Evaluation --> roc_auc_ovr: 0.9993  Best roc_auc_ovr: 1.0000
    Time iteration: 3.675s   Total time: 11.302s
    Iteration 4 -------------------------------------
    Parameters --> {'penalty': 'none', 'solver': 'newton-cg', 'max_iter': 378}
    Evaluation --> roc_auc_ovr: 1.0000  Best roc_auc_ovr: 1.0000
    Time iteration: 3.007s   Total time: 14.545s
    
    Results for Logistic Regression:         
    Bayesian Optimization ---------------------------
    Best parameters --> {'penalty': 'l2', 'C': 46.003, 'solver': 'lbfgs', 'max_iter': 745}
    Best evaluation --> roc_auc_ovr: 1.0000
    Time elapsed: 14.760s
    Fit ---------------------------------------------
    Train evaluation --> roc_auc_ovr: 1.0000
    Test evaluation --> roc_auc_ovr: 0.9965
    Time elapsed: 0.025s
    Bagging -----------------------------------------
    Evaluation --> roc_auc_ovr: 0.9942 ± 0.0026
    Time elapsed: 0.097s
    -------------------------------------------------
    Total time: 14.886s
    
    
    Running BO for Linear Discriminant Analysis...
    Initial point 1 ---------------------------------
    Parameters --> {'solver': 'eigen', 'shrinkage': 1.0}
    Evaluation --> roc_auc_ovr: 0.8975  Best roc_auc_ovr: 0.8975
    Time iteration: 0.030s   Total time: 0.032s
    Initial point 2 ---------------------------------
    Parameters --> {'solver': 'svd'}
    Evaluation --> roc_auc_ovr: 1.0000  Best roc_auc_ovr: 1.0000
    Time iteration: 0.030s   Total time: 0.066s
    Initial point 3 ---------------------------------
    Parameters --> {'solver': 'svd'}
    Evaluation --> roc_auc_ovr: 1.0000  Best roc_auc_ovr: 1.0000
    Time iteration: 0.028s   Total time: 0.099s
    Iteration 4 -------------------------------------
    Parameters --> {'solver': 'lsqr', 'shrinkage': 0.7}
    Evaluation --> roc_auc_ovr: 0.8996  Best roc_auc_ovr: 1.0000
    Time iteration: 0.024s   Total time: 0.309s
    
    Results for Linear Discriminant Analysis:         
    Bayesian Optimization ---------------------------
    Best parameters --> {'solver': 'svd'}
    Best evaluation --> roc_auc_ovr: 1.0000
    Time elapsed: 0.502s
    Fit ---------------------------------------------
    Train evaluation --> roc_auc_ovr: 1.0000
    Test evaluation --> roc_auc_ovr: 1.0000
    Time elapsed: 0.077s
    Bagging -----------------------------------------
    Evaluation --> roc_auc_ovr: 0.9998 ± 0.0005
    Time elapsed: 0.037s
    -------------------------------------------------
    Total time: 0.617s
    
    
    Running BO for Random Forest...
    Initial point 1 ---------------------------------
    Parameters --> {'n_estimators': 245, 'criterion': 'entropy', 'max_depth': None, 'min_samples_split': 13, 'min_samples_leaf': 6, 'max_features': None, 'bootstrap': True, 'ccp_alpha': 0.007, 'max_samples': 0.6}
    Evaluation --> roc_auc_ovr: 0.9921  Best roc_auc_ovr: 0.9921
    Time iteration: 0.441s   Total time: 0.449s
    Initial point 2 ---------------------------------
    Parameters --> {'n_estimators': 400, 'criterion': 'entropy', 'max_depth': 8, 'min_samples_split': 7, 'min_samples_leaf': 19, 'max_features': 0.7, 'bootstrap': True, 'ccp_alpha': 0.008, 'max_samples': 0.7}
    Evaluation --> roc_auc_ovr: 0.9927  Best roc_auc_ovr: 0.9927
    Time iteration: 0.648s   Total time: 1.102s
    Initial point 3 ---------------------------------
    Parameters --> {'n_estimators': 78, 'criterion': 'gini', 'max_depth': 5, 'min_samples_split': 2, 'min_samples_leaf': 14, 'max_features': 0.8, 'bootstrap': False, 'ccp_alpha': 0.003}
    Evaluation --> roc_auc_ovr: 0.9851  Best roc_auc_ovr: 0.9927
    Time iteration: 0.129s   Total time: 1.236s
    Iteration 4 -------------------------------------
    Parameters --> {'n_estimators': 394, 'criterion': 'entropy', 'max_depth': 3, 'min_samples_split': 19, 'min_samples_leaf': 14, 'max_features': 0.8, 'bootstrap': False, 'ccp_alpha': 0.015}
    Evaluation --> roc_auc_ovr: 0.9897  Best roc_auc_ovr: 0.9927
    Time iteration: 0.497s   Total time: 2.036s
    
    Results for Random Forest:         
    Bayesian Optimization ---------------------------
    Best parameters --> {'n_estimators': 400, 'criterion': 'entropy', 'max_depth': 8, 'min_samples_split': 7, 'min_samples_leaf': 19, 'max_features': 0.7, 'bootstrap': True, 'ccp_alpha': 0.008, 'max_samples': 0.7}
    Best evaluation --> roc_auc_ovr: 0.9927
    Time elapsed: 2.333s
    Fit ---------------------------------------------
    Train evaluation --> roc_auc_ovr: 0.9997
    Test evaluation --> roc_auc_ovr: 0.9802
    Time elapsed: 0.605s
    Bagging -----------------------------------------
    Evaluation --> roc_auc_ovr: 0.9740 ± 0.0074
    Time elapsed: 2.643s
    -------------------------------------------------
    Total time: 5.583s
    
    
    Final results ========================= >>
    Duration: 21.088s
    ------------------------------------------
    Logistic Regression          --> roc_auc_ovr: 0.994 ± 0.003
    Linear Discriminant Analysis --> roc_auc_ovr: 1.000 ± 0.000 !
    Random Forest                --> roc_auc_ovr: 0.974 ± 0.007
    

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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LR</th>
      <td>1</td>
      <td>14.760s</td>
      <td>1</td>
      <td>0.996503</td>
      <td>0.025s</td>
      <td>0.994172</td>
      <td>0.00255349</td>
      <td>0.097s</td>
      <td>14.886s</td>
    </tr>
    <tr>
      <th>LDA</th>
      <td>1</td>
      <td>0.502s</td>
      <td>1</td>
      <td>1</td>
      <td>0.077s</td>
      <td>0.999767</td>
      <td>0.0004662</td>
      <td>0.037s</td>
      <td>0.617s</td>
    </tr>
    <tr>
      <th>RF</th>
      <td>0.992716</td>
      <td>2.333s</td>
      <td>0.999654</td>
      <td>0.980186</td>
      <td>0.605s</td>
      <td>0.974022</td>
      <td>0.00735105</td>
      <td>2.643s</td>
      <td>5.583s</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Show the scoring for a different metric than the one we trained on
atom.scoring('precision_macro')
```

    Results ===================== >>
    Logistic Regression          --> precision_macro: 0.949
    Linear Discriminant Analysis --> precision_macro: 1.0
    Random Forest                --> precision_macro: 0.919
    

**Let's have a closer look at the Random Forest**


```python
# Get the results on some other metrics
print('Jaccard score:', atom.rf.scoring('jaccard_weighted'))
print('Recall score:', atom.rf.scoring('recall_macro'))
```

    Jaccard score: 0.8428571428571429
    Recall score: 0.923076923076923
    


```python
# Plot the confusion matrix
atom.RF.plot_confusion_matrix(figsize=(9, 9))
```


![png](output_11_0.png)



```python
# Save the estimator as a pickle file
atom.RF.save_estimator('Random_Forest_model')
```

    Random Forest estimator saved successfully!
    
