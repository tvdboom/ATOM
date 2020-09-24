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
      <th>101</th>
      <td>12.60</td>
      <td>1.34</td>
      <td>1.90</td>
      <td>18.5</td>
      <td>88.0</td>
      <td>1.45</td>
      <td>1.36</td>
      <td>0.29</td>
    </tr>
    <tr>
      <th>133</th>
      <td>12.70</td>
      <td>3.55</td>
      <td>2.36</td>
      <td>21.5</td>
      <td>106.0</td>
      <td>1.70</td>
      <td>1.20</td>
      <td>0.17</td>
    </tr>
    <tr>
      <th>86</th>
      <td>12.16</td>
      <td>1.61</td>
      <td>2.31</td>
      <td>22.8</td>
      <td>90.0</td>
      <td>1.78</td>
      <td>1.69</td>
      <td>0.43</td>
    </tr>
    <tr>
      <th>93</th>
      <td>12.29</td>
      <td>2.83</td>
      <td>2.22</td>
      <td>18.0</td>
      <td>88.0</td>
      <td>2.45</td>
      <td>2.25</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>92</th>
      <td>12.69</td>
      <td>1.53</td>
      <td>2.26</td>
      <td>20.7</td>
      <td>80.0</td>
      <td>1.38</td>
      <td>1.46</td>
      <td>0.58</td>
    </tr>
  </tbody>
</table>
</div>



## Run the pipeline


```python
atom = ATOMClassifier(X, y, n_jobs=-1, warnings='ignore', verbose=2, random_state=1)

# Fit the pipeline with the selected models
atom.run(models=['LR','LDA', 'RF'],
         metric='roc_auc_ovr',
         n_calls=4,
         n_initial_points=3,
         bo_params={'base_estimator': 'rf', 'max_time': 100},
         bagging=5)
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
    Random start 1 ----------------------------------
    Parameters --> {'max_iter': 335, 'solver': 'sag', 'penalty': 'l2', 'C': 0.001}
    Evaluation --> roc_auc_ovr: 0.9970  Best roc_auc_ovr: 0.9970
    Time iteration: 3.971s   Total time: 3.975s
    Random start 2 ----------------------------------
    Parameters --> {'max_iter': 244, 'solver': 'lbfgs', 'penalty': 'l2', 'C': 0.087}
    Evaluation --> roc_auc_ovr: 1.0000  Best roc_auc_ovr: 1.0000
    Time iteration: 3.769s   Total time: 7.748s
    Random start 3 ----------------------------------
    Parameters --> {'max_iter': 376, 'solver': 'liblinear', 'penalty': 'l2', 'C': 2.667}
    Evaluation --> roc_auc_ovr: 1.0000  Best roc_auc_ovr: 1.0000
    Time iteration: 3.589s   Total time: 11.342s
    Iteration 4 -------------------------------------
    Parameters --> {'max_iter': 498, 'solver': 'sag', 'penalty': 'l2', 'C': 0.882}
    Evaluation --> roc_auc_ovr: 1.0000  Best roc_auc_ovr: 1.0000
    Time iteration: 4.328s   Total time: 15.920s
    
    Results for Logistic Regression:         
    Bayesian Optimization ---------------------------
    Best parameters --> {'max_iter': 244, 'solver': 'lbfgs', 'penalty': 'l2', 'C': 0.087}
    Best evaluation --> roc_auc_ovr: 1.0000
    Time elapsed: 16.151s
    Fitting -----------------------------------------
    Score on the train set --> roc_auc_ovr: 1.0000
    Score on the test set  --> roc_auc_ovr: 0.9988
    Time elapsed: 0.020s
    Bagging -----------------------------------------
    Score --> roc_auc_ovr: 0.9991 ± 0.0009
    Time elapsed: 0.072s
    -------------------------------------------------
    Total time: 16.249s
    
    
    Running BO for Linear Discriminant Analysis...
    Random start 1 ----------------------------------
    Parameters --> {'solver': 'eigen', 'shrinkage': 1.0}
    Evaluation --> roc_auc_ovr: 0.8975  Best roc_auc_ovr: 0.8975
    Time iteration: 0.021s   Total time: 0.022s
    Random start 2 ----------------------------------
    Parameters --> {'solver': 'svd'}
    Evaluation --> roc_auc_ovr: 1.0000  Best roc_auc_ovr: 1.0000
    Time iteration: 0.021s   Total time: 0.047s
    Random start 3 ----------------------------------
    Parameters --> {'solver': 'svd'}
    Evaluation --> roc_auc_ovr: 1.0000  Best roc_auc_ovr: 1.0000
    Time iteration: 0.018s   Total time: 0.068s
    Iteration 4 -------------------------------------
    Parameters --> {'solver': 'lsqr', 'shrinkage': 0.7}
    Evaluation --> roc_auc_ovr: 0.8996  Best roc_auc_ovr: 1.0000
    Time iteration: 0.020s   Total time: 0.279s
    
    Results for Linear Discriminant Analysis:         
    Bayesian Optimization ---------------------------
    Best parameters --> {'solver': 'svd'}
    Best evaluation --> roc_auc_ovr: 1.0000
    Time elapsed: 0.474s
    Fitting -----------------------------------------
    Score on the train set --> roc_auc_ovr: 1.0000
    Score on the test set  --> roc_auc_ovr: 1.0000
    Time elapsed: 0.010s
    Bagging -----------------------------------------
    Score --> roc_auc_ovr: 0.9998 ± 0.0005
    Time elapsed: 0.024s
    -------------------------------------------------
    Total time: 0.510s
    
    
    Running BO for Random Forest...
    Random start 1 ----------------------------------
    Parameters --> {'n_estimators': 245, 'max_depth': 7, 'max_features': 1.0, 'criterion': 'gini', 'min_samples_split': 7, 'min_samples_leaf': 16, 'ccp_alpha': 0.008, 'bootstrap': True, 'max_samples': 0.6}
    Evaluation --> roc_auc_ovr: 0.9853  Best roc_auc_ovr: 0.9853
    Time iteration: 0.412s   Total time: 0.418s
    Random start 2 ----------------------------------
    Parameters --> {'n_estimators': 400, 'max_depth': 4, 'max_features': 0.8, 'criterion': 'gini', 'min_samples_split': 20, 'min_samples_leaf': 12, 'ccp_alpha': 0.016, 'bootstrap': True, 'max_samples': 0.7}
    Evaluation --> roc_auc_ovr: 0.9937  Best roc_auc_ovr: 0.9937
    Time iteration: 0.642s   Total time: 1.063s
    Random start 3 ----------------------------------
    Parameters --> {'n_estimators': 78, 'max_depth': 10, 'max_features': 0.7, 'criterion': 'gini', 'min_samples_split': 2, 'min_samples_leaf': 14, 'ccp_alpha': 0.025, 'bootstrap': False}
    Evaluation --> roc_auc_ovr: 0.9865  Best roc_auc_ovr: 0.9937
    Time iteration: 0.122s   Total time: 1.190s
    Iteration 4 -------------------------------------
    Parameters --> {'n_estimators': 323, 'max_depth': 7, 'max_features': 1.0, 'criterion': 'gini', 'min_samples_split': 16, 'min_samples_leaf': 1, 'ccp_alpha': 0.007, 'bootstrap': False}
    Evaluation --> roc_auc_ovr: 0.9315  Best roc_auc_ovr: 0.9937
    Time iteration: 0.405s   Total time: 1.823s
    
    Results for Random Forest:         
    Bayesian Optimization ---------------------------
    Best parameters --> {'n_estimators': 400, 'max_depth': 4, 'max_features': 0.8, 'criterion': 'gini', 'min_samples_split': 20, 'min_samples_leaf': 12, 'ccp_alpha': 0.016, 'bootstrap': True, 'max_samples': 0.7}
    Best evaluation --> roc_auc_ovr: 0.9937
    Time elapsed: 2.056s
    Fitting -----------------------------------------
    Score on the train set --> roc_auc_ovr: 0.9997
    Score on the test set  --> roc_auc_ovr: 0.9825
    Time elapsed: 0.588s
    Bagging -----------------------------------------
    Score --> roc_auc_ovr: 0.9737 ± 0.0116
    Time elapsed: 2.716s
    -------------------------------------------------
    Total time: 5.363s
    
    
    Final results ========================= >>
    Duration: 22.125s
    ------------------------------------------
    Logistic Regression          --> roc_auc_ovr: 0.999 ± 0.001
    Linear Discriminant Analysis --> roc_auc_ovr: 1.000 ± 0.000 !
    Random Forest                --> roc_auc_ovr: 0.974 ± 0.012
    

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
      <td>1.000000</td>
      <td>16.151s</td>
      <td>1.000000</td>
      <td>0.998834</td>
      <td>0.020s</td>
      <td>0.999068</td>
      <td>0.000872</td>
      <td>0.072s</td>
      <td>16.249s</td>
    </tr>
    <tr>
      <th>LDA</th>
      <td>Linear Discriminant Analysis</td>
      <td>1.000000</td>
      <td>0.474s</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.010s</td>
      <td>0.999767</td>
      <td>0.000466</td>
      <td>0.024s</td>
      <td>0.510s</td>
    </tr>
    <tr>
      <th>RF</th>
      <td>Random Forest</td>
      <td>0.993712</td>
      <td>2.056s</td>
      <td>0.999725</td>
      <td>0.982517</td>
      <td>0.588s</td>
      <td>0.973686</td>
      <td>0.011577</td>
      <td>2.716s</td>
      <td>5.363s</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Show the scoring for a different metric than the one we trained on
atom.scoring('precision_macro')
```

    Results ===================== >>
    Logistic Regression          --> precision_macro: 1.0
    Linear Discriminant Analysis --> precision_macro: 0.976
    Random Forest                --> precision_macro: 0.9
    

**Let's have a closer look at the Random Forest**


```python
# Get the results on some other metrics
print('Jaccard score:', atom.rf.scoring('jaccard_weighted'))
print('Recall score:', atom.rf.scoring('recall_macro'))
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
atom.RF.save_estimator('Random_Forest_model')
```

    Random Forest estimator saved successfully!
    
