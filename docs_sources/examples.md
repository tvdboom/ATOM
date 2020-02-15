# Binary classification
---------------------------------

Download the Australian weather dataset from [https://www.kaggle.com/jsphyg/weather-dataset-rattle-package](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package).
 This dataset tries to predict whether or not it will rain tomorrow by
 training a classification model on target `RainTomorrow`.

**Load the data**


```python
# Import packages
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score
from atom import ATOMClassifier

# Load the Australian weather dataset
X = pd.read_csv('../weatherAUS.csv')
X = X.drop(['RISK_MM', 'Date'], axis=1)  # Drop unrelated features
```

**Run the pipeline**


```python
# Call ATOM using only a percentage of the complete dataset (for explanatory purposes)
atom = ATOMClassifier(X, y="RainTomorrow", percentage=5, log='auto', n_jobs=2, verbose=3)
```

    <<=============== ATOM ===============>>
    Parallel processing with 2 cores.
    Initial data cleaning...
     --> Dropping 45 duplicate rows.
    Algorithm task: binary classification.
    
    Dataset stats ===================>
    Shape: (7107, 22)
    Missing values: 15346
    Categorical columns: 5
    Scaled: False
    ----------------------------------
    Size of training set: 4974
    Size of test set: 2133
    ----------------------------------
    Instances in RainTomorrow per class:
    |        |    total |    train_set |    test_set |
    |:-------|---------:|-------------:|------------:|
    | 0: No  |     5520 |         3867 |        1653 |
    | 1: Yes |     1587 |         1107 |         480 |
    
    


```python
# If we change a column during the pre-processing,
# we need to call the update method to update all data attributes

atom.X['MaxTemp'] = np.log(atom.X['MaxTemp'])

# MaxTemp has now been changed for atom.X, but not for in atom.X_train, atom.dataset, etc...
# To do se, we do...
atom.update('X')

assert atom.X['MaxTemp'].equals(atom.dataset['MaxTemp'])
```


```python
# Impute missing values
atom.impute(strat_num='knn', strat_cat='missing', max_frac_rows=0.8)
```

    Imputing missing values...
     --> Removing 702 rows for containing too many missing values.
     --> Imputing 4 missing values using the KNN imputer in feature MinTemp.
     --> Imputing 4 missing values using the KNN imputer in feature MaxTemp.
     --> Imputing 26 missing values using the KNN imputer in feature Rainfall.
     --> Imputing 2318 missing values using the KNN imputer in feature Evaporation.
     --> Imputing 2624 missing values using the KNN imputer in feature Sunshine.
     --> Imputing 239 missing values with missing in feature WindGustDir.
     --> Imputing 239 missing values using the KNN imputer in feature WindGustSpeed.
     --> Imputing 309 missing values with missing in feature WindDir9am.
     --> Imputing 24 missing values with missing in feature WindDir3pm.
     --> Imputing 1 missing values using the KNN imputer in feature WindSpeed9am.
     --> Imputing 1 missing values using the KNN imputer in feature WindSpeed3pm.
     --> Imputing 38 missing values using the KNN imputer in feature Humidity9am.
     --> Imputing 64 missing values using the KNN imputer in feature Humidity3pm.
     --> Imputing 48 missing values using the KNN imputer in feature Pressure9am.
     --> Imputing 45 missing values using the KNN imputer in feature Pressure3pm.
     --> Imputing 2086 missing values using the KNN imputer in feature Cloud9am.
     --> Imputing 2190 missing values using the KNN imputer in feature Cloud3pm.
     --> Imputing 4 missing values using the KNN imputer in feature Temp9am.
     --> Imputing 29 missing values using the KNN imputer in feature Temp3pm.
     --> Imputing 26 missing values with missing in feature RainToday.
    


```python
# Encode the categorical features
atom.encode(max_onehot=10, frac_to_other=0.04)
```

    Encoding categorical features...
     --> Target-encoding feature Location.  Contains 1 unique categories.
     --> Target-encoding feature WindGustDir.  Contains 17 unique categories.
     --> Target-encoding feature WindDir9am.  Contains 17 unique categories.
     --> Target-encoding feature WindDir3pm.  Contains 17 unique categories.
     --> One-hot-encoding feature RainToday. Contains 3 unique categories.
    


```python
# Select only the best 10 features
atom.feature_selection(strategy="univariate", solver='f_classif', max_features=15, max_correlation=0.8)

# See which features were removed due to collinearity
atom.collinear
```

    Performing feature selection...
     --> Feature Location was removed due to low variance: 0.00.
     --> Feature Pressure3pm was removed due to collinearity with another feature.
     --> Feature Temp9am was removed due to collinearity with another feature.
     --> Feature Temp3pm was removed due to collinearity with another feature.
     --> Feature RainToday_Yes was removed due to collinearity with another feature.
     --> Feature WindSpeed9am was removed after the univariate test (score: 36.90  p-value: 0.00).
     --> Feature WindSpeed3pm was removed after the univariate test (score: 55.43  p-value: 0.00).
     --> Feature RainToday_other was removed after the univariate test (score: 23.65  p-value: 0.00).
    




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
      <th>drop_feature</th>
      <th>correlated_feature</th>
      <th>correlation_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Pressure3pm</td>
      <td>Pressure9am</td>
      <td>0.96065</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Temp9am</td>
      <td>MinTemp, MaxTemp</td>
      <td>0.90137, 0.87496</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Temp3pm</td>
      <td>MaxTemp, Temp9am</td>
      <td>0.96166, 0.84807</td>
    </tr>
    <tr>
      <th>3</th>
      <td>RainToday_Yes</td>
      <td>RainToday_No</td>
      <td>-0.98762</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Perform undersampling of the majority class to balance the dataset
atom.balance(undersample=0.8)
```

    Using TensorFlow backend.
    

    Performing undersampling...
    


```python
# Remove outliers from the training set
atom.outliers(max_sigma=5)
```

    Handling outliers...
     --> Dropping 30 rows due to outliers.
    


```python
# Change the verbosity of ATOM to not print too much details while fitting
atom.verbose = 2

# Let's use a custom metric
def f2_score(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=2)

# Let's compare the performance of various gradient boosting algorithms
atom.pipeline(['gbm', 'lgb', 'catb'],
              metric=f2_score,
              max_iter=5,
              init_points=5,
              cv=1,
              bagging=5)
```

    
    Running pipeline =================>
    Models in pipeline: GBM, LGB, CatB
    Metric: f2_score
    
    
    Running BO for Gradient Boosting Machine...
    Final results for Gradient Boosting Machine:         
    Bayesian Optimization ---------------------------
    Best hyperparameters: {'n_estimators': 348, 'learning_rate': 1.0, 'subsample': 1.0, 'max_depth': 4, 'max_features': 0.6, 'criterion': 'mse', 'min_samples_split': 7, 'min_samples_leaf': 8, 'ccp_alpha': 0.0}
    Best score on the BO: 0.7208
    Time elapsed: 36.286s
    Fitting -----------------------------------------
    Score on the training set: 1.0000
    Score on the test set: 0.5660
    Time elapsed: 1.601s
    Bagging -----------------------------------------
    Mean: 0.5531   Std: 0.0154
    Time elapsed: 3.781s
    -------------------------------------------------
    Total time: 41.672s
    
    
    Running BO for LightGBM...
    Final results for LightGBM:         
    Bayesian Optimization ---------------------------
    Best hyperparameters: {'n_estimators': 500, 'learning_rate': 0.75, 'max_depth': 10, 'num_leaves': 29, 'min_child_weight': 12, 'min_child_samples': 13, 'subsample': 0.6, 'colsample_bytree': 0.3, 'reg_alpha': 0.0, 'reg_lambda': 0.0}
    Best score on the BO: 0.7371
    Time elapsed: 2.250s
    Fitting -----------------------------------------
    Score on the training set: 1.0000
    Score on the test set: 0.5622
    Time elapsed: 1.348s
    Bagging -----------------------------------------
    Mean: 0.5752   Std: 0.0197
    Time elapsed: 0.579s
    -------------------------------------------------
    Total time: 4.178s
    
    
    Running BO for CatBoost...
    Final results for CatBoost:         
    Bayesian Optimization ---------------------------
    Best hyperparameters: {'n_estimators': 172, 'learning_rate': 0.68, 'max_depth': 6, 'subsample': 0.6, 'colsample_bylevel': 0.3, 'reg_lambda': 0.01}
    Best score on the BO: 0.7606
    Time elapsed: 14.099s
    Fitting -----------------------------------------
    Score on the training set: 1.0000
    Score on the test set: 0.5861
    Time elapsed: 0.349s
    Bagging -----------------------------------------
    Mean: 0.5817   Std: 0.0078
    Time elapsed: 1.097s
    -------------------------------------------------
    Total time: 15.547s
    
    
    Final results ================>>
    Duration: 1m:01s
    Metric: f2_score
    --------------------------------
    Gradient Boosting Machine --> 0.553 ± 0.015 ~
    LightGBM                  --> 0.575 ± 0.020 ~
    CatBoost                  --> 0.582 ± 0.008 !! ~
    

**Analyze the results**


```python
# Let's have a look at the best model
print('And the winner is...', atom.winner.longname)

print('Score on the training set: ', atom.winner.score_train)
print('Score on the test set: ', atom.winner.score_test)
```

    And the winner is... CatBoost
    Score on the training set:  1.0
    Score on the test set:  0.5860702151755378
    


```python
# Make some plots to analyze the results
atom.winner.plot_confusion_matrix(normalize=True, figsize=(7, 7), filename='confusion_matrix.png')
atom.winner.plot_probabilities()

# Change plots aesthetics
ATOMClassifier.set_style('whitegrid')
ATOMClassifier.set_title_fontsize(24)

atom.plot_ROC(models=('LGB', 'CatB'), title="ROC for the LightGBM vs CatBoost model")
atom.plot_PRC(title="PRC comparison of the models")
atom.catb.plot_threshold(metric=['f1', 'accuracy', 'average_precision'], steps=50, filename='thresholds.png')
```


![png](img/confusion_matrix.png)



![png](img/probabilities.png)



![png](img/ROC.png)



![png](img/PRC.png)



![png](img/threshold.png)




<br><br>

# Multiclass classification
---------------------------------

Import the wine dataset from [sklearn.datasets](https://scikit-learn.org/stable/datasets/index.html#wine-dataset).
 This is a small and easy to train dataset which goal is to classify wines
 into three groups (which cultivator it's from) using features based on the results
 of chemical analysis.


**Load the data**

```python
# Import packages
import numpy as np
import pandas as pd
from sklearn.datasets load_wine
from atom.atom import ATOMClassifier

# Load the dataset's features and targets
dataset = load_wine()

# Convert to pd.DtaFrame to get the names of the features
data = np.c_[dataset.data, dataset.target]
columns = np.append(dataset.feature_names, ["target"])
data = pd.DataFrame(data, columns=columns)
X = data.drop('target', axis=1)
y = data['target']
```

**Run the pipeline**


```python
# Call ATOMclass for ML task exploration
atom = ATOMClassifier(X, y, n_jobs=-1, verbose=3)

# Fit the pipeline with the selected models
atom.pipeline(models=['LDA','RF', 'lSVM'],
              metric='f1_macro',
              max_iter=4,
              init_points=3,
              cv=3,
              bagging=10)
```

    <<=============== ATOM ===============>>
    Parallel processing with 4 cores.
    Initial data cleaning...
    Algorithm task: multiclass classification. Number of classes: 3.
    
    Dataset stats ===================>
    Shape: (178, 14)
    Scaled: False
    ----------------------------------
    Size of training set: 124
    Size of test set: 54
    ----------------------------------
    Instances in target per class:
    |    |    total |    train_set |    test_set |
    |---:|---------:|-------------:|------------:|
    |  0 |       59 |           42 |          17 |
    |  1 |       71 |           47 |          24 |
    |  2 |       48 |           35 |          13 |
    
    
    Running pipeline =================>
    Models in pipeline: LDA, RF, lSVM
    Metric: f1_macro
    
    
    Running BO for Linear Discriminant Analysis...
    Initial point: 1 --------------------------------
    Parameters --> {'solver': 'lsqr', 'shrinkage': 0.9}
    Evaluation --> f1_macro: 0.6787
    Time elapsed: 0.815s   Total time: 0.816s
    Initial point: 2 --------------------------------
    Parameters --> {'solver': 'lsqr', 'shrinkage': 0.8}
    Evaluation --> f1_macro: 0.6865
    Time elapsed: 0.505s   Total time: 1.320s
    Initial point: 3 --------------------------------
    Parameters --> {'solver': 'eigen', 'shrinkage': 0.7}
    Evaluation --> f1_macro: 0.6667
    Time elapsed: 0.021s   Total time: 1.341s
    Iteration: 1 ------------------------------------
    Parameters --> {'solver': 'svd'}
    Evaluation --> f1_macro: 0.9753
    Time elapsed: 0.020s   Total time: 1.560s
    Iteration: 2 ------------------------------------
    Parameters --> {'solver': 'svd'}
    Evaluation --> f1_macro: 0.9753
    Time elapsed: 0.026s   Total time: 1.796s
    
    Final results for Linear Discriminant Analysis:         
    Bayesian Optimization ---------------------------
    Best hyperparameters: {'solver': 'svd'}
    Best score on the BO: 0.9753
    Time elapsed: 1.936s
    Fitting -----------------------------------------
    Score on the training set: 1.0000
    Score on the test set: 0.9617
    Time elapsed: 0.079s
    Bagging -----------------------------------------
    Mean: 0.9788   Std: 0.0159
    Time elapsed: 0.035s
    -------------------------------------------------
    Total time: 2.050s
    
    
    Running BO for Random Forest...
    Initial point: 1 --------------------------------
    Parameters --> {'n_estimators': 460, 'max_depth': 5, 'max_features': 0.9, 'criterion': 'entropy', 'min_samples_split': 10, 'min_samples_leaf': 20, 'ccp_alpha': 0.03, 'bootstrap': True, 'max_samples': 0.7}
    Evaluation --> f1_macro: 0.8673
    Time elapsed: 0.835s   Total time: 0.836s
    Initial point: 2 --------------------------------
    Parameters --> {'n_estimators': 210, 'max_depth': 6, 'max_features': 0.5, 'criterion': 'entropy', 'min_samples_split': 11, 'min_samples_leaf': 14, 'ccp_alpha': 0.025, 'bootstrap': False}
    Evaluation --> f1_macro: 0.9357
    Time elapsed: 0.449s   Total time: 1.286s
    Initial point: 3 --------------------------------
    Parameters --> {'n_estimators': 155, 'max_depth': 7, 'max_features': 0.7, 'criterion': 'entropy', 'min_samples_split': 18, 'min_samples_leaf': 13, 'ccp_alpha': 0.02, 'bootstrap': False}
    Evaluation --> f1_macro: 0.8638
    Time elapsed: 0.405s   Total time: 1.691s
    Iteration: 1 ------------------------------------
    Parameters --> {'n_estimators': 460, 'max_depth': 6, 'max_features': 0.9, 'criterion': 'entropy', 'min_samples_split': 10, 'min_samples_leaf': 20, 'ccp_alpha': 0.035, 'bootstrap': True, 'max_samples': 0.7}
    Evaluation --> f1_macro: 0.9073
    Time elapsed: 0.827s   Total time: 2.801s
    Iteration: 2 ------------------------------------
    Parameters --> {'n_estimators': 20, 'max_depth': 3, 'max_features': 0.7, 'criterion': 'gini', 'min_samples_split': 3, 'min_samples_leaf': 18, 'ccp_alpha': 0.015, 'bootstrap': False}
    Evaluation --> f1_macro: 0.8953
    Time elapsed: 0.234s   Total time: 3.362s
    Iteration: 3 ------------------------------------
    Parameters --> {'n_estimators': 20, 'max_depth': 8, 'max_features': 0.6, 'criterion': 'entropy', 'min_samples_split': 14, 'min_samples_leaf': 3, 'ccp_alpha': 0.03, 'bootstrap': True, 'max_samples': 0.6}
    Evaluation --> f1_macro: 0.9512
    Time elapsed: 0.231s   Total time: 3.822s
    Iteration: 4 ------------------------------------
    Parameters --> {'n_estimators': 20, 'max_depth': 9, 'max_features': 1.0, 'criterion': 'entropy', 'min_samples_split': 20, 'min_samples_leaf': 7, 'ccp_alpha': 0.02, 'bootstrap': False}
    Evaluation --> f1_macro: 0.8560
    Time elapsed: 0.235s   Total time: 4.563s
    
    Final results for Random Forest:         
    Bayesian Optimization ---------------------------
    Best hyperparameters: {'n_estimators': 20, 'max_depth': 8, 'max_features': 0.6, 'criterion': 'entropy', 'min_samples_split': 14, 'min_samples_leaf': 3, 'ccp_alpha': 0.03, 'bootstrap': True, 'max_samples': 0.6}
    Best score on the BO: 0.9512
    Time elapsed: 4.790s
    Fitting -----------------------------------------
    Score on the training set: 1.0000
    Score on the test set: 0.9448
    Time elapsed: 5.671s
    Bagging -----------------------------------------
    Mean: 0.9240   Std: 0.0274
    Time elapsed: 2.345s
    -------------------------------------------------
    Total time: 12.806s
    
    
    Running BO for Linear SVM...
    Initial point: 1 --------------------------------
    Parameters --> {'C': 0.01, 'loss': 'squared_hinge', 'dual': True, 'penalty': 'l2'}
    Evaluation --> f1_macro: 0.9833
    Time elapsed: 0.031s   Total time: 0.031s
    Initial point: 2 --------------------------------
    Parameters --> {'C': 0.001, 'loss': 'hinge', 'dual': True, 'penalty': 'l2'}
    Evaluation --> f1_macro: 0.9290
    Time elapsed: 0.016s   Total time: 0.047s
    Initial point: 3 --------------------------------
    Parameters --> {'C': 0.001, 'loss': 'squared_hinge', 'dual': True, 'penalty': 'l2'}
    Evaluation --> f1_macro: 0.9601
    Time elapsed: 0.031s   Total time: 0.078s
    Iteration: 1 ------------------------------------
    Parameters --> {'C': 10, 'loss': 'squared_hinge', 'dual': False, 'penalty': 'l1'}
    Evaluation --> f1_macro: 0.9842
    Time elapsed: 0.028s   Total time: 0.359s
    Iteration: 2 ------------------------------------
    Parameters --> {'C': 100, 'loss': 'squared_hinge', 'dual': False, 'penalty': 'l1'}
    Evaluation --> f1_macro: 0.9842
    Time elapsed: 0.025s   Total time: 0.730s
    Iteration: 3 ------------------------------------
    Parameters --> {'C': 100, 'loss': 'squared_hinge', 'dual': False, 'penalty': 'l1'}
    Evaluation --> f1_macro: 0.9842
    Time elapsed: 0.016s   Total time: 1.059s
    
    Final results for Linear SVM:         
    Bayesian Optimization ---------------------------
    Best hyperparameters: {'C': 10.0, 'loss': 'squared_hinge', 'dual': False, 'penalty': 'l1'}
    Best score on the BO: 0.9842
    Time elapsed: 1.230s
    Fitting -----------------------------------------
    Score on the training set: 1.0000
    Score on the test set: 0.9617
    Time elapsed: 0.047s
    Bagging -----------------------------------------
    Mean: 0.9498   Std: 0.0117
    Time elapsed: 0.101s
    -------------------------------------------------
    Total time: 1.379s
    
    
    Final results ================>>
    Duration: 16.238s
    Metric: f1_macro
    --------------------------------
    Linear Discriminant Analysis --> 0.979 ± 0.016 !!
    Random Forest                --> 0.924 ± 0.027
    Linear SVM                   --> 0.950 ± 0.012
    

**Analyze the results**


```python
atom.scores
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
      <th>model</th>
      <th>total_time</th>
      <th>score_train</th>
      <th>score_test</th>
      <th>fit_time</th>
      <th>bagging_mean</th>
      <th>bagging_std</th>
      <th>bagging_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LDA</td>
      <td>2.050s</td>
      <td>1.0</td>
      <td>0.961698</td>
      <td>0.079s</td>
      <td>0.978848</td>
      <td>0.015898</td>
      <td>0.035s</td>
    </tr>
    <tr>
      <th>1</th>
      <td>RF</td>
      <td>12.806s</td>
      <td>1.0</td>
      <td>0.944813</td>
      <td>5.671s</td>
      <td>0.923975</td>
      <td>0.027393</td>
      <td>2.345s</td>
    </tr>
    <tr>
      <th>2</th>
      <td>lSVM</td>
      <td>1.379s</td>
      <td>1.0</td>
      <td>0.961698</td>
      <td>0.047s</td>
      <td>0.949818</td>
      <td>0.011717</td>
      <td>0.101s</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Show the results for a different metric
atom.results('precision_macro')
```

    
    Final results ================>>
    Metric: precision_macro
    --------------------------------
    Linear Discriminant Analysis --> 0.956 !!
    Random Forest                --> 0.941
    Linear SVM                   --> 0.956 !!
    


```python
atom.plot_bagging()
```


![png](img/bagging_results.png)


**Let's have a closer look at the Random Forest**


```python
# Get the results on some other metrics
print('Jaccard score:', atom.rf.jaccard_weighted)
print('Recall score:', atom.rf.recall_macro)
```

    Jaccard score: 0.8960493827160495
    Recall score: 0.9526143790849674
    


```python
# Plot the feature importance and compare it to the permutation importance of the LDA
atom.rf.plot_feature_importance(show=10)
atom.lda.plot_permutation_importance(show=10)
```


![png](img/feature_importance.png)



![png](img/permutation_importance.png)



```python
# Save the random forest class for production
atom.RF.save('Random_Forest_class')
```

    Random Forest model subclass saved successfully!
    
<br><br>

# Regression
---------------------------------

Download the abalone dataset from [https://archive.ics.uci.edu/ml/datasets/Abalone](https://archive.ics.uci.edu/ml/datasets/Abalone).
 The goal of this dataset is to predict the age of abalone shells from physical measurements.

**Load the data**

```python
# Import packages
import pandas as pd
from atom import ATOMRegressor

# Load the abalone dataset
X = pd.read_csv('../abalone.csv')
atom = ATOMRegressor(X, y="Rings", percentage=10, warnings=False, verbose=1, random_state=42)

# Encode categorical features
atom.encode()

# Apply PCA for dimensionality reduction
atom.feature_selection(strategy="pca", max_features=6)
atom.plot_PCA(figsize=(8, 6), filename='atom_PCA_plot')
```

    <<=============== ATOM ===============>>
    Algorithm task: regression.
    


![png](img/PCA.png)


**Run the pipeline**

```python
# Let's compare tree-based models using a successive halving approach
atom.pipeline(['tree', 'bag', 'et', 'rf', 'gbm', 'lgb'],
              successive_halving=True,
              metric='neg_mean_squared_error',
              max_iter=5,
              init_points=5,
              cv=1,
              bagging=5)
```

                                                                                   

    
    Running pipeline =================>
    Metric: neg_mean_squared_error
    
    
    <<=============== Iteration 0 ==============>>
    Models in pipeline: Tree, Bag, ET, RF, GBM, LGB
    

    Processing: 100%|████████████████████████████████| 6/6 [00:25<00:00,  4.18s/it]
                                                                                   

    
    
    Final results ================>>
    Duration: 25.079s
    Metric: neg_mean_squared_error
    --------------------------------
    Decision Tree             --> -9.479 ± 0.667 !! ~
    Bagging Regressor         --> -11.409 ± 2.167 ~
    Extra-Trees               --> -11.788 ± 1.270 ~
    Random Forest             --> -11.441 ± 1.059 ~
    Gradient Boosting Machine --> -11.044 ± 2.575 ~
    LightGBM                  --> -12.929 ± 3.211 ~
    
    
    <<=============== Iteration 1 ==============>>
    Models in pipeline: Tree, Bag, GBM
    

    Processing: 100%|████████████████████████████████| 3/3 [00:12<00:00,  4.03s/it]
                                                                                   

    
    
    Final results ================>>
    Duration: 37.229s
    Metric: neg_mean_squared_error
    --------------------------------
    Decision Tree             --> -11.110 ± 5.487 ~
    Bagging Regressor         --> -6.780 ± 1.605 !! ~
    Gradient Boosting Machine --> -8.079 ± 0.545 ~
    
    
    <<=============== Iteration 2 ==============>>
    Model in pipeline: Bag
    

    Processing: 100%|████████████████████████████████| 1/1 [00:10<00:00, 10.36s/it]

    
    
    Final results ================>>
    Duration: 47.619s
    Metric: neg_mean_squared_error
    --------------------------------
    Bagging Regressor --> -4.925 ± 0.403 ~
    

    
    
**Analyze the results**

```python
# Plot successive halving results
atom.plot_successive_halving()
```


![png](img/successive_halving.png)

