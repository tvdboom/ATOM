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


```python
# Let's have a look at the dataset
X.head()
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
      <th>Location</th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustDir</th>
      <th>WindGustSpeed</th>
      <th>WindDir9am</th>
      <th>WindDir3pm</th>
      <th>WindSpeed9am</th>
      <th>WindSpeed3pm</th>
      <th>Humidity9am</th>
      <th>Humidity3pm</th>
      <th>Pressure9am</th>
      <th>Pressure3pm</th>
      <th>Cloud9am</th>
      <th>Cloud3pm</th>
      <th>Temp9am</th>
      <th>Temp3pm</th>
      <th>RainToday</th>
      <th>RainTomorrow</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Albury</td>
      <td>13.4</td>
      <td>22.9</td>
      <td>0.6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>W</td>
      <td>44.0</td>
      <td>W</td>
      <td>WNW</td>
      <td>20.0</td>
      <td>24.0</td>
      <td>71.0</td>
      <td>22.0</td>
      <td>1007.7</td>
      <td>1007.1</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>16.9</td>
      <td>21.8</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Albury</td>
      <td>7.4</td>
      <td>25.1</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WNW</td>
      <td>44.0</td>
      <td>NNW</td>
      <td>WSW</td>
      <td>4.0</td>
      <td>22.0</td>
      <td>44.0</td>
      <td>25.0</td>
      <td>1010.6</td>
      <td>1007.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.2</td>
      <td>24.3</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Albury</td>
      <td>12.9</td>
      <td>25.7</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WSW</td>
      <td>46.0</td>
      <td>W</td>
      <td>WSW</td>
      <td>19.0</td>
      <td>26.0</td>
      <td>38.0</td>
      <td>30.0</td>
      <td>1007.6</td>
      <td>1008.7</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>21.0</td>
      <td>23.2</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Albury</td>
      <td>9.2</td>
      <td>28.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NE</td>
      <td>24.0</td>
      <td>SE</td>
      <td>E</td>
      <td>11.0</td>
      <td>9.0</td>
      <td>45.0</td>
      <td>16.0</td>
      <td>1017.6</td>
      <td>1012.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>18.1</td>
      <td>26.5</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Albury</td>
      <td>17.5</td>
      <td>32.3</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>W</td>
      <td>41.0</td>
      <td>ENE</td>
      <td>NW</td>
      <td>7.0</td>
      <td>20.0</td>
      <td>82.0</td>
      <td>33.0</td>
      <td>1010.8</td>
      <td>1006.0</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>17.8</td>
      <td>29.7</td>
      <td>No</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>



**Run the pipeline**


```python
# Call ATOM using only a percentage of the complete dataset (for explanatory purposes)
atom = ATOMClassifier(X, y="RainTomorrow", percentage=5, log='auto', n_jobs=2, verbose=3)
```

    <<=============== ATOM ===============>>
    Parallel processing with 2 cores.
    Initial data cleaning...
    Algorithm task: binary classification.
    
    Dataset stats ===================>
    Shape: (7110, 22)
    Missing values: 15836
    Categorical columns: 5
    Scaled: False
    ----------------------------------
    Size of training set: 4977
    Size of test set: 2133
    ----------------------------------
    Class balance: No:Yes <==> 3.6:1.0
    Instances in RainTomorrow per class:
    |        |    total |    train_set |    test_set |
    |:-------|---------:|-------------:|------------:|
    | 0: No  |     5562 |         3896 |        1666 |
    | 1: Yes |     1548 |         1081 |         467 |
    
    


```python
# If we change a column during the pre-processing,
# we need to call the update method to update all data attributes

atom.X['MaxTemp'] = np.log(atom.X['MaxTemp'])  # Random operator on column MaxTemp

# MaxTemp has now been changed for atom.X, but not in atom.X_train, atom.dataset, etc...
# To do so, we use the update method, where the parameter is a string of the changed attribute
atom.update('X')

assert atom.X['MaxTemp'].equals(atom.dataset['MaxTemp'])
```


```python
# Impute missing values
atom.impute(strat_num='knn', strat_cat='remove', min_frac_rows=0.8)
```

    Imputing missing values...
     --> Removing 736 rows for containing too many missing values.
     --> Imputing 10 missing values using the KNN imputer in feature MinTemp.
     --> Imputing 2 missing values using the KNN imputer in feature MaxTemp.
     --> Imputing 35 missing values using the KNN imputer in feature Rainfall.
     --> Imputing 2365 missing values using the KNN imputer in feature Evaporation.
     --> Imputing 2666 missing values using the KNN imputer in feature Sunshine.
     --> Removing 224 rows due to missing values in feature WindGustDir.
     --> Imputing 223 missing values using the KNN imputer in feature WindGustSpeed.
     --> Removing 327 rows due to missing values in feature WindDir9am.
     --> Removing 26 rows due to missing values in feature WindDir3pm.
     --> Imputing 4 missing values using the KNN imputer in feature WindSpeed9am.
     --> Imputing 5 missing values using the KNN imputer in feature WindSpeed3pm.
     --> Imputing 25 missing values using the KNN imputer in feature Humidity9am.
     --> Imputing 59 missing values using the KNN imputer in feature Humidity3pm.
     --> Imputing 55 missing values using the KNN imputer in feature Pressure9am.
     --> Imputing 52 missing values using the KNN imputer in feature Pressure3pm.
     --> Imputing 2127 missing values using the KNN imputer in feature Cloud9am.
     --> Imputing 2209 missing values using the KNN imputer in feature Cloud3pm.
     --> Imputing 5 missing values using the KNN imputer in feature Temp9am.
     --> Imputing 41 missing values using the KNN imputer in feature Temp3pm.
     --> Removing 35 rows due to missing values in feature RainToday.
    


```python
# Encode the categorical features
atom.encode(max_onehot=10, frac_to_other=0.04)
```

    Encoding categorical features...
     --> Target-encoding feature Location.  Contains 1 unique categories.
     --> Target-encoding feature WindGustDir.  Contains 16 unique categories.
     --> Target-encoding feature WindDir9am.  Contains 16 unique categories.
     --> Target-encoding feature WindDir3pm.  Contains 16 unique categories.
     --> Label-encoding feature RainToday. Contains 2 unique categories.
    


```python
# Perform undersampling of the majority class to balance the dataset
atom.balance(undersample=0.8)
```

    Performing undersampling...
     --> Removing 239 rows from class No.
    

    Using TensorFlow backend.
    


```python
# Remove outliers from the training set
atom.outliers(max_sigma=5)
```

    Handling outliers...
     --> Dropping 22 rows due to outliers.
    


```python
# Select only the best 10 features
atom.feature_selection(strategy="univariate", max_features=15, max_correlation=0.8)

# See which features were removed due to collinearity
atom.collinear
```

    Performing feature selection...
     --> Feature Pressure3pm was removed due to collinearity with another feature.
     --> Feature Temp9am was removed due to collinearity with another feature.
     --> Feature Temp3pm was removed due to collinearity with another feature.
    




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
      <td>0.95258</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Temp9am</td>
      <td>MinTemp, MaxTemp</td>
      <td>0.93135, 0.89859</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Temp3pm</td>
      <td>MaxTemp, Temp9am</td>
      <td>0.95784, 0.88146</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Change the verbosity of ATOM to not print too much details while fitting
atom.verbose = 2

# Define a custom metric
def f2_score(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=2)

# We can compare the performance of various gradient boosting algorithms
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
    Best hyperparameters: {'n_estimators': 249, 'learning_rate': 1.0, 'subsample': 0.7, 'max_depth': 2, 'max_features': 0.6, 'criterion': 'friedman_mse', 'min_samples_split': 20, 'min_samples_leaf': 11, 'ccp_alpha': 0.02}
    Best score on the BO: 0.7684
    Time elapsed: 6.856s
    Fitting -----------------------------------------
    Score on the training set: 0.7305
    Score on the test set: 0.6330
    Time elapsed: 0.491s
    Bagging -----------------------------------------
    Mean: 0.6086   Std: 0.0269
    Time elapsed: 2.117s
    -------------------------------------------------
    Total time: 9.469s
    
    
    Running BO for LightGBM...
    Final results for LightGBM:         
    Bayesian Optimization ---------------------------
    Best hyperparameters: {'n_estimators': 361, 'learning_rate': 0.8, 'max_depth': 1, 'num_leaves': 37, 'min_child_weight': 11, 'min_child_samples': 14, 'subsample': 1.0, 'colsample_bytree': 0.5, 'reg_alpha': 0.1, 'reg_lambda': 100.0}
    Best score on the BO: 0.7471
    Time elapsed: 3.781s
    Fitting -----------------------------------------
    Score on the training set: 0.7702
    Score on the test set: 0.6831
    Time elapsed: 0.139s
    Bagging -----------------------------------------
    Mean: 0.6671   Std: 0.0173
    Time elapsed: 0.258s
    -------------------------------------------------
    Total time: 4.181s
    
    
    Running BO for CatBoost...
    Final results for CatBoost:         
    Bayesian Optimization ---------------------------
    Best hyperparameters: {'n_estimators': 461, 'learning_rate': 0.38, 'max_depth': 3, 'subsample': 0.6, 'colsample_bylevel': 1.0, 'reg_lambda': 30.0}
    Best score on the BO: 0.7465
    Time elapsed: 24.624s
    Fitting -----------------------------------------
    Score on the training set: 0.9644
    Score on the test set: 0.6531
    Time elapsed: 0.596s
    Bagging -----------------------------------------
    Mean: 0.6606   Std: 0.0074
    Time elapsed: 2.815s
    -------------------------------------------------
    Total time: 28.040s
    
    
    Final results ================>>
    Duration: 41.841s
    Metric: f2_score
    --------------------------------
    Gradient Boosting Machine --> 0.609 ± 0.027
    LightGBM                  --> 0.667 ± 0.017 !!
    CatBoost                  --> 0.661 ± 0.007 ~
    

**Analyze the results**


```python
# Let's have a look at the best model
print('And the winner is...', atom.winner.longname)

# The ~ symbol indicates that the model is probably overfitting the training set
# This happens because we only use 5% of the dataset
# We can see that the training score is >20% of the test score
print('Score on the training set: ', atom.winner.score_train)
print('Score on the test set: ', atom.winner.score_test)
```

    And the winner is... LightGBM
    Score on the training set:  0.7702407002188184
    Score on the test set:  0.6830769230769231
    


```python
# Check the winner's confusion matrix and probability distribution
atom.winner.plot_confusion_matrix(normalize=True, figsize=(7, 7), filename='confusion_matrix.png')
atom.winner.plot_probabilities()

# How do other metrics perform for different thresholds on the winning model
atom.winner.plot_threshold(metric=['f1', 'accuracy', 'average_precision'], steps=50, filename='thresholds.png')

# Change plots aesthetics
ATOMClassifier.set_style('whitegrid')
ATOMClassifier.set_title_fontsize(24)

# Make some plots to compare the models
atom.plot_ROC(title="ROC for the LightGBM vs CatBoost model")
atom.plot_PRC(title="PRC comparison of the models")
```


![png](img/examples/output_15_0.png)



![png](img/examples/output_15_1.png)



![png](img/examples/output_15_2.png)



![png](img/examples/output_15_3.png)



![png](img/examples/output_15_4.png)



![png](img/examples/output_15_5.png)



![png](img/examples/output_15_6.png)






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
from sklearn.datasets import load_wine
from atom import ATOMClassifier

# Load the dataset's features and targets
dataset = load_wine()

# Convert to pd.DataFrame to get the names of the features
data = np.c_[dataset.data, dataset.target]
columns = np.append(dataset.feature_names, ["target"])
data = pd.DataFrame(data, columns=columns)
X = data.drop('target', axis=1)
y = data['target']
```


```python
# Let's have a look at the dataset
X.head()
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
      <th>proanthocyanins</th>
      <th>color_intensity</th>
      <th>hue</th>
      <th>od280/od315_of_diluted_wines</th>
      <th>proline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14.23</td>
      <td>1.71</td>
      <td>2.43</td>
      <td>15.6</td>
      <td>127.0</td>
      <td>2.80</td>
      <td>3.06</td>
      <td>0.28</td>
      <td>2.29</td>
      <td>5.64</td>
      <td>1.04</td>
      <td>3.92</td>
      <td>1065.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13.20</td>
      <td>1.78</td>
      <td>2.14</td>
      <td>11.2</td>
      <td>100.0</td>
      <td>2.65</td>
      <td>2.76</td>
      <td>0.26</td>
      <td>1.28</td>
      <td>4.38</td>
      <td>1.05</td>
      <td>3.40</td>
      <td>1050.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13.16</td>
      <td>2.36</td>
      <td>2.67</td>
      <td>18.6</td>
      <td>101.0</td>
      <td>2.80</td>
      <td>3.24</td>
      <td>0.30</td>
      <td>2.81</td>
      <td>5.68</td>
      <td>1.03</td>
      <td>3.17</td>
      <td>1185.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14.37</td>
      <td>1.95</td>
      <td>2.50</td>
      <td>16.8</td>
      <td>113.0</td>
      <td>3.85</td>
      <td>3.49</td>
      <td>0.24</td>
      <td>2.18</td>
      <td>7.80</td>
      <td>0.86</td>
      <td>3.45</td>
      <td>1480.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13.24</td>
      <td>2.59</td>
      <td>2.87</td>
      <td>21.0</td>
      <td>118.0</td>
      <td>2.80</td>
      <td>2.69</td>
      <td>0.39</td>
      <td>1.82</td>
      <td>4.32</td>
      <td>1.04</td>
      <td>2.93</td>
      <td>735.0</td>
    </tr>
  </tbody>
</table>
</div>



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
    Class balance: 0:1:2 <==> 1.2:1.5:1.0
    Instances in target per class:
    |    |    total |    train_set |    test_set |
    |---:|---------:|-------------:|------------:|
    |  0 |       59 |           40 |          19 |
    |  1 |       71 |           53 |          18 |
    |  2 |       48 |           31 |          17 |
    
    
    Running pipeline =================>
    Models in pipeline: LDA, RF, lSVM
    Metric: f1_macro
    
    
    Running BO for Linear Discriminant Analysis...
    Initial point: 1 --------------------------------
    Parameters --> {'solver': 'svd'}
    Evaluation --> f1_macro: 0.9754
    Time elapsed: 0.774s   Total time: 0.774s
    Initial point: 2 --------------------------------
    Parameters --> {'solver': 'eigen', 'shrinkage': 0.3}
    Evaluation --> f1_macro: 0.6488
    Time elapsed: 0.477s   Total time: 1.251s
    Initial point: 3 --------------------------------
    Parameters --> {'solver': 'svd'}
    Evaluation --> f1_macro: 0.9754
    Time elapsed: 0.019s   Total time: 1.270s
    Iteration: 1 ------------------------------------
    Parameters --> {'solver': 'svd'}
    Evaluation --> f1_macro: 0.9754
    Time elapsed: 0.019s   Total time: 1.485s
    Iteration: 2 ------------------------------------
    Parameters --> {'solver': 'svd'}
    Evaluation --> f1_macro: 0.9754
    Time elapsed: 0.024s   Total time: 1.690s
    Iteration: 3 ------------------------------------
    Parameters --> {'solver': 'eigen', 'shrinkage': 0.7}
    Evaluation --> f1_macro: 0.6951
    Time elapsed: 0.020s   Total time: 1.895s
    Iteration: 4 ------------------------------------
    Parameters --> {'solver': 'lsqr', 'shrinkage': 1.0}
    Evaluation --> f1_macro: 0.7274
    Time elapsed: 0.024s   Total time: 2.107s
    
    Final results for Linear Discriminant Analysis:         
    Bayesian Optimization ---------------------------
    Best hyperparameters: {'solver': 'svd'}
    Best score on the BO: 0.9754
    Time elapsed: 2.240s
    Fitting -----------------------------------------
    Score on the training set: 1.0000
    Score on the test set: 1.0000
    Time elapsed: 0.043s
    Bagging -----------------------------------------
    Mean: 1.0000   Std: 0.0000
    Time elapsed: 0.039s
    -------------------------------------------------
    Total time: 2.324s
    
    
    Running BO for Random Forest...
    Initial point: 1 --------------------------------
    Parameters --> {'n_estimators': 361, 'max_depth': 7, 'max_features': 0.7, 'criterion': 'gini', 'min_samples_split': 6, 'min_samples_leaf': 2, 'ccp_alpha': 0.03, 'bootstrap': True, 'max_samples': 0.7}
    Evaluation --> f1_macro: 0.9377
    Time elapsed: 0.497s   Total time: 0.498s
    Initial point: 2 --------------------------------
    Parameters --> {'n_estimators': 39, 'max_depth': 7, 'max_features': 0.6, 'criterion': 'gini', 'min_samples_split': 17, 'min_samples_leaf': 1, 'ccp_alpha': 0.035, 'bootstrap': True, 'max_samples': 0.8}
    Evaluation --> f1_macro: 0.9454
    Time elapsed: 0.271s   Total time: 0.769s
    Initial point: 3 --------------------------------
    Parameters --> {'n_estimators': 62, 'max_depth': 9, 'max_features': 0.9, 'criterion': 'entropy', 'min_samples_split': 19, 'min_samples_leaf': 4, 'ccp_alpha': 0.005, 'bootstrap': False}
    Evaluation --> f1_macro: 0.8743
    Time elapsed: 0.274s   Total time: 1.044s
    Iteration: 1 ------------------------------------
    Parameters --> {'n_estimators': 40, 'max_depth': 7, 'max_features': 0.6, 'criterion': 'gini', 'min_samples_split': 17, 'min_samples_leaf': 1, 'ccp_alpha': 0.03, 'bootstrap': True, 'max_samples': 0.8}
    Evaluation --> f1_macro: 0.9229
    Time elapsed: 0.243s   Total time: 1.556s
    Iteration: 2 ------------------------------------
    Parameters --> {'n_estimators': 500, 'max_depth': 3, 'max_features': 0.5, 'criterion': 'gini', 'min_samples_split': 2, 'min_samples_leaf': 1, 'ccp_alpha': 0.035, 'bootstrap': True, 'max_samples': 0.9}
    Evaluation --> f1_macro: 0.9377
    Time elapsed: 0.624s   Total time: 2.408s
    Iteration: 3 ------------------------------------
    Parameters --> {'n_estimators': 500, 'max_depth': 3, 'max_features': 0.6, 'criterion': 'entropy', 'min_samples_split': 18, 'min_samples_leaf': 4, 'ccp_alpha': 0.02, 'bootstrap': False}
    Evaluation --> f1_macro: 0.9141
    Time elapsed: 0.754s   Total time: 3.437s
    Iteration: 4 ------------------------------------
    Parameters --> {'n_estimators': 499, 'max_depth': 5, 'max_features': 0.5, 'criterion': 'gini', 'min_samples_split': 2, 'min_samples_leaf': 7, 'ccp_alpha': 0.015, 'bootstrap': True, 'max_samples': 0.5}
    Evaluation --> f1_macro: 0.9460
    Time elapsed: 0.632s   Total time: 4.321s
    
    Final results for Random Forest:         
    Bayesian Optimization ---------------------------
    Best hyperparameters: {'n_estimators': 499, 'max_depth': 5, 'max_features': 0.5, 'criterion': 'gini', 'min_samples_split': 2, 'min_samples_leaf': 7, 'ccp_alpha': 0.015, 'bootstrap': True, 'max_samples': 0.5}
    Best score on the BO: 0.9460
    Time elapsed: 4.527s
    Fitting -----------------------------------------
    Score on the training set: 0.9758
    Score on the test set: 1.0000
    Time elapsed: 1.150s
    Bagging -----------------------------------------
    Mean: 0.9817   Std: 0.0184
    Time elapsed: 6.119s
    -------------------------------------------------
    Total time: 11.797s
    
    
    Running BO for Linear SVM...
    Initial point: 1 --------------------------------
    Parameters --> {'C': 0.1, 'loss': 'squared_hinge', 'dual': False, 'penalty': 'l1'}
    Evaluation --> f1_macro: 0.9680
    Time elapsed: 0.023s   Total time: 0.024s
    Initial point: 2 --------------------------------
    Parameters --> {'C': 0.1, 'loss': 'hinge', 'dual': True, 'penalty': 'l2'}
    Evaluation --> f1_macro: 0.9913
    Time elapsed: 0.020s   Total time: 0.045s
    Initial point: 3 --------------------------------
    Parameters --> {'C': 0.001, 'loss': 'hinge', 'dual': True, 'penalty': 'l2'}
    Evaluation --> f1_macro: 0.9215
    Time elapsed: 0.019s   Total time: 0.065s
    Iteration: 1 ------------------------------------
    Parameters --> {'C': 100, 'loss': 'squared_hinge', 'dual': False, 'penalty': 'l1'}
    Evaluation --> f1_macro: 0.9503
    Time elapsed: 0.025s   Total time: 0.282s
    Iteration: 2 ------------------------------------
    Parameters --> {'C': 0.001, 'loss': 'squared_hinge', 'dual': False, 'penalty': 'l1'}
    Evaluation --> f1_macro: 0.1626
    Time elapsed: 0.024s   Total time: 0.563s
    Iteration: 3 ------------------------------------
    Parameters --> {'C': 100, 'loss': 'hinge', 'dual': True, 'penalty': 'l2'}
    Evaluation --> f1_macro: 0.9503
    Time elapsed: 0.021s   Total time: 0.806s
    Iteration: 4 ------------------------------------
    Parameters --> {'C': 100, 'loss': 'hinge', 'dual': True, 'penalty': 'l2'}
    Evaluation --> f1_macro: 0.9503
    Time elapsed: 0.021s   Total time: 1.056s
    
    Final results for Linear SVM:         
    Bayesian Optimization ---------------------------
    Best hyperparameters: {'C': 0.1, 'loss': 'hinge', 'dual': True, 'penalty': 'l2'}
    Best score on the BO: 0.9913
    Time elapsed: 1.247s
    Fitting -----------------------------------------
    Score on the training set: 0.9842
    Score on the test set: 1.0000
    Time elapsed: 0.032s
    Bagging -----------------------------------------
    Mean: 1.0000   Std: 0.0000
    Time elapsed: 0.030s
    -------------------------------------------------
    Total time: 1.309s
    
    
    Final results ================>>
    Duration: 15.434s
    Metric: f1_macro
    --------------------------------
    Linear Discriminant Analysis --> 1.000 ± 0.000 !!
    Random Forest                --> 0.982 ± 0.018
    Linear SVM                   --> 1.000 ± 0.000 !!
    

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
      <td>2.324s</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>0.043s</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.039s</td>
    </tr>
    <tr>
      <th>1</th>
      <td>RF</td>
      <td>11.797s</td>
      <td>0.975759</td>
      <td>1.0</td>
      <td>1.150s</td>
      <td>0.981706</td>
      <td>0.018371</td>
      <td>6.119s</td>
    </tr>
    <tr>
      <th>2</th>
      <td>lSVM</td>
      <td>1.309s</td>
      <td>0.984184</td>
      <td>1.0</td>
      <td>0.032s</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.030s</td>
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
    Linear Discriminant Analysis --> 1.000 !!
    Random Forest                --> 1.000 !!
    Linear SVM                   --> 1.000 !!
    


```python
atom.plot_bagging()
```


![png](img/examples/output_9_0.png)


**Let's have a closer look at the Random Forest**


```python
# Get the results on some other metrics
print('Jaccard score:', atom.rf.jaccard_weighted)
print('Recall score:', atom.rf.recall_macro)
```

    Jaccard score: 1.0
    Recall score: 1.0
    


```python
# Check the winner's confusion matrix
atom.RF.plot_confusion_matrix()
```


![png](img/examples/output_12_0.png)



![png](img/examples/output_12_1.png)



![png](img/examples/output_12_2.png)



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
```


```python
# Let's have a look at the dataset
X.head()
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
      <th>Sex</th>
      <th>Length</th>
      <th>Diameter</th>
      <th>Height</th>
      <th>Whole weight</th>
      <th>Shucked weight</th>
      <th>Viscera weight</th>
      <th>Shell weight</th>
      <th>Rings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M</td>
      <td>0.455</td>
      <td>0.365</td>
      <td>0.095</td>
      <td>0.5140</td>
      <td>0.2245</td>
      <td>0.1010</td>
      <td>0.150</td>
      <td>15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M</td>
      <td>0.350</td>
      <td>0.265</td>
      <td>0.090</td>
      <td>0.2255</td>
      <td>0.0995</td>
      <td>0.0485</td>
      <td>0.070</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>F</td>
      <td>0.530</td>
      <td>0.420</td>
      <td>0.135</td>
      <td>0.6770</td>
      <td>0.2565</td>
      <td>0.1415</td>
      <td>0.210</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M</td>
      <td>0.440</td>
      <td>0.365</td>
      <td>0.125</td>
      <td>0.5160</td>
      <td>0.2155</td>
      <td>0.1140</td>
      <td>0.155</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>I</td>
      <td>0.330</td>
      <td>0.255</td>
      <td>0.080</td>
      <td>0.2050</td>
      <td>0.0895</td>
      <td>0.0395</td>
      <td>0.055</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Initlaize ATOM for regression tasks and encode the categorical features
atom = ATOMRegressor(X, y="Rings", verbose=2, random_state=42)
atom.encode()
```

    <<=============== ATOM ===============>>
    Initial data cleaning...
    Algorithm task: regression.
    
    Dataset stats ===================>
    Shape: (4177, 9)
    Categorical columns: 1
    Scaled: False
    ----------------------------------
    Size of training set: 2923
    Size of test set: 1254
    
    Encoding categorical features...
    


```python
# Plot the dataset's correlation matrix
atom.plot_correlation()
```


![png](img/examples/output_5_0.png)



```python
# Apply PCA for dimensionality reduction
atom.feature_selection(strategy="pca", n_features=6)
atom.plot_PCA()
atom.plot_components(figsize=(8, 6), filename='atom_PCA_plot')
```

    Performing feature selection...
    


![png](img/examples/output_6_1.png)



![png](img/examples/output_6_2.png)


**Run the pipeline**


```python
atom.pipeline(['tree', 'bag', 'et'],
              metric='neg_mean_squared_error',
              max_iter=5,
              init_points=2,
              cv=1,
              bagging=5)
```

    
    Running pipeline =================>
    Models in pipeline: Tree, Bag, ET
    Metric: neg_mean_squared_error
    
    
    Running BO for Decision Tree...
    Final results for Decision Tree:         
    Bayesian Optimization ---------------------------
    Best hyperparameters: {'criterion': 'mse', 'splitter': 'random', 'max_depth': 10, 'max_features': 0.5, 'min_samples_split': 2, 'min_samples_leaf': 20, 'ccp_alpha': 0.0}
    Best score on the BO: -6.6417
    Time elapsed: 0.894s
    Fitting -----------------------------------------
    Score on the training set: -10.0595
    Score on the test set: -8.7809
    Time elapsed: 0.024s
    Bagging -----------------------------------------
    Mean: -7.5249   Std: 1.2204
    Time elapsed: 0.013s
    -------------------------------------------------
    Total time: 0.932s
    
    
    Running BO for Bagging Regressor...
    Final results for Bagging Regressor:         
    Bayesian Optimization ---------------------------
    Best hyperparameters: {'n_estimators': 93, 'max_samples': 0.5, 'max_features': 1.0, 'bootstrap': False, 'bootstrap_features': True}
    Best score on the BO: -5.2595
    Time elapsed: 7.585s
    Fitting -----------------------------------------
    Score on the training set: -1.4264
    Score on the test set: -4.9367
    Time elapsed: 0.971s
    Bagging -----------------------------------------
    Mean: -5.0745   Std: 0.0345
    Time elapsed: 3.621s
    -------------------------------------------------
    Total time: 12.178s
    
    
    Running BO for Extra-Trees...
    Final results for Extra-Trees:         
    Bayesian Optimization ---------------------------
    Best hyperparameters: {'n_estimators': 159, 'max_depth': 10, 'max_features': 0.9, 'criterion': 'mae', 'min_samples_split': 10, 'min_samples_leaf': 15, 'ccp_alpha': 0.01, 'bootstrap': True, 'max_samples': 0.8}
    Best score on the BO: -6.2369
    Time elapsed: 11.282s
    Fitting -----------------------------------------
    Score on the training set: -7.2610
    Score on the test set: -6.4164
    Time elapsed: 3.024s
    Bagging -----------------------------------------
    Mean: -6.2672   Std: 0.0772
    Time elapsed: 15.529s
    -------------------------------------------------
    Total time: 29.835s
    
    
    Final results ================>>
    Duration: 42.945s
    Metric: neg_mean_squared_error
    --------------------------------
    Decision Tree     --> -7.525 ± 1.220 ~
    Bagging Regressor --> -5.075 ± 0.035 !! ~
    Extra-Trees       --> -6.267 ± 0.077 ~
    

<br><br>

# Successive halving
---------------------------------

Import the boston dataset from [sklearn.datasets](https://scikit-learn.org/stable/datasets/index.html#boston-dataset).
 This is a small and easy to train dataset which goal is to predict house prices.

**Load the data**


```python
# Import packages
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from atom import ATOMRegressor

# Load the dataset's features and targets
dataset = load_boston()

# Convert to pd.DataFrame to get the names of the features
data = np.c_[dataset.data, dataset.target]
columns = np.append(dataset.feature_names, ["target"])
data = pd.DataFrame(data, columns=columns)
X = data.drop('target', axis=1)
y = data['target']
```


```python
# Let's have a look at the dataset
X.head()
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
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1.0</td>
      <td>296.0</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Initialize ATOM
atom = ATOMRegressor(X, y, verbose=1, random_state=42)

# Select best features with the RFECV strategy
atom.feature_selection('RFECV', solver='OLS', scoring='r2')
atom.plot_RFECV()
```

    <<=============== ATOM ===============>>
    Algorithm task: regression.
    


![png](img/examples/output_4_1.png)


**Run the pipeline**


```python
# We can compare tree-based models via successive halving
atom.successive_halving(['tree', 'bag', 'et', 'rf', 'lgb', 'catb'],
                        metric='neg_mean_squared_error',
                        max_iter=5,
                        init_points=2,
                        cv=1,
                        bagging=5)
```

                                                                                   

    
    Running pipeline =================>
    Metric: neg_mean_squared_error
    
    
    <<=============== Iteration 0 ==============>>
    Models in pipeline: Tree, Bag, ET, RF, LGB, CatB
    Percentage of data: 16.7%
    Size of training set: 58
    Size of test set: 26
    

    Processing: 100%|████████████████████████████████| 6/6 [00:12<00:00,  2.04s/it]
                                                                                   

    
    
    Final results ================>>
    Duration: 12.251s
    Metric: neg_mean_squared_error
    --------------------------------
    Decision Tree     --> -49.394 ± 27.206 ~
    Bagging Regressor --> -38.655 ± 3.232 ~
    Extra-Trees       --> -49.144 ± 13.045 ~
    Random Forest     --> -100.813 ± 101.754 ~
    LightGBM          --> -31.724 ± 8.447 !! ~
    
    
    <<=============== Iteration 1 ==============>>
    Models in pipeline: Bag, LGB
    Percentage of data: 50.0%
    Size of training set: 177
    Size of test set: 76
    

    Processing: 100%|████████████████████████████████| 2/2 [00:05<00:00,  2.58s/it]
                                                                                   

    
    
    Final results ================>>
    Duration: 17.454s
    Metric: neg_mean_squared_error
    --------------------------------
    Bagging Regressor --> -27.847 ± 4.189 !! ~
    LightGBM          --> -32.049 ± 6.159 ~
    
    
    <<=============== Iteration 2 ==============>>
    Model in pipeline: Bag
    Percentage of data: 100.0%
    Size of training set: 354
    Size of test set: 152
    

    Processing: 100%|████████████████████████████████| 1/1 [00:05<00:00,  5.73s/it]

    
    
    Final results ================>>
    Duration: 23.209s
    Metric: neg_mean_squared_error
    --------------------------------
    Bagging Regressor --> -16.360 ± 3.548 ~
    

    
    


```python
atom.plot_successive_halving()
```


![png](img/examples/output_7_0.png)


<br><br>


# Train sizing
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


```python
# Let's have a look at the dataset
X.head()
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
      <th>Location</th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustDir</th>
      <th>WindGustSpeed</th>
      <th>WindDir9am</th>
      <th>WindDir3pm</th>
      <th>...</th>
      <th>Humidity9am</th>
      <th>Humidity3pm</th>
      <th>Pressure9am</th>
      <th>Pressure3pm</th>
      <th>Cloud9am</th>
      <th>Cloud3pm</th>
      <th>Temp9am</th>
      <th>Temp3pm</th>
      <th>RainToday</th>
      <th>RainTomorrow</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Albury</td>
      <td>13.4</td>
      <td>22.9</td>
      <td>0.6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>W</td>
      <td>44.0</td>
      <td>W</td>
      <td>WNW</td>
      <td>...</td>
      <td>71.0</td>
      <td>22.0</td>
      <td>1007.7</td>
      <td>1007.1</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>16.9</td>
      <td>21.8</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Albury</td>
      <td>7.4</td>
      <td>25.1</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WNW</td>
      <td>44.0</td>
      <td>NNW</td>
      <td>WSW</td>
      <td>...</td>
      <td>44.0</td>
      <td>25.0</td>
      <td>1010.6</td>
      <td>1007.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.2</td>
      <td>24.3</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Albury</td>
      <td>12.9</td>
      <td>25.7</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WSW</td>
      <td>46.0</td>
      <td>W</td>
      <td>WSW</td>
      <td>...</td>
      <td>38.0</td>
      <td>30.0</td>
      <td>1007.6</td>
      <td>1008.7</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>21.0</td>
      <td>23.2</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Albury</td>
      <td>9.2</td>
      <td>28.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NE</td>
      <td>24.0</td>
      <td>SE</td>
      <td>E</td>
      <td>...</td>
      <td>45.0</td>
      <td>16.0</td>
      <td>1017.6</td>
      <td>1012.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>18.1</td>
      <td>26.5</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Albury</td>
      <td>17.5</td>
      <td>32.3</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>W</td>
      <td>41.0</td>
      <td>ENE</td>
      <td>NW</td>
      <td>...</td>
      <td>82.0</td>
      <td>33.0</td>
      <td>1010.8</td>
      <td>1006.0</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>17.8</td>
      <td>29.7</td>
      <td>No</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>



**Run the pipeline**


```python
# Initialize ATOM
atom = ATOMClassifier(X, verbose=1, random_state=42)
atom.impute(strat_num='median', strat_cat='most_frequent', min_frac_rows=0.8)
atom.encode()

# We can compare tree-based models via successive halving
atom.train_sizing('lgb',
                  metric='accuracy',
                  max_iter=5,
                  init_points=2,
                  cv=3)
```

    <<=============== ATOM ===============>>
    Algorithm task: binary classification.
    

                                                                                   

    
    Running pipeline =================>
    Model in pipeline: LGB
    Metric: accuracy
    
    
    <<=============== Iteration 0 ==============>>
    Percentage of data: 10.0%
    Size of training set: 8890
    Size of test set: 3811
    

    Processing: 100%|████████████████████████████████| 1/1 [00:08<00:00,  8.79s/it]
                                                                                   

    
    
    Final results ================>>
    Duration: 8.837s
    Metric: accuracy
    --------------------------------
    LightGBM --> 0.845
    
    
    <<=============== Iteration 1 ==============>>
    Percentage of data: 20.0%
    Size of training set: 17781
    Size of test set: 7621
    

    Processing: 100%|████████████████████████████████| 1/1 [00:09<00:00,  9.97s/it]
                                                                                   

    
    
    Final results ================>>
    Duration: 18.863s
    Metric: accuracy
    --------------------------------
    LightGBM --> 0.837
    
    
    <<=============== Iteration 2 ==============>>
    Percentage of data: 30.0%
    Size of training set: 26672
    Size of test set: 11431
    

    Processing: 100%|████████████████████████████████| 1/1 [00:17<00:00, 17.26s/it]
                                                                                   

    
    
    Final results ================>>
    Duration: 36.187s
    Metric: accuracy
    --------------------------------
    LightGBM --> 0.852
    
    
    <<=============== Iteration 3 ==============>>
    Percentage of data: 40.0%
    Size of training set: 35562
    Size of test set: 15242
    

    Processing: 100%|████████████████████████████████| 1/1 [00:29<00:00, 29.42s/it]
    

    
    
    Final results ================>>
    Duration: 1m:05s
    Metric: accuracy
    --------------------------------
    LightGBM --> 0.862
    

                                                                                   

    
    
    <<=============== Iteration 4 ==============>>
    Percentage of data: 50.0%
    Size of training set: 44454
    Size of test set: 19052
    

    Processing: 100%|████████████████████████████████| 1/1 [00:24<00:00, 24.44s/it]
    

    
    
    Final results ================>>
    Duration: 1m:30s
    Metric: accuracy
    --------------------------------
    LightGBM --> 0.857
    

                                                                                   

    
    
    <<=============== Iteration 5 ==============>>
    Percentage of data: 60.0%
    Size of training set: 53344
    Size of test set: 22863
    

    Processing: 100%|████████████████████████████████| 1/1 [00:30<00:00, 30.59s/it]
    

    
    
    Final results ================>>
    Duration: 2m:01s
    Metric: accuracy
    --------------------------------
    LightGBM --> 0.859
    

    Processing:   0%|                                        | 0/1 [00:00<?, ?it/s]

    
    
    <<=============== Iteration 6 ==============>>
    Percentage of data: 70.0%
    Size of training set: 62235
    Size of test set: 26673
    

    Processing: 100%|████████████████████████████████| 1/1 [00:47<00:00, 47.88s/it]
                                                                                   

    
    
    Final results ================>>
    Duration: 2m:50s
    Metric: accuracy
    --------------------------------
    LightGBM --> 0.862
    
    
    <<=============== Iteration 7 ==============>>
    Percentage of data: 80.0%
    Size of training set: 71126
    Size of test set: 30483
    

    Processing: 100%|████████████████████████████████| 1/1 [00:58<00:00, 58.07s/it]
    

    
    
    Final results ================>>
    Duration: 3m:48s
    Metric: accuracy
    --------------------------------
    LightGBM --> 0.854
    

                                                                                   

    
    
    <<=============== Iteration 8 ==============>>
    Percentage of data: 90.0%
    Size of training set: 80017
    Size of test set: 34293
    

    Processing: 100%|████████████████████████████████| 1/1 [00:57<00:00, 57.22s/it]
    

    
    
    Final results ================>>
    Duration: 4m:46s
    Metric: accuracy
    --------------------------------
    LightGBM --> 0.860
    

                                                                                   

    
    
    <<=============== Iteration 9 ==============>>
    Percentage of data: 100.0%
    Size of training set: 88907
    Size of test set: 38104
    

    Processing: 100%|████████████████████████████████| 1/1 [01:15<00:00, 75.39s/it]

    
    
    Final results ================>>
    Duration: 6m:02s
    Metric: accuracy
    --------------------------------
    LightGBM --> 0.855
    

    
    


```python
atom.plot_learning_curve()
```


![png](img/examples/output_6_0.png)

