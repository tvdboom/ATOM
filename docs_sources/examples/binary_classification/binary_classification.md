# Binary classification
---------------------------------

This example shows how we can use ATOM to perform a variety of data cleaning steps in order to prepare the data for modelling. Then, we compare the prediction performance of an Extra-Trees and a Random Forest.

The data used is a variation on the Australian weather dataset from [https://www.kaggle.com/jsphyg/weather-dataset-rattle-package](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package). The goal of this dataset is to predict whether or not it will rain tomorrow training a binay classifier on target `RainTomorrow`.

## Load the data


```python
# Import packages
import pandas as pd
from sklearn.metrics import fbeta_score
from atom import ATOMClassifier
```


```python
# Load data
X = pd.read_csv('./datasets/weatherAUS.csv')

# Let's have a look at a subset of the data
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
      <th>Location</th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustDir</th>
      <th>WindGustSpeed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>65083</th>
      <td>MelbourneAirport</td>
      <td>9.3</td>
      <td>13.9</td>
      <td>3.8</td>
      <td>2.4</td>
      <td>3.3</td>
      <td>S</td>
      <td>48.0</td>
    </tr>
    <tr>
      <th>105643</th>
      <td>Woomera</td>
      <td>5.7</td>
      <td>17.4</td>
      <td>0.0</td>
      <td>2.2</td>
      <td>NaN</td>
      <td>W</td>
      <td>54.0</td>
    </tr>
    <tr>
      <th>98697</th>
      <td>MountGambier</td>
      <td>6.7</td>
      <td>20.9</td>
      <td>0.0</td>
      <td>6.4</td>
      <td>10.9</td>
      <td>S</td>
      <td>31.0</td>
    </tr>
    <tr>
      <th>106637</th>
      <td>Albany</td>
      <td>14.4</td>
      <td>20.2</td>
      <td>0.0</td>
      <td>4.2</td>
      <td>11.2</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>69629</th>
      <td>Mildura</td>
      <td>10.7</td>
      <td>27.9</td>
      <td>0.0</td>
      <td>5.8</td>
      <td>11.3</td>
      <td>SE</td>
      <td>30.0</td>
    </tr>
  </tbody>
</table>
</div>



## Run the pipeline


```python
# Call ATOM using only 5% of the complete dataset (for explanatory purposes)
atom = ATOMClassifier(X, y='RainTomorrow', n_rows=0.05, n_jobs=8, warnings=False, verbose=2, random_state=1)
```

    << ================== ATOM ================== >>
    Algorithm task: binary classification.
    Parallel processing with 8 cores.
    Applying data cleaning...
    
    Dataset stats ================= >>
    Shape: (7110, 22)
    Missing values: 14621
    Categorical columns: 5
    Scaled: False
    ----------------------------------
    Size of training set: 5688
    Size of test set: 1422
    ----------------------------------
    Class balance: No:Yes <==> 3.8:1.0
    Instances in RainTomorrow per class:
    |        |    total |    train_set |    test_set |
    |:-------|---------:|-------------:|------------:|
    | 0: No  |     5615 |         4473 |        1142 |
    | 1: Yes |     1495 |         1215 |         280 |
    
    


```python
# We can change the data properties in the pipeline
# Note that we can only replace the property with a new df
new_train = atom.X
new_train.insert(loc=3, column='AvgTemp', value=(atom.X['MaxTemp'] + atom.X['MinTemp'])/2)
atom.X = new_train

# This will automatically update all other data properties
assert 'AvgTemp' in atom.dataset
```


```python
# Impute missing values
atom.impute(strat_num='knn', strat_cat='remove', min_frac_rows=0.8)
```

    Fitting Imputer...
    Imputing missing values...
     --> Dropping 778 rows for containing less than 80% non-missing values.
     --> Imputing 5 missing values using the KNN imputer in feature MinTemp.
     --> Imputing 3 missing values using the KNN imputer in feature MaxTemp.
     --> Imputing 8 missing values using the KNN imputer in feature AvgTemp.
     --> Imputing 31 missing values using the KNN imputer in feature Rainfall.
     --> Imputing 2314 missing values using the KNN imputer in feature Evaporation.
     --> Imputing 2645 missing values using the KNN imputer in feature Sunshine.
     --> Imputing 201 missing values with remove in feature WindGustDir.
     --> Imputing 199 missing values using the KNN imputer in feature WindGustSpeed.
     --> Imputing 365 missing values with remove in feature WindDir9am.
     --> Imputing 24 missing values with remove in feature WindDir3pm.
     --> Imputing 4 missing values using the KNN imputer in feature WindSpeed9am.
     --> Imputing 3 missing values using the KNN imputer in feature WindSpeed3pm.
     --> Imputing 23 missing values using the KNN imputer in feature Humidity9am.
     --> Imputing 55 missing values using the KNN imputer in feature Humidity3pm.
     --> Imputing 42 missing values using the KNN imputer in feature Pressure9am.
     --> Imputing 40 missing values using the KNN imputer in feature Pressure3pm.
     --> Imputing 2112 missing values using the KNN imputer in feature Cloud9am.
     --> Imputing 2198 missing values using the KNN imputer in feature Cloud3pm.
     --> Imputing 5 missing values using the KNN imputer in feature Temp9am.
     --> Imputing 32 missing values using the KNN imputer in feature Temp3pm.
     --> Imputing 31 missing values with remove in feature RainToday.
    


```python
# Encode the categorical features
atom.encode(max_onehot=10, frac_to_other=0.04)
```

    Fitting Encoder...
    Encoding categorical columns...
     --> Target-encoding feature Location. Contains 1 unique categories.
     --> Target-encoding feature WindGustDir. Contains 17 unique categories.
     --> Target-encoding feature WindDir9am. Contains 17 unique categories.
     --> Target-encoding feature WindDir3pm. Contains 17 unique categories.
     --> One-hot-encoding feature RainToday. Contains 3 unique categories.
    


```python
# Perform undersampling of the majority class
atom.balance(oversample=None, undersample=0.9)
atom.stats()  # Note the balanced training set
```

    Performing undersampling...
     --> Removing 3145 rows from category: No.
    
    Dataset stats ================= >>
    Shape: (3965, 24)
    Scaled: False
    ----------------------------------
    Size of training set: 2543
    Size of test set: 1422
    ----------------------------------
    Class balance: No:Yes <==> 1.7:1.0
    Instances in RainTomorrow per class:
    |        |    total |    train_set |    test_set |
    |:-------|---------:|-------------:|------------:|
    | 0: No  |     2473 |         1338 |        1135 |
    | 1: Yes |     1492 |         1205 |         287 |
    
    


```python
# Define a custom metric_
def f2_score(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=2)

# Fit the EXtra-Trees and Random Forest to the data
atom.run(models=['et', 'rf'],
         metric=f2_score,
         n_calls=0,
         bagging=5,
         verbose=1)
```

    
    Running pipeline ============================= >>
    Models in pipeline: ET, RF
    Metric: f2_score
    
    
    Results for Extra-Trees:         
    Fitting -----------------------------------------
    Score on the train set --> f2_score: 1.0000
    Score on the test set  --> f2_score: 0.7509
    Time elapsed: 0.345s
    Bagging -----------------------------------------
    Score --> f2_score: 0.7165 ± 0.0036
    Time elapsed: 1.209s
    -------------------------------------------------
    Total time: 1.559s
    
    
    Results for Random Forest:         
    Fitting -----------------------------------------
    Score on the train set --> f2_score: 1.0000
    Score on the test set  --> f2_score: 0.7194
    Time elapsed: 0.449s
    Bagging -----------------------------------------
    Score --> f2_score: 0.6933 ± 0.0088
    Time elapsed: 1.718s
    -------------------------------------------------
    Total time: 2.172s
    
    
    Final results ========================= >>
    Duration: 3.733s
    ------------------------------------------
    Extra-Trees   --> f2_score: 0.717 ± 0.004 ~ !
    Random Forest --> f2_score: 0.693 ± 0.009 ~
    

## Analyze the results


```python
# Let's have a look at the final scoring
atom.scoring()

# The winning model is indicated with a ! and can be accessed through the winner attribute
# The ~ indicates that the model is probably overfitting. If we look at the train and test
# score we see a difference of more than 20%
print(f'\n\nAnd the winner is the {atom.winner.longname} model!!')
print('Score on the training set: ', atom.winner.metric_train)
print('Score on the test set: ', atom.winner.metric_test)
```

    Results ===================== >>
    Extra-Trees   --> f2_score: 0.717 ± 0.004 ~
    Random Forest --> f2_score: 0.693 ± 0.009 ~
    
    
    And the winner is the Extra-Trees model!!
    Score on the training set:  1.0
    Score on the test set:  0.7508630609896432
    

**We can make many plots to check the performance of the models**


```python
# The probabilties plot shows the distribution of predicted
# probabilities for the positive class
atom.winner.plot_probabilities()
```


![png](output_14_0.png)



```python
# The threshold plot let us compare how different metrics
# perform for different thresholds
atom.winner.plot_threshold(metric=['f1', 'accuracy', 'average_precision'], steps=50, filename='thresholds.png')
```


![png](output_15_0.png)



```python
# The ROC and PRC curve are also typical ways of measuring performance 
atom.plot_roc(title="ROC for the LightGBM vs CatBoost model")
atom.plot_prc(title="PRC comparison of the models")
```


![png](output_16_0.png)



![png](output_16_1.png)

