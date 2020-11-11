# Imbalanced datasets
------------------------------------

This example shows the different approaches we can take to handle imbalanced datasets.

The data used is a variation on the Australian weather dataset from [https://www.kaggle.com/jsphyg/weather-dataset-rattle-package](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package). The goal of this dataset is to predict whether or not it will rain tomorrow training a binay classifier on target `RainTomorrow`.

## Load the data


```python
# Import packages
import pandas as pd
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
      <th>116787</th>
      <td>PerthAirport</td>
      <td>15.1</td>
      <td>39.2</td>
      <td>0.0</td>
      <td>11.6</td>
      <td>13.2</td>
      <td>E</td>
      <td>48.0</td>
    </tr>
    <tr>
      <th>124452</th>
      <td>Walpole</td>
      <td>9.1</td>
      <td>22.7</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NNE</td>
      <td>41.0</td>
    </tr>
    <tr>
      <th>5086</th>
      <td>BadgerysCreek</td>
      <td>12.5</td>
      <td>28.4</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>ESE</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>77899</th>
      <td>Watsonia</td>
      <td>16.5</td>
      <td>26.3</td>
      <td>0.0</td>
      <td>10.6</td>
      <td>12.4</td>
      <td>WSW</td>
      <td>43.0</td>
    </tr>
    <tr>
      <th>124901</th>
      <td>Walpole</td>
      <td>13.1</td>
      <td>22.9</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>SSE</td>
      <td>39.0</td>
    </tr>
  </tbody>
</table>
</div>



## Run the pipeline


```python
# Initialize ATOM with the created dataset
atom = ATOMClassifier(X, n_rows=0.3, test_size=0.3, verbose=2, random_state=1)
atom.clean()
atom.impute()
atom.encode()
```

    << ================== ATOM ================== >>
    Algorithm task: binary classification.
    
    Dataset stats ================== >>
    Shape: (42658, 22)
    Missing values: 95216
    Categorical columns: 5
    Scaled: False
    -----------------------------------
    Train set size: 29861
    Test set size: 12797
    -----------------------------------
    Train set balance: No:Yes <==> 3.5:1.0
    Test set balance: No:Yes <==> 3.4:1.0
    -----------------------------------
    Distribution of classes:
    |     |   dataset |   train |   test |
    |:----|----------:|--------:|-------:|
    | No  |     33139 |   23247 |   9892 |
    | Yes |      9519 |    6614 |   2905 |
    
    Applying data cleaning...
     --> Label-encoding the target column.
    Fitting Imputer...
    Imputing missing values...
     --> Dropping 352 rows for containing less than 50% non-missing values.
     --> Dropping 92 rows due to missing values in feature MinTemp.
     --> Dropping 56 rows due to missing values in feature MaxTemp.
     --> Dropping 350 rows due to missing values in feature Rainfall.
     --> Dropping 17551 rows due to missing values in feature Evaporation.
     --> Dropping 3229 rows due to missing values in feature Sunshine.
     --> Dropping 1258 rows due to missing values in feature WindGustDir.
     --> Dropping 655 rows due to missing values in feature WindDir9am.
     --> Dropping 69 rows due to missing values in feature WindDir3pm.
     --> Dropping 73 rows due to missing values in feature Humidity9am.
     --> Dropping 20 rows due to missing values in feature Humidity3pm.
     --> Dropping 18 rows due to missing values in feature Pressure9am.
     --> Dropping 5 rows due to missing values in feature Pressure3pm.
     --> Dropping 1609 rows due to missing values in feature Cloud9am.
     --> Dropping 426 rows due to missing values in feature Cloud3pm.
    Fitting Encoder...
    Encoding categorical columns...
     --> LeaveOneOut-encoding feature Location. Contains 26 unique classes.
     --> LeaveOneOut-encoding feature WindGustDir. Contains 16 unique classes.
     --> LeaveOneOut-encoding feature WindDir9am. Contains 16 unique classes.
     --> LeaveOneOut-encoding feature WindDir3pm. Contains 16 unique classes.
     --> Label-encoding feature RainToday. Contains 2 unique classes.
    


```python
# First, we fit a logistic regression model directly on the imbalanced data
atom.run("LR", metric="f1", bagging=5)
```

    
    Training ===================================== >>
    Models: LR
    Metric: f1
    
    
    Results for Logistic Regression:         
    Fit ---------------------------------------------
    Train evaluation --> f1: 0.6174
    Test evaluation --> f1: 0.6096
    Time elapsed: 0.078s
    Bagging -----------------------------------------
    Evaluation --> f1: 0.6078 ± 0.0048
    Time elapsed: 0.342s
    -------------------------------------------------
    Total time: 0.430s
    
    
    Final results ========================= >>
    Duration: 0.432s
    ------------------------------------------
    Logistic Regression --> f1: 0.608 ± 0.005
    

## Use weighted classes


```python
# Add the class weights through the est_params parameter
atom.run("LR_cw", est_params={"class_weight": atom.get_class_weight()}, bagging=5)
```

    
    Training ===================================== >>
    Models: LR_cw
    Metric: f1
    
    
    Results for Logistic Regression:         
    Fit ---------------------------------------------
    Train evaluation --> f1: 0.6174
    Test evaluation --> f1: 0.6096
    Time elapsed: 0.078s
    Bagging -----------------------------------------
    Evaluation --> f1: 0.6078 ± 0.0048
    Time elapsed: 0.338s
    -------------------------------------------------
    Total time: 0.422s
    
    
    Final results ========================= >>
    Duration: 0.422s
    ------------------------------------------
    Logistic Regression --> f1: 0.608 ± 0.005
    

## Use sample weights


```python
# Remember to add "_fit" to the est_params key to add the parameter to the fit method
atom.run("LR_sw", est_params={"sample_weight_fit": atom.get_sample_weight()}, bagging=5)
```

    
    Training ===================================== >>
    Models: LR_sw
    Metric: f1
    
    
    Results for Logistic Regression:         
    Fit ---------------------------------------------
    Train evaluation --> f1: 0.6174
    Test evaluation --> f1: 0.6096
    Time elapsed: 0.076s
    Bagging -----------------------------------------
    Evaluation --> f1: 0.6078 ± 0.0048
    Time elapsed: 0.343s
    -------------------------------------------------
    Total time: 0.426s
    
    
    Final results ========================= >>
    Duration: 0.427s
    ------------------------------------------
    Logistic Regression --> f1: 0.608 ± 0.005
    

## Use oversampling


```python
# Perform oversampling of the minority class
atom.balance(strategy='smote', sampling_strategy=0.9)
```

    Oversampling with SMOTE...
     --> Adding 5830 rows to class: Yes.
    


```python
atom.classes  # Note the balanced training set!
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
      <th>dataset</th>
      <th>train</th>
      <th>test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>13189</td>
      <td>9317</td>
      <td>3872</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9536</td>
      <td>8385</td>
      <td>1151</td>
    </tr>
  </tbody>
</table>
</div>




```python
atom.run("LR_os", bagging=5)
```

    
    Training ===================================== >>
    Models: LR_os
    Metric: f1
    
    
    Results for Logistic Regression:         
    Fit ---------------------------------------------
    Train evaluation --> f1: 0.7918
    Test evaluation --> f1: 0.6505
    Time elapsed: 0.093s
    Bagging -----------------------------------------
    Evaluation --> f1: 0.6489 ± 0.0031
    Time elapsed: 0.400s
    -------------------------------------------------
    Total time: 0.504s
    
    
    Final results ========================= >>
    Duration: 0.505s
    ------------------------------------------
    Logistic Regression --> f1: 0.649 ± 0.003
    

## Analyze results


```python
# Clearly, oversampling proves to be the best way to handle this imbalanced dataset
atom.plot_bagging()
```


![png](output_16_0.png)

