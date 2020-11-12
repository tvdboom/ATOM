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
      <th>86433</th>
      <td>Cairns</td>
      <td>14.7</td>
      <td>26.4</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>7.6</td>
      <td>ESE</td>
      <td>35.0</td>
    </tr>
    <tr>
      <th>5964</th>
      <td>Cobar</td>
      <td>23.8</td>
      <td>39.9</td>
      <td>0.0</td>
      <td>12.6</td>
      <td>13.2</td>
      <td>S</td>
      <td>31.0</td>
    </tr>
    <tr>
      <th>18064</th>
      <td>NorahHead</td>
      <td>21.8</td>
      <td>26.7</td>
      <td>4.2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NNE</td>
      <td>48.0</td>
    </tr>
    <tr>
      <th>107088</th>
      <td>Albany</td>
      <td>19.5</td>
      <td>25.9</td>
      <td>0.0</td>
      <td>5.2</td>
      <td>12.2</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>79530</th>
      <td>Dartmoor</td>
      <td>9.6</td>
      <td>14.2</td>
      <td>11.8</td>
      <td>1.4</td>
      <td>0.1</td>
      <td>SSW</td>
      <td>37.0</td>
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
    Time elapsed: 0.083s
    Bagging -----------------------------------------
    Evaluation --> f1: 0.6078 ± 0.0048
    Time elapsed: 0.353s
    -------------------------------------------------
    Total time: 0.446s
    
    
    Final results ========================= >>
    Duration: 0.448s
    ------------------------------------------
    Logistic Regression --> f1: 0.608 ± 0.005
    

## Class weights


```python
# Add class weights through the est_params parameter
atom.run("LR_cw", est_params={"class_weight": atom.get_class_weight()}, bagging=5)
```

    
    Training ===================================== >>
    Models: LR_cw
    Metric: f1
    
    
    Results for Logistic Regression:         
    Fit ---------------------------------------------
    Train evaluation --> f1: 0.6449
    Test evaluation --> f1: 0.6472
    Time elapsed: 0.080s
    Bagging -----------------------------------------
    Evaluation --> f1: 0.6483 ± 0.0014
    Time elapsed: 0.334s
    -------------------------------------------------
    Total time: 0.423s
    
    
    Final results ========================= >>
    Duration: 0.424s
    ------------------------------------------
    Logistic Regression --> f1: 0.648 ± 0.001
    

## Oversampling & undersampling


```python
# Since we are going to use two different approaches that
# change the dataset, we will split atom in two branches

# We will use the current branch later for undersampling
atom.branch.rename("undersampling")
```

    Branch renamed successfully!
    


```python
atom.classes
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
      <td>3706</td>
      <td>2555</td>
      <td>1151</td>
    </tr>
  </tbody>
</table>
</div>




```python
# And we need to create a new branch for oversampling
atom.branch = "oversampling"
```

    New branch 'oversampling' successfully created!
    


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
    Time elapsed: 0.107s
    Bagging -----------------------------------------
    Evaluation --> f1: 0.6489 ± 0.0031
    Time elapsed: 0.405s
    -------------------------------------------------
    Total time: 0.523s
    
    
    Final results ========================= >>
    Duration: 0.524s
    ------------------------------------------
    Logistic Regression --> f1: 0.649 ± 0.003
    


```python
# Now, let's return to the undersampling branch
atom.branch = "undersampling"

# Note that here the data is still imbalanced!
atom.classes
```

    Switched to branch 'undersampling'.
    




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
      <td>3706</td>
      <td>2555</td>
      <td>1151</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Perform undersampling of the majority class
atom.balance(strategy='NearMiss', sampling_strategy=0.9)
```

    Undersampling with NearMiss...
     --> Removing 6479 rows from class: No.
    


```python
atom.run("LR_us", bagging=5)
```

    
    Training ===================================== >>
    Models: LR_us
    Metric: f1
    
    
    Results for Logistic Regression:         
    Fit ---------------------------------------------
    Train evaluation --> f1: 0.7829
    Test evaluation --> f1: 0.6061
    Time elapsed: 0.060s
    Bagging -----------------------------------------
    Evaluation --> f1: 0.6037 ± 0.0071
    Time elapsed: 0.239s
    -------------------------------------------------
    Total time: 0.305s
    
    
    Final results ========================= >>
    Duration: 0.306s
    ------------------------------------------
    Logistic Regression --> f1: 0.604 ± 0.007 ~
    

## Analyze results


```python
atom.plot_bagging()
```


![png](output_20_0.png)

