# Train sizing
---------------------------------

This example shows how to asses a model's performance based on the size of the training set.

The data used is a variation on the Australian weather dataset from https://www.kaggle.com/jsphyg/weather-dataset-rattle-package. The goal of this dataset is to predict whether or not it will rain tomorrow training a binay classifier on target RainTomorrow.

## Load the data


```python
# Import packages
import numpy as np
import pandas as pd
from atom import ATOMClassifier
```


```python
# Load the Australian weather dataset
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
      <th>10829</th>
      <td>CoffsHarbour</td>
      <td>2.1</td>
      <td>17.7</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>15108</th>
      <td>Newcastle</td>
      <td>20.2</td>
      <td>29.7</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>117467</th>
      <td>PerthAirport</td>
      <td>11.7</td>
      <td>20.5</td>
      <td>0.4</td>
      <td>6.0</td>
      <td>8.4</td>
      <td>WSW</td>
      <td>57.0</td>
    </tr>
    <tr>
      <th>80685</th>
      <td>Dartmoor</td>
      <td>-0.8</td>
      <td>15.3</td>
      <td>0.0</td>
      <td>1.8</td>
      <td>7.2</td>
      <td>WSW</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>122490</th>
      <td>SalmonGums</td>
      <td>8.6</td>
      <td>21.2</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NW</td>
      <td>67.0</td>
    </tr>
  </tbody>
</table>
</div>



## Run the pipeline


```python
# Initialize ATOM and prepare the data
atom = ATOMClassifier(X, verbose=2, random_state=1)
atom.clean()
atom.impute(strat_num='median', strat_cat='most_frequent', min_frac_rows=0.8)
atom.encode()
```

    << ================== ATOM ================== >>
    Algorithm task: binary classification.
    
    Dataset stats ================== >>
    Shape: (142193, 22)
    Missing values: 316559
    Categorical columns: 5
    Scaled: False
    -----------------------------------
    Train set size: 113755
    Test set size: 28438
    -----------------------------------
    Dataset balance: No:Yes <==> 3.5:1.0
    -----------------------------------
    Distribution of classes:
    |     |   dataset |   train |   test |
    |:----|----------:|--------:|-------:|
    | No  |    110316 |   88263 |  22053 |
    | Yes |     31877 |   25492 |   6385 |
    
    Applying data cleaning...
     --> Label-encoding the target column.
    Fitting Imputer...
    Imputing missing values...
     --> Dropping 15182 rows for containing less than 80% non-missing values.
     --> Imputing 100 missing values with median in feature MinTemp.
     --> Imputing 57 missing values with median in feature MaxTemp.
     --> Imputing 640 missing values with median in feature Rainfall.
     --> Imputing 46535 missing values with median in feature Evaporation.
     --> Imputing 53034 missing values with median in feature Sunshine.
     --> Imputing 4381 missing values with most_frequent in feature WindGustDir.
     --> Imputing 4359 missing values with median in feature WindGustSpeed.
     --> Imputing 6624 missing values with most_frequent in feature WindDir9am.
     --> Imputing 612 missing values with most_frequent in feature WindDir3pm.
     --> Imputing 80 missing values with median in feature WindSpeed9am.
     --> Imputing 49 missing values with median in feature WindSpeed3pm.
     --> Imputing 532 missing values with median in feature Humidity9am.
     --> Imputing 1168 missing values with median in feature Humidity3pm.
     --> Imputing 1028 missing values with median in feature Pressure9am.
     --> Imputing 972 missing values with median in feature Pressure3pm.
     --> Imputing 42172 missing values with median in feature Cloud9am.
     --> Imputing 44251 missing values with median in feature Cloud3pm.
     --> Imputing 98 missing values with median in feature Temp9am.
     --> Imputing 702 missing values with median in feature Temp3pm.
     --> Imputing 640 missing values with most_frequent in feature RainToday.
    Fitting Encoder...
    Encoding categorical columns...
     --> LeaveOneOut-encoding feature Location. Contains 45 unique classes.
     --> LeaveOneOut-encoding feature WindGustDir. Contains 16 unique classes.
     --> LeaveOneOut-encoding feature WindDir9am. Contains 16 unique classes.
     --> LeaveOneOut-encoding feature WindDir3pm. Contains 16 unique classes.
     --> Label-encoding feature RainToday. Contains 2 unique classes.
    


```python
# We can analyze the impact of the training set's size on a LightGBM model
atom.train_sizing('lgb', train_sizes=np.linspace(0.1, 1, 9), bagging=4)
```

    
    Training ===================================== >>
    Models: LGB
    Metric: f1
    
    
    Run: 0 ================================ >>
    Size of training set: 10164 (10%)
    Size of test set: 25366
    
    
    Results for LightGBM:         
    Fit ---------------------------------------------
    Train evaluation --> f1: 0.8175
    Test evaluation --> f1: 0.6109
    Time elapsed: 0.715s
    Bagging -----------------------------------------
    Evaluation --> f1: 0.5928 ± 0.0033
    Time elapsed: 1.566s
    -------------------------------------------------
    Total time: 2.291s
    
    
    Final results ========================= >>
    Duration: 2.293s
    ------------------------------------------
    LightGBM --> f1: 0.593 ± 0.003 ~
    
    
    Run: 1 ================================ >>
    Size of training set: 21599 (21%)
    Size of test set: 25366
    
    
    Results for LightGBM:         
    Fit ---------------------------------------------
    Train evaluation --> f1: 0.7398
    Test evaluation --> f1: 0.6248
    Time elapsed: 0.950s
    Bagging -----------------------------------------
    Evaluation --> f1: 0.6159 ± 0.0042
    Time elapsed: 2.027s
    -------------------------------------------------
    Total time: 2.979s
    
    
    Final results ========================= >>
    Duration: 2.981s
    ------------------------------------------
    LightGBM --> f1: 0.616 ± 0.004
    
    
    Run: 2 ================================ >>
    Size of training set: 33034 (32%)
    Size of test set: 25366
    
    
    Results for LightGBM:         
    Fit ---------------------------------------------
    Train evaluation --> f1: 0.7035
    Test evaluation --> f1: 0.6314
    Time elapsed: 1.166s
    Bagging -----------------------------------------
    Evaluation --> f1: 0.6208 ± 0.0028
    Time elapsed: 2.432s
    -------------------------------------------------
    Total time: 3.599s
    
    
    Final results ========================= >>
    Duration: 3.601s
    ------------------------------------------
    LightGBM --> f1: 0.621 ± 0.003
    
    
    Run: 3 ================================ >>
    Size of training set: 44469 (44%)
    Size of test set: 25366
    
    
    Results for LightGBM:         
    Fit ---------------------------------------------
    Train evaluation --> f1: 0.6862
    Test evaluation --> f1: 0.6313
    Time elapsed: 1.304s
    Bagging -----------------------------------------
    Evaluation --> f1: 0.6277 ± 0.0047
    Time elapsed: 3.011s
    -------------------------------------------------
    Total time: 4.316s
    
    
    Final results ========================= >>
    Duration: 4.319s
    ------------------------------------------
    LightGBM --> f1: 0.628 ± 0.005
    
    
    Run: 4 ================================ >>
    Size of training set: 55904 (55%)
    Size of test set: 25366
    
    
    Results for LightGBM:         
    Fit ---------------------------------------------
    Train evaluation --> f1: 0.6803
    Test evaluation --> f1: 0.6396
    Time elapsed: 1.643s
    Bagging -----------------------------------------
    Evaluation --> f1: 0.6277 ± 0.0044
    Time elapsed: 3.521s
    -------------------------------------------------
    Total time: 5.165s
    
    
    Final results ========================= >>
    Duration: 5.167s
    ------------------------------------------
    LightGBM --> f1: 0.628 ± 0.004
    
    
    Run: 5 ================================ >>
    Size of training set: 67339 (66%)
    Size of test set: 25366
    
    
    Results for LightGBM:         
    Fit ---------------------------------------------
    Train evaluation --> f1: 0.6770
    Test evaluation --> f1: 0.6382
    Time elapsed: 1.928s
    Bagging -----------------------------------------
    Evaluation --> f1: 0.6365 ± 0.0018
    Time elapsed: 4.042s
    -------------------------------------------------
    Total time: 5.970s
    
    
    Final results ========================= >>
    Duration: 5.973s
    ------------------------------------------
    LightGBM --> f1: 0.637 ± 0.002
    
    
    Run: 6 ================================ >>
    Size of training set: 78774 (77%)
    Size of test set: 25366
    
    
    Results for LightGBM:         
    Fit ---------------------------------------------
    Train evaluation --> f1: 0.6752
    Test evaluation --> f1: 0.6398
    Time elapsed: 2.085s
    Bagging -----------------------------------------
    Evaluation --> f1: 0.6330 ± 0.0034
    Time elapsed: 4.610s
    -------------------------------------------------
    Total time: 6.697s
    
    
    Final results ========================= >>
    Duration: 6.700s
    ------------------------------------------
    LightGBM --> f1: 0.633 ± 0.003
    
    
    Run: 7 ================================ >>
    Size of training set: 90209 (89%)
    Size of test set: 25366
    
    
    Results for LightGBM:         
    Fit ---------------------------------------------
    Train evaluation --> f1: 0.6693
    Test evaluation --> f1: 0.6395
    Time elapsed: 2.356s
    Bagging -----------------------------------------
    Evaluation --> f1: 0.6348 ± 0.0033
    Time elapsed: 5.087s
    -------------------------------------------------
    Total time: 7.444s
    
    
    Final results ========================= >>
    Duration: 7.446s
    ------------------------------------------
    LightGBM --> f1: 0.635 ± 0.003
    
    
    Run: 8 ================================ >>
    Size of training set: 101645 (100%)
    Size of test set: 25366
    
    
    Results for LightGBM:         
    Fit ---------------------------------------------
    Train evaluation --> f1: 0.6642
    Test evaluation --> f1: 0.6396
    Time elapsed: 2.552s
    Bagging -----------------------------------------
    Evaluation --> f1: 0.6383 ± 0.0011
    Time elapsed: 5.604s
    -------------------------------------------------
    Total time: 8.158s
    
    
    Final results ========================= >>
    Duration: 8.161s
    ------------------------------------------
    LightGBM --> f1: 0.638 ± 0.001
    

## Analyze the results


```python
# Note that the results dataframe now is multi-index
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
      <th></th>
      <th>metric_train</th>
      <th>metric_test</th>
      <th>time_fit</th>
      <th>mean_bagging</th>
      <th>std_bagging</th>
      <th>time_bagging</th>
      <th>time</th>
    </tr>
    <tr>
      <th>frac</th>
      <th>model</th>
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
      <th>0.100</th>
      <th>LGB</th>
      <td>0.817545</td>
      <td>0.610896</td>
      <td>0.715s</td>
      <td>0.592843</td>
      <td>0.00326925</td>
      <td>1.566s</td>
      <td>2.291s</td>
    </tr>
    <tr>
      <th>0.213</th>
      <th>LGB</th>
      <td>0.739836</td>
      <td>0.624805</td>
      <td>0.950s</td>
      <td>0.615899</td>
      <td>0.00419954</td>
      <td>2.027s</td>
      <td>2.979s</td>
    </tr>
    <tr>
      <th>0.325</th>
      <th>LGB</th>
      <td>0.703472</td>
      <td>0.631394</td>
      <td>1.166s</td>
      <td>0.620782</td>
      <td>0.00276679</td>
      <td>2.432s</td>
      <td>3.599s</td>
    </tr>
    <tr>
      <th>0.438</th>
      <th>LGB</th>
      <td>0.686179</td>
      <td>0.631308</td>
      <td>1.304s</td>
      <td>0.627692</td>
      <td>0.00471833</td>
      <td>3.011s</td>
      <td>4.316s</td>
    </tr>
    <tr>
      <th>0.550</th>
      <th>LGB</th>
      <td>0.680266</td>
      <td>0.639622</td>
      <td>1.643s</td>
      <td>0.6277</td>
      <td>0.00435635</td>
      <td>3.521s</td>
      <td>5.165s</td>
    </tr>
    <tr>
      <th>0.662</th>
      <th>LGB</th>
      <td>0.677</td>
      <td>0.638232</td>
      <td>1.928s</td>
      <td>0.636536</td>
      <td>0.00181403</td>
      <td>4.042s</td>
      <td>5.970s</td>
    </tr>
    <tr>
      <th>0.775</th>
      <th>LGB</th>
      <td>0.675226</td>
      <td>0.639803</td>
      <td>2.085s</td>
      <td>0.63298</td>
      <td>0.00338554</td>
      <td>4.610s</td>
      <td>6.697s</td>
    </tr>
    <tr>
      <th>0.888</th>
      <th>LGB</th>
      <td>0.66932</td>
      <td>0.639532</td>
      <td>2.356s</td>
      <td>0.634824</td>
      <td>0.00334286</td>
      <td>5.087s</td>
      <td>7.444s</td>
    </tr>
    <tr>
      <th>1.000</th>
      <th>LGB</th>
      <td>0.664209</td>
      <td>0.639555</td>
      <td>2.552s</td>
      <td>0.638334</td>
      <td>0.00110171</td>
      <td>5.604s</td>
      <td>8.158s</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot the train sizing's results
atom.plot_learning_curve()
```


![png](output_9_0.png)

