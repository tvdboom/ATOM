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
      <th>3118</th>
      <td>BadgerysCreek</td>
      <td>11.7</td>
      <td>23.2</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>SW</td>
      <td>28.0</td>
    </tr>
    <tr>
      <th>18965</th>
      <td>NorahHead</td>
      <td>10.2</td>
      <td>19.4</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>SSE</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>11196</th>
      <td>CoffsHarbour</td>
      <td>9.7</td>
      <td>21.2</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NW</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>62283</th>
      <td>Sale</td>
      <td>8.4</td>
      <td>21.7</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WSW</td>
      <td>41.0</td>
    </tr>
    <tr>
      <th>92461</th>
      <td>Townsville</td>
      <td>11.1</td>
      <td>27.1</td>
      <td>0.0</td>
      <td>7.6</td>
      <td>10.7</td>
      <td>ENE</td>
      <td>37.0</td>
    </tr>
  </tbody>
</table>
</div>



## Run the pipeline


```python
# Initialize ATOM and prepare the data
atom = ATOMClassifier(X, verbose=2, random_state=1)
atom.impute(strat_num='median', strat_cat='most_frequent', min_frac_rows=0.8)
atom.encode()
```

    << ================== ATOM ================== >>
    Algorithm task: binary classification.
    Applying data cleaning...
    
    Dataset stats ================= >>
    Shape: (142193, 22)
    Missing values: 292032
    Categorical columns: 5
    Scaled: False
    ----------------------------------
    Size of training set: 113755
    Size of test set: 28438
    ----------------------------------
    Class balance: No:Yes <==> 3.5:1.0
    Instances in RainTomorrow per class:
    |        |    total |    train_set |    test_set |
    |:-------|---------:|-------------:|------------:|
    | 0: No  |   110316 |        88263 |       22053 |
    | 1: Yes |    31877 |        25492 |        6385 |
    
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
     --> Target-encoding feature Location. Contains 45 unique categories.
     --> Target-encoding feature WindGustDir. Contains 16 unique categories.
     --> Target-encoding feature WindDir9am. Contains 16 unique categories.
     --> Target-encoding feature WindDir3pm. Contains 16 unique categories.
     --> Label-encoding feature RainToday. Contains 2 unique categories.
    


```python
# We can analyze the impact of the training set's size on a LightGBM model
atom.train_sizing('lgb', train_sizes=np.linspace(0.1, 1, 9), bagging=4)
```

    
    Running pipeline ============================= >>
    Models in pipeline: LGB
    Metric: f1
    
    
    Run 0 (10% of set) ============================>>
    Size of training set: 11375
    Size of test set: 28438
    
    
    Results for LightGBM:         
    Fitting -----------------------------------------
    Score on the train set --> f1: 0.8029
    Score on the test set  --> f1: 0.6086
    Time elapsed: 0.998s
    Bagging -----------------------------------------
    Score --> f1: 0.5945 ± 0.0073
    Time elapsed: 2.229s
    -------------------------------------------------
    Total time: 3.242s
    
    
    Final results ========================= >>
    Duration: 3.244s
    ------------------------------------------
    LightGBM --> f1: 0.594 ± 0.007 ~
    
    
    Run 1 (21% of set) ============================>>
    Size of training set: 24172
    Size of test set: 28438
    
    
    Results for LightGBM:         
    Fitting -----------------------------------------
    Score on the train set --> f1: 0.7292
    Score on the test set  --> f1: 0.6273
    Time elapsed: 1.244s
    Bagging -----------------------------------------
    Score --> f1: 0.6166 ± 0.0053
    Time elapsed: 2.879s
    -------------------------------------------------
    Total time: 4.129s
    
    
    Final results ========================= >>
    Duration: 4.131s
    ------------------------------------------
    LightGBM --> f1: 0.617 ± 0.005
    
    
    Run 2 (32% of set) ============================>>
    Size of training set: 36970
    Size of test set: 28438
    
    
    Results for LightGBM:         
    Fitting -----------------------------------------
    Score on the train set --> f1: 0.6955
    Score on the test set  --> f1: 0.6325
    Time elapsed: 1.533s
    Bagging -----------------------------------------
    Score --> f1: 0.6199 ± 0.0038
    Time elapsed: 3.502s
    -------------------------------------------------
    Total time: 5.039s
    
    
    Final results ========================= >>
    Duration: 5.042s
    ------------------------------------------
    LightGBM --> f1: 0.620 ± 0.004
    
    
    Run 3 (44% of set) ============================>>
    Size of training set: 49767
    Size of test set: 28438
    
    
    Results for LightGBM:         
    Fitting -----------------------------------------
    Score on the train set --> f1: 0.6832
    Score on the test set  --> f1: 0.6386
    Time elapsed: 1.825s
    Bagging -----------------------------------------
    Score --> f1: 0.6256 ± 0.0036
    Time elapsed: 4.148s
    -------------------------------------------------
    Total time: 5.979s
    
    
    Final results ========================= >>
    Duration: 5.981s
    ------------------------------------------
    LightGBM --> f1: 0.626 ± 0.004
    
    
    Run 4 (55% of set) ============================>>
    Size of training set: 62565
    Size of test set: 28438
    
    
    Results for LightGBM:         
    Fitting -----------------------------------------
    Score on the train set --> f1: 0.6818
    Score on the test set  --> f1: 0.6391
    Time elapsed: 2.152s
    Bagging -----------------------------------------
    Score --> f1: 0.6271 ± 0.0025
    Time elapsed: 4.838s
    -------------------------------------------------
    Total time: 6.996s
    
    
    Final results ========================= >>
    Duration: 6.998s
    ------------------------------------------
    LightGBM --> f1: 0.627 ± 0.002
    
    
    Run 5 (66% of set) ============================>>
    Size of training set: 75362
    Size of test set: 28438
    
    
    Results for LightGBM:         
    Fitting -----------------------------------------
    Score on the train set --> f1: 0.6767
    Score on the test set  --> f1: 0.6399
    Time elapsed: 2.418s
    Bagging -----------------------------------------
    Score --> f1: 0.6346 ± 0.0021
    Time elapsed: 5.622s
    -------------------------------------------------
    Total time: 8.045s
    
    
    Final results ========================= >>
    Duration: 8.047s
    ------------------------------------------
    LightGBM --> f1: 0.635 ± 0.002
    
    
    Run 6 (77% of set) ============================>>
    Size of training set: 88160
    Size of test set: 28438
    
    
    Results for LightGBM:         
    Fitting -----------------------------------------
    Score on the train set --> f1: 0.6665
    Score on the test set  --> f1: 0.6384
    Time elapsed: 2.810s
    Bagging -----------------------------------------
    Score --> f1: 0.6342 ± 0.0021
    Time elapsed: 6.240s
    -------------------------------------------------
    Total time: 9.058s
    
    
    Final results ========================= >>
    Duration: 9.060s
    ------------------------------------------
    LightGBM --> f1: 0.634 ± 0.002
    
    
    Run 7 (89% of set) ============================>>
    Size of training set: 100957
    Size of test set: 28438
    
    
    Results for LightGBM:         
    Fitting -----------------------------------------
    Score on the train set --> f1: 0.6651
    Score on the test set  --> f1: 0.6432
    Time elapsed: 3.063s
    Bagging -----------------------------------------
    Score --> f1: 0.6372 ± 0.0025
    Time elapsed: 6.888s
    -------------------------------------------------
    Total time: 9.958s
    
    
    Final results ========================= >>
    Duration: 9.960s
    ------------------------------------------
    LightGBM --> f1: 0.637 ± 0.003
    
    
    Run 8 (100% of set) ===========================>>
    Size of training set: 113755
    Size of test set: 28438
    
    
    Results for LightGBM:         
    Fitting -----------------------------------------
    Score on the train set --> f1: 0.6650
    Score on the test set  --> f1: 0.6549
    Time elapsed: 3.379s
    Bagging -----------------------------------------
    Score --> f1: 0.6508 ± 0.0026
    Time elapsed: 7.621s
    -------------------------------------------------
    Total time: 11.009s
    
    
    Final results ========================= >>
    Duration: 11.012s
    ------------------------------------------
    LightGBM --> f1: 0.651 ± 0.003
    

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
      <th>name</th>
      <th>score_train</th>
      <th>score_test</th>
      <th>time_fit</th>
      <th>mean_bagging</th>
      <th>std_bagging</th>
      <th>time_bagging</th>
      <th>time</th>
    </tr>
    <tr>
      <th>run</th>
      <th>model</th>
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
      <th>0</th>
      <th>LGB</th>
      <td>LightGBM</td>
      <td>0.802859</td>
      <td>0.608590</td>
      <td>0.998s</td>
      <td>0.594472</td>
      <td>0.007341</td>
      <td>2.229s</td>
      <td>3.242s</td>
    </tr>
    <tr>
      <th>1</th>
      <th>LGB</th>
      <td>LightGBM</td>
      <td>0.729212</td>
      <td>0.627277</td>
      <td>1.244s</td>
      <td>0.616583</td>
      <td>0.005321</td>
      <td>2.879s</td>
      <td>4.129s</td>
    </tr>
    <tr>
      <th>2</th>
      <th>LGB</th>
      <td>LightGBM</td>
      <td>0.695463</td>
      <td>0.632544</td>
      <td>1.533s</td>
      <td>0.619899</td>
      <td>0.003822</td>
      <td>3.502s</td>
      <td>5.039s</td>
    </tr>
    <tr>
      <th>3</th>
      <th>LGB</th>
      <td>LightGBM</td>
      <td>0.683228</td>
      <td>0.638575</td>
      <td>1.825s</td>
      <td>0.625589</td>
      <td>0.003608</td>
      <td>4.148s</td>
      <td>5.979s</td>
    </tr>
    <tr>
      <th>4</th>
      <th>LGB</th>
      <td>LightGBM</td>
      <td>0.681811</td>
      <td>0.639062</td>
      <td>2.152s</td>
      <td>0.627105</td>
      <td>0.002460</td>
      <td>4.838s</td>
      <td>6.996s</td>
    </tr>
    <tr>
      <th>5</th>
      <th>LGB</th>
      <td>LightGBM</td>
      <td>0.676747</td>
      <td>0.639897</td>
      <td>2.418s</td>
      <td>0.634642</td>
      <td>0.002138</td>
      <td>5.622s</td>
      <td>8.045s</td>
    </tr>
    <tr>
      <th>6</th>
      <th>LGB</th>
      <td>LightGBM</td>
      <td>0.666471</td>
      <td>0.638376</td>
      <td>2.810s</td>
      <td>0.634245</td>
      <td>0.002098</td>
      <td>6.240s</td>
      <td>9.058s</td>
    </tr>
    <tr>
      <th>7</th>
      <th>LGB</th>
      <td>LightGBM</td>
      <td>0.665065</td>
      <td>0.643197</td>
      <td>3.063s</td>
      <td>0.637232</td>
      <td>0.002537</td>
      <td>6.888s</td>
      <td>9.958s</td>
    </tr>
    <tr>
      <th>8</th>
      <th>LGB</th>
      <td>LightGBM</td>
      <td>0.665018</td>
      <td>0.654904</td>
      <td>3.379s</td>
      <td>0.650772</td>
      <td>0.002577</td>
      <td>7.621s</td>
      <td>11.009s</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot the train sizing's results
atom.plot_learning_curve()
```


![png](output_9_0.png)

