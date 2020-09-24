# Calibration
---------------------------------

This example shows us how to use the calibration method to calibrate a classifier.

The data used is a variation on the Australian weather dataset from [https://www.kaggle.com/jsphyg/weather-dataset-rattle-package](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package). The goal of this dataset is to predict whether or not it will rain tomorrow training a binay classifier on target `RainTomorrow`.

## Load the data


```python
# Import packages
import pandas as pd
from atom import ATOMClassifier
```


```python
# Get the dataset's features and targets
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
      <th>40667</th>
      <td>Williamtown</td>
      <td>10.0</td>
      <td>20.4</td>
      <td>0.0</td>
      <td>5.4</td>
      <td>NaN</td>
      <td>NW</td>
      <td>48.0</td>
    </tr>
    <tr>
      <th>43490</th>
      <td>Wollongong</td>
      <td>15.0</td>
      <td>22.0</td>
      <td>0.4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>SSW</td>
      <td>59.0</td>
    </tr>
    <tr>
      <th>102419</th>
      <td>Nuriootpa</td>
      <td>2.6</td>
      <td>23.9</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>12.8</td>
      <td>ESE</td>
      <td>35.0</td>
    </tr>
    <tr>
      <th>123437</th>
      <td>SalmonGums</td>
      <td>3.4</td>
      <td>18.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WSW</td>
      <td>33.0</td>
    </tr>
    <tr>
      <th>18121</th>
      <td>NorahHead</td>
      <td>16.5</td>
      <td>22.3</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>S</td>
      <td>46.0</td>
    </tr>
  </tbody>
</table>
</div>



## Run the pipeline


```python
# Initialize the ATOM class
atom = ATOMClassifier(X, y='RainTomorrow', n_rows=1e4, verbose=1, warnings='ignore', random_state=1)

# Handle missing values and categorical columns in the dataset
atom.impute(strat_num='median', strat_cat='most_frequent')
atom.encode(strategy='target', max_onehot=5, frac_to_other=0.05)

# Fit a linear SVM to the data
atom.run('lsvm')
```

    << ================== ATOM ================== >>
    Algorithm task: binary classification.
    Applying data cleaning...
    
    Dataset stats ================= >>
    Shape: (10000, 22)
    Missing values: 22613
    Categorical columns: 5
    Scaled: False
    ----------------------------------
    Train set size: 8000
    Test set size: 2000
    
    Fitting Imputer...
    Imputing missing values...
    Fitting Encoder...
    Encoding categorical columns...
    
    Running pipeline ============================= >>
    Models in pipeline: lSVM
    Metric: f1
    
    
    Results for Linear SVM:         
    Fitting -----------------------------------------
    Score on the train set --> f1: 0.5639
    Score on the test set  --> f1: 0.5929
    Time elapsed: 0.444s
    -------------------------------------------------
    Total time: 0.444s
    
    
    Final results ========================= >>
    Duration: 0.444s
    ------------------------------------------
    Linear SVM --> f1: 0.593
    

## Analyze the results


```python
# Check our model's calibration
atom.plot_calibration()
```


![png](output_7_0.png)



```python
# Let's try to improve it using the calibrate method
atom.calibrate(method='isotonic', cv=5)
atom.plot_calibration()
```


![png](output_8_0.png)

