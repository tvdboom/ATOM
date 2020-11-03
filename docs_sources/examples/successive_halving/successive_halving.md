# Successive halving
---------------------------------

This example shows how to compare multiple tree-based models using successive halving.

Import the boston dataset from [sklearn.datasets](https://scikit-learn.org/stable/datasets/index.html#boston-dataset).
 This is a small and easy to train dataset whose goal is to predict house prices.

## Load the data


```python
# Import packages
from sklearn.datasets import load_boston
from atom import ATOMRegressor
```


```python
# Load the dataset's features and targets
X, y = load_boston(return_X_y=True)
```

## Run the pipeline


```python
atom = ATOMRegressor(X, y, verbose=1, random_state=1)
```

    << ================== ATOM ================== >>
    Algorithm task: regression.
    
    Dataset stats ================== >>
    Shape: (506, 14)
    Scaled: False
    -----------------------------------
    Train set size: 405
    Test set size: 101
    
    


```python
# We can compare tree-based models via successive halving
atom.successive_halving(
    models=['tree', 'bag', 'et', 'rf', 'lgb', 'catb'],
    metric='mae',
    bagging=5
)
```

    
    Training ===================================== >>
    Metric: neg_mean_absolute_error
    
    
    Run: 0 ================================ >>
    Models: Tree, Bag, ET, RF, LGB, CatB
    Size of training set: 67 (17%)
    Size of test set: 101
    
    
    Results for Decision Tree:         
    Fit ---------------------------------------------
    Train evaluation --> neg_mean_absolute_error: -0.0000
    Test evaluation --> neg_mean_absolute_error: -3.3257
    Time elapsed: 0.007s
    Bagging -----------------------------------------
    Evaluation --> neg_mean_absolute_error: -4.3307 ± 0.5250
    Time elapsed: 0.022s
    -------------------------------------------------
    Total time: 0.029s
    
    
    Results for Bagging Regressor:         
    Fit ---------------------------------------------
    Train evaluation --> neg_mean_absolute_error: -1.3054
    Test evaluation --> neg_mean_absolute_error: -2.6950
    Time elapsed: 0.020s
    Bagging -----------------------------------------
    Evaluation --> neg_mean_absolute_error: -3.0957 ± 0.2677
    Time elapsed: 0.082s
    -------------------------------------------------
    Total time: 0.104s
    
    
    Results for Extra-Trees:         
    Fit ---------------------------------------------
    Train evaluation --> neg_mean_absolute_error: -0.0000
    Test evaluation --> neg_mean_absolute_error: -2.1541
    Time elapsed: 0.086s
    Bagging -----------------------------------------
    Evaluation --> neg_mean_absolute_error: -2.5554 ± 0.1708
    Time elapsed: 0.360s
    -------------------------------------------------
    Total time: 0.446s
    
    
    Results for Random Forest:         
    Fit ---------------------------------------------
    Train evaluation --> neg_mean_absolute_error: -1.1509
    Test evaluation --> neg_mean_absolute_error: -2.4143
    Time elapsed: 0.109s
    Bagging -----------------------------------------
    Evaluation --> neg_mean_absolute_error: -2.9574 ± 0.2253
    Time elapsed: 0.506s
    -------------------------------------------------
    Total time: 0.615s
    
    
    Results for LightGBM:         
    Fit ---------------------------------------------
    Train evaluation --> neg_mean_absolute_error: -3.4205
    Test evaluation --> neg_mean_absolute_error: -4.5600
    Time elapsed: 0.027s
    Bagging -----------------------------------------
    Evaluation --> neg_mean_absolute_error: -4.8393 ± 0.2682
    Time elapsed: 0.074s
    -------------------------------------------------
    Total time: 0.103s
    
    
    Results for CatBoost:         
    Fit ---------------------------------------------
    Train evaluation --> neg_mean_absolute_error: -0.0806
    Test evaluation --> neg_mean_absolute_error: -2.3984
    Time elapsed: 1.333s
    Bagging -----------------------------------------
    Evaluation --> neg_mean_absolute_error: -2.9165 ± 0.2564
    Time elapsed: 4.065s
    -------------------------------------------------
    Total time: 5.399s
    
    
    Final results ========================= >>
    Duration: 6.699s
    ------------------------------------------
    Decision Tree     --> neg_mean_absolute_error: -4.331 ± 0.525 ~
    Bagging Regressor --> neg_mean_absolute_error: -3.096 ± 0.268 ~
    Extra-Trees       --> neg_mean_absolute_error: -2.555 ± 0.171 ~ !
    Random Forest     --> neg_mean_absolute_error: -2.957 ± 0.225 ~
    LightGBM          --> neg_mean_absolute_error: -4.839 ± 0.268 ~
    CatBoost          --> neg_mean_absolute_error: -2.916 ± 0.256 ~
    
    
    Run: 1 ================================ >>
    Models: ET, CatB, RF
    Size of training set: 135 (33%)
    Size of test set: 101
    
    
    Results for Extra-Trees:         
    Fit ---------------------------------------------
    Train evaluation --> neg_mean_absolute_error: -0.0000
    Test evaluation --> neg_mean_absolute_error: -2.2361
    Time elapsed: 0.099s
    Bagging -----------------------------------------
    Evaluation --> neg_mean_absolute_error: -2.6016 ± 0.2890
    Time elapsed: 0.418s
    -------------------------------------------------
    Total time: 0.517s
    
    
    Results for CatBoost:         
    Fit ---------------------------------------------
    Train evaluation --> neg_mean_absolute_error: -0.2835
    Test evaluation --> neg_mean_absolute_error: -2.4196
    Time elapsed: 1.847s
    Bagging -----------------------------------------
    Evaluation --> neg_mean_absolute_error: -2.5681 ± 0.2119
    Time elapsed: 6.494s
    -------------------------------------------------
    Total time: 8.343s
    
    
    Results for Random Forest:         
    Fit ---------------------------------------------
    Train evaluation --> neg_mean_absolute_error: -0.9820
    Test evaluation --> neg_mean_absolute_error: -2.5055
    Time elapsed: 0.131s
    Bagging -----------------------------------------
    Evaluation --> neg_mean_absolute_error: -2.6144 ± 0.1188
    Time elapsed: 0.662s
    -------------------------------------------------
    Total time: 0.793s
    
    
    Final results ========================= >>
    Duration: 9.655s
    ------------------------------------------
    Extra-Trees   --> neg_mean_absolute_error: -2.602 ± 0.289 ~
    CatBoost      --> neg_mean_absolute_error: -2.568 ± 0.212 ~ !
    Random Forest --> neg_mean_absolute_error: -2.614 ± 0.119 ~
    
    
    Run: 2 ================================ >>
    Models: CatB
    Size of training set: 405 (100%)
    Size of test set: 101
    
    
    Results for CatBoost:         
    Fit ---------------------------------------------
    Train evaluation --> neg_mean_absolute_error: -0.3978
    Test evaluation --> neg_mean_absolute_error: -1.8772
    Time elapsed: 3.348s
    Bagging -----------------------------------------
    Evaluation --> neg_mean_absolute_error: -2.0501 ± 0.0892
    Time elapsed: 14.871s
    -------------------------------------------------
    Total time: 18.221s
    
    
    Final results ========================= >>
    Duration: 18.223s
    ------------------------------------------
    CatBoost --> neg_mean_absolute_error: -2.050 ± 0.089 ~
    

## Analyze results


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
      <th>n_models</th>
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
      <th>1</th>
      <th>CatB</th>
      <td>-0.397799</td>
      <td>-1.87721</td>
      <td>3.348s</td>
      <td>-2.05012</td>
      <td>0.0891846</td>
      <td>14.871s</td>
      <td>18.221s</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">3</th>
      <th>ET</th>
      <td>-2.31519e-14</td>
      <td>-2.23608</td>
      <td>0.099s</td>
      <td>-2.60165</td>
      <td>0.289034</td>
      <td>0.418s</td>
      <td>0.517s</td>
    </tr>
    <tr>
      <th>RF</th>
      <td>-0.981978</td>
      <td>-2.50547</td>
      <td>0.131s</td>
      <td>-2.61442</td>
      <td>0.118758</td>
      <td>0.662s</td>
      <td>0.793s</td>
    </tr>
    <tr>
      <th>CatB</th>
      <td>-0.28355</td>
      <td>-2.41962</td>
      <td>1.847s</td>
      <td>-2.56808</td>
      <td>0.211868</td>
      <td>6.494s</td>
      <td>8.343s</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">6</th>
      <th>Tree</th>
      <td>-0</td>
      <td>-3.32574</td>
      <td>0.007s</td>
      <td>-4.33069</td>
      <td>0.525026</td>
      <td>0.022s</td>
      <td>0.029s</td>
    </tr>
    <tr>
      <th>Bag</th>
      <td>-1.30537</td>
      <td>-2.69505</td>
      <td>0.020s</td>
      <td>-3.09566</td>
      <td>0.267668</td>
      <td>0.082s</td>
      <td>0.104s</td>
    </tr>
    <tr>
      <th>ET</th>
      <td>-2.25624e-14</td>
      <td>-2.15409</td>
      <td>0.086s</td>
      <td>-2.55543</td>
      <td>0.170823</td>
      <td>0.360s</td>
      <td>0.446s</td>
    </tr>
    <tr>
      <th>RF</th>
      <td>-1.15087</td>
      <td>-2.4143</td>
      <td>0.109s</td>
      <td>-2.9574</td>
      <td>0.225311</td>
      <td>0.506s</td>
      <td>0.615s</td>
    </tr>
    <tr>
      <th>LGB</th>
      <td>-3.42052</td>
      <td>-4.55996</td>
      <td>0.027s</td>
      <td>-4.83931</td>
      <td>0.268167</td>
      <td>0.074s</td>
      <td>0.103s</td>
    </tr>
    <tr>
      <th>CatB</th>
      <td>-0.080555</td>
      <td>-2.39843</td>
      <td>1.333s</td>
      <td>-2.91647</td>
      <td>0.256428</td>
      <td>4.065s</td>
      <td>5.399s</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot the successive halving's results
atom.plot_successive_halving()
```


![png](output_9_0.png)

