# Regression
---------------------------------

This example shows how to use ATOM to apply PCA on the data and run a regression pipeline.

Download the abalone dataset from [https://archive.ics.uci.edu/ml/datasets/Abalone](https://archive.ics.uci.edu/ml/datasets/Abalone). The goal of this dataset is to predict the rings (age) of abalone shells from physical measurements.

### Load the data


```python
# Import packages
import pandas as pd
from atom import ATOMRegressor

# Load the abalone dataset
X = pd.read_csv('./datasets/abalone.csv')
```


```python
# Let's have a look at the data
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
# Initialize ATOM for regression tasks and encode the categorical features
atom = ATOMRegressor(X, y="Rings", verbose=2, random_state=42)
atom.encode()
```

    << ================== ATOM ================== >>
    Algorithm task: regression.
    Applying data cleaning...
    
    Dataset stats ================= >>
    Shape: (4177, 9)
    Categorical columns: 1
    Scaled: False
    ----------------------------------
    Size of training set: 3342
    Size of test set: 835
    
    Fitting Encoder...
    Encoding categorical columns...
     --> One-hot-encoding feature Sex. Contains 3 unique categories.
    


```python
# Plot the dataset's correlation matrix
atom.plot_correlation()
```


![png](output_5_0.png)



```python
# Apply PCA for dimensionality reduction
atom.feature_selection(strategy="pca", n_features=6)
```

    Fitting FeatureSelector...
    Performing feature selection ...
     --> Feature Diameter was removed due to collinearity with another feature.
     --> Applying Principal Component Analysis...
       >>> Scaling features...
       >>> Total explained variance: 0.977
    


```python
# Use the plotting methods to see the retained variance ratio
atom.plot_pca()
atom.plot_components(figsize=(8, 6), filename='atom_PCA_plot')
```


![png](output_7_0.png)



![png](output_7_1.png)


### Run the pipeline


```python
atom.run(['Tree', 'Bag', 'ET'],
         metric='MSE',
         n_calls=5,
         n_random_starts=2,
         bo_params={'base_estimator': 'GBRT', 'cv': 1},
         bagging=5)
```

    
    Running pipeline ============================= >>
    Models in pipeline: Tree, Bag, ET
    Metric: neg_mean_squared_error
    
    
    Running BO for Decision Tree...
    Random start 1 ----------------------------------
    Parameters --> {'criterion': 'mae', 'splitter': 'random', 'max_depth': 5, 'max_features': 0.9, 'min_samples_split': 8, 'min_samples_leaf': 19, 'ccp_alpha': 0.003}
    Evaluation --> neg_mean_squared_error: -8.2257  Best neg_mean_squared_error: -8.2257
    Time iteration: 0.038s   Total time: 0.042s
    Random start 2 ----------------------------------
    Parameters --> {'criterion': 'mae', 'splitter': 'best', 'max_depth': 10, 'max_features': 0.9, 'min_samples_split': 3, 'min_samples_leaf': 12, 'ccp_alpha': 0.033}
    Evaluation --> neg_mean_squared_error: -9.0433  Best neg_mean_squared_error: -8.2257
    Time iteration: 0.197s   Total time: 0.243s
    Iteration 3 -------------------------------------
    Parameters --> {'criterion': 'friedman_mse', 'splitter': 'random', 'max_depth': 7, 'max_features': 0.6, 'min_samples_split': 17, 'min_samples_leaf': 19, 'ccp_alpha': 0.015}
    Evaluation --> neg_mean_squared_error: -6.3817  Best neg_mean_squared_error: -6.3817
    Time iteration: 0.006s   Total time: 0.342s
    Iteration 4 -------------------------------------
    Parameters --> {'criterion': 'friedman_mse', 'splitter': 'best', 'max_depth': 9, 'max_features': 0.6, 'min_samples_split': 19, 'min_samples_leaf': 13, 'ccp_alpha': 0.013}
    Evaluation --> neg_mean_squared_error: -6.5175  Best neg_mean_squared_error: -6.3817
    Time iteration: 0.012s   Total time: 0.448s
    Iteration 5 -------------------------------------
    Parameters --> {'criterion': 'friedman_mse', 'splitter': 'random', 'max_depth': 10, 'max_features': 0.6, 'min_samples_split': 16, 'min_samples_leaf': 4, 'ccp_alpha': 0.011}
    Evaluation --> neg_mean_squared_error: -10.0056  Best neg_mean_squared_error: -6.3817
    Time iteration: 0.007s   Total time: 0.551s
    
    Results for Decision Tree:         
    Bayesian Optimization ---------------------------
    Best parameters --> {'criterion': 'friedman_mse', 'splitter': 'random', 'max_depth': 7, 'max_features': 0.6, 'min_samples_split': 17, 'min_samples_leaf': 19, 'ccp_alpha': 0.015}
    Best evaluation --> neg_mean_squared_error: -6.3817
    Time elapsed: 0.645s
    Fitting -----------------------------------------
    Score on the train set --> neg_mean_squared_error: -8.3743
    Score on the test set  --> neg_mean_squared_error: -6.9551
    Time elapsed: 0.007s
    Bagging -----------------------------------------
    Score --> neg_mean_squared_error: -6.6744 ± 0.9527
    Time elapsed: 0.020s
    -------------------------------------------------
    Total time: 0.678s
    
    
    Running BO for Bagging Regressor...
    Random start 1 ----------------------------------
    Parameters --> {'n_estimators': 112, 'max_samples': 0.9, 'max_features': 0.6, 'bootstrap': False, 'bootstrap_features': False}
    Evaluation --> neg_mean_squared_error: -5.8182  Best neg_mean_squared_error: -5.8182
    Time iteration: 0.873s   Total time: 0.876s
    Random start 2 ----------------------------------
    Parameters --> {'n_estimators': 131, 'max_samples': 0.5, 'max_features': 0.5, 'bootstrap': False, 'bootstrap_features': False}
    Evaluation --> neg_mean_squared_error: -6.7970  Best neg_mean_squared_error: -5.8182
    Time iteration: 0.589s   Total time: 1.467s
    Iteration 3 -------------------------------------
    Parameters --> {'n_estimators': 50, 'max_samples': 0.9, 'max_features': 0.6, 'bootstrap': False, 'bootstrap_features': True}
    Evaluation --> neg_mean_squared_error: -5.4292  Best neg_mean_squared_error: -5.4292
    Time iteration: 0.383s   Total time: 2.015s
    Iteration 4 -------------------------------------
    Parameters --> {'n_estimators': 74, 'max_samples': 0.5, 'max_features': 0.5, 'bootstrap': False, 'bootstrap_features': True}
    Evaluation --> neg_mean_squared_error: -5.9847  Best neg_mean_squared_error: -5.4292
    Time iteration: 0.327s   Total time: 2.434s
    Iteration 5 -------------------------------------
    Parameters --> {'n_estimators': 18, 'max_samples': 0.8, 'max_features': 0.6, 'bootstrap': True, 'bootstrap_features': False}
    Evaluation --> neg_mean_squared_error: -6.1556  Best neg_mean_squared_error: -5.4292
    Time iteration: 0.092s   Total time: 2.613s
    
    Results for Bagging Regressor:         
    Bayesian Optimization ---------------------------
    Best parameters --> {'n_estimators': 50, 'max_samples': 0.9, 'max_features': 0.6, 'bootstrap': False, 'bootstrap_features': True}
    Best evaluation --> neg_mean_squared_error: -5.4292
    Time elapsed: 2.708s
    Fitting -----------------------------------------
    Score on the train set --> neg_mean_squared_error: -0.0861
    Score on the test set  --> neg_mean_squared_error: -4.9042
    Time elapsed: 0.514s
    Bagging -----------------------------------------
    Score --> neg_mean_squared_error: -4.9562 ± 0.1064
    Time elapsed: 2.145s
    -------------------------------------------------
    Total time: 5.374s
    
    
    Running BO for Extra-Trees...
    Random start 1 ----------------------------------
    Parameters --> {'n_estimators': 112, 'max_depth': 6, 'max_features': 1.0, 'criterion': 'mae', 'min_samples_split': 8, 'min_samples_leaf': 19, 'ccp_alpha': 0.003, 'bootstrap': True, 'max_samples': 0.6}
    Evaluation --> neg_mean_squared_error: -6.7733  Best neg_mean_squared_error: -6.7733
    Time iteration: 0.995s   Total time: 1.000s
    Random start 2 ----------------------------------
    Parameters --> {'n_estimators': 369, 'max_depth': 10, 'max_features': 0.8, 'criterion': 'mse', 'min_samples_split': 13, 'min_samples_leaf': 6, 'ccp_alpha': 0.0, 'bootstrap': False}
    Evaluation --> neg_mean_squared_error: -6.6959  Best neg_mean_squared_error: -6.6959
    Time iteration: 0.475s   Total time: 1.479s
    Iteration 3 -------------------------------------
    Parameters --> {'n_estimators': 481, 'max_depth': 10, 'max_features': 0.8, 'criterion': 'mse', 'min_samples_split': 7, 'min_samples_leaf': 2, 'ccp_alpha': 0.001, 'bootstrap': False}
    Evaluation --> neg_mean_squared_error: -4.8752  Best neg_mean_squared_error: -4.8752
    Time iteration: 0.726s   Total time: 2.310s
    Iteration 4 -------------------------------------
    Parameters --> {'n_estimators': 460, 'max_depth': 5, 'max_features': 1.0, 'criterion': 'mae', 'min_samples_split': 5, 'min_samples_leaf': 4, 'ccp_alpha': 0.034, 'bootstrap': True, 'max_samples': 0.6}
    Evaluation --> neg_mean_squared_error: -7.0711  Best neg_mean_squared_error: -4.8752
    Time iteration: 4.778s   Total time: 7.196s
    Iteration 5 -------------------------------------
    Parameters --> {'n_estimators': 474, 'max_depth': 4, 'max_features': 0.8, 'criterion': 'mae', 'min_samples_split': 20, 'min_samples_leaf': 1, 'ccp_alpha': 0.018, 'bootstrap': True, 'max_samples': 0.6}
    Evaluation --> neg_mean_squared_error: -7.2239  Best neg_mean_squared_error: -4.8752
    Time iteration: 3.961s   Total time: 11.260s
    
    Results for Extra-Trees:         
    Bayesian Optimization ---------------------------
    Best parameters --> {'n_estimators': 481, 'max_depth': 10, 'max_features': 0.8, 'criterion': 'mse', 'min_samples_split': 7, 'min_samples_leaf': 2, 'ccp_alpha': 0.001, 'bootstrap': False}
    Best evaluation --> neg_mean_squared_error: -4.8752
    Time elapsed: 11.359s
    Fitting -----------------------------------------
    Score on the train set --> neg_mean_squared_error: -4.4007
    Score on the test set  --> neg_mean_squared_error: -4.1469
    Time elapsed: 0.924s
    Bagging -----------------------------------------
    Score --> neg_mean_squared_error: -4.2367 ± 0.0422
    Time elapsed: 4.064s
    -------------------------------------------------
    Total time: 16.352s
    
    
    Final results ========================= >>
    Duration: 22.406s
    ------------------------------------------
    Decision Tree     --> neg_mean_squared_error: -6.674 ± 0.953 ~
    Bagging Regressor --> neg_mean_squared_error: -4.956 ± 0.106 ~
    Extra-Trees       --> neg_mean_squared_error: -4.237 ± 0.042 ~ !
    
