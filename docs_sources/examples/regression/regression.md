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
atom = ATOMRegressor(X, "Rings", verbose=2, random_state=42)
atom.encode()
```

    << ================== ATOM ================== >>
    Algorithm task: regression.
    
    Dataset stats ================== >>
    Shape: (4177, 9)
    Categorical columns: 1
    Scaled: False
    -----------------------------------
    Train set size: 3342
    Test set size: 835
    
    Fitting Encoder...
    Encoding categorical columns...
     --> OneHot-encoding feature Sex. Contains 3 unique classes.
    


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
     --> Applying Principal Component Analysis...
       >>> Scaling features...
       >>> Total explained variance: 0.976
    


```python
# Use the plotting methods to see the retained variance ratio
atom.plot_pca()
atom.plot_components(figsize=(8, 6), filename='atom_PCA_plot')
```


![png](output_7_0.png)



![png](output_7_1.png)


### Run the pipeline


```python
atom.run(
    models=['Tree', 'Bag', 'ET'],
    metric='MSE',
    n_calls=5,
    n_initial_points=2,
    bo_params={'base_estimator': 'GBRT', 'cv': 1},
    bagging=5
)
```

    
    Training ===================================== >>
    Models: Tree, Bag, ET
    Metric: neg_mean_squared_error
    
    
    Running BO for Decision Tree...
    Initial point 1 ---------------------------------
    Parameters --> {'criterion': 'mae', 'splitter': 'random', 'max_depth': 7, 'min_samples_split': 8, 'min_samples_leaf': 19, 'max_features': None, 'ccp_alpha': 0.016}
    Evaluation --> neg_mean_squared_error: -8.3677  Best neg_mean_squared_error: -8.3677
    Time iteration: 0.043s   Total time: 0.049s
    Initial point 2 ---------------------------------
    Parameters --> {'criterion': 'mae', 'splitter': 'best', 'max_depth': 6, 'min_samples_split': 3, 'min_samples_leaf': 12, 'max_features': 0.9, 'ccp_alpha': 0.0}
    Evaluation --> neg_mean_squared_error: -8.2055  Best neg_mean_squared_error: -8.2055
    Time iteration: 0.186s   Total time: 0.240s
    Iteration 3 -------------------------------------
    Parameters --> {'criterion': 'mae', 'splitter': 'best', 'max_depth': 6, 'min_samples_split': 14, 'min_samples_leaf': 9, 'max_features': 0.9, 'ccp_alpha': 0.005}
    Evaluation --> neg_mean_squared_error: -6.1540  Best neg_mean_squared_error: -6.1540
    Time iteration: 0.172s   Total time: 0.619s
    Iteration 4 -------------------------------------
    Parameters --> {'criterion': 'mae', 'splitter': 'random', 'max_depth': 7, 'min_samples_split': 15, 'min_samples_leaf': 4, 'max_features': 0.7, 'ccp_alpha': 0.018}
    Evaluation --> neg_mean_squared_error: -7.9567  Best neg_mean_squared_error: -6.1540
    Time iteration: 0.070s   Total time: 0.797s
    Iteration 5 -------------------------------------
    Parameters --> {'criterion': 'mae', 'splitter': 'best', 'max_depth': 6, 'min_samples_split': 14, 'min_samples_leaf': 5, 'max_features': 0.9, 'ccp_alpha': 0.009}
    Evaluation --> neg_mean_squared_error: -7.1330  Best neg_mean_squared_error: -6.1540
    Time iteration: 0.171s   Total time: 1.079s
    
    Results for Decision Tree:         
    Bayesian Optimization ---------------------------
    Best parameters --> {'criterion': 'mae', 'splitter': 'best', 'max_depth': 6, 'min_samples_split': 14, 'min_samples_leaf': 9, 'max_features': 0.9, 'ccp_alpha': 0.005}
    Best evaluation --> neg_mean_squared_error: -6.1540
    Time elapsed: 1.187s
    Fit ---------------------------------------------
    Train evaluation --> neg_mean_squared_error: -6.3073
    Test evaluation --> neg_mean_squared_error: -5.5317
    Time elapsed: 0.262s
    Bagging -----------------------------------------
    Evaluation --> neg_mean_squared_error: -5.6780 ± 0.2464
    Time elapsed: 1.056s
    -------------------------------------------------
    Total time: 2.507s
    
    
    Running BO for Bagging Regressor...
    Initial point 1 ---------------------------------
    Parameters --> {'n_estimators': 112, 'max_samples': 0.9, 'max_features': 0.6, 'bootstrap': False, 'bootstrap_features': False}
    Evaluation --> neg_mean_squared_error: -5.7680  Best neg_mean_squared_error: -5.7680
    Time iteration: 0.887s   Total time: 0.891s
    Initial point 2 ---------------------------------
    Parameters --> {'n_estimators': 131, 'max_samples': 0.5, 'max_features': 0.5, 'bootstrap': False, 'bootstrap_features': False}
    Evaluation --> neg_mean_squared_error: -6.8254  Best neg_mean_squared_error: -5.7680
    Time iteration: 0.598s   Total time: 1.495s
    Iteration 3 -------------------------------------
    Parameters --> {'n_estimators': 50, 'max_samples': 0.9, 'max_features': 0.6, 'bootstrap': False, 'bootstrap_features': True}
    Evaluation --> neg_mean_squared_error: -5.4895  Best neg_mean_squared_error: -5.4895
    Time iteration: 0.392s   Total time: 1.980s
    Iteration 4 -------------------------------------
    Parameters --> {'n_estimators': 74, 'max_samples': 0.5, 'max_features': 0.5, 'bootstrap': False, 'bootstrap_features': True}
    Evaluation --> neg_mean_squared_error: -6.0363  Best neg_mean_squared_error: -5.4895
    Time iteration: 0.335s   Total time: 2.413s
    Iteration 5 -------------------------------------
    Parameters --> {'n_estimators': 36, 'max_samples': 0.9, 'max_features': 0.6, 'bootstrap': True, 'bootstrap_features': False}
    Evaluation --> neg_mean_squared_error: -6.0037  Best neg_mean_squared_error: -5.4895
    Time iteration: 0.193s   Total time: 2.696s
    
    Results for Bagging Regressor:         
    Bayesian Optimization ---------------------------
    Best parameters --> {'n_estimators': 50, 'max_samples': 0.9, 'max_features': 0.6, 'bootstrap': False, 'bootstrap_features': True}
    Best evaluation --> neg_mean_squared_error: -5.4895
    Time elapsed: 2.793s
    Fit ---------------------------------------------
    Train evaluation --> neg_mean_squared_error: -0.0867
    Test evaluation --> neg_mean_squared_error: -4.9533
    Time elapsed: 0.515s
    Bagging -----------------------------------------
    Evaluation --> neg_mean_squared_error: -5.2363 ± 0.1099
    Time elapsed: 2.156s
    -------------------------------------------------
    Total time: 5.465s
    
    
    Running BO for Extra-Trees...
    Initial point 1 ---------------------------------
    Parameters --> {'n_estimators': 112, 'criterion': 'mae', 'max_depth': 1, 'min_samples_split': 9, 'min_samples_leaf': 7, 'max_features': 0.6, 'bootstrap': True, 'ccp_alpha': 0.016, 'max_samples': 0.6}
    Evaluation --> neg_mean_squared_error: -10.2607  Best neg_mean_squared_error: -10.2607
    Time iteration: 0.366s   Total time: 0.373s
    Initial point 2 ---------------------------------
    Parameters --> {'n_estimators': 369, 'criterion': 'mae', 'max_depth': None, 'min_samples_split': 3, 'min_samples_leaf': 12, 'max_features': 0.9, 'bootstrap': True, 'ccp_alpha': 0.035, 'max_samples': 0.8}
    Evaluation --> neg_mean_squared_error: -9.4727  Best neg_mean_squared_error: -9.4727
    Time iteration: 4.781s   Total time: 5.159s
    Iteration 3 -------------------------------------
    Parameters --> {'n_estimators': 385, 'criterion': 'mse', 'max_depth': None, 'min_samples_split': 6, 'min_samples_leaf': 18, 'max_features': 0.9, 'bootstrap': False, 'ccp_alpha': 0.02}
    Evaluation --> neg_mean_squared_error: -5.5174  Best neg_mean_squared_error: -5.5174
    Time iteration: 0.508s   Total time: 5.793s
    Iteration 4 -------------------------------------
    Parameters --> {'n_estimators': 425, 'criterion': 'mse', 'max_depth': 1, 'min_samples_split': 20, 'min_samples_leaf': 19, 'max_features': 0.7, 'bootstrap': False, 'ccp_alpha': 0.016}
    Evaluation --> neg_mean_squared_error: -9.1980  Best neg_mean_squared_error: -5.5174
    Time iteration: 0.314s   Total time: 6.231s
    Iteration 5 -------------------------------------
    Parameters --> {'n_estimators': 445, 'criterion': 'mse', 'max_depth': None, 'min_samples_split': 7, 'min_samples_leaf': 20, 'max_features': 0.6, 'bootstrap': False, 'ccp_alpha': 0.004}
    Evaluation --> neg_mean_squared_error: -6.9959  Best neg_mean_squared_error: -5.5174
    Time iteration: 0.428s   Total time: 6.782s
    
    Results for Extra-Trees:         
    Bayesian Optimization ---------------------------
    Best parameters --> {'n_estimators': 385, 'criterion': 'mse', 'max_depth': None, 'min_samples_split': 6, 'min_samples_leaf': 18, 'max_features': 0.9, 'bootstrap': False, 'ccp_alpha': 0.02}
    Best evaluation --> neg_mean_squared_error: -5.5174
    Time elapsed: 6.909s
    Fit ---------------------------------------------
    Train evaluation --> neg_mean_squared_error: -6.1021
    Test evaluation --> neg_mean_squared_error: -5.0002
    Time elapsed: 0.656s
    Bagging -----------------------------------------
    Evaluation --> neg_mean_squared_error: -4.9204 ± 0.0591
    Time elapsed: 3.082s
    -------------------------------------------------
    Total time: 10.647s
    
    
    Final results ========================= >>
    Duration: 18.623s
    ------------------------------------------
    Decision Tree     --> neg_mean_squared_error: -5.678 ± 0.246 ~
    Bagging Regressor --> neg_mean_squared_error: -5.236 ± 0.110 ~
    Extra-Trees       --> neg_mean_squared_error: -4.920 ± 0.059 ~ !
    

### Analyze the results


```python
# For regression tasks, use the errors or residuals plots to check the model performances
atom.plot_residuals()
```


![png](output_11_0.png)



```python
# Use the partial dependence plot to analyze the relation between the target response and the features
atom.n_jobs = 8  # The method can be slow...
atom.ET.plot_partial_dependence(features=(0, 1, (2, 3)), figsize=(12, 8))
```


![png](output_12_0.png)

