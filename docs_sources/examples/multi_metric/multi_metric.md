# Multi-metric
-----------------------

This example shows how we can evaluate an ATOM pipeline on multiple metrics.

Import the breast cancer dataset from [sklearn.datasets](https://scikit-learn.org/stable/datasets/index.html#wine-dataset). This is a small and easy to train dataset whose goal is to predict whether a patient has breast cancer or not.

## Load the data


```python
# Import packages
from sklearn.datasets import load_breast_cancer
from atom import ATOMClassifier
```


```python
# Get the dataset's features and targets
X, y = load_breast_cancer(return_X_y=True)
```

## Run the pipeline


```python
# Call ATOM and run the pipeline using multipe metrics
# Note that for every step of the BO, both metrics are calculated, but only the first is used for optimization!
atom = ATOMClassifier(X, y, n_jobs=2, verbose=2, warnings=False, random_state=1)
atom.run(['MNB', 'QDA'], metric=('f1', 'recall'), n_calls=3, n_initial_points=1, bagging=4)
```

    << ================== ATOM ================== >>
    Algorithm task: binary classification.
    Parallel processing with 2 cores.
    
    Dataset stats ================== >>
    Shape: (569, 31)
    Scaled: False
    -----------------------------------
    Train set size: 456
    Test set size: 113
    -----------------------------------
    Train set balance: 0:1 <==> 1.0:1.7
    Test set balance: 0:1 <==> 1.0:1.5
    -----------------------------------
    Distribution of classes:
    |    |   dataset |   train |   test |
    |---:|----------:|--------:|-------:|
    |  0 |       212 |     167 |     45 |
    |  1 |       357 |     289 |     68 |
    
    
    Training ===================================== >>
    Models: MNB, QDA
    Metric: f1, recall
    
    
    Running BO for Multinomial Naive Bayes...
    Initial point 1 ---------------------------------
    Parameters --> {'alpha': 1.0, 'fit_prior': True}
    Evaluation --> f1: 0.9260  Best f1: 0.9260   recall: 0.9722  Best recall: 0.9722
    Time iteration: 3.553s   Total time: 3.559s
    Iteration 2 -------------------------------------
    Parameters --> {'alpha': 9.744, 'fit_prior': True}
    Evaluation --> f1: 0.9225  Best f1: 0.9260   recall: 0.9688  Best recall: 0.9722
    Time iteration: 0.031s   Total time: 3.595s
    Iteration 3 -------------------------------------
    Parameters --> {'alpha': 0.66, 'fit_prior': False}
    Evaluation --> f1: 0.9223  Best f1: 0.9260   recall: 0.9655  Best recall: 0.9722
    Time iteration: 0.031s   Total time: 3.758s
    
    Results for Multinomial Naive Bayes:         
    Bayesian Optimization ---------------------------
    Best parameters --> {'alpha': 1.0, 'fit_prior': True}
    Best evaluation --> f1: 0.9260   recall: 0.9722
    Time elapsed: 3.879s
    Fit ---------------------------------------------
    Train evaluation --> f1: 0.9243   recall: 0.9723
    Test evaluation --> f1: 0.9103   recall: 0.9706
    Time elapsed: 0.012s
    Bagging -----------------------------------------
    Evaluation --> f1: 0.9100 ± 0.0005   recall: 0.9669 ± 0.0064
    Time elapsed: 0.028s
    -------------------------------------------------
    Total time: 3.921s
    
    
    Running BO for Quadratic Discriminant Analysis...
    Initial point 1 ---------------------------------
    Parameters --> {'reg_param': 0}
    Evaluation --> f1: 0.9654  Best f1: 0.9654   recall: 0.9619  Best recall: 0.9619
    Time iteration: 0.039s   Total time: 0.042s
    Iteration 2 -------------------------------------
    Parameters --> {'reg_param': 1.0}
    Evaluation --> f1: 0.9245  Best f1: 0.9654   recall: 0.9897  Best recall: 0.9897
    Time iteration: 0.034s   Total time: 0.080s
    Iteration 3 -------------------------------------
    Parameters --> {'reg_param': 0.0}
    Evaluation --> f1: 0.9633  Best f1: 0.9654   recall: 0.9549  Best recall: 0.9897
    Time iteration: 0.034s   Total time: 0.211s
    
    Results for Quadratic Discriminant Analysis:         
    Bayesian Optimization ---------------------------
    Best parameters --> {'reg_param': 0}
    Best evaluation --> f1: 0.9654   recall: 0.9619
    Time elapsed: 0.315s
    Fit ---------------------------------------------
    Train evaluation --> f1: 0.9828   recall: 0.9896
    Test evaluation --> f1: 0.9710   recall: 0.9853
    Time elapsed: 0.014s
    Bagging -----------------------------------------
    Evaluation --> f1: 0.9606 ± 0.0081   recall: 0.9853 ± 0.0104
    Time elapsed: 0.033s
    -------------------------------------------------
    Total time: 0.363s
    
    
    Final results ========================= >>
    Duration: 4.286s
    ------------------------------------------
    Multinomial Naive Bayes         --> f1: 0.910 ± 0.001   recall: 0.967 ± 0.006
    Quadratic Discriminant Analysis --> f1: 0.961 ± 0.008   recall: 0.985 ± 0.010 !
    

## Analyze the results


```python
# Note that some columns in the results dataframe now contain a list of scores,
# one for each metric, in the same order as you called them
atom.results[['metric_bo', 'metric_train', 'metric_test']]
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
      <th>metric_bo</th>
      <th>metric_train</th>
      <th>metric_test</th>
    </tr>
    <tr>
      <th>model</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MNB</th>
      <td>[0.9259597646215939, 0.9722323049001815]</td>
      <td>[0.924342105263158, 0.972318339100346]</td>
      <td>[0.9103448275862068, 0.9705882352941176]</td>
    </tr>
    <tr>
      <th>QDA</th>
      <td>[0.965402611638704, 0.9618874773139746]</td>
      <td>[0.9828178694158075, 0.9896193771626297]</td>
      <td>[0.9710144927536232, 0.9852941176470589]</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Some plots allow us to choose the metric we want to show
atom.plot_bagging(metric='recall')
```


![png](output_8_0.png)

