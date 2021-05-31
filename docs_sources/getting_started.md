# Installation
--------------

Install ATOM's newest release easily via `pip`:

    $ pip install -U atom-ml

or via `conda`:

    $ conda install -c conda-forge atom-ml

Note that these commands will also install all [required dependencies](../dependencies/#required).
To install the [optional dependencies](../dependencies/#optional) as well, add [models]
after the package's name.

    $ pip install -U atom-ml[models]

!!! info
    Since atom was already taken, download the package under the name `atom-ml`!


<br><br>

# Usage
-------

Call the `ATOMClassifier` or `ATOMRegressor` class and provide the data you want to use:

```python
from sklearn.datasets import load_breast_cancer
from atom import ATOMClassifier

X, y = load_breast_cancer(return_X_y)
atom = ATOMClassifier(X, y, logger="auto", n_jobs=2, verbose=2)
```

ATOM has multiple data cleaning methods to help you prepare the data for modelling:

```python
atom.impute(strat_num="knn", strat_cat="most_frequent", min_frac_rows=0.1)  
atom.encode(strategy="LeaveOneOut", max_onehot=8, frac_to_other=0.05)  
atom.feature_selection(strategy="PCA", n_features=12)
```

Train and evaluate the models you want to compare:

```python
atom.run(
    models=["LR", "LDA", "XGB", "lSVM"],
    metric="f1",
    n_calls=25,
    n_initial_points=10,
    n_bootstrap=4,
)
```

Make plots to analyze the results: 

```python
atom.plot_results(figsize=(9, 6), filename="bootstrap_results.png")  
atom.lda.plot_confusion_matrix(normalize=True, filename="cm.png")
```
