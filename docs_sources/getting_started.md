# Getting started
-----------------

## Installation

Install ATOM's newest release easily via `pip`:

    $ pip install -U atom-ml

or via `conda`:

    $ conda install -c conda-forge atom-ml

These commands will install ATOM and all [required dependencies](../dependencies/#required).
To install the [optional dependencies](../dependencies/#optional) as well, add [models]
after the package's name.

    $ pip install -U atom-ml[models]

!!! note
    Since atom was already taken, download the package under the name `atom-ml`!

Sometimes, new features and bug fixes are already implemented in the
`development` branch, but waiting for the next release to be made
available. If you can't wait for that, it's possible to install the
package directly from git.

    $ pip install git+https://github.com/tvdboom/ATOM.git@development#egg=atom-ml

Don't forget to include `#egg=atom-ml` to explicitly name the project,
this way pip can track metadata for it without having to have run the
`setup.py` script.

<br><br>

## Usage

Call the `ATOMClassifier` or `ATOMRegressor` class and provide the data you want to use:

```python
from sklearn.datasets import load_breast_cancer
from atom import ATOMClassifier

X, y = load_breast_cancer(return_X_y=True)
atom = ATOMClassifier(X, y, logger="auto", n_jobs=2, verbose=2)
```

ATOM has multiple data cleaning methods to help you prepare the data for modelling:

```python
atom.impute(strat_num="knn", strat_cat="most_frequent", max_nan_rows=0.1)  
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
