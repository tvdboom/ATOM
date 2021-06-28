# First steps
-------------

You can quickly install atom using `pip` or `conda`, see the [installation guide](../../getting_started/#installation).
ATOM contains a variety of classes and functions to perform data cleaning,
feature engineering, model training, plotting and much more. The easiest
way to use everything ATOM has to offer is through one of the main classes:

* [ATOMClassifier](../../API/ATOM/atomclassifier) for binary or multiclass classification tasks.
* [ATOMRegressor](../../API/ATOM/atomregressor) for regression tasks.

These two classes are convenient wrappers for the whole machine learning
pipeline. Like a sklearn [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html),
they assemble several steps that can be cross-validated together while
setting different parameters. There are, however, some important
differences with sklearn's API:

1. atom is initialized with the data you want to manipulate. This data can be
   accessed at any moment through atom's [data attributes](../../API/ATOM/atomclassifier/#data-attributes).
2. The classes in ATOM's API are reached through atom's methods. For example,
   calling the [encode](../../API/ATOM/atomclassifier/#encode) method will
   initialize an [Encoder](../../API/data_cleaning/encoder) instance, fit it on
   the training set and transform the whole dataset.
3. The transformations are applied immediately after calling the method from
   atom (there is no fit command). This approach saves lines of code and gives
   the user a clearer overview and more control over every step in the pipeline.

Let's get started with an example!

First, initialize atom and provide it the data you want to use. You can
either input a dataset and let ATOM split the train and test set or provide
a train and test set already split. Note that if a dataframe is provided,
the indices are reset by atom.

```python
atom = ATOMClassifier(X, y, test_size=0.25)
```

Apply data cleaning steps through atom's methods. For example, calling
[impute](../../API/ATOM/atomclassifier/#impute) will handle all missing
values in the dataset.

```python
atom.impute(strat_num="median", strat_cat="most_frequent", max_nan_rows=0.1)
```

When the data is ready for modelling, call the [run](../../API/ATOM/atomclassifier/#run)
method. Here, we [tune hyperparameters](../training/#hyperparameter-tuning) and
fit a [Random Forest](../../API/models/rf) and [AdaBoost](../../API/models/adab)
model to the data.

```python
atom.run(["RF", "AdaB"], metric="accuracy", n_calls=25, n_initial_points=10)
```

Finally, visualize the result using the integrated [plots](../plots).

```python
atom.plot_feature_importance(show=10, filename="feature_importance_plot")
atom.plot_prc(title="Precision-recall curve comparison plot")
```
