# Predicting
------------

After training a model, you probably want to make predictions on new,
unseen data. Just like a sklearn estimator, you can call the prediction
methods from the model, e.g., `#!python atom.tree.predict(X)`.

All prediction methods transform the provided data through the pipeline
in the model's branch before making the predictions. Transformers that
should only be applied on the training set are excluded from this step
(e.g., outlier pruning or class balancing).

The available prediction methods are the standard methods for estimators
in sklearn's and sktime's API.

For classification and regression tasks:

:: atom.models:AdaBoost
    :: methods:
        toc_only: True
        include:
            - decision_function
            - predict
            - predict_log_proba
            - predict_proba
            - score

For forecast tasks:

:: atom.models:ARIMA
    :: methods:
        toc_only: True
        include:
            - predict
            - predict_interval
            - predict_proba
            - predict_quantiles
            - predict_var
            - score


!!! warning
    The `score` method return atom's metric score, not the metric returned
    by sklearn's score method for estimators. Use the method's `metric`
    parameter to calculate a different metric.

!!! note
    * The output of ATOM's methods are pandas objects, not numpy arrays.
    * The `predict_proba` method of some meta-estimators for [multioutput tasks][]
    (such as [MultioutputClassifier][]) return 3 dimensions, namely, a list of
    arrays with shape=(n_samples, n_classes). One array per target column. Since
    ATOM's prediction methods return pandas objects, such 3-dimensional arrays
    are converted to a multiindex pd.DataFrame, where the first level of the row
    indices are the target columns, and the second level are the classes.


It's also possible to get the prediction for a specific row or rows in
the dataset. See the [row and column selection][] section in the user guide
to learn how to select the rows, e.g., `#!python atom.rf.predict("test")`
or `#!python atom.rf.predict_proba(range(100))`.

!!! note
    For forecast models, prediction on rows follow the [ForecastingHorizon][]
    API. That means that using the row index works, but for example using
    `#!python atom.arima.predict(1)` returns the prediction on the first row
    of the test set (instead of the second row of the train set).
