# Predicting
------------

## Prediction methods

After training a model, you probably want to make predictions on new,
unseen data. Just like a sklearn estimator, you can call the prediction
methods from the model, e.g. `atom.tree.predict(X)`.

All prediction methods transform the provided data through the pipeline
in the model's branch before making the predictions. Transformers that
should only be applied on the training set are excluded from this step
(e.g. outlier pruning or class balancing).

The available prediction methods are the most common methods for estimators
in sklearn's API:

:: atom.basemodel:BaseModel
    :: members:
        toc_only: True
        url: ../../API/predicting/
        include:
            - decision_function
            - predict
            - predict_log_proba
            - predict_proba
            - score


## Prediction attributes

The prediction methods can be calculated on the train, test and
holdout set. You can access them through attributes of the form
[method]_[data_set], e.g. `atom.mnb.predict_train`, `atom.mnb.predict_test`
or `atom.mnb.predict_holdout`. The predictions for these attributes
are not calculated until the attribute is called for the first time.
This mechanism avoids having to make (perhaps) expensive calculations
that are never used, saving time and memory.

!!! note
    Many of the [plots][] use the prediction attributes. This can considerably
    increase the size of the instance for large datasets. Use the [clear](../../API/ATOM/atomclassifier/#clear)
    method if you need to free some memory.


## Predictions on rows in the dataset

It's also possible to get the prediction for a specific row or rows in
the dataset, providing the names or positions of their indices, e.g.
`atom.rf.predict(10)` returns the random forest's prediction on the 10th
row in the dataset, or `atom.rf.predict_proba(["index1", "index2"])`
returns the class probabilities for the rows in the dataset with indices
`index1` and `index2`.

