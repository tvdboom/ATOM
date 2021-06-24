# Predicting
------------

After running a successful pipeline, it's possible you would like to
apply all used transformations onto new data, or make predictions using
one of the trained models. Just like a sklearn estimator, you can call
the prediction methods from a fitted trainer, e.g. `atom.predict(X)`.
Calling the method without specifying a model will use the winning model
in the pipeline (under attribute `winner`). To use a different model,
simply call the method from a model, e.g. `atom.AdaB.predict(X)`.

All prediction methods transform the provided data through all
transformers in the current branch before making the predictions.
By default, this excludes transformers that should only be applied
on the training set, like outlier pruning and balancing the dataset.
Use the method's `pipeline` parameter to customize which
transformations to apply with every call.

The available prediction methods are a selection of the most common
methods for estimators in sklearn's API:

<table>
<tr>
<td><a href="../../API/predicting/transform">transform</a></td>
<td>Transform new data through all transformers in a branch.</td>
</tr>

<tr>
<td><a href="../../API/predicting/predict">predict</a></td>
<td>Transform new data through all transformers in a branch and return class predictions.</td>
</tr>

<tr>
<td><a href="../../API/predicting/predict_proba">predict_proba</a></td>
<td>Transform new data through all transformers in a branch and return class probabilities.</td>
</tr>

<tr>
<td><a href="../../API/predicting/predict_log_proba">predict_log_proba</a></td>
<td>Transform new data through all transformers in a branch and return class log-probabilities. </td>
</tr>

<tr>
<td><a href="../../API/predicting/decision_function">decision_function</a></td>
<td>Transform new data through all transformers in a branch and return confidence scores.</td>
</tr>

<tr>
<td><a href="../../API/predicting/score">score</a></td>
<td>Transform new data through all transformers in a branch and return a metric score.</td>
</tr>
</table>

Except for transform, the prediction methods can be calculated on the train
and test set. You can access them through the model's prediction attributes,
e.g. `atom.mnb.predict_train` or ` atom.mnb.predict_test`. Keep in mind that
the results are not calculated until the attribute is called for the first
time. This mechanism avoids having to calculate attributes that are never
used, saving time and memory.

!!! note
    Many of the [plots](../plots) use the prediction attributes. This can
    considerably increase the size of the instance for large datasets. Use
    the [reset_predictions](../../API/ATOM/atomclassifier/#reset-predictions)
    method if you need to free some memory!

