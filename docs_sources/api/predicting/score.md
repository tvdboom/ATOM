# score
-------

<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">score</strong>(X,
y, metric=None, sample_weights=None, pipeline=None, verbose=None)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/basepredictor.py#L210">[source]</a>
</span>
</div>

Transform new data through all transformers in the current branch and
return a metric score. If called from a trainer, the best model in
the pipeline (under the `winner` attribute) is used. If called from a
model, that model is used.

<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>X: dict, list, tuple, np.array, sps.matrix or pd.DataFrame</strong><br>
Feature set with shape=(n_samples, n_features).
</p>
<strong>y: int, str or sequence</strong><br>
<ul style="line-height:1.2em;margin-top:5px">
<li>If int: Position of the target column in X.</li>
<li>If str: Name of the target column in X.</li>
<li>Else: Target column with shape=(n_samples,).</li>
</ul>
<p>
<strong>metric: str, func, scorer or None, optional (default=None)</strong><br>
Metric to calculate. Choose from any of sklearn's <a href="https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules">SCORERS</a>,
a function with signature <code>metric(y_true, y_pred)</code> or
a scorer object. If None, it returns mean accuracy for classification
tasks and r2 for regression tasks.
</p>
<p>
<strong>sample_weights: sequence or None, optional (default=None)</strong><br>
Sample weights corresponding to y.
</p>
<strong>pipeline: bool, sequence or None, optional (default=None)</strong><br>
Transformers to use on the data before predicting.
<ul style="line-height:1.2em;margin-top:5px">
<li>If None: Only transformers that are applied on the whole dataset are used.</li>
<li>If False: Don't use any transformers.</li>
<li>If True: Use all transformers in the pipeline.</li>
<li>If sequence: Transformers to use, selected by their index in the pipeline.</li>
</ul>
<p>
<strong>verbose: int or None, optional (default=None)</strong><br>
Verbosity level of the output. If None, it uses the transformer's own verbosity.
</p>
</td>
</tr>
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Returns:</strong></td>
<td width="80%" class="td_params">
<strong>score: np.float64</strong><br>
Metric score of X with respect to y.
</td>
</tr>
</table>

!!! note
    If the `metric` parameter is left to its default value, the method
    outputs the same value as sklearn's score method for an estimator. 

!!! info
    This method is intended to calculate metric scores on new data.
    To get the metric results on the train or test set, use the
    [evaluate](../../ATOM/atomclassifier/#evaluate) method.

<br />



## Example

```python
from atom import ATOMClassifier

atom = ATOMClassifier(X, y)
atom.run(["MNB", "KNN", "kSVM"], metric="precision")

# Get the mean accuracy on new data
predictions = atom.mnb.score(X_new, y_new)
```