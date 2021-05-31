# score
-------

<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">score</strong>(X,
y, sample_weights=None, pipeline=None, verbose=None)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/basepredictor.py#L203">[source]</a>
</span>
</div>

Transform new data through all transformers in the current branch and
return model's score. If called from a trainer, the best model in
the pipeline (under the `winner` attribute) is used. If called from a
model, that model is used. The estimator must have a `score` method.

<table style="font-size:16px">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<p>
<strong>X: dict, list, tuple, np.ndarray or pd.DataFrame</strong><br>
Feature set with shape=(n_samples, n_features).
</p>
<strong>y: int, str or sequence</strong><br>
<ul style="line-height:1.2em;margin-top:5px">
<li>If int: Position of the target column in X.</li>
<li>If str: Name of the target column in X.</li>
<li>Else: Target column with shape=(n_samples,).</li>
</ul>
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
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Returns:</strong></td>
<td width="80%" style="background:white;">
<strong>score: np.float64</strong><br>
Mean accuracy or r2 (depending on the task) of predict(X) with respect to y.
</td>
</tr>
</table>
<br />



## Example

```python
from atom import ATOMClassifier

atom = ATOMClassifier(X, y)
atom.run(["MNB", "KNN", "kSVM"], metric="precision")

# Get the mean accuracy on new data
predictions = atom.mnb.score(X_new, y_new)
```