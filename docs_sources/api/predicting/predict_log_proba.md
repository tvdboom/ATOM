# predict_log_proba
-------------------

<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">predict_log_proba</strong>
(X, pipeline=None, verbose=None)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/basepredictor.py#L140">[source]</a>
</span>
</div>

Transform new data through all transformers in the current branch and
return class log-probabilities. If called from a trainer, the best model
in the pipeline (under the `winner` attribute) is used. If called from a
model, that model is used. The estimator must have a `predict_log_proba`
method.

<table style="font-size:16px">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<p>
<strong>X: dict, list, tuple, np.ndarray or pd.DataFrame</strong><br>
Feature set with shape=(n_samples, n_features).
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
</tr>
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Returns:</strong></td>
<td width="80%" style="background:white;">
<strong>p: np.ndarray</strong><br>
The class log-probabilities of the input samples, with shape=(n_samples,)
for binary classification tasks and (n_samples, n_classes) for multiclass
classification tasks.
</td>
</tr>
</table>
<br />



## Example

```python
from atom import ATOMClassifier

atom = ATOMClassifier(X, y)
atom.run(["Tree", "AdaB"], metric="AP", n_calls=10)

# Make predictions on new data
predictions = atom.adab.predict_log_proba(X_new)
```