# predict_proba
---------------

<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">predict_proba</strong>(X, verbose=None)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/basepredictor.py#L193">[source]</a>
</span>
</div>

Transform new data through the current branch and return class
probabilities. Transformers that are only applied on the training set
are skipped. If called from a trainer, the best model in the pipeline
(under the `winner` attribute) is used. If called from a model, that
model is used. The estimator must have a `predict_proba` method.

<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>X: dataframe-like</strong><br>
Feature set with shape=(n_samples, n_features).
</p>
<p>
<strong>verbose: int or None, optional (default=None)</strong><br>
Verbosity level of the output. If None, it uses the transformer's own verbosity.
</p>
</td>
</tr>
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Returns:</strong></td>
<td width="80%" class="td_params">
<strong>p: np.ndarray</strong><br>
The class probabilities of the input samples, with shape=(n_samples,)
for binary classification tasks and (n_samples, n_classes) for
multiclass classification tasks.
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
predictions = atom.adab.predict_proba(X_new)
```
