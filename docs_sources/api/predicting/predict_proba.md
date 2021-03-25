# predict_proba
---------------

<pre><em>method</em> <strong style="color:#008AB8">predict_proba</strong>(X, pipeline=None, verbose=None) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basepredictor.py#L132">[source]</a></div></pre>
Transform new data through all transformers in a branch and return
class probabilities. If called from a trainer, it will use the best
model in the pipeline (under the `winner` attribute). If called from
a model, it will use that model. The estimator must have a
`predict_proba` method.
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>X: dict, list, tuple, np.ndarray or pd.DataFrame</strong>
<blockquote>
Feature set with shape=(n_samples, n_features).
</blockquote>
<strong>pipeline: bool, sequence or None, optional (default=None)</strong>
<blockquote>
Transformers to use on the data before predicting.
<ul>
<li>If None: Only transformers that are applied on the whole dataset are used.</li>
<li>If False: Don't use any transformers.</li>
<li>If True: Use all transformers in the pipeline.</li>
<li>If sequence: Transformers to use, selected by their index in the pipeline.</li>
</ul>
</blockquote>
<strong>verbose: int or None, optional (default=None)</strong>
<blockquote>
Verbosity level of the output. If None, it uses the transformer's own verbosity.
</blockquote>
</tr>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Returns:</strong></td>
<td width="75%" style="background:white;">
<strong>p: np.ndarray</strong>
<blockquote>
The class probabilities of the input samples, with shape=(n_samples,) for binary
classification tasks and (n_samples, n_classes) for multiclass classification tasks.
</blockquote>
</td>
</tr>
</table>
<br />


## Example
----------

```python
from atom import ATOMClassifier

atom = ATOMClassifier(X, y)
atom.run(["Tree", "AdaB"], metric="AP", n_calls=10)

# Make predictions on new data
predictions = atom.adab.predict_proba(X_new)
```
