# predict_log_proba
-------------------

<a name="atom"></a>
<pre><em>method</em> <strong style="color:#008AB8">predict_log_proba</strong>(X, verbose=None, \*\*kwargs) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basepredictor.py#L134">[source]</a></div></pre>
<div style="padding-left:3%">
Transform the data and make logarithmic probability predictions on new data. If
 called from a `training` instance, it will use the best model in the pipeline (under
 the `winner` attribute). If called from a `model`, it will use that model. The
 estimator must have a `predict_proba` method.
<br /><br />
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>X: dict, list, tuple, np.array or pd.DataFrame</strong>
<blockquote>
Data containing the features, with shape=(n_samples, n_features).
</blockquote>
<strong>verbose: int or None, optional (default=None)</strong>
<blockquote>
Verbosity level of the output. If None, it uses the `training`"s verbosity. Possible values are:
<ul>
<li>0 to not print anything.</li>
<li>1 to print basic information.</li>
<li>2 to print detailed information.</li>
</ul>
</blockquote>
<strong>**kwargs</strong>
<blockquote>
Same keyword arguments as the [transform](transform.md) method to
 include/exclude transformers from the transformations.
</blockquote>
</tr>
</table>
</div>
<br />


## Example
----------

```python
from atom import ATOMClassifier

atom = ATOMClassifier(X, y)
atom.run(["Tree", "AdaB"], metric="AP", n_calls=10)

# Make predictions on new data
predictions = atom.adab.predict_log_proba(X_new)
```