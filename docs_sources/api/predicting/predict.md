# predict
---------

<pre><em>method</em> <strong style="color:#008AB8">predict</strong>(X, verbose=None, **kwargs) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basepredictor.py#L124">[source]</a></div></pre>
Transform new data through all transformers in a branch and return
class predictions. If called from a trainer, it will use the best model
in the pipeline (under the `winner` attribute). If called from a
model, it will use that model. The estimator must have a
`predict` method.
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>X: dict, list, tuple, np.ndarray or pd.DataFrame</strong>
<blockquote>
Feature set with shape=(n_samples, n_features).
</blockquote>
<strong>verbose: int or None, optional (default=None)</strong>
<blockquote>
Verbosity level of the output. If None, it uses the trainer's
verbosity. Possible values are:
<ul>
<li>0 to not print anything.</li>
<li>1 to print basic information.</li>
<li>2 to print detailed information.</li>
</ul>
</blockquote>
<strong>**kwargs</strong>
<blockquote>
Same keyword arguments as the <a href="../transform">transform</a> method to
include/exclude transformers from the transformations.
</blockquote>
</tr>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Returns:</strong></td>
<td width="75%" style="background:white;">
<strong>p: np.ndarray</strong>
<blockquote>
Predicted targets with shape=(n_samples,).
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
predictions = atom.adab.predict(X_new)
```
