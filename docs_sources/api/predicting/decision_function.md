# decision_function
-------------------

<pre><em>method</em> <strong style="color:#008AB8">decision_function</strong>(X, verbose=None, **kwargs) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basepredictor.py#L142">[source]</a></div></pre>
Transform new data through all transformers in a branch and return
predicted confidence scores. If called from a trainer, it will use the
best model in the pipeline (under the `winner` attribute). If called
from a model, it will use that model. The estimator must have a
<code>decision_function</code> method.
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
Predicted confidence scores of the input samples, with shape=(n_samples,) for binary
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
atom.run("kSVM", metric="accuracy")

# Predict confidence scores on new data
predictions = atom.ksvm.decision_function(X_new)
```