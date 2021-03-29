# score
-------

<pre><em>method</em> <strong style="color:#008AB8">score</strong>(X, y, sample_weights=None, pipeline=None, verbose=None) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basepredictor.py#L152">[source]</a></div></pre>
Transform new data through all transformers in a branch and return
the model's score. If called from a trainer, it will use the
best model in the pipeline (under the `winner` attribute). If called
from a model, it will use that model. The estimator must have a
`score` method.
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>X: dict, list, tuple, np.ndarray or pd.DataFrame</strong>
<blockquote>
Feature set with shape=(n_samples, n_features).
</blockquote>
<strong>y: int, str or sequence</strong>
<blockquote>
<ul>
<li>If int: Position of the target column in X.</li>
<li>If str: Name of the target column in X.</li>
<li>Else: Target column with shape=(n_samples,).</li>
</ul>
</blockquote>
<strong>sample_weights: sequence or None, optional (default=None)</strong>
<blockquote>
Sample weights corresponding to y.
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
<strong>score: np.float64</strong>
<blockquote>
Mean accuracy or r2 (depending on the task) of self.predict(X) with respect to y.
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
atom.run(["MNB", "KNN", "kSVM"], metric="precision")

# Get the mean accuracy on new data
predictions = atom.mnb.score(X_new, y_new)
```