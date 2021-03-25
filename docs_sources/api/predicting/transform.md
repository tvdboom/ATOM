# transform
-----------

<pre><em>method</em> <strong style="color:#008AB8">transform</strong>(X, y=None, pipeline=None, verbose=None) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L380">[source]</a></div></pre>
Transform new data through all transformers in a branch. By default,
transformers that are applied on the training set only are not used
during the transformations. Use the `pipeline` parameter to customize
this behaviour. This method can only be called from atom, not from the models.
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>X: dict, list, tuple, np.ndarray or pd.DataFrame</strong>
<blockquote>
Features to transform, with shape=(n_samples, n_features).
</blockquote>
<strong>y: int, str, sequence or None, optional (default=None)</strong>
<blockquote>
<ul>
<li>If None: y is ignored in the transformers.</li>
<li>If int: Position of the target column in X.</li>
<li>If str: Name of the target column in X.</li>
<li>Else: Target column with shape=(n_samples,).</li>
</ul>
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
<strong>X: pd.DataFrame</strong>
<blockquote>
Transformed feature set.
</blockquote>
<strong>y: pd.Series</strong>
<blockquote>
Transformed target column. Only returned if provided.
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
atom.clean()
atom.impute(strat_num="knn", strat_cat="drop")
atom.prune(strategy="z-score", method="min_max", max_sigma=2)

# Transform new data through all data cleaning steps
X_transformed = atom.transform(X_new)
```
