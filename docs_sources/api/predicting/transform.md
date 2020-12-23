# transform
-----------

<pre><em>method</em> <strong style="color:#008AB8">transform</strong>(X, y=None, verbose=None, **kwargs) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L264">[source]</a></div></pre>
Transform new data through all transformers in a branch. By default,
 all transformers are included except [outliers](../../ATOM/atomclassifier/#outliers)
 and [balance](../../ATOM/atomclassifier/#balance) since they should
 only be applied on the training set. Can only be called from atom.
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
<strong>verbose: int or None, optional (default=None)</strong>
<blockquote>
Verbosity level of the output. If None, it uses the `training`'s verbosity. Possible values are:
<ul>
<li>0 to not print anything.</li>
<li>1 to print basic information.</li>
<li>2 to print detailed information.</li>
</ul>
</blockquote>
<strong>**kwargs</strong>
<blockquote>
Additional keyword arguments to customize which transformers to apply. You can
 either select them including their index in the <code>pipeline</code> parameter,
 e.g. <code>pipeline=[0, 1, 4]</code> or include/exclude them individually using their
 methods, e.g. <code>outliers=True</code> or <code>feature_selection=False</code>.
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
atom.outliers(strategy="min_max", max_sigma=2)

# Transform new data through all data cleaning steps
X_transformed = atom.transform(X_new)
```
