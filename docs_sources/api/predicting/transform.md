# transform
-----------

<a name="atom"></a>
<pre><em>method</em> <strong style="color:#008AB8">transform</strong>(X, y=None, verbose=None, **kwargs) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L304">[source]</a></div></pre>
<div style="padding-left:3%">
Transform new data through all the pre-processing steps in the pipeline. By default,
 all transformers are included except [outliers](../../ATOM/atomclassifier/#atomclassifier-outliers)
 and [balance](../../ATOM/atomclassifier/#atomclassifier-balance) since they should
 only be applied on the training set. Can only be called from `atom`.
<br /><br />
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>X: dict, sequence, np.array or pd.DataFrame</strong>
<blockquote>
Features to transform, with shape=(n_samples, n_features).
</blockquote>
<strong>y: int, str, sequence, np.array, pd.Series or None, optional (default=None)</strong>
<blockquote>
<ul>
<li>If None: y is ignored in the transformers.</li>
<li>If int: Position of the target column in X.</li>
<li>If string: Name of the target column in X.</li>
<li>Else: Target column with shape=(n_samples,).</li>
</ul>
</blockquote>
<strong>verbose: int or None, optional (default=None)</strong>
<blockquote>
Verbosity level of the output. If None, it uses ATOM's verbosity. Possible values are:
<ul>
<li>0 to not print anything.</li>
<li>1 to print basic information.</li>
<li>2 to print detailed information.</li>
</ul>
</blockquote>
<strong>\*\*kwargs</strong>
<blockquote>
Additional keyword arguments to customize which transformers to apply. You can
 either select them including their index in the `pipeline` parameter,
 e.g. `pipeline=[0, 1, 4]` or include/exclude them individually using their
 methods, e.g. `impute=True` or `feature_selection=False`.
</blockquote>
</tr>
</table>
</div>
<br />

!!! note
    When using the pipeline parameter to include/exclude transformers, remember
    that the first transformer (index 0) in `atom`'s pipeline is always the
    [StandardCleaner](../data_cleaning/standard_cleaner.md) called during
    initialization.

<br>


## Example
----------

```python
from atom import ATOMClassifier

atom = ATOMClassifier(X, y)
atom.impute(strat_num='knn', strat_cat='drop')
atom.outliers(strategy='min_max', max_sigma=2)
atom.feature_generation(strategy='gfg', n_features=3, generations=10, population=1000)

# Apply only the StandardCleaner and Imputer on new data
X_transformed = atom.transform(X_new, pipeline=[0, 1])
```
