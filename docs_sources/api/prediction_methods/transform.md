# transform
-----------

<pre><em>function</em> atom.atom.<strong style="color:#008AB8">transform</strong>(X, standard_cleaner=True, scale=True, impute=True, encode=True, outliers=False,
                             balance=False, feature_generation=True, feature_selection=True, verbose=None) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L711">[source]</a></div></pre>
<div style="padding-left:3%">
Transform new data through all the pre-processing steps in the pipeline. The outliers
 and balancer steps are not included in the default steps since they should only be
 applied on the training set.
 <br /><br />
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>X: dict, sequence, np.array or pd.DataFrame</strong>
<blockquote>
Features to transform, with shape=(n_samples, n_features).
</blockquote>
<strong>standard_cleaner: bool, optional (default=True)</strong>
<blockquote>
Whether to apply the standard cleaning step in the transformer.
</blockquote>
<strong>scale: bool, optional (default=True)</strong>
<blockquote>
Whether to apply the scaler step in the transformer.
</blockquote>
<strong>impute: bool, optional (default=True)</strong>
<blockquote>
Whether to apply the imputer step in the transformer.
</blockquote>
<strong>encode: bool, optional (default=True)</strong>
<blockquote>
Whether to apply the encoder step in the transformer.
</blockquote>
<strong>outliers: bool, optional (default=False)</strong>
<blockquote>
Whether to apply the outliers step in the transformer.
</blockquote>
<strong>balance: bool, optional (default=False)</strong>
<blockquote>
Whether to apply the balance step in the transformer.
</blockquote>
<strong>feature_generation: bool, optional (default=True)</strong>
<blockquote>
Whether to apply the feature_generation step in the transformer.
</blockquote>
<strong>feature_selection: bool, optional (default=True)</strong>
<blockquote>
Whether to apply the feature_selection step in the transformer.
</blockquote>
<strong>verbose: int, optional (default=None)</strong>
<blockquote>
Verbosity level of the output. If None, it uses the instance's verbosity.
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
atom.impute(strat_num='knn', strat_cat='drop')
atom.feature_generation(strategy='gfg', n_features=3, generations=10, population=1000)

# Apply imputing and feature generation on new data
X_transformed = atom.transform(X_new)
```
