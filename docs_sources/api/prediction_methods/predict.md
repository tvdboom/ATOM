# predict
---------

<pre><em>function</em> <strong style="color:#008AB8">predict</strong>(X, \*\*kwargs) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L711">[source]</a></div></pre>
<div style="padding-left:3%">
Transform the data and make predictions using the winning model in the pipeline.
The model has to have a `predict` method.
 <br /><br />
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>X: dict, sequence, np.array or pd.DataFrame</strong>
<blockquote>
Data containing the features, with shape=(n_samples, n_features).
</blockquote>
<strong>**kwargs</strong>
<blockquote>
Same arguments as the <a href="#atomclassifier-transform">transform</a> method to
 include/exclude pre-processing steps from the transformer.
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
