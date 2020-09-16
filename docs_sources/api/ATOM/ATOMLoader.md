# ATOMLoader
------------

<a name="atom"></a>
<pre><em>function</em> <strong style="color:#008AB8">ATOMLoader</strong>(filename=None, X=None, y=-1, transform_data=True, verbose=None)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/api.py#L25">[source]</a></div></pre>
<div style="padding-left:3%">
Load a class instance from a pickle file. If the file is a `training` instance that
 was saved using `save_data=False`, you can load new data into it. If the file is an
 `atom` instance, you can also apply all data transformations in the pipeline to
 the provided data.
<br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>filename: str</strong>
<blockquote>
Name of the pickle file to load.
</blockquote>
<strong>X: dict, sequence, np.array, pd.DataFrame or None, optional (default=None)</strong>
<blockquote>
Data containing the features, with shape=(n_samples, n_features). Only use this
 parameter if the file is a `training` instance that was saved using
 `save_data=False`. See the [save](../atomclassifier/#atomclassifier-save) method.
</blockquote>
<strong>y: int, str, sequence, np.array or pd.Series, optional (default=-1)</strong>
<blockquote>
<ul>
<li>If int: Index of the target column in X.</li>
<li>If string: Name of the target column in X.</li>
<li>Else: Target column with shape=(n_samples,).</li>
</ul>
This parameter is ignored if X=None.
</blockquote>
<strong>transform_data: bool, optional (default=True)</strong>
<blockquote>
Whether to transform the provided data through all the steps in the instance's
 pipeline. This parameter is ignored if the loaded file is not an `atom` instance.
</blockquote>
<strong>verbose: int or None, optional (default=None)</strong>
<blockquote>
Verbosity level of the transformations applied on the new data. If None, use the
 verbosity from the loaded instance. This parameter is ignored if the loaded file
 is not an `atom` instance.
</blockquote>
</tr>
</table>
</div>
<br />



## Example
----------

```python
from atom import ATOMClassifier, ATOMLoader

# Save an atom instance to a pickle file
atom = ATOMClassifier(X, y)
atom.encode(strategy='Helmert', max_onehot=12)
atom.run('LR', metric='AP', n_calls=25, n_initial_points=10)
atom.save('atom_lr', save_data=False)

# Load the class and add the transformed data to the new instance
atom_2 = ATOMLoader('atom_lr', X, y, verbose=0)
```
