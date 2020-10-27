# ATOMLoader
------------

<a name="atom"></a>
<pre><em>function</em> <strong style="color:#008AB8">ATOMLoader</strong>(filename=None, data=None, transform_data=True, verbose=None)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/api.py#L76">[source]</a></div></pre>
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
<strong>data: tuple of indexables or None, optional (default=None)</strong>
<blockquote>
Tuple containing the features and target data. Only use this parameter if the
 file is a `training` instance that was saved using `save_data=False` (see the
 [save](../atomclassifier/#save) method). Allowed formats are:
<ul>
<li>X, y</li>
<li>train, test</li>
<li>X_train, X_test, y_train, y_test</li>
<li>(X_train, y_train), (X_test, y_test)</li>
</ul>
X, train, test: dict, list, tuple, np.array or pd.DataFrame<br>
&nbsp;&nbsp;&nbsp;&nbsp;
Feature set with shape=(n_features, n_samples). If no y is provided, the
 last column is used as target.<br><br>
y: int, str, list, tuple,  np.array or pd.Series<br>
<ul>
<li>If int: Position of the target column in X.</li>
<li>If str: Name of the target column in X.</li>
<li>Else: Data target column with shape=(n_samples,).</li>
</ul>
</blockquote>
<strong>transform_data: bool, optional (default=True)</strong>
<blockquote>
If False, the `data` is left as provided. If True, the `data` is transformed through
 all the steps in the instance's pipeline. This parameter is ignored if the loaded
 file is not an `atom` instance.
</blockquote>
<strong>verbose: int or None, optional (default=None)</strong>
<blockquote>
Verbosity level of the transformations applied on the new data. If None, use the
 verbosity from the loaded instance. This parameter is ignored if `transform_data=False`.
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
atom.encode(strategy="Helmert", max_onehot=12)
atom.run("LR", metric="AP", n_calls=25, n_initial_points=10)
atom.save("atom_lr", save_data=False)

# Load the class and add the transformed data to the new instance
atom_2 = ATOMLoader("atom_lr", data=(X, y), verbose=0)
```
