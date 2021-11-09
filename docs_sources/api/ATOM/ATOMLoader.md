# ATOMLoader
------------

<div style="font-size:20px">
<em>function</em> atom.api.<strong style="color:#008AB8">ATOMLoader</strong>(filename,
data=None, transform_data=True, verbose=None)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/api.py#L70">[source]</a>
</span>
</div>

Load a class instance from a pickle file. If the file is a trainer that
was saved using `save_data=False`, it is possible to load new data into
it. For atom pickles, all data transformations in the pipeline can be
applied to the loaded data.
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>filename: str</strong><br>
Name of the pickle file to load.
</p>
<strong>data: tuple of indexables or None, optional (default=None)</strong><br>
Tuple containing the features and target data. Only use this parameter
if the file is a trainer that was saved using <code>save_data=False</code> (see
the <a href="../atomclassifier/#save">save</a> method). Allowed formats are:
<ul style="line-height:1.2em;margin-top:5px">
<li>X</li>
<li>X, y</li>
<li>train, test</li>
<li>train, test, holdout</li>
<li>X_train, X_test, y_train, y_test</li>
<li>X_train, X_test, X_holdout, y_train, y_test, y_holdout</li>
<li>(X_train, y_train), (X_test, y_test)</li>
<li>(X_train, y_train), (X_test, y_test), (X_holdout, y_holdout)</li>
</ul>
X, train, test: dict, list, tuple, np.array, sps.matrix or pd.DataFrame<br>
<p style="margin-top:0;margin-left:15px">
Feature set with shape=(n_samples, n_features). If no y is provided, the
last column is used as target.</p>
y: int, str or sequence<br>
<ul style="line-height:1.2em;margin-top:5px">
<li>If int: Position of the target column in X.</li>
<li>If str: Name of the target column in X.</li>
<li>Else: Target column with shape=(n_samples,).</li>
</ul>
<strong>transform_data: bool, optional (default=True)</strong><br>
If False, the <code>data</code> is left as provided. If True, it is transformed
through all the steps in the instance's pipeline. This parameter is
ignored if the loaded file is not an atom pickle.
<p>
<strong>verbose: int or None, optional (default=None)</strong><br>
Verbosity level of the transformations applied to the new data. If
None, use the verbosity from the loaded instance. This parameter
is ignored if <code>transform_data=False</code>.
</p>
</td>
</tr>
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Returns:</strong></td>
<td width="80%" class="td_params">
<strong>cls: class instance</strong><br>
Un-pickled instance.
</td>
</tr>
</table>
<br />



## Example

```python
from atom import ATOMClassifier, ATOMLoader

atom = ATOMClassifier(X, y)
atom.run("LR", metric="AP", n_calls=25, n_initial_points=10)
atom.save("atom", save_data=False)  # Save atom to a pickle file

# Load the class and add the data to the new instance
atom_2 = ATOMLoader("atom", data=(X, y), verbose=0)
```
