# ATOMLoader
------------

<pre><em>function</em> atom.api.<strong style="color:#008AB8">ATOMLoader</strong>(filename=None, data=None, transform_data=True, verbose=0)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L336">[source]</a></div></pre>
<div style="padding-left:3%">
Load a class from a pickle file. If its an ATOM or training instance, you can load 
 new data in it.
<br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>filename: str</strong>
<blockquote>
Name of the pickle file to load.
</blockquote>
<strong>dict, sequence, np.array, pd.DataFrame or tuple, optional (default=None)</strong>
<blockquote>
If not tuple, dataset containing the features, with shape=(n_samples, n_features + 1).
 The last column will be used as target column. If tuple, it should be of the form
 (X, y), where X is the feature set and y the corresponding target column as
 array-like, index or name. Only use this parameter if the file is an ATOM or
 training instance that was saved using save_data=False. See the
 <a href="../atomclassifier/#atomclassifier-save">save</a> method.
</blockquote>
<strong>transform_data: bool, optional (default=True)</strong>
<blockquote>
Whether to transform the provided data through all the steps in the instance's
 pipeline. Only if the loaded file is an ATOM instance.
</blockquote>
<strong>verbose: int or None, optional (default=None)</strong>
<blockquote>
Verbosity level of the transformations applied on the new data. If None, use the
 verbosity from the loaded instance.
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
atom.run('LR')
atom.save('atom_lr', save_data=False)

# Load the class again
atom = ATOMLoader('atom_lr', data=(X, y))
```
