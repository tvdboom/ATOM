# load
------

<pre><em>function</em> atom.api.<strong style="color:#008AB8">load</strong>(filename=None, X=None, y=-1)
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
<strong>X: dict, sequence, np.array, pd.DataFrame or None, optional (default=None)</strong>
<blockquote>
Dataset containing the features, with shape=(n_samples, n_features). Only if the file
 is an ATOM or training instance that was saved using save_data=False. See the
 <a href="../atomclassifier/#atomclassifier-save">save</a> method.
</blockquote>
<strong>y: int, str, sequence, np.array or pd.Series, optional (default=-1)</strong>
<blockquote>
<ul>
<li>If int: Position of the target column in X.</li>
<li>If string: Name of the target column in X</li>
<li>Else: Data target column with shape=(n_samples,)</li>
</ul>
Only used if X is provided.
</blockquote>
</tr>
</table>
</div>
<br />



## Example
----------
```python
from atom import ATOMClassifier, load

# Save an atom instance to a pickle file
atom = ATOMClassifier(X, y)
atom.run('LR')
atom.save('atom_lr', save_data=False)

# Load the class again
atom = load('atom_lr', X, y)
```
