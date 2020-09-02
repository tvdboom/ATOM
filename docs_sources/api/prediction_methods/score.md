# score
-------

<a name="atom"></a>
<pre><em>method</em> <strong style="color:#008AB8">score</strong>(X, y, verbose=None, \*\*kwargs) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basepredictor.py#L131">[source]</a></div></pre>
<div style="padding-left:3%">
Transform the data and return the model's score on new data. If called from a
 [training instance](../../../user_guide/#training) instance, it will use
 the best model in the pipeline (under the `winner` attribute). If called from a
 [model subclass](../../../user_guide/#model-subclasses), it will use that model.
 The model has to have a `score` method.
<br /><br />
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>X: dict, sequence, np.array or pd.DataFrame</strong>
<blockquote>
Data containing the features, with shape=(n_samples, n_features).
</blockquote>
<strong>y: int, str, sequence, np.array or pd.Series</strong>
<blockquote>
<ul>
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
<strong>**kwargs</strong>
<blockquote>
Same keyword arguments as the [transform](transform.md) method to
 include/exclude transformers from the transformations.
</blockquote>
</tr>
</table>
</div>
<br />

!!! note
    The returned metric is determined by each model's score method pre-defined
    by its respective package. See the corresponding documentation for further
    details.

<br>

## Example
----------
```python
from atom import ATOMClassifier

atom = ATOMClassifier(X, y)
atom.run(['MNB', 'KNN', 'kSVM'], metric='precision')

# Get the mean accuracy on new data
predictions = atom.kSVM.score(X_new, y_new)
```