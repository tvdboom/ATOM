# transform
-----------

<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">transform</strong>(X, y=None, verbose=None)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L750">[source]</a>
</span>
</div>

Transform unseen data through the current branch. Transformers that are
only applied on the training set are skipped. This method can only be
called from atom, not from the models.

<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>X: dataframe-like</strong><br>
Features to transform, with shape=(n_samples, n_features).
</p>
<strong>y: int, str, sequence or None, optional (default=None)</strong><br>
<ul style="line-height:1.2em;margin-top:5px">
<li>If None: y is ignored in the transformers.</li>
<li>If int: Position of the target column in X.</li>
<li>If str: Name of the target column in X.</li>
<li>Else: Target column with shape=(n_samples,).</li>
</ul>
<p>
<strong>verbose: int or None, optional (default=None)</strong><br>
Verbosity level of the output. If None, it uses the transformer's own verbosity.
</p>
</td>
</tr>
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Returns:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>pd.DataFrame</strong><br>
Transformed feature set.
</p>
<p>
<strong>pd.Series</strong><br>
Transformed target column. Only returned if provided.
</p>
</td>
</tr>
</table>
<br />



## Example

```python
from atom import ATOMClassifier

atom = ATOMClassifier(X, y)
atom.clean()
atom.impute(strat_num="knn", strat_cat="drop")
atom.prune(strategy="z-score", method="min_max", max_sigma=2)

# Transform new data through all transformers in the branch
X_transformed = atom.transform(X_new)
```
