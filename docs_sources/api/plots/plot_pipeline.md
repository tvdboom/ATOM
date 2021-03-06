# plot_pipeline
---------------

<pre><em>method</em> <strong style="color:#008AB8">plot_pipeline</strong>(show_params=True, branch=None, title=None, figsize=None, filename=None, display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L3408">[source]</a></div></pre>
Plot a diagram of every estimator in a branch.
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>show_params: bool, optional (default=True)</strong>
<blockquote>
Whether to show the parameters used for every estimator.
</blockquote>
<strong>branch: str or None, optional (default=None)</strong>
<blockquote>
Name of the branch to plot. If None, plot the current branch.
</blockquote>
<strong>title: str or None, optional (default=None)</strong>
<blockquote>
Plot's title. If None, the title is left empty.
</blockquote>
<strong>figsize: tuple or None, optional (default=None)</strong>
<blockquote>
Figure's size, format as (x, y). If None, adapts size to the length of the pipeline.
</blockquote>
<strong>filename: str or None, optional (default=None)</strong>
<blockquote>
Name of the file. If None, the figure is not saved.
</blockquote>
<strong>display: bool, optional (default=True)</strong>
<blockquote>
Whether to render the plot.
</blockquote>
</tr>
</table>
<br />



## Example
----------

```python
from atom import ATOMClassifier

atom = ATOMClassifier(X, y)
atom.impute(strat_num="median", strat_cat="drop", min_frac_rows=0.8)
atom.encode(strategy="LeaveOneOut", max_onehot=8, frac_to_other=0.02)
atom.prune(strategy="drop", max_sigma=4, include_target=False)
atom.feature_selection(
    strategy="PCA",
    n_features=10,
    max_frac_repeated=1.,
    max_correlation=0.7
)

atom.plot_pipeline()
```

<div align="center">
    <img src="../../../img/plots/plot_pipeline.png" alt="plot_pipeline" width="700" height="700"/>
</div>
