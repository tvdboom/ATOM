# plot_pipeline
---------------

<a name="atom"></a>
<pre><em>method</em> <strong style="color:#008AB8">plot_pipeline</strong>(show_params=True, title=None, figsize=None, filename=None, display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L2621">[source]</a></div></pre>
<div style="padding-left:3%">
Plot a diagram of every estimator in `atom`"s pipeline.
 <br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>show_params: bool, optional (default=True)</strong>
<blockquote>
Whether to show the parameters used for every estimator.
</blockquote>
<strong>title: str or None, optional (default=None)</strong>
<blockquote>
Plot's title. If None, the default option is used.
</blockquote>
<strong>figsize: tuple or None, optional (default=None)</strong>
<blockquote>
Figure's size, format as (x, y). If None, adapts size to the length of the pipeline.
</blockquote>
<strong>filename: str or None, optional (default=None)</strong>
<blockquote>
Name of the file (to save). If None, the figure is not saved.
</blockquote>
<strong>display: bool, optional (default=True)</strong>
<blockquote>
Whether to render the plot.
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
atom.impute(strat_num="median", strat_cat="drop", min_frac_rows=0.8)
atom.encode(strategy="LeaveOneOut", max_onehot=8, frac_to_other=0.02)
atom.outliers(strategy="drop", max_sigma=4, include_target=False)
atom.feature_selection(
    strategy="PCA",
    n_features=10,
    max_frac_repeated=1.,
    max_correlation=0.7
)

atom.run(
    models=["GBM", "LGB"],
    metric="recall_weighted",
    n_calls=(10, 20),
    n_initial_points=(5, 12),
    bo_params={"base_estimator": "RF", "cv": 1, "max_time": 1000},
    bagging=4
)

atom.plot_pipeline()
```
<div align="center">
    <img src="../../../img/plots/plot_pipeline.png" alt="plot_pipeline" width="700" height="1200"/>
</div>
