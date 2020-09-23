# plot_successive_halving
-------------------------

<a name="atom"></a>
<pre><em>method</em> <strong style="color:#008AB8">plot_successive_halving</strong>(models=None, metric=0, title=None,
                               figsize=(10, 6), filename=None, display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L2407">[source]</a></div></pre>
<div style="padding-left:3%">
Plot of the models' scores per iteration of the successive halving. Only
 available if the models were fitted using [successive halving](../../../user_guide/#successive-halving).
<br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>models: str, sequence or None, optional (default=None)</strong>
<blockquote>
Name of the models to plot. If None, all the models in the pipeline are selected.
</blockquote>
<strong>metric: int or str, optional (default=0)</strong>
<blockquote>
Index or name of the metric to plot. Only for [multi-metric](../../../user_guide/#metric) runs.
</blockquote>
<strong>title: str or None, optional (default=None)</strong>
<blockquote>
Plot's title. If None, the default option is used.
</blockquote>
<strong>figsize: tuple, optional (default=(10, 6))</strong>
<blockquote>
Figure's size, format as (x, y).
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
from atom import ATOMRegressor

atom = ATOMRegressor(X, y)
atom.successive_halving(['tree', 'bag', 'adab', 'et', 'rf', 'gbm', 'xgb', 'lgb'], metric='mse')
atom.plot_successive_halving()
```
<div align="center">
    <img src="/img/plots/plot_successive_halving.png" alt="plot_successive_halving" width="700" height="420"/>
</div>
