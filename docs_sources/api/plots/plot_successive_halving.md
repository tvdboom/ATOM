# plot_successive_halving
-------------------------

<a name="atom-plot-successive-halving"></a>
<pre><em>function</em> atom.plots.<strong style="color:#008AB8">plot_successive_halving</strong>(models=None, metric=0, title=None,
                                            figsize=(10, 6), filename=None, display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L336">[source]</a></div></pre>
<div style="padding-left:3%">
Plot of the models' scores per iteration of the successive halving. Only
 available if the models were fitted with a
 [SuccessiveHalvingClassifier](../training/successivehalvingclassifier.md)/[SuccessiveHalvingRegressor](../training/successivehalvingregressor.md)
 instance.
<br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>models: str, sequence or None, optional (default=None)</strong>
<blockquote>
Name of the models to plot. If None, all the models in the pipeline are selected.
 If you call the plot from a model subclass, the parameter will automatically
 be filled with the model's name.
</blockquote>
<strong>metric: int or str, optional (default=0)</strong>
<blockquote>
Index or name of the metric to plot. Only for multi-metric runs.
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
atom successive_halving(['tree', 'bag', 'et', 'rf', 'gbm', 'lgb'], metric='neg_mean_squared_error')
atom.plot_successive_halving()
```
![plot_successive_halving](../../img/plots/plot_successive_halving.png)
