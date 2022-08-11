# beeswarm_plot
---------------

<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">beeswarm_plot</strong>(models=None,
index=None, show=None, target=1, title=None, figsize=None, filename=None,
display=True, **kwargs)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L3274">[source]</a>
</span>
</div>

Plot SHAP's beeswarm plot. The plot is colored by feature values.
Read more about SHAP plots in the [user guide](../../../user_guide/plots/#shap).

<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>models: int, str, slice, sequence or None, default=None</strong><br>
Name of the model to plot. If None, all models are selected.
Note that leaving the default option could raise an exception
if there are multiple models. To avoid this, call the plot from a
model, e.g. <code>atom.xgb.beeswarm_plot()</code>.
</p>
<p>
<strong>index: slice, sequence or None, default=None</strong><br>
Index names or positions of the rows in the dataset to plot.
If None, it selects all rows in the test set. The beeswarm
plot does not support plotting a single sample.
</p>
<p>
<strong>show: int or None, default=None</strong><br>
Number of features (ordered by importance) to show. None to show all.
</p>
<p>
<strong>target: int or str, default=1</strong><br>
Index or name of the class in the target column to look at. Only
for multi-class classification tasks.
</p>
<p>
<strong>title: str or None, default=None</strong><br>
Plot's title. If None, the title is left empty.
</p>
<p>
<strong>figsize: tuple or None, default=None</strong><br>
Figure's size, format as (x, y). If None, it adapts the size to the
number of features shown.
</p>
<p>
<strong>filename: str or None, default=None</strong><br>
Name of the file. Use "auto" for automatic naming.
If None, the figure is not saved.
</p>
<p>
<strong>display: bool or None, default=True</strong><br>
Whether to render the plot. If None, it returns the matplotlib figure.
</p>
<p>
<strong>**kwargs</strong><br>
Additional keyword arguments for SHAP's <a href="https://shap.readthedocs.io/en/latest/generated/shap.plots.beeswarm.html">beeswarm plot</a>.
</p>
</td>
</tr>
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Returns:</strong></td>
<td width="80%" class="td_params">
<strong>matplotlib.figure.Figure</strong><br>
Plot object. Only returned if <code>display=None</code>.
</td>
</tr>
</table>
<br />



## Example

```python
from atom import ATOMRegressor

atom = ATOMRegressor(X, y)
atom.run("RF")
atom.beeswarm_plot()
```

<div align="center">
    <img src="../../../img/plots/beeswarm_plot.png" alt="beeswarm_plot" width="700" height="700"/>
</div>
