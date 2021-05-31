# scatter_plot
--------------

<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">scatter_plot</strong>(models=None,
index=None, feature=0, target=1, title=None, figsize=(10, 6),
filename=None, display=True, **kwargs)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L3006">[source]</a>
</span>
</div>

Plot SHAP's scatter plot. Plots the value of the feature on the x-axis
and the SHAP value of the same feature on the y-axis. This shows how
the model depends on the given feature, and is like a richer extension
of the classical partial dependence plots. Vertical dispersion of the
data points represents interaction effects. Read more about SHAP plots
in the [user guide](../../../user_guide/plots/#shap).

<table style="font-size:16px">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<p>
<strong>models: str, sequence or None, optional (default=None)</strong><br>
Name of the model to plot. If None, all models in the pipeline are
selected. Note that leaving the default option could raise an
exception if there are multiple models in the pipeline. To avoid
this, call the plot from a model, e.g. <code>atom.xgb.scatter_plot()</code>.
</p>
<p>
<strong>index: tuple, slice or None, optional (default=None)</strong><br>
Indices of the rows in the dataset to plot. If tuple (n, m), it selects
rows n until m. If None, it selects all rows in the test set. The scatter
plot does not support plotting a single sample.
</p>
<p>
<strong>feature: int or str, optional (default=0)</strong><br>
Index or name of the feature to plot.
</p>
<p>
<strong>target: int or str, optional (default=1)</strong><br>
Index or name of the class in the target column to look at. Only for
multi-class classification tasks.
</p>
<p>
<strong>title: str or None, optional (default=None)</strong><br>
Plot's title. If None, the title is left empty.
</p>
<p>
<strong>figsize: tuple, optional (default=(10, 6)))</strong><br>
Figure's size, format as (x, y).
</p>
<p>
<strong>filename: str or None, optional (default=None)</strong><br>
Name of the file. Use "auto" for automatic naming.
If None, the figure is not saved.
</p>
<p>
<strong>display: bool or None, optional (default=True)</strong><br>
Whether to render the plot. If None, it returns the matplotlib figure.
</p>
<p>
<strong>**kwargs</strong><br>
Additional keyword arguments for SHAP's <a href="https://shap.readthedocs.io/en/latest/generated/shap.plots.scatter.html">scatter plot</a>.
</p>
</td>
</tr>
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Returns:</strong></td>
<td width="80%" style="background:white;">
<strong>fig: matplotlib.figure.Figure</strong><br>
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
atom.scatter_plot(feature="bmi")
```
<div align="center">
    <img src="../../../img/plots/scatter_plot.png" alt="scatter_plot" width="700" height="420"/>
</div>
