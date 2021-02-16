# heatmap_plot
--------------

<pre><em>method</em> <strong style="color:#008AB8">heatmap_plot</strong>(models=None, index=None, show=None, target=1,
                    title=None, figsize=(8, 6), filename=None, display=True, **kwargs)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L2663">[source]</a></div></pre>
Plot SHAP's heatmap plot. This plot is designed to show the population
substructure of a dataset using supervised clustering and a heatmap.
Supervised clustering involves clustering data points not by their original
feature values but by their explanations. Read more about SHAP plots in the
 [user guide](../../../user_guide/#shap).
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>models: str, sequence or None, optional (default=None)</strong>
<blockquote>
Name of the models to plot. If None, all models in the pipeline are selected. Note
 that selecting multiple models will raise an exception. To avoid this, call the
 plot from a model.
</blockquote>
<strong>index: tuple, slice or None, optional (default=None)</strong>
<blockquote>
Indices of the rows in the dataset to plot. If tuple (n, m), it selects rows
n until m. If None, it selects all rows in the test set. The heatmap plot does
not support plotting a single sample.
</blockquote>
<strong>show: int or None, optional (default=None)</strong>
<blockquote>
Number of features (ordered by importance) to show in the plot. None to show all.
</blockquote>
<strong>target: int or str, optional (default=1)</strong>
<blockquote>
Index or name of the class in the target column to look at. Only for multi-class
 classification tasks.
</blockquote>
<strong>title: str or None, optional (default=None)</strong>
<blockquote>
Plot's title. If None, the title is left empty.
</blockquote>
<strong>figsize: tuple, optional (default=(8, 6)))</strong>
<blockquote>
Figure's size, format as (x, y).
</blockquote>
<strong>filename: str or None, optional (default=None)</strong>
<blockquote>
Name of the file. If None, the figure is not saved.
</blockquote>
<strong>display: bool, optional (default=True)</strong>
<blockquote>
Whether to render the plot.
</blockquote>
<strong>**kwargs</strong>
<blockquote>
Additional keyword arguments for SHAP's <a href="https://shap.readthedocs.io/en/latest/generated/shap.plots.heatmap.html">heatmap plot</a>.
</blockquote>
</tr>
</table>
<br />



## Example
----------

```python
from atom import ATOMRegressor

atom = ATOMRegressor(X, y)
atom.run("RF")
atom.heatmap_plot()
```
<div align="center">
    <img src="../../../img/plots/heatmap_plot.png" alt="heatmap_plot" width="540" height="420"/>
</div>
