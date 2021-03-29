# scatter_plot
--------------

<pre><em>method</em> <strong style="color:#008AB8">scatter_plot</strong>(models=None, index=None, feature=0, target=1,
                    title=None, figsize=(10, 6), filename=None, display=True, **kwargs)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L2713">[source]</a></div></pre>
Plot SHAP's scatter plot. Plots the value of the feature on the x-axis
and the SHAP value of the same feature on the y-axis. This shows how
the model depends on the given feature, and is like a richer extension
of the classical partial dependence plots. Vertical dispersion of the
data points represents interaction effects. Read more about SHAP plots
in the [user guide](../../../user_guide/#shap).
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>models: str, sequence or None, optional (default=None)</strong>
<blockquote>
Name of the models to plot. If None, all models in the pipeline are
selected. Note that selecting multiple models will raise an exception.
To avoid this, call the plot from a model.
</blockquote>
<strong>index: tuple, slice or None, optional (default=None)</strong>
<blockquote>
Indices of the rows in the dataset to plot. If tuple (n, m), it selects
rows n until m. If None, it selects all rows in the test set. The scatter
plot does not support plotting a single sample.
</blockquote>
<strong>feature: int or str, optional (default=0)</strong>
<blockquote>
Index or name of the feature to plot.
</blockquote>
<strong>target: int or str, optional (default=1)</strong>
<blockquote>
Index or name of the class in the target column to look at. Only for
multi-class classification tasks.
</blockquote>
<strong>title: str or None, optional (default=None)</strong>
<blockquote>
Plot's title. If None, the title is left empty.
</blockquote>
<strong>figsize: tuple, optional (default=(10, 6)))</strong>
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
Additional keyword arguments for SHAP's <a href="https://shap.readthedocs.io/en/latest/generated/shap.plots.scatter.html">scatter plot</a>.
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
atom.scatter_plot(feature="bmi")
```
<div align="center">
    <img src="../../../img/plots/scatter_plot.png" alt="scatter_plot" width="700" height="420"/>
</div>
