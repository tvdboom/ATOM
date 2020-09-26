# dependence_plot
-----------------

<a name="atom"></a>
<pre><em>method</em> <strong style="color:#008AB8">dependence_plot</strong>(models=None, index='rank(1)', target=1,
                       title=None, figsize=(10, 6), filename=None, display=True, **kwargs)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L2161">[source]</a></div></pre>
<div style="padding-left:3%">
Plot SHAP's dependence plot. Plots the value of the feature on the x-axis and the
 SHAP value of the same feature on the y-axis. This shows how the model depends on
 the given feature, and is like a richer extension of the classical partial dependence
 plots. Vertical dispersion of the data points represents interaction effects. Grey
 ticks along the y-axis are data points where the feature's value was NaN. The
 explainer will be chosen automatically based on the model's type. Read more about
 SHAP plots in the [user guide](../../../user_guide/#shap).
<br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>models: str, sequence or None, optional (default=None)</strong>
<blockquote>
Name of the models to plot. If None, all models in the pipeline are selected. Note
 that selecting multiple models will raise an exception. To avoid this, call the
 plot from a `model`.
</blockquote>
<strong>index: int, sequence or None, optional (default='rank(1)')</strong>
<blockquote>
If this is an int, it is the index of the feature to plot. If this is a
 string it is either the name of the feature to plot, or it can have the
 form 'rank(int)' to specify the feature with that rank (ordered by mean
 absolute SHAP value over all the samples).
</blockquote>
<strong>target: int or str, optional (default=1)</strong>
<blockquote>
Category to look at in the target class as index or name. Only for multi-class
 classification tasks.
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
<strong>\*\*kwargs</strong>
<blockquote>
Additional keyword arguments for shap's dependence_plot.
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
atom.run('RF')
atom.dependence_plot(index='rank(3)')
```
<div align="center">
    <img src="../../../img/plots/dependence_plot.png" alt="dependence_plot" width="700" height="420"/>
</div>

