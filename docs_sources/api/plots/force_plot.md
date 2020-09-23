# force_plot
------------

<a name="atom"></a>
<pre><em>method</em> <strong style="color:#008AB8">force_plot</strong>(models=None, index=None, target=1, title=None, figsize=(14, 6), filename=None, display=True, **kwargs)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L2075">[source]</a></div></pre>
<div style="padding-left:3%">
Plot SHAP's force plot. Visualize the given SHAP values with an additive force layout.
 The explainer will be chosen automatically based on the model's type. Note that by
 default this plot will render using javascript. For a regular figure use `matplotlib=True`
 (this option is only available when only 1 row is selected through the `index` parameter).
 Read more about SHAP plots in the [user guide](../../../user_guide/#shap).
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
<strong>index: int, sequence or None, optional (default=None)</strong>
<blockquote>
Indices of the rows in the dataset to plot. If tuple (n, m), select rows n until m.
 If None, select all rows in the test set.
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
<strong>figsize: tuple, optional (default=(14, 6))</strong>
<blockquote>
Figure's size, format as (x, y).
</blockquote>
<strong>filename: str or None, optional (default=None)</strong>
<blockquote>
Name of the file (to save). If matplotlib=False, the figure will be saved as an html
 file. If None, the figure is not saved.
</blockquote>
<strong>display: bool, optional (default=True)</strong>
<blockquote>
Whether to render the plot.
</blockquote>
<strong>\*\*kwargs</strong>
<blockquote>
Additional keyword arguments for shap's force_plot.
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
atom.run('lr')
atom.force_plot(index=atom.X_test.index[0], matplotlib=True, filename='force_plot')
```
<div align="center">
    <img src="/img/plots/force_plot.png" alt="force_plot" width="1000" height="420"/>
</div>

