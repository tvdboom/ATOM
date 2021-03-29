# decision_plot
---------------

<pre><em>method</em> <strong style="color:#008AB8">decision_plot</strong>(models=None, index=None, show=None, target=1,
                     title=None, figsize=None, filename=None, display=True, **kwargs)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L2429">[source]</a></div></pre>
Plot SHAP's decision plot. Visualize model decisions using cumulative SHAP values.
 Each plotted line explains a single model prediction. If a single prediction is
 plotted, feature values will be printed in the plot (if supplied). If multiple
 predictions are plotted together, feature values will not be printed. Plotting too
 many predictions together will make the plot unintelligible. Read more about
SHAP plots in the [user guide](../../../user_guide/#shap).
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
<strong>index: int, tuple, slice or None, optional (default=None)</strong>
<blockquote>
Indices of the rows in the dataset to plot. If tuple (n, m), it selects rows
n until m. If None, select all rows in the test set.
</blockquote>
<strong>show: int or None, optional (default=None)</strong>
<blockquote>
Number of features (ordered by importance) to show in the plot. None to show all.
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
<strong>figsize: tuple or None, optional (default=None)</strong>
<blockquote>
Figure's size, format as (x, y). If None, it adapts the size to the number of features shown.
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
Additional keyword arguments for SHAP's <a href="https://shap.readthedocs.io/en/latest/generated/shap.plots.decision.html">decision plot</a>.
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
atom.decision_plot()
```
<div align="center">
    <img src="../../../img/plots/decision_plot_1.png" alt="decision_plot_1" width="700" height="700"/>
</div>
<br>
```python
atom.decision_plot(index=120)
```
<div align="center">
    <img src="../../../img/plots/decision_plot_2.png" alt="decision_plot_2" width="700" height="700"/>
</div>
