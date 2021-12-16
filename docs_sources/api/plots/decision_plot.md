# decision_plot
---------------

<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">decision_plot</strong>(models=None,
index=None, show=None, target=1, title=None, figsize=None, filename=None,
display=True, **kwargs)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L3148">[source]</a>
</span>
</div>

Plot SHAP's decision plot. Visualize model decisions using cumulative
SHAP values. Each plotted line explains a single model prediction. If
a single prediction is plotted, feature values will be printed in the
plot (if supplied). If multiple predictions are plotted together,
feature values will not be printed. Plotting too many predictions
together will make the plot unintelligible. Read more about SHAP plots
in the [user guide](../../../user_guide/plots/#shap).

<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>models: str, sequence or None, optional (default=None)</strong><br>
Name of the model to plot. If None, all models in the pipeline are
selected. Note that leaving the default option could raise an
exception if there are multiple models in the pipeline. To avoid
this, call the plot from a model, e.g. <code>atom.xgb.decision_plot()</code>.
</p>
<p>
<strong>index: int, str, sequence or None, optional (default=None)</strong><br>
Index names or positions of the rows in the dataset to plot.
If None, it selects all rows in the test set.
</p>
<p>
<strong>show: int or None, optional (default=None)</strong><br>
Number of features (ordered by importance) to show. None to show all.
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
<strong>figsize: tuple or None, optional (default=None)</strong><br>
Figure's size, format as (x, y). If None, it adapts the size to the
number of features shown.
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
Additional keyword arguments for SHAP's <a href="https://shap.readthedocs.io/en/latest/generated/shap.plots.decision.html">decision plot</a>.
</p>
</td>
</tr>
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Returns:</strong></td>
<td width="80%" class="td_params">
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
atom.decision_plot()  # For multiple samples
```

<div align="center">
    <img src="../../../img/plots/decision_plot_1.png" alt="decision_plot_1" width="700" height="700"/>
</div>
<br>
```python
atom.decision_plot(index=120)  # For a single sample
```

<div align="center">
    <img src="../../../img/plots/decision_plot_2.png" alt="decision_plot_2" width="700" height="700"/>
</div>
