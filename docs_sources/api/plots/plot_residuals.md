# plot_residuals
----------------

<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">plot_residuals</strong>(models=None,
dataset="test", title=None, figsize=(10, 6), filename=None, display=True)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L1745">[source]</a>
</span>
</div>

The plot shows the residuals (difference between the predicted and the
true value) on the vertical axis and the independent variable on the
horizontal axis. The gray, intersected line shows the identity line.
This plot can be useful to analyze the variance of the error of the
regressor. If the points are randomly dispersed around the horizontal
axis, a linear regression model is appropriate for the data; otherwise,
a non-linear model is more appropriate. Only for regression tasks.

<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>models: str, sequence or None, optional (default=None)</strong><br>
Name of the models to plot. If None, all models in the pipeline are selected.
</p>
<p>
<strong>dataset: str, optional (default="test")</strong><br>
Data set on which to calculate the residuals. Choose from:
"train", "test", "both" (train and test) or "holdout".
</p>
<p>
<strong>title: str or None, optional (default=None)</strong><br>
Plot's title. If None, the title is left empty.
</p>
<p>
<strong>figsize: tuple, optional (default=(10, 6))</strong><br>
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
atom.run(["OLS", "LGB"], metric="MAE")
atom.plot_residuals()
```

<div align="center">
    <img src="../../../img/plots/plot_residuals.png" alt="plot_residuals" width="700" height="420"/>
</div>
