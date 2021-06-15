# plot_bo
---------

<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">plot_bo</strong>(models=None,
metric=0, title=None, figsize=(10, 8), filename=None, display=True)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L858">[source]</a>
</span>
</div>

Plot the bayesian optimization scoring. Only for models that ran
hyperparameter tuning. This is the same plot as the one produced
by `bo_params={"plot": True}` while running the BO. Creates a
canvas with two plots: the first plot shows the score of every trial
and the second shows the distance between the last consecutive steps.

<table style="font-size:16px">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<p>
<strong>models: str, sequence or None, optional (default=None)</strong><br>
Name of the models to plot. If None, all models in the pipeline that used bayesian
optimization are selected.
</p>
<p>
<strong>metric: int or str, optional (default=0)</strong><br>
Index or name of the metric to plot. Only for <a href="../../../user_guide/training/#metric">multi-metric</a> runs.
</p>
<p>
<strong>title: str or None, optional (default=None)</strong><br>
Plot's title. If None, the title is left empty.
</p>
<p>
<strong>figsize: tuple, optional (default=(10, 8))</strong><br>
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
from atom import ATOMClassifier

atom = ATOMClassifier(X, y)
atom.run(["LDA", "LGB"], metric="f1", n_calls=24, n_initial_points=10)
atom.plot_bo()
```
<div align="center">
    <img src="../../../img/plots/plot_bo.png" alt="plot_bo" width="700" height="560"/>
</div>
