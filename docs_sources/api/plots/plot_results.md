# plot_results
--------------

<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">plot_results</strong>(models=None,
metric=0, title=None, figsize=None, filename=None, display=True)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L921">[source]</a>
</span>
</div>

Plot of the model results after the evaluation. If all models applied
bootstrap, the plot is a boxplot. If not, the plot is a barplot. Models
are ordered based on their score from the top down. The score is either
the `mean_bootstrap` or `metric_test` attribute of the model, selected in
that order.

<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>models: str, sequence or None, optional (default=None)</strong><br>
Name of the models to plot. If None, all models in the pipeline are selected.
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
<strong>figsize: tuple, optional (default=None)</strong><br>
Figure's size, format as (x, y). If None, it adapts the size to
the number of models shown.
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
from atom import ATOMClassifier

atom = ATOMClassifier(X, y)
atom.run(["QDA", "Tree", "RF", "ET", "LGB"], metric="f1", n_bootstrap=5)
atom.plot_results()  # With bootstrap...
```

<div align="center">
    <img src="../../../img/plots/plot_results_1.png" alt="plot_results" width="700" height="420"/>
</div>

```python
# And without bootstrap...
atom.run(["QDA", "Tree", "RF", "ET", "LGB"], metric="f1", n_bootstrap=0)
atom.plot_results()
```

<div align="center">
    <img src="../../../img/plots/plot_results_2.png" alt="plot_results" width="700" height="420"/>
</div>