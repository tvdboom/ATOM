# plot_results
--------------

<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">plot_results</strong>
(models=None, metric=0, title=None, figsize=None, filename=None, display=True)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L2021">[source]</a>
</span>
</div>

Plot of the model results after the evaluation. If all models applied
bagging, the plot is a boxplot. If not, the plot is a barplot. Models
are ordered based on their score from the top down. The score is either
the `mean_bagging` or `metric_test` attribute of the model, selected in
that order.

<table style="font-size:16px">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<p>
<strong>models: str, sequence or None, optional (default=None)</strong><br>
Name of the models to plot. If None, all models in the pipeline are selected.
</p>
<p>
<strong>metric: int or str, optional (default=0)</strong><br>
Index or name of the metric to plot. Only for <a href="../../../user_guide/#metric">multi-metric</a> runs.
</p>
<p>
<strong>title: str or None, optional (default=None)</strong><br>
Plot's title. If None, the title is left empty.
</p>
<p>
<strong>figsize: tuple, optional (default=None)</strong><br>
Figure's size, format as (x, y). If None, adapts size the to number of models.
</p>
<p>
<strong>filename: str or None, optional (default=None)</strong><br>
Name of the file. Use "auto" for automatic naming.
If None, the figure is not saved.
</p>
<p>
<strong>display: bool, optional (default=True)</strong><br>
Whether to render the plot.
</p>
</td>
</tr>
</table>
<br />



## Example

```python
from atom import ATOMClassifier

atom = ATOMClassifier(X, y)
atom.run(["QDA", "Tree", "RF", "ET", "LGB"], metric="f1", bagging=5)
atom.plot_results()  # With bagging...
```
<div align="center">
    <img src="../../../img/plots/plot_results_1.png" alt="plot_results" width="700" height="420"/>
</div>

```python
# And without bagging...
atom.run(["QDA", "Tree", "RF", "ET", "LGB"], metric="f1", bagging=0)
atom.plot_results()
```
<div align="center">
    <img src="../../../img/plots/plot_results_2.png" alt="plot_results" width="700" height="420"/>
</div>