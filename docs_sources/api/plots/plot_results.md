# plot_results
--------------

<pre><em>method</em> <strong style="color:#008AB8">plot_results</strong>(models=None, metric=0, title=None, figsize=None, filename=None, display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L706">[source]</a></div></pre>
Plot of the model results after the evaluation.
If all models applied bagging, the plot will be a boxplot.
If not, the plot will be a barplot. Models are ordered based
on their score from the top down. The score is either the
`mean_bagging` or `metric_test` attribute of the model,
selected in that order.
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>models: str, sequence or None, optional (default=None)</strong>
<blockquote>
Name of the models to plot. If None, all models in the pipeline are selected.
</blockquote>
<strong>metric: int or str, optional (default=0)</strong>
<blockquote>
Index or name of the metric to plot. Only for <a href="../../../user_guide/#metric">multi-metric</a> runs.
</blockquote>
<strong>title: str or None, optional (default=None)</strong>
<blockquote>
Plot's title. If None, the default option is used.
</blockquote>
<strong>figsize: tuple, optional (default=None)</strong>
<blockquote>
Figure's size, format as (x, y). If None, adapts size the to number of models.
</blockquote>
<strong>filename: str or None, optional (default=None)</strong>
<blockquote>
Name of the file. If None, the figure is not saved.
</blockquote>
<strong>display: bool, optional (default=True)</strong>
<blockquote>
Whether to render the plot.
</blockquote>
</tr>
</table>
<br />



## Example
----------

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