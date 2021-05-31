# plot_threshold
----------------

<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">plot_threshold</strong>(models=None,
metric=None, dataset="test", steps=100, title=None, figsize=(10, 6),
filename=None, display=True)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L2024">[source]</a>
</span>
</div>

Plot metric performances against threshold values. Only for binary
classification tasks.

<table style="font-size:16px">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<p>
<strong>models: str, sequence or None, optional (default=None)</strong><br>
Name of the models to plot. If None, all models in the pipeline are selected.
</p>
<p>
<strong>metric: str, func, scorer, sequence or None, optional (default=None)</strong><br>
Metric to plot. Choose from any of sklearn's <a href="https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules">SCORERS</a>,
a function with signature <code>metric(y_true, y_pred)</code>,
a scorer object or a sequence of these. If None, the metric
used to run the pipeline is plotted.
</p>
<p>
<strong>dataset: str, optional (default="test")</strong><br>
Data set on which to calculate the metric. Options are "train", "test" or "both".
</p>
<p>
<strong>steps: int, optional (default=100)</strong><br>
Number of thresholds measured.
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
from sklearn.metrics import recall_score

atom = ATOMClassifier(X, y)
atom.run("LGB")
atom.plot_threshold(metric=["accuracy", "f1", recall_score])
```
<div align="center">
    <img src="../../../img/plots/plot_threshold.png" alt="plot_threshold" width="700" height="420"/>
</div>

