# plot_threshold
----------------

<pre><em>method</em> <strong style="color:#008AB8">plot_threshold</strong>(models=None, metric=None, dataset="test", steps=100,
                      title=None, figsize=(10, 6), filename=None, display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L1859">[source]</a></div></pre>
Plot metric performances against threshold values. Only for binary classification tasks.
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>models: str, sequence or None, optional (default=None)</strong>
<blockquote>
Name of the models to plot. If None, all models in the pipeline are selected.
</blockquote>
<strong>metric: str, callable, list, tuple or None, optional (default=None)</strong>
<blockquote>
Metric(s) to plot. These can be one of sklearn's predefined scorers, a metric function
 or a sklearn scorer object (see the <a href="../../../user_guide/#metric">user guide</a>). If
 None, the metric used to run the pipeline is used.
</blockquote>
<strong>dataset: str, optional (default="test")</strong>
<blockquote>
Data set on which to calculate the metric. Options are "train", "test" or "both".
</blockquote>
<strong>steps: int, optional (default=100)</strong>
<blockquote>
Number of thresholds measured.
</blockquote>
<strong>title: str or None, optional (default=None)</strong>
<blockquote>
Plot's title. If None, the title is left empty.
</blockquote>
<strong>figsize: tuple, optional (default=(10, 6))</strong>
<blockquote>
Figure's size, format as (x, y).
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
from sklearn.metrics import recall_score

atom = ATOMClassifier(X, y)
atom.run("LGB")
atom.plot_threshold(metric=["accuracy", "f1", recall_score])
```
<div align="center">
    <img src="../../../img/plots/plot_threshold.png" alt="plot_threshold" width="700" height="420"/>
</div>

