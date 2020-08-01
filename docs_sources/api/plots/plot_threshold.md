# plot_threshold
----------------

<a name="atom-plot-threshold"></a>
<pre><em>function</em> atom.plots.<strong style="color:#008AB8">plot_threshold</strong>(models=None, metric=None, steps=100,
                                   title=None, figsize=(10, 6), filename=None, display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L336">[source]</a></div></pre>
<div style="padding-left:3%">
Plot performance metric(s) against multiple threshold values.
<br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>models: str, sequence or None, optional (default=None)</strong>
<blockquote>
Name of the models to plot. If None, all models in the pipeline are selected.
</blockquote>
<strong>metric: str, callable, sequence or None, optional (default=None)</strong>
<blockquote>
Metric(s) to plot. These can be one of the pre-defined sklearn scorers as string,
 a metric function or a sklearn scorer object. If None, the metric used to run
 the pipeline is used.
</blockquote>
<strong>steps: int, optional (default=100)</strong>
<blockquote>
Number of thresholds measured.
</blockquote>
<strong>title: str or None, optional (default=None)</strong>
<blockquote>
Plot's title. If None, the default option is used.
</blockquote>
<strong>figsize: tuple, optional (default=(10, 6))</strong>
<blockquote>
Figure's size, format as (x, y).
</blockquote>
<strong>filename: str or None, optional (default=None)</strong>
<blockquote>
Name of the file (to save). If None, the figure is not saved.
</blockquote>
<strong>display: bool, optional (default=True)</strong>
<blockquote>
Whether to render the plot.
</blockquote>
</tr>
</table>
</div>
<br />



## Example
----------
```python
from atom import ATOMClassifier
from sklearn.metrics import recall_score

atom = ATOMClassifier(X, y)
atom.run('KNN')
atom.plot_threshold(metric=['accuracy', 'f1', recall_score])
```
![plot_threshold](./img/plot_threshold.png)
