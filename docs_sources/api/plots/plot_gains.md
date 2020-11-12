# plot_gains
------------

<a name="atom"></a>
<pre><em>method</em> <strong style="color:#008AB8">plot_gains</strong>(models=None, dataset="test", title=None, figsize=(10, 6), filename=None, display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L1905">[source]</a></div></pre>
<div style="padding-left:3%">
Plot the cumulative gains curve. Only for binary classification tasks.
<br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>models: str, list, tuple or None, optional (default=None)</strong>
<blockquote>
Name of the models to plot. If None, all models in the pipeline are selected.
</blockquote>
<strong>dataset: str, optional (default="test")</strong>
<blockquote>
Data set on which to calculate the gains curve. Options are "train", "test" or "both".
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
Name of the file. If None, the figure is not saved.
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

atom = ATOMClassifier(X, y)
atom.run(["GNB", "RF", "LGB"], metric="roc_auc")
atom.plot_gains(filename="cumulative_gains_curve.png")
```
<div align="center">
    <img src="../../../img/plots/plot_gains.png" alt="plot_gains" width="700" height="420"/>
</div>
