# plot_successive_halving
-------------------------

<pre><em>method</em> <strong style="color:#008AB8">plot_successive_halving</strong>(models=None, metric=0, title=None,
                               figsize=(10, 6), filename=None, display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L2878">[source]</a></div></pre>
Plot of the models' scores per iteration of the successive halving. Only
available if the models were fitted using [successive halving](../../../user_guide/#successive-halving).
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>models: str, sequence or None, optional (default=None)</strong>
<blockquote>
Name of the models to plot. If None, all the models in the pipeline are selected.
</blockquote>
<strong>metric: int or str, optional (default=0)</strong>
<blockquote>
Index or name of the metric to plot. Only for <a href="../../../user_guide/#metric">multi-metric</a> runs.
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

atom = ATOMClassifier(X, y)
atom.successive_halving(["bag", "adab", "et", "lgb"], metric="accuracy", bagging=5)
atom.plot_successive_halving(filename="plot_successive_halving")
```
<div align="center">
    <img src="../../../img/plots/plot_successive_halving.png" alt="plot_successive_halving" width="700" height="420"/>
</div>
