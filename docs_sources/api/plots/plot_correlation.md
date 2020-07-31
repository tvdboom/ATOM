# plot_correlation
------------------

<a name="plot-correlation"></a>
<pre><em>function</em> atom.plots.<strong style="color:#008AB8">plot_correlation</strong>(title=None, figsize=(10, 10), filename=None, display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L41">[source]</a></div></pre>
<div style="padding-left:3%">
Correlation matrix plot of the dataset. Ignores non-numeric columns. Can only be
 called from an [ATOMClassifier](../ATOM/atomclassifier.md)/[ATOMRegressor](../ATOM/atomregressor.md)
 instance. Can't be called from the model subclasses.
<br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>title: str or None, optional (default=None)</strong>
<blockquote>
Plot's title. If None, the default option is used.
</blockquote>
<strong>figsize: tuple, optional (default=(10, 10))</strong>
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

atom = ATOMClassifier(X, y)
atom.plot_correlation()
```
![plot_correlation](../../img/plots/plot_correlation.png)
