# plot_correlation
------------------

<a name="atom"></a>
<pre><em>method</em> <strong style="color:#008AB8">plot_correlation</strong>(method='pearson', title=None, figsize=(8, 8), filename=None, display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L2550">[source]</a></div></pre>
<div style="padding-left:3%">
Plot the data's correlation matrix. Ignores non-numeric columns.
<br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>method: str, optional (default='pearson')</strong>
<blockquote>
Method of correlation. Choose from 'pearson', 'kendall' or 'spearman'.
</blockquote>
<strong>title: str or None, optional (default=None)</strong>
<blockquote>
Plot's title. If None, the default option is used.
</blockquote>
<strong>figsize: tuple, optional (default=(8, 8))</strong>
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

atom = ATOMClassifier(X, y='RainTomorrow')
atom.plot_correlation()
```
<div align="center">
    <img src="/img/plots/plot_correlation.png" alt="plot_correlation" width="560" height="560"/>
</div>
