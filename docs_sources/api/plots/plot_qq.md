# plot_qq
---------

<a name="atom"></a>
<pre><em>method</em> <strong style="color:#008AB8">plot_qq</strong>(columns=0, distribution="norm", title=None, figsize=None, filename=None, display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L2960">[source]</a></div></pre>
Plot a quantile-quantile plot.
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>columns: int, str, slice or sequence, optional (default=0)</strong>
<blockquote>
Slice, names or indices of the columns to plot. Selected
categorical columns are ignored.
</blockquote>
<strong>distribution: str, sequence or None, optional (default="norm")</strong>
<blockquote>
Name of the <code>scipy.stats</code> distribution to fit to the columns.
</blockquote>
<strong>title: str or None, optional (default=None)</strong>
<blockquote>
Plot's title. If None, the title is left empty.
</blockquote>
<strong>figsize: tuple, optional (default=(10, 6)))</strong>
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
atom.plot_qq(columns=[0, 1], distribution="triang")
```
<div align="center">
    <img src="../../../img/plots/plot_qq.png" alt="plot_qq" width="720" height="460"/>
</div>
