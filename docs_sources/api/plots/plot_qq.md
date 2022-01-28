# plot_qq
---------

<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">plot_qq</strong>(columns=0,
distributions="norm", title=None, figsize=(10, 6), filename=None, display=True)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L3956">[source]</a>
</span>
</div>

Plot a quantile-quantile plot.

<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>columns: int, str, slice or sequence, optional (default=0)</strong><br>
Slice, names or indices of the columns to plot. Selected
categorical columns are ignored.
</p>
<p>
<strong>distributions: str, sequence or None, optional (default="norm")</strong><br>
Name of the <code>scipy.stats</code> distributions to fit to the columns.
</p>
<p>
<strong>title: str or None, optional (default=None)</strong><br>
Plot's title. If None, the title is left empty.
</p>
<p>
<strong>figsize: tuple, optional (default=(10, 6)))</strong><br>
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
<td width="20%" class="td_title" style="vertical-align:top"><strong>Returns:</strong></td>
<td width="80%" class="td_params">
<strong>matplotlib.figure.Figure</strong><br>
Plot object. Only returned if <code>display=None</code>.
</td>
</tr>
</table>
<br />



## Example

```python
from atom import ATOMClassifier

atom = ATOMClassifier(X, y)
atom.plot_qq(columns=[0, 1], distributions="triang")
```

<div align="center">
    <img src="../../../img/plots/plot_qq.png" alt="plot_qq" width="720" height="460"/>
</div>
