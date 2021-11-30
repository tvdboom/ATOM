# plot_correlation
------------------

<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">plot_correlation</strong>(columns=None,
method="pearson", title=None, figsize=(8, 7), filename=None, display=True)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L3541">[source]</a>
</span>
</div>

Plot the data's correlation matrix.

<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>columns: slice, sequence or None, optional (default=None)</strong><br>
Slice, names or indices of the columns to plot. If None,
plot all columns in the dataset. Selected categorical
columns are ignored.
</p>
<p>
<strong>method: str, optional (default="pearson")</strong><br>
Method of correlation. Choose from "pearson", "kendall" or "spearman".
</p>
<p>
<strong>title: str or None, optional (default=None)</strong><br>
Plot's title. If None, the title is left empty.
</p>
<p>
<strong>figsize: tuple, optional (default=(8, 7))</strong><br>
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
<strong>fig: matplotlib.figure.Figure</strong><br>
Plot object. Only returned if <code>display=None</code>.
</td>
</tr>
</table>
<br />



## Example

```python
from atom import ATOMClassifier

atom = ATOMClassifier(X, y)
atom.plot_correlation()
```

<div align="center">
    <img src="../../../img/plots/plot_correlation.png" alt="plot_correlation" width="560" height="480"/>
</div>
