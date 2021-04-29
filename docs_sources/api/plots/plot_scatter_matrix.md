# plot_scatter_matrix
---------------------

<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">plot_scatter_matrix</strong>
(columns=None, title=None, figsize=(10, 10), filename=None, display=True)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L2021">[source]</a>
</span>
</div>

Plot a matrix of scatter plots. A subset of max 250 random samples
are selected from every column to not clutter the plot.

<table style="font-size:16px">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<p>
<strong>columns: slice, sequence or None, optional (default=None)</strong><br>
Slice, names or indices of the columns to plot. If None,
plot all columns in the dataset. Selected categorical
columns are ignored.
</p>
<p>
<strong>title: str or None, optional (default=None)</strong><br>
Plot's title. If None, the title is left empty.
</p>
<p>
<strong>figsize: tuple, optional (default=(10, 10))</strong><br>
Figure's size, format as (x, y).
</p>
<p>
<strong>filename: str or None, optional (default=None)</strong><br>
Name of the file. Use "auto" for automatic naming.
If None, the figure is not saved.
</p>
<p>
<strong>display: bool, optional (default=True)</strong><br>
Whether to render the plot.
</p>
<p>
<strong>**kwargs</strong><br>
Additional keyword arguments for seaborn's <a href="https://seaborn.pydata.org/generated/seaborn.pairplot.html">pairplot</a>.
</p>
</td>
</tr>
</table>
<br />



## Example

```python
from atom import ATOMClassifier

atom = ATOMClassifier(X, y)
atom.plot_scatter_matrix(columns=slice(0, 5))
```
<div align="center">
    <img src="../../../img/plots/plot_scatter_matrix.png" alt="plot_scatter_matrix" width="720" height="720"/>
</div>
