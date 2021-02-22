# plot_scatter_matrix
---------------------

<a name="atom"></a>
<pre><em>method</em> <strong style="color:#008AB8">plot_correlation</strong>(columns=None, title=None, figsize=(10, 10), filename=None, display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L2960">[source]</a></div></pre>
Plot the data's scatter matrix. Ignores categorical columns.
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>columns: slice, sequence or None, optional (default=None)</strong>
<blockquote>
Slice, names or indices of the columns to plot. If None, plot
all columns in the dataset.
</blockquote>
<strong>title: str or None, optional (default=None)</strong>
<blockquote>
Plot's title. If None, the title is left empty.
</blockquote>
<strong>figsize: tuple, optional (default=(10, 10))</strong>
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
atom.plot_scatter_matrix(columns=slice(0, 5))
```
<div align="center">
    <img src="../../../img/plots/plot_scatter_matrix.png" alt="plot_scatter_matrix" width="720" height="720"/>
</div>
