# plot_correlation
------------------

<a name="atom"></a>
<pre><em>method</em> <strong style="color:#008AB8">plot_correlation</strong>(columns=None, method="pearson", title=None, figsize=(8, 7), filename=None, display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L3056">[source]</a></div></pre>
Plot the data's correlation matrix.
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>columns: slice, sequence or None, optional (default=None)</strong>
<blockquote>
Slice, names or indices of the columns to plot. If None,
plot all columns in the dataset. Selected categorical
columns are ignored.
</blockquote>
<strong>method: str, optional (default="pearson")</strong>
<blockquote>
Method of correlation. Choose from "pearson", "kendall" or "spearman".
</blockquote>
<strong>title: str or None, optional (default=None)</strong>
<blockquote>
Plot's title. If None, the title is left empty.
</blockquote>
<strong>figsize: tuple, optional (default=(8, 7))</strong>
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
atom.plot_correlation()
```
<div align="center">
    <img src="../../../img/plots/plot_correlation.png" alt="plot_correlation" width="560" height="480"/>
</div>
