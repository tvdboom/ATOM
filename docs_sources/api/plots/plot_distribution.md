# plot_distribution
--------------------

<a name="atom"></a>
<pre><em>method</em> <strong style="color:#008AB8">plot_distribution</strong>(columns=0, distribution=None, show=None,
                         title=None, figsize=None, filename=None, display=True, **kwargs)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L3202">[source]</a></div></pre>
Plot column distributions. Additionally, it is possible to plot any of
`scipy.stats` probability distributions fitted to the column.

!!!tip
    Use atom's [distribution](../../ATOM/atomclassifier/#distribution) method to
    check which distribution fits the column best.

<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>columns: int, str, slice or sequence, optional (default=0)</strong>
<blockquote>
Slice, names or indices of the columns to plot. It is only
possible to plot one categorical column. If more than just
the one categorical column is selected, all categorical
columns are ignored.
</blockquote>
<strong>distribution: str, sequence or None, optional (default=None)</strong>
<blockquote>
Names of the <code>scipy.stats</code> distributions to fit to the column.
If None, no distribution is fitted. Only for numerical columns.
</blockquote>
<strong>show: int or None, optional (default=None)</strong>
<blockquote>
Number of classes (ordered by number of occurrences) to show in
the plot. None to show all. Only for categorical columns.
</blockquote>
<strong>title: str or None, optional (default=None)</strong>
<blockquote>
Plot's title. If None, the title is left empty.
</blockquote>
<strong>figsize: tuple, optional (default=None)</strong>
<blockquote>
Figure's size, format as (x, y). If None, adapts size to
the plot's type.
</blockquote>
<strong>filename: str or None, optional (default=None)</strong>
<blockquote>
Name of the file. If None, the figure is not saved.
</blockquote>
<strong>display: bool, optional (default=True)</strong>
<blockquote>
Whether to render the plot.
</blockquote>
<strong>**kwargs</strong>
<blockquote>
Additional keyword arguments for seaborn's <a href="https://seaborn.pydata.org/generated/seaborn.histplot.html">histplot</a>.
</blockquote>
</tr>
</table>
<br />


## Example
----------

```python
from atom import ATOMClassifier

atom = ATOMClassifier(X, y)
atom.plot_distribution(columns=[1, 2])  # With numerical columns
```
<div align="center">
    <img src="../../../img/plots/plot_distribution_1.png" alt="plot_distribution_1" width="720" height="460"/>
</div>

```python
atom.plot_distribution(columns="mean radius", distribution=["norm", "triang", "pearson3"])  # With fitted distributions
```
<div align="center">
    <img src="../../../img/plots/plot_distribution_2.png" alt="plot_distribution_2" width="720" height="460"/>
</div>

```python
atom.plot_distribution(columns="Location", show=11)  # With categorical columns
```
<div align="center">
    <img src="../../../img/plots/plot_distribution_3.png" alt="plot_distribution_3" width="700" height="700"/>
</div>