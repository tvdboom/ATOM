# plot_distribution
--------------------

<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">plot_distribution</strong>(columns=0,
distributions=None, show=None, title=None, figsize=None, filename=None,
display=True, **kwargs)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L3804">[source]</a>
</span>
</div>

Plot column distributions. Additionally, it is possible to plot any of
`scipy.stats` probability distributions fitted to the column. Missing
values are ignored.

<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>columns: int, str, slice or sequence, optional (default=0)</strong><br>
Slice, names or indices of the columns to plot. It is only
possible to plot one categorical column. If more than just
the one categorical column is selected, all categorical
columns are ignored.
</p>
<p>
<strong>distributions: str, sequence or None, optional (default=None)</strong><br>
Names of the <code>scipy.stats</code> distributions to fit to the column.
If None, no distribution is fitted. Only for numerical columns.
</p>
<p>
<strong>show: int or None, optional (default=None)</strong><br>
Number of classes (ordered by number of occurrences) to show in
the plot. None to show all. Only for categorical columns.
</p>
<p>
<strong>title: str or None, optional (default=None)</strong><br>
Plot's title. If None, the title is left empty.
</p>
<p>
<strong>figsize: tuple, optional (default=None)</strong><br>
Figure's size, format as (x, y). If None, adapts size to
the plot's type.
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
<p>
<strong>**kwargs</strong><br>
Additional keyword arguments for seaborn's <a href="https://seaborn.pydata.org/generated/seaborn.histplot.html">histplot</a>.
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

!!! tip
    Use atom's [distribution](../../ATOM/atomclassifier/#distribution) method to
    check which distribution fits the column best.

<br>



## Example

```python
from atom import ATOMClassifier

atom = ATOMClassifier(X, y)
atom.plot_distribution(columns=[1, 2])  # With numerical columns
```

<div align="center">
    <img src="../../../img/plots/plot_distribution_1.png" alt="plot_distribution_1" width="720" height="460"/>
</div>

```python
# With fitted distributions
atom.plot_distribution(columns="mean radius", distributions=["norm", "triang"])
```

<div align="center">
    <img src="../../../img/plots/plot_distribution_2.png" alt="plot_distribution_2" width="720" height="460"/>
</div>

```python
# With categorical columns
atom.plot_distribution(columns="Location", show=11)
```

<div align="center">
    <img src="../../../img/plots/plot_distribution_3.png" alt="plot_distribution_3" width="700" height="700"/>
</div>