# plot_distributions
--------------------

<a name="atom"></a>
<pre><em>method</em> <strong style="color:#008AB8">plot_distributions</strong>(columns=0, title=None, figsize=(10, 6), filename=None, display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L2960">[source]</a></div></pre>
Plot column distributions.
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>columns: int, str, slice or sequence, optional (default=0)</strong>
<blockquote>
Slice, names or indices of the columns to plot. It is only
possible to plot one categorical column (which will show the
seven most frequent values). If more than one categorical
columns are selected, the categorical features are ignored.
</blockquote>
<strong>title: str or None, optional (default=None)</strong>
<blockquote>
Plot's title. If None, the title is left empty.
</blockquote>
<strong>figsize: tuple, optional (default=(10, 6))</strong>
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
atom.plot_distributions(columns=[1, 2])  # With numerical features
```
<div align="center">
    <img src="../../../img/plots/plot_distributions_1.png" alt="plot_distributions_1" width="720" height="460"/>
</div>

```python
atom.plot_distributions(columns="Location")  # With categorical columns
```
<div align="center">
    <img src="../../../img/plots/plot_distributions_2.png" alt="plot_distributions_2" width="720" height="460"/>
</div>