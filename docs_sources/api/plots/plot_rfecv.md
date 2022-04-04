# plot_rfecv
------------

<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">plot_rfecv</strong>(title=None,
figsize=(10, 6), filename=None, display=True)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L642">[source]</a>
</span>
</div>

Plot the rfecv results, i.e. the scores obtained by the estimator
fitted on every subset of the dataset. Only available if
[rfecv](../../../user_guide/feature_engineering/#rfe) was applied
on the data.

<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>title: str or None, optional (default=None)</strong><br>
Plot's title. If None, the title is left empty.
</p>
<p>
<strong>figsize: tuple, optional (default=(10, 6))</strong><br>
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
atom.feature_selection(strategy="rfecv", solver="LGB", scoring="precision")
atom.plot_rfecv()
```

<div align="center">
    <img src="../../../img/plots/plot_rfecv.png" alt="plot_rfecv" width="700" height="420"/>
</div>
