# plot_rfecv
------------

<a name="atom"></a>
<pre><em>method</em> <strong style="color:#008AB8">plot_rfecv</strong>(title=None, figsize=(10, 6), filename=None, display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L471">[source]</a></div></pre>
<div style="padding-left:3%">
Plot the RFECV results, i.e. the scores obtained by the estimator fitted on every
 subset of the dataset. Only available if RFECV was applied on the data.
<br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>title: str or None, optional (default=None)</strong>
<blockquote>
Plot's title. If None, the default option is used.
</blockquote>
<strong>figsize: tuple, optional (default=(10, 6))</strong>
<blockquote>
Figure's size, format as (x, y).
</blockquote>
<strong>filename: str or None, optional (default=None)</strong>
<blockquote>
Name of the file (to save). If None, the figure is not saved.
</blockquote>
<strong>display: bool, optional (default=True)</strong>
<blockquote>
Whether to render the plot.
</blockquote>
</tr>
</table>
</div>
<br />



## Example
----------

```python
from atom import ATOMClassifier

atom = ATOMClassifier(X, y)
atom.feature_selection(strategy='RFECV', solver='LGB', scoring='precision')
atom.plot_rfecv()
```
<div align="center">
    <img src="../../../img/plots/plot_rfecv.png" alt="plot_rfecv" width="700" height="420"/>
</div>
