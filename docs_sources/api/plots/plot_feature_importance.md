# plot_feature_importance
-------------------------

<pre><em>method</em> <strong style="color:#008AB8">plot_feature_importance</strong>(models=None, show=None, title=None, figsize=None, filename=None, display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L1240">[source]</a></div></pre>
Plot a tree-based model's feature importance. The importances are normalized in order
 to be able to compare them between models. The `feature_importance` attribute is
 updated with the extracted importance ranking.
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>models: str, sequence or None, optional (default=None)</strong>
<blockquote>
Name of the models to plot. If None, all the models in the pipeline are selected.
</blockquote>
<strong>show: int, optional (default=None)</strong>
<blockquote>
Number of features (ordered by importance) to show in the plot. None to show all.
</blockquote>
<strong>title: str or None, optional (default=None)</strong>
<blockquote>
Plot's title. If None, the title is left empty.
</blockquote>
<strong>figsize: tuple or None, optional (default=None)</strong>
<blockquote>
Figure's size, format as (x, y). If None, it adapts the size to the
number of features shown.
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
atom.run(["LR", "RF"], metric="recall_weighted")
atom.RF.plot_feature_importance(show=11, filename="random_forest_importance")
```
<div align="center">
    <img src="../../../img/plots/plot_feature_importance.png" alt="plot_feature_importance" width="700" height="700"/>
</div>
