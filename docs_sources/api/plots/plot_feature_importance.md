# plot_feature_importance
-------------------------

<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">plot_feature_importance</strong>(models=None,
show=None, title=None, figsize=None, filename=None, display=True)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L1424">[source]</a>
</span>
</div>

Plot a model's feature importance. The importances are normalized in
order to be able to compare them between models. The `feature_importance`
attribute is updated with the extracted importance ranking. Only for
models whose estimator has a `feature_importances_` attribute.

<table style="font-size:16px">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<p>
<strong>models: str, sequence or None, optional (default=None)</strong><br>
Name of the models to plot. If None, all the models in the pipeline are selected.
</p>
<p>
<strong>show: int, optional (default=None)</strong><br>
Number of features (ordered by importance) to show. None to show all.
</p>
<p>
<strong>title: str or None, optional (default=None)</strong><br>
Plot's title. If None, the title is left empty.
</p>
<p>
<strong>figsize: tuple or None, optional (default=None)</strong><br>
Figure's size, format as (x, y). If None, it adapts the size to the
number of features shown.
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
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Returns:</strong></td>
<td width="80%" style="background:white;">
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
atom.run(["LR", "RF"], metric="recall_weighted")
atom.RF.plot_feature_importance(show=11, filename="random_forest_importance")
```
<div align="center">
    <img src="../../../img/plots/plot_feature_importance.png" alt="plot_feature_importance" width="700" height="700"/>
</div>
