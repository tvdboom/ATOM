# plot_successive_halving
-------------------------

<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">plot_successive_halving</strong>(models=None,
metric=0, title=None, figsize=(10, 6), filename=None, display=True)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L736">[source]</a>
</span>
</div>

Plot of the models' scores per iteration of the successive
halving. Only use with models fitted using [successive halving](../../../user_guide/training/#successive-halving).
[Ensemble](../../../user_guide/models/#ensembles) models are
ignored.

<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>models: str, sequence or None, optional (default=None)</strong><br>
Name of the models to plot. If None, all the models in the pipeline are selected.
</p>
<p>
<strong>metric: int or str, optional (default=0)</strong><br>
Index or name of the metric to plot. Only for <a href="../../../user_guide/training/#metric">multi-metric</a> runs.
</p>
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
atom.successive_halving(
    models=["tree", "et", "rf", "xgb", "lgb", "catb"],
    metric="f1_weighted",
    n_bootstrap=6,
)
atom.plot_successive_halving(filename="successive_halving")
```

<div align="center">
    <img src="../../../img/plots/plot_successive_halving.png" alt="plot_successive_halving" width="700" height="420"/>
</div>
