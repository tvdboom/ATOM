# plot_evals
------------

<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">plot_evals</strong>(models=None,
dataset="both", title=None, figsize=(10, 6), filename=None, display=True)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L1135">[source]</a>
</span>
</div>

Plot evaluation curves for the train and test set. Only for models that
allow in-training evaluation ([XGB](../../models/xgb), [LGB](../../models/lgb),
[CatB](../../models/catb)). The metric is provided by the estimator's
package and is different for every model and every task. For this reason,
the method only allows plotting one model.

<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>models: str, sequence or None, optional (default=None)</strong><br>
Name of the model to plot. If None, all models in the pipeline are
selected. Note that leaving the default option could raise an
exception if there are multiple models in the pipeline. To avoid
this, call the plot from a model, e.g. <code>atom.lgb.plot_evals()</code>.
</p>
<p>
<strong>dataset: str, optional (default="both")</strong><br>
Data set on which to calculate the evaluation curves. Options
 are "train", "test" or "both".
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
from atom import ATOMRegressor

atom = ATOMRegressor(X, y)
atom.run(["Bag", "LGB"])
atom.lgb.plot_evals()
```

<div align="center">
    <img src="../../../img/plots/plot_evals.png" alt="plot_evals" width="700" height="420"/>
</div>

