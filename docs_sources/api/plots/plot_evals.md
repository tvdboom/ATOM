# plot_evals
------------

<pre><em>method</em> <strong style="color:#008AB8">plot_evals</strong>(models=None, dataset="both", title=None, figsize=(10, 6), filename=None, display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L871">[source]</a></div></pre>
Plot evaluation curves for the train and test set. Only for models that allow
 in-training evaluation ([XGB](../../models/xgb), [LGB](../../models/lgb),
 [CatB](../../models/catb)). The metric is provided by the estimator's
 package and is different for every model and every task. For this reason, the
 method only allows plotting one model at a time.
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>models: str, sequence or None, optional (default=None)</strong>
<blockquote>
Name of the model to plot. If None, all models in the pipeline are selected. Note
 that leaving the default option could raise an exception if there are multiple
 models in the pipeline. To avoid this, call the plot from a model, e.g. <code>atom.lgb.plot_evals()</code>.
</blockquote>
<strong>dataset: str, optional (default="both")</strong>
<blockquote>
Data set on which to calculate the evaluation curves. Options
 are "train", "test" or "both".
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
from atom import ATOMRegressor

atom = ATOMRegressor(X, y)
atom.run(["Bag", "LGB"])
atom.lgb.plot_evals()
```
<div align="center">
    <img src="../../../img/plots/plot_evals.png" alt="plot_evals" width="700" height="420"/>
</div>

