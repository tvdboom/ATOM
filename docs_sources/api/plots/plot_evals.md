# plot_evals
------------

<a name="atom-plot-evals"></a>
<pre><em>function</em> atom.plots.<strong style="color:#008AB8">plot_evals</strong>(models=None, title=None, figsize=(10, 6), filename=None, display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L336">[source]</a></div></pre>
<div style="padding-left:3%">
Plot evaluation curves for the train and test set. Only for models that allow
 in-training evaluation (XGB, LGB, CatB). Only allows plotting of one model at a time.
<br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>models: str, sequence or None, optional (default=None)</strong>
<blockquote>
Name of the models to plot. If None, all models in the pipeline are selected. Note
 that this will raise an exception if there are multiple models in the pipeline.
</blockquote>
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
from atom import ATOMRegressor
atom = ATOMRegressor(X, y)
atom.run('XGB')
atom.XGB.plot_evals()
```
![plot_evals](./img/plot_evals.png)
