# plot_prc
-----------

<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">plot_prc</strong>(models=None,
dataset="test", title=None, figsize=(10, 6), filename=None, display=True)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L1294">[source]</a>
</span>
</div>

Plot the Precision-Recall Curve (PRC). The legend shows the average
precision (AP) score. Only for binary classification tasks. Read more
about PRC in sklearn's [documentation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html).

<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>models: str, sequence or None, optional (default=None)</strong><br>
Name of the models to plot. If None, all models in the pipeline are selected.
</p>
<p>
<strong>dataset: str, optional (default="test")</strong><br>
Data set on which to calculate the metric. Choose from:
"train", "test", "both" (train and test) or "holdout".
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
atom.run(["LR", "RF", "LGB"], metric="average_precision")
atom.plot_prc()
```

<div align="center">
    <img src="../../../img/plots/plot_prc.png" alt="plot_prc" width="700" height="420"/>
</div>
