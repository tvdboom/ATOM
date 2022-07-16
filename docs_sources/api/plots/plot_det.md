# plot_det
-----------

<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">plot_det</strong>(models=None,
dataset="test", title=None, figsize=(10, 6), filename=None, display=True)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L1619">[source]</a>
</span>
</div>

Plot the Detection Error Tradeoff (DET) curve. Only for binary
classification tasks. Read more about DET in sklearn's [documentation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_det.html).

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
atom.run(["LR", "RF", "LGB"], metric="fpr")
atom.plot_det()
```

<div align="center">
    <img src="../../../img/plots/plot_det.png" alt="plot_det" width="700" height="420"/>
</div>
