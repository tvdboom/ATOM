# plot_probabilities
--------------------

<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">plot_probabilities</strong>(models=None,
dataset="test", target=1, title=None, figsize=(10, 6), filename=None, display=True)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L2896">[source]</a>
</span>
</div>

Plot the probability distribution of the classes in the target column.
Only for classification tasks.

<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>models: int, str, slice, sequence or None, default=None</strong><br>
Name or index of the models to plot. If None, all models are selected.
</p>
<p>
<strong>dataset: str, default="test"</strong><br>
Data set on which to calculate the metric. Choose from:
"train", "test", "both" (train and test) or "holdout".
</p>
<p>
<strong>target: int or str, default=1</strong><br>
Probability of being that class in the target column (as index or
name). Only for multiclass classification tasks.
</p>
<p>
<strong>title: str or None, default=None</strong><br>
Plot's title. If None, the title is left empty.
</p>
<p>
<strong>figsize: tuple, default=(10, 6)</strong><br>
Figure's size, format as (x, y).
</p>
<p>
<strong>filename: str or None, default=None</strong><br>
Name of the file. Use "auto" for automatic naming.
If None, the figure is not saved.
</p>
<p>
<strong>display: bool or None, default=True</strong><br>
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

atom = ATOMClassifier(X, y="RainTomorrow")
atom.run("rf")
atom.plot_probabilities()
```

<div align="center">
    <img src="../../../img/plots/plot_probabilities.png" alt="plot_probabilities" width="700" height="420"/>
</div>
