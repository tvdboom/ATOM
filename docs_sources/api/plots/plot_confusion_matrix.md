# plot_confusion_matrix
-----------------------

<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">plot_confusion_matrix</strong>(models=None,
dataset="test", normalize=False, title=None, figsize=None, filename=None,
display=True)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L2725">[source]</a>
</span>
</div>

Plot a model's confusion matrix. For one model, the plot shows a
heatmap. For multiple models, it compares TP, FP, FN and TN in a
barplot (not implemented for multiclass classification tasks).
Only for classification tasks.

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
Data set on which to calculate the confusion matrix. Choose from:
"train", "test" or "holdout".
</p>
<p>
<strong>normalize: bool, optional (default=False)</strong><br>
Whether to normalize the matrix.
</p>
<p>
<strong>title: str or None, optional (default=None)</strong><br>
Plot's title. If None, the title is left empty.
</p>
<p>
<strong>figsize: tuple, optional (default=None)</strong><br>
Figure's size, format as (x, y). If None, it adapts the size
to plot's type.
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
atom.run(["Tree", "Bag"])
atom.Tree.plot_confusion_matrix(normalize=True)  # For one model
```

<div align="center">
    <img src="../../../img/plots/plot_confusion_matrix_1.png" alt="plot_confusion_matrix_1" width="560" height="420"/>
</div>

```python
atom.plot_confusion_matrix()  # For multiple models
```

<div align="center">
    <img src="../../../img/plots/plot_confusion_matrix_2.png" alt="plot_confusion_matrix_2" width="700" height="420"/>
</div>