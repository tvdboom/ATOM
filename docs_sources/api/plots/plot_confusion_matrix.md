# plot_confusion_matrix
-----------------------

<a name="atom"></a>
<pre><em>method</em> <strong style="color:#008AB8">plot_confusion_matrix</strong>(models=None, dataset="test", normalize=False,
                             title=None, figsize=None, filename=None, display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L1469">[source]</a></div></pre>
<div style="padding-left:3%">
Plot a model's confusion matrix. Only for classification tasks.

* For 1 model: plot the confusion matrix in a heatmap.
* For multiple models: compare TP, FP, FN and TN in a barplot (not implemented for multiclass classification tasks).

<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>models: str, list, tuple or None, optional (default=None)</strong>
<blockquote>
Name of the models to plot. If None, all models in the pipeline are selected.
</blockquote>
<strong>dataset: str, optional (default="test")</strong>
<blockquote>
Data set on which to calculate the confusion matrix. Options are "train" or "test".
</blockquote>
<strong>normalize: bool, optional (default=False)</strong>
<blockquote>
Whether to normalize the matrix. Only for the heatmap plot.
</blockquote>
<strong>title: str or None, optional (default=None)</strong>
<blockquote>
Plot's title. If None, the default option is used.
</blockquote>
<strong>figsize: tuple, optional (default=None)</strong>
<blockquote>
Figure's size, format as (x, y). If None, adapts size to plot type.
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
from atom import ATOMClassifier

atom = ATOMClassifier(X, y)
atom.run(["Tree", "Bag"])
atom.Tree.plot_confusion_matrix(normalize=True)
```
<div align="center">
    <img src="../../../img/plots/plot_confusion_matrix_1.png" alt="plot_confusion_matrix_1" width="560" height="560"/>
</div>
```python
atom.plot_confusion_matrix()
```
<div align="center">
    <img src="../../../img/plots/plot_confusion_matrix_2.png" alt="plot_confusion_matrix_2" width="700" height="420"/>
</div>