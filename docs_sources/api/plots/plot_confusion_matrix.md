# plot_confusion_matrix
-------------------------

<a name="atom-plot-confusion-matrix"></a>
<pre><em>function</em> atom.plots.<strong style="color:#008AB8">plot_confusion_matrix</strong>(models=None, normalize=False, title=None,
                                          figsize=None, filename=None, display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L336">[source]</a></div></pre>
<div style="padding-left:3%">
For 1 model: plot the confusion matrix in a heatmap. For multiple models: compare TP,
 FP, FN and TN in a barplot (not implemented for multiclass classification).
<br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>models: str, sequence or None, optional (default=None)</strong>
<blockquote>
Name of the models to plot. If None, all models in the pipeline are selected.
</blockquote>
<strong>normalize: bool, optional (default=False)</strong>
<blockquote>
Whether to normalize the matrix.
</blockquote>
<strong>title: str or None, optional (default=None)</strong>
<blockquote>
Plot's title. If None, the default option is used.
</blockquote>
<strong>figsize: tuple, optional (default=None)</strong>
<blockquote>
Figure's size, format as (x, y). If None, 
</blockquote>
<strong>filename: str or None, optional (default=None)</strong>
<blockquote>
Name of the file (to save). If None, adapts size to plot type.
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
atom.run(['Tree', 'Bag'])
atom.Tree.plot_confusion_matrix(normalize=True)
```
![plot_confusion_matrix_1](../../img/plots/plot_confusion_matrix_1.png)
```python
atom.plot_confusion_matrix()
```
![plot_confusion_matrix_2](../../img/plots/plot_confusion_matrix_2.png)