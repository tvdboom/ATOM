# plot_permutation_importance
-----------------------------

<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">plot_permutation_importance</strong>
(models=None, show=None, n_repeats=10, title=None, figsize=None, filename=None, display=True)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L2021">[source]</a>
</span>
</div>

Plot the feature permutation importance of models. Calculating
permutations can be time-consuming, especially if `n_repeats`
is high. For this reason, the permutations are stored under the
`permutations` attribute. If the plot is called again for the
same model with the same `n_repeats`, it will use the stored
values, making the method considerably faster. The
`feature_importance` attribute is updated with the extracted
importance ranking.

<table style="font-size:16px">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<p>
<strong>models: str, sequence or None, optional (default=None)</strong><br>
Name of the models to plot. If None, all models in the pipeline are selected.
</p>
<p>
<strong>show: int, optional (default=None)</strong><br>
Number of features (ordered by importance) to show. None to show all.
</p>
<p>
<strong>n_repeats: int, optional (default=10)</strong><br>
Number of times to permute each feature.
</p>
<p>
<strong>title: str or None, optional (default=None)</strong><br>
Plot's title. If None, the title is left empty.
</p>
<p>
<strong>figsize: tuple or None, optional (default=None)</strong><br>
Figure's size, format as (x, y). If None, it adapts the size to the
number of features shown.
</p>
<p>
<strong>filename: str or None, optional (default=None)</strong><br>
Name of the file. Use "auto" for automatic naming.
If None, the figure is not saved.
</p>
<p>
<strong>display: bool, optional (default=True)</strong><br>
Whether to render the plot.
</p>
</td>
</tr>
</table>
<br />



## Example

```python
from atom import ATOMClassifier

atom = ATOMClassifier(X, y)
atom.run(["LR", "LDA"], metric="average_precision")
atom.lda.plot_permutation_importance(show=10, n_repeats=7)
```
<div align="center">
    <img src="../../../img/plots/plot_permutation_importance.png" alt="plot_permutation_importance" width="700" height="700"/>
</div>
