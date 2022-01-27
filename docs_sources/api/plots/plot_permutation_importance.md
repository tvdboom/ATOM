# plot_permutation_importance
-----------------------------

<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">plot_permutation_importance</strong>(models=None,
show=None, n_repeats=10, title=None, figsize=None, filename=None, display=True)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L1939">[source]</a>
</span>
</div>

Plot the feature permutation importance of models. Calculating
permutations can be time-consuming, especially if `n_repeats`
is high. For this reason, the permutations are stored under the
`permutations` attribute. If the plot is called again for the
same model with the same `n_repeats`, it will use the stored
values, making the method considerably faster. The trainer's
`feature_importance` attribute is updated with the extracted
importance ranking.

<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
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
atom.run(["LR", "LDA"], metric="average_precision")
atom.lda.plot_permutation_importance(show=10, n_repeats=7)
```

<div align="center">
    <img src="../../../img/plots/plot_permutation_importance.png" alt="plot_permutation_importance" width="700" height="700"/>
</div>
