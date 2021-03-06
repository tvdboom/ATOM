# plot_permutation_importance
-----------------------------

<pre><em>method</em> <strong style="color:#008AB8">plot_permutation_importance</strong>(models=None, show=None, n_repeats=10,
                                   title=None, figsize=None, filename=None, display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L1106">[source]</a></div></pre>
Plot the feature permutation importance of models. Calculating all permutations can
 be time-consuming, especially if `n_repeats` is high. They are stored under
 the attribute `permutations`. This means that if a plot is repeated for
 the same model with the same `n_repeats`, it will be considerably faster.
 The `feature_importance` attribute is updated with the extracted importance ranking.
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>models: str, sequence or None, optional (default=None)</strong>
<blockquote>
Name of the models to plot. If None, all models in the pipeline are selected.
</blockquote>
<strong>show: int, optional (default=None)</strong>
<blockquote>
Number of features (ordered by importance) to show in the plot. None to show all.
</blockquote>
<strong>n_repeats: int, optional (default=10)</strong>
<blockquote>
Number of times to permute each feature.
</blockquote>
<strong>title: str or None, optional (default=None)</strong>
<blockquote>
Plot's title. If None, the title is left empty.
</blockquote>
<strong>figsize: tuple or None, optional (default=None)</strong>
<blockquote>
Figure's size, format as (x, y). If None, it adapts the size to the
number of features shown.
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
from atom import ATOMClassifier

atom = ATOMClassifier(X, y)
atom.run(["LR", "LDA"], metric="average_precision")
atom.lda.plot_permutation_importance(show=10, n_repeats=7)
```
<div align="center">
    <img src="../../../img/plots/plot_permutation_importance.png" alt="plot_permutation_importance" width="700" height="700"/>
</div>
