# plot_partial_dependence
-------------------------

<a name="atomclassifier-plot"></a>
<pre><em>function</em> atom.plots.<strong style="color:#008AB8">plot_partial_dependence</strong>(models=None, features=None, target=None, title=None,
                                            figsize=(10, 6), filename=None, display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L336">[source]</a></div></pre>
<div style="padding-left:3%">
Plot the partial dependence of features. The partial dependence of a feature (or a
 set of features) corresponds to the average response of the model for each possible
 value of the feature. Two-way partial dependence plots are plotted as contour plots
 (only allowed for single model plots). The deciles of the feature values will be
 shown with tick marks on the x-axes for one-way plots, and on both axes for two-way
 plots.
<br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>models: str, sequence or None, optional (default=None)</strong>
<blockquote>
Name of the models to plot. If None, all the models in the pipeline are selected.
</blockquote>
<strong>features: int, str, sequence or None, optional (default=None)</strong>
<blockquote>
Features or feature pairs (name or index) to get the partial dependence from. Maximum
of 3 allowed. If None, it uses the top 3 features if `best_features`
is defined (see [plot_feature_importance](./plot_feature_importance.md) or
 [plot_permutation_importance](./plot_permutation_importance.md)), else it uses the
 first 3 features in the dataset.
</blockquote>
<strong>target: int or str, optional (default=1)</strong>
<blockquote>
Category to look at in the target class as index or name. Only for multi-class classification.
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
from atom import ATOMClassifier

atom = ATOMClassifier(X, y)
atom.feature_selection(strategy='PCA', n_features=4)
atom.run(['Tree', 'Bag'], metric='precision')
atom.Tree.plot_partial_dependence(features=[0, 1, (1, 3)])
```
![plot_partial_dependence](./img/plot_partial_dependence.png)
