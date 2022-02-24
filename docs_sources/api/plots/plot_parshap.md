# plot_parshap
--------------

<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">plot_parshap</strong>(models=None,
columns=None, target=1, title=None, figsize=(10, 6), filename=None, display=True)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L2332">[source]</a>
</span>
</div>

Plots the train and test correlation between the shap value of
every feature with its target value, after removing the effect
of all other features (partial correlation). This plot is
useful to identify the features that are contributing most to
overfitting. Features that lie below the bisector (diagonal
line) performed worse on the test set than on the training set.
If the estimator has a `feature_importances_` or `coef_` attribute,
its normalized values are shown in a color map. Read more about
this plot [here](https://towardsdatascience.com/which-of-your-features-are-overfitting-c46d0762e769).

<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>models: str, sequence or None, optional (default=None)</strong><br>
Name of the models to plot. If None, all the models in the pipeline are selected.
</p>
<p>
<strong>columns: int, str, sequence or None, optional (default=None)</strong><br>
Names or indices of the features to plot. None to show all.
</p>
<p>
<strong>target: int or str, optional (default=1)</strong><br>
Index or name of the class in the target column to look at.
Only for multi-class classification tasks.
</p>
<p>
<strong>title: str or None, optional (default=None)</strong><br>
Plot's title. If None, the title is left empty.
</p>
<p>
<strong>figsize: tuple, optional (default=(10, 6))</strong><br>
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
atom.run(["GNB", "LGB"])
atom.gnb.plot_parshap()
```

<div align="center">
    <img src="../../../img/plots/plot_parshap_1.png" alt="plot_parshap" width="700" height="420"/>
</div>

```python
atom.lgb.plot_parshap()
```

<div align="center">
    <img src="../../../img/plots/plot_parshap_2.png" alt="plot_parshap" width="700" height="420"/>
</div>