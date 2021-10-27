# plot_partial_dependence
-------------------------

<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">plot_partial_dependence</strong>(models=None,
features=None, kind="average", target=None, title=None, figsize=(10, 6),
filename=None, display=True)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L1939">[source]</a>
</span>
</div>

Plot the partial dependence of features. The partial dependence of a
feature (or a set of features) corresponds to the response of
the model for each possible value of the feature. Two-way partial
dependence plots are plotted as contour plots (only allowed for single
model plots). The deciles of the feature values will be shown with tick
marks on the x-axes for one-way plots, and on both axes for two-way
plots. Read more about partial dependence on sklearn's [documentation](https://scikit-learn.org/stable/modules/partial_dependence.html#partial-dependence-and-individual-conditional-expectation-plots).

<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>models: str, sequence or None, optional (default=None)</strong><br>
Name of the models to plot. If None, all the models in the pipeline are selected.
</p>
<p>
<strong>features: int, str, sequence or None, optional (default=None)</strong><br>
Features or feature pairs (name or index) to get the partial dependence
from. Maximum of 3 allowed. If None, it uses the top 3 features if the
<code>feature_importance</code> attribute is defined, else it uses the
first 3 features in the dataset.
</p>
<strong>kind: str, optional (default="average")</strong><br>
<ul style="line-height:1.2em;margin-top:5px;margin-bottom:0">
<li>"average": Plot the partial dependence averaged across
 all the samples in the dataset.</li>
<li>"individual": Plot the partial dependence per sample
(Individual Conditional Expectation).</li>
<li>"both": Plot both the average (as a thick line) and
the individual (thin lines) partial dependence.</li>
</ul>
<p style="margin-top:5px">
This parameter is ignored when plotting feature pairs.
</p>
<p>
<strong>target: int or str, optional (default=1)</strong><br>
Index or name of the class in the target column to look at. Only for
multi-class classification tasks.
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
<strong>fig: matplotlib.figure.Figure</strong><br>
Plot object. Only returned if <code>display=None</code>.
</td>
</tr>
</table>
<br />



## Example

```python
from atom import ATOMClassifier

atom = ATOMClassifier(X, y)
atom.feature_selection(strategy="PCA", n_features=6)
atom.run(["Tree", "Bag"], metric="precision")
atom.plot_partial_dependence()
```
<div align="center">
    <img src="../../../img/plots/plot_partial_dependence_1.png" alt="plot_partial_dependence_1" width="700" height="420"/>
</div>
<br>
```python
atom.tree.plot_partial_dependence(features=(4, (3, 4)))
```
<div align="center">
    <img src="../../../img/plots/plot_partial_dependence_2.png" alt="plot_partial_dependence_2" width="700" height="420"/>
</div>