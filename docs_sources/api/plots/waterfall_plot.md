# waterfall_plot
----------------

<pre><em>method</em> <strong style="color:#008AB8">waterfall_plot</strong>(models=None, index=None, show=None, target=1,
                      title=None, figsize=None, filename=None, display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L2691">[source]</a></div></pre>
Plot SHAP's waterfall plot for a single prediction.
 The SHAP value of a feature represents the impact of the evidence
 provided by that feature on the model’s output. The waterfall plot
 is designed to visually display how the SHAP values (evidence) of
 each feature move the model output from our prior expectation under
 the background data distribution, to the final model prediction
 given the evidence of all the features. Features are sorted by
 the magnitude of their SHAP values with the smallest magnitude
 features grouped together at the bottom of the plot when the
 number of features in the models exceeds the `show` parameter.
 Read more about SHAP plots in the [user guide](../../../user_guide/#shap).
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>models: str, sequence or None, optional (default=None)</strong>
<blockquote>
Name of the models to plot. If None, all models in the pipeline are selected. Note
 that selecting multiple models will raise an exception. To avoid this, call the
 plot from a model.
</blockquote>
<strong>index: int or None, optional (default=None)</strong>
<blockquote>
Index of the row in the dataset to plot. If None, selects a random row in the test set.
</blockquote>
<strong>show: int or None, optional (default=None)</strong>
<blockquote>
Number of features to show in the plot. None to show all.
</blockquote>
<strong>target: int or str, optional (default=1)</strong>
<blockquote>
Index or name of the class in the target column to look at. Only for multi-class
 classification tasks.
</blockquote>
<strong>title: str or None, optional (default=None)</strong>
<blockquote>
Plot's title. If None, the default option is used.
</blockquote>
<strong>figsize: tuple or None, optional (default=None)</strong>
<blockquote>
Figure's size, format as (x, y). If None, adapts size to <code>show</code> parameter.
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
atom.run("Tree")
atom.tree.waterfall_plot(show=11)
```
<div align="center">
    <img src="../../../img/plots/waterfall_plot.png" alt="waterfall_plot" width="700" height="700"/>
</div>

