# plot_calibration
------------------

<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">plot_calibration</strong>(models=None,
n_bins=10, title=None, figsize=(10, 10), filename=None, display=True)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L3074">[source]</a>
</span>
</div>

Plot the calibration curve for a binary classifier. Well calibrated
classifiers are probabilistic classifiers for which the output of the
`predict_proba` method can be directly interpreted as a confidence
level. For instance a well calibrated (binary) classifier should
classify the samples such that among the samples to which it gave a
`predict_proba` value close to 0.8, approx. 80% actually belong to the
positive class. Read more in sklearn's [documentation](https://scikit-learn.org/stable/modules/calibration.html).

This figure shows two plots: the calibration curve, where the x-axis
represents the average predicted probability in each bin, and the y-axis
is the fraction of positives, i.e. the proportion of samples whose
class is the positive class (in each bin); and a distribution of all
predicted probabilities of the classifier.

<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>models: str, sequence or None, optional (default=None)</strong><br>
Name of the models to plot. If None, all models in the pipeline are selected.
</p>
<p>
<strong>n_bins: int, optional (default=10)</strong><br>
Number of bins used for calibration. Minimum of 5 required.
</p>
<p>
<strong>title: str or None, optional (default=None)</strong><br>
Plot's title. If None, the title is left empty.
</p>
<p>
<strong>figsize: tuple, optional (default=(10, 10))</strong><br>
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
<strong>matplotlib.figure.Figure</strong><br>
Plot object. Only returned if <code>display=None</code>.
</td>
</tr>
</table>

!!! tip
    Use the [calibrate](../../ATOM/atomclassifier/#calibrate) method to
    calibrate the winning model.

<br>



## Example

```python
from atom import ATOMClassifier

atom = ATOMClassifier(X, y)
atom.run(["GNB", "LR", "LGB"], metric="average_precision")
atom.plot_calibration()
```

<div align="center">
    <img src="../../../img/plots/plot_calibration.png" alt="plot_calibration" width="700" height="700"/>
</div>
