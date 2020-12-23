# plot_calibration
------------------

<pre><em>method</em> <strong style="color:#008AB8">plot_calibration</strong>(models=None, n_bins=10, title=None, figsize=(10, 10), filename=None, display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L2051">[source]</a></div></pre>
Plot the calibration curve for a binary classifier.
 Well calibrated classifiers are probabilistic classifiers for which the
 output of the `predict_proba` method can be directly interpreted as a
 confidence level. For instance a well calibrated (binary) classifier
 should classify the samples such that among the samples to which it gave
 a `predict_proba` value close to 0.8, approx. 80% actually belong to the
 positive class. Read more in sklearn's
 [documentation](https://scikit-learn.org/stable/modules/calibration.html).
 
 This figure shows two plots: the calibration curve, where
 the x-axis represents the average predicted probability in each bin and the
 y-axis is the fraction of positives, i.e. the proportion of samples whose
 class is the positive class (in each bin); and a distribution of all
 predicted probabilities of the classifier.
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>models: str, sequence or None, optional (default=None)</strong>
<blockquote>
Name of the models to plot. If None, all models in the pipeline are selected.
</blockquote>
<strong>n_bins: int, optional (default=10)</strong>
<blockquote>
Number of bins for the calibration calculation and the histogram.
 Minimum of 5 required.
</blockquote>
<strong>title: str or None, optional (default=None)</strong>
<blockquote>
Plot's title. If None, the default option is used.
</blockquote>
<strong>figsize: tuple, optional (default=(10, 10))</strong>
<blockquote>
Figure's size, format as (x, y).
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
atom.run(["GNB", "LR", "LGB"], metric="average_precision")
atom.plot_calibration()
```
<div align="center">
    <img src="../../../img/plots/plot_calibration.png" alt="plot_calibration" width="700" height="700"/>
</div>
