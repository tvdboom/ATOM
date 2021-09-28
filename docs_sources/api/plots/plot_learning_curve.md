# plot_learning_curve
---------------------

<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">plot_learning_curve</strong>(models=None,
metric=0, title=None, figsize=(10, 6), filename=None, display=True)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L3429">[source]</a>
</span>
</div>

Plot the model's learning curve: score vs number of training
samples. Only available if the models were fitted using
[train sizing](../../../user_guide/training/#train-sizing).

<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>models: str, sequence or None, optional (default=None)</strong><br>
Name of the models to plot. If None, all models in the pipeline are selected.
</p>
<p>
<strong>metric: int or str, optional (default=0)</strong><br>
Index or name of the metric to plot. Only for <a href="../../../user_guide/training/#metric">multi-metric</a> runs.
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
import numpy as np
from atom import ATOMClassifier

atom = ATOMClassifier(X, y)
atom.train_sizing(
    models=["GNB", "LDA"],
    metric="accuracy",
    train_sizes=9,
    n_bootstrap=5,
)
atom.plot_learning_curve()
```
<div align="center">
    <img src="../../../img/plots/plot_learning_curve.png" alt="plot_learning_curve" width="700" height="420"/>
</div>
