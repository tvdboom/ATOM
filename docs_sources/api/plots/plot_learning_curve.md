# plot_learning_curve
---------------------

<a name="atom"></a>
<pre><em>method</em> <strong style="color:#008AB8">plot_learning_curve</strong>(models=None, metric=0, title=None, figsize=(10, 6), filename=None, display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L2474">[source]</a></div></pre>
<div style="padding-left:3%">
Plot the model's learning curve: score vs number of training samples. Only
 available if the models were fitted using [train sizing](../../../user_guide/#train-sizing).
<br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>models: str, sequence or None, optional (default=None)</strong>
<blockquote>
Name of the models to plot. If None, all models in the pipeline are selected.
</blockquote>
<strong>metric: int or str, optional (default=0)</strong>
<blockquote>
Index or name of the metric to plot. Only for [multi-metric](../../../user_guide/#metric) runs.
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
import numpy as np
from atom import ATOMClassifier

atom = ATOMClassifier(X, y)
atom.train_sizing(['GNB', 'LDA'], metric='accuracy', train_sizes=np.linspace(0.1, 1.0, 9), bagging=5)
atom.plot_learning_curve()
```
<div align="center">
    <img src="/img/plots/plot_learning_curve.png" alt="plot_learning_curve" width="700" height="420"/>
</div>
