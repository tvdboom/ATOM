# plot_pipeline
---------------

<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">plot_pipeline</strong>(models=None,
draw_hyperparameter_tuning=True, color_branches=None, title=None,
figsize=None, filename=None, display=True)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L733">[source]</a>
</span>
</div>

Plot a diagram of the pipeline.

<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>model: str or None, default=None</strong><br>
Name or index of the models to plot. If None, all models are selected.
</p>
<p>
<strong>draw_hyperparameter_tuning: bool, default=True</strong><br>
Whether to draw if the models used Hyperparameter Tuning.
</p>
<p>
<strong>color_branches: bool or None, default=None</strong><br>
Whether to draw every branch in a different color. If None,
branches are colored when there is more than one.
</p>
<p>
<strong>title: str or None, default=None</strong><br>
Plot's title. If None, the title is left empty.
</p>
<p>
<strong>figsize: tuple or None, default=None</strong><br>
Figure's size, format as (x, y). If None, it adapts the size to the
pipeline drawn.
</p>
<p>
<strong>filename: str or None, default=None</strong><br>
Name of the file. Use "auto" for automatic naming.
If None, the figure is not saved.
</p>
<p>
<strong>display: bool or None, default=True</strong><br>
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

!!! tip
    Print `atom.pipeline` in a notebook for [sklearn's interactive visualization](https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_pipeline_display.html)
    of the current pipeline.



## Example

```python
from atom import ATOMClassifier

atom = ATOMClassifier(X, y)
atom.impute(strat_num="median")
atom.encode(max_onehot=5)
atom.run(["GNB", "RNN", "SGD", "MLP"])
atom.voting(models=atom.winners[:2])

atom.plot_pipeline()  # For a single branch
```

<div align="center">
    <img src="../../../img/plots/plot_pipeline_1.png" alt="plot_pipeline_1" width="700" height="400"/>
</div>


```python
from atom import ATOMClassifier

atom = ATOMClassifier(X, y)
atom.scale()
atom.prune()
atom.run("RF", n_trials=10)

atom.branch = "oversample"
atom.balance(strategy="adasyn")
atom.run("RF_os")

atom.branch = "undersample_from_master"
atom.balance(strategy="nearmiss")
atom.run("RF_us")

atom.plot_pipeline()  # For multiple branches
```

<div align="center">
    <img src="../../../img/plots/plot_pipeline_2.png" alt="plot_pipeline_2" width="700" height="400"/>
</div>
