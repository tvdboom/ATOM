# Plots
-------

ATOM provides many plotting methods to analyze the data or compare the
model performances. Descriptions and examples can be found in the API
section. ATOM mainly uses the [plotly](https://plotly.com/python/) library
for plotting. Plotly makes interactive, publication-quality graphs that
are rendered using html. Some plots require other libraries like
[matplotlib](https://matplotlib.org/), [shap](https://github.com/slundberg/shap),
[wordcloud](http://amueller.github.io/word_cloud/) and [schemdraw](https://schemdraw.readthedocs.io/en/latest/).

Plots that compare model performances (methods with the `models`
parameter) can be called directly from atom, e.g., `#!python atom.plot_roc()`,
or from one of the models, e.g., `#!python atom.adab.plot_roc()`. If called from
atom, use the `models` parameter to specify which models to plot. If
called from a specific model, it makes the plot only for that model and
the `models` parameter becomes unavailable.

Plots that analyze the data (methods without the `models` parameter)
can only be called from atom, and not from the models.

<br>

## Parameters

Apart from the plot-specific parameters, all plots have five parameters
in common:

* The `title` parameter adds a title to the plot. The default value doesn't
  show any title. Provide a configuration (as dictionary) to customize its
  appearance, e.g., `#!python title=dict(text="Awesome plot", color="red")`.
  Read more in plotly's [documentation](https://plotly.com/python/figure-labels/).
* The `legend` parameter is used to show/hide, position or customize the
  plot's legend. Provide a configuration (as dictionary) to customize its
  appearance (e.g., `#!python legend=dict(title="Title for legend", title_font_color="red")`)
  or choose one of the following locations:

    - upper left
    - upper right
    - lower left
    - lower right
    - upper center
    - lower center
    - center left
    - center right
    - center
    - out: Position the legend outside the axis, on the right hand side. This
      is plotly's default position. Note that this shrinks the size of the axis
      to fit both legend and axes in the specified `figsize`.

* The `figsize` parameter adjust the plot's size.
* The `filename` parameter is used to save the plot.
* The `display` parameter determines whether to show or return the plot.

!!! info
    In some [plotting methods][prediction-plots], it's possible to plot separate
    lines for different subsets of the rows. For example, to compare the results
    on the train and test set. For these cases, either provide a sequence to the
    `rows` parameter for every line you want to draw, e.g., `#!python atom.plot_roc(rows=("train", "test"))`,
    or provide a dictionary where the keys are the names of the sets (used in the
    legend) and the values are the corresponding selection of rows, selected using
    any of the aforementioned approaches, e.g, `#!python atom.plot_roc(rows={"0-99": range(100), "100-199": range(100, 200})`.
    Note that for these methods, using `#!python atom.plot_roc(rows="train+test")`,
    only plots one line with the data from both sets. See the
    [advanced plotting example][example-advanced-plotting].

<br>

## Aesthetics

The plot's aesthetics can be customized using the plot attributes prior
to calling the plotting method, e.g., `#!python atom.title_fontsize = 30`.
The default values are:

* **palette:** ["rgb(0, 98, 98)", "rgb(56, 166, 165)", "rgb(115, 175, 72)",
  "rgb(237, 173, 8)", "rgb(225, 124, 5)", "rgb(204, 80, 62)", "rgb(148, 52, 110)",
  "rgb(111, 64, 112)", "rgb(102, 102, 102)"]
* **title_fontsize:** 24
* **label_fontsize:** 16
* **tick_fontsize:** 12

Use atom's [update_layout][atomclassifier-update_layout] method to further
customize the plot's layout using any of plotly's [layout properties](https://plotly.com/python/reference/layout/),
e.g., `#!python atom.update_layout(template="plotly_dark")`. Similarly, use
the [update_traces][atomclassifier-update_traces] method to customize the
[traces properties](https://plotly.com/python/reference/scatter/), e.g.
`#!python atom.update_traces(mode="lines+markers")`.

The [reset_aesthetics][atomclassifier-reset_aesthetics] method allows you
to reset all aesthetics to their default value. See [advanced plotting][example-advanced-plotting]
for an example.

<br>

## Canvas

Use the [canvas][atomclassifier-canvas] method to draw multiple plots side
by side, for example to make it easier to compare similar results. The canvas
method is a `@contextmanager`, i.e., it's used through Python's `with` command.
Plots in a canvas ignore the legend, figsize, filename and display parameters.
Instead, specify these parameters in the canvas. If a variable is assigned to
the canvas (e.g., `#!python with atom.canvas() as fig`), it yields the resulting
figure.

For example, we can use a canvas to compare the results of a XGBoost and
LightGBM model on the train and test set. We could also draw the lines for
both models in the same axes, but that would clutter the plot too much.
Click [here][example-advanced-plotting] for more examples.

```python
from atom import ATOMClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, flip_y=0.2, random_state=1)

atom = ATOMClassifier(X, y)
atom.run(["XGB", "LGB"])

with atom.canvas(2, 2, title="XGBoost vs LightGBM"):
    atom.xgb.plot_roc(rows="train+test", title="ROC - XGBoost")
    atom.lgb.plot_roc(rows="train+test", title="ROC - LightGBM")
    atom.xgb.plot_prc(rows="train+test", title="PRC - XGBoost")
    atom.lgb.plot_prc(rows="train+test", title="PRC - LightGBM")
```

<br>

## SHAP

The [SHAP](https://github.com/slundberg/shap) (SHapley Additive exPlanations)
python package uses a game theoretic approach to explain the output of
any machine learning model. It connects optimal credit allocation with
local explanations using the classic [Shapley values](https://en.wikipedia.org/wiki/Shapley_value)
from game theory and their related extensions. ATOM implements methods
to plot 7 of SHAP's plotting functions directly from its API. A list of
available shap plots can be found [here][shap-plots].

Calculating the Shapley values is computationally expensive, especially
for model agnostic explainers like [Permutation](https://shap.readthedocs.io/en/latest/generated/shap.explainers.Permutation.html).
To avoid having to recalculate the values for every plot, ATOM stores
the shapley values internally after the first calculation, and access
them later when needed again.

!!! warning
    * It's not possible to draw multiple models in the same figure.
      Selecting more than one model will raise an exception. To avoid
      this, call the plot directly from a model, e.g., `#!python atom.lr.plot_shap_force()`.
    * The returned plot is a matplotlib figure, not plotly's.
    * SHAP plots aren't available for [forecast][time-series] tasks.

<br>

## Available plots

A list of available plots can be found hereunder. Note that not all
plots can be called from every class and that their availability can
depend on the task at hand.

### Data plots

:: atom.plots:DataPlot
    :: methods:
        toc_only: True
        solo_link: True
        exclude:
            - canvas
            - reset_aesthetics
            - update_layout
            - update_traces

### Hyperparameter tuning plots

:: atom.plots:HyperparameterTuningPlot
    :: methods:
        toc_only: True
        solo_link: True
        exclude:
            - canvas
            - reset_aesthetics
            - update_layout
            - update_traces

### Prediction plots

:: atom.plots:PredictionPlot
    :: methods:
        toc_only: True
        solo_link: True
        exclude:
            - canvas
            - reset_aesthetics
            - update_layout
            - update_traces

### Shap plots

:: atom.plots:ShapPlot
    :: methods:
        toc_only: True
        solo_link: True
        exclude:
            - canvas
            - reset_aesthetics
            - update_layout
            - update_traces
