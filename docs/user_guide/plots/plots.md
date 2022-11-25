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
parameter) can be called directly from atom, e.g. `#!python atom.plot_roc()`,
or from one of the models, e.g. `#!python atom.adab.plot_roc()`. If called from
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
  appearance, e.g. `#!python title=dict(text="Awesome plot", color="red")`.
  Read more in plotly's [documentation](https://plotly.com/python/figure-labels/).
* The `legend` parameter is used to show/hide, position or customize the
  plot's legend. Provide a configuration (as dictionary) to customize its
  appearance (e.g. `#!python legend=dict(title="Title for legend", title_font_color="red")`)
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

<br>

## Aesthetics

The plot's aesthetics can be customized using the plot attributes, e.g.
`#!python atom.title_fontsize = 30`. The default values are:

* **palette:** ["rgb(0, 98, 98)", "rgb(56, 166, 165)", "rgb(115, 175, 72)",
  "rgb(237, 173, 8)", "rgb(225, 124, 5)", "rgb(204, 80, 62)", "rgb(148, 52, 110)",
  "rgb(111, 64, 112)", "rgb(102, 102, 102)"]
* **title_fontsize:** 24
* **label_fontsize:** 16
* **tick_fontsize:** 12

Use atom's [update_layout][atomclassifier-update_layout] method to further
customize the plot's aesthetics using any of plotly's [layout properties](https://plotly.com/python/reference/layout/),
e.g. `#!python atom.update_layout(template="plotly_dark")`. Use the [reset_aesthetics][atomclassifier-reset_aesthetics]
method to reset the aesthetics to their default value. See [advanced plotting][example-advanced-plotting]
for various examples.

<br>

## Canvas

Use the [canvas][atomclassifier-canvas] method to draw multiple plots side
by side, for example to make it easier to compare similar results. The canvas
method is a `@contextmanager`, i.e. it's used through Python's `with` command.
Plots in a canvas ignore the legend, figsize, filename and display parameters.
Instead, specify these parameters in the canvas. If a variable is assigned to
the canvas (e.g. `#!python with atom.canvas() as fig`), it yields the resulting figure.

For example, we can use a canvas to compare the results of a XGBoost and
LightGBM model on the train and test set. We could also draw the lines for
both models in the same axes, but that would clutter the plot too much.
Click [here][example-advanced-plotting] for more examples.

```pycon
>>> from atom import ATOMClassifier
>>> import pandas as pd

>>> X = pd.read_csv("./examples/datasets/weatherAUS.csv")

>>> atom = ATOMClassifier(X, y="RainTomorrow")
>>> atom.impute()
>>> atom.encode()
>>> atom.run(["xgb", "lgb"])

>>> with atom.canvas(2, 2, title="XGBoost vs LightGBM"):
...     atom.xgb.plot_roc(dataset="both", title="ROC - XGBoost")
...     atom.lgb.plot_roc(dataset="both", title="ROC - LightGBM")
...     atom.xgb.plot_prc(dataset="both", title="PRC - XGBoost")
...     atom.lgb.plot_prc(dataset="both", title="PRC - LightGBM")

```

:: insert:
    url: /img/plots/canvas.html

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

!!! note
    Since the plot figures are not made by ATOM, note the following:

    * It's not possible to draw multiple models in the same figure.
      Selecting more than one model will raise an exception. To avoid
      this, call the plot directly from a model, e.g. `#!python atom.lr.plot_shap_force()`.
    * The returned plot is a matplotlib figure, not plotly's.

<br>

## Available plots

A list of available plots can be found hereunder. Note that not all
plots can be called from every class and that their availability can
depend on the task at hand.

### Feature selection plots

:: atom.plots:FeatureSelectorPlot
    :: methods:
        toc_only: True
        exclude:
            - canvas
            - reset_aesthetics
            - update_layout

### Data plots

:: atom.plots:DataPlot
    :: methods:
        toc_only: True
        exclude:
            - canvas
            - reset_aesthetics
            - update_layout

### Hyperparameter tuning plots

:: atom.plots:HTPlot
    :: methods:
        toc_only: True
        exclude:
            - canvas
            - reset_aesthetics
            - update_layout

### Prediction plots

:: atom.plots:PredictionPlot
    :: methods:
        toc_only: True
        exclude:
            - canvas
            - reset_aesthetics
            - update_layout

### Shap plots

:: atom.plots:ShapPlot
    :: methods:
        toc_only: True
        exclude:
            - canvas
            - reset_aesthetics
            - update_layout
