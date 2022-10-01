# Plots
-------

ATOM provides many plotting methods to analyze the data or compare the
model performances. Descriptions and examples can be found in the API
section. ATOM uses the packages [matplotlib](https://matplotlib.org/),
[seaborn](https://seaborn.pydata.org/), [shap](https://github.com/slundberg/shap),
[wordcloud](http://amueller.github.io/word_cloud/) and [schemdraw](https://schemdraw.readthedocs.io/en/latest/)
for plotting.

Plots that compare model performances (methods with the `models`
parameter) can be called directly from atom, e.g. `atom.plot_roc()`,
or from one of the models, e.g. `atom.adab.plot_roc()`. If called from
atom, use the `models` parameter to specify which models to plot. If
called from a specific model, it makes the plot only for that model and
the `models` parameter becomes unavailable.

Plots that analyze the data (methods without the `models` parameter)
can only be called from atom, and not from the models.

<br>

## Parameters

Apart from the plot-specific parameters, all plots have four
parameters in common:

* The `title` parameter allows you to add a title to the plot.
* The `figsize` parameter adjust the plot's size.
* The `filename` parameter is used to save the plot.
* The `display` parameter determines whether to show or return the plot.

<br>

## Aesthetics

The plot aesthetics can be customized using the plot attributes, e.g.
`atom.style = "white"`. These attributes can be called from any instance
with plotting methods. Note that the plot attributes are attached to the
class and not the instance. This means that changing the attribute will
also change it for all other instances in the module. Use the
[reset_aesthetics][atomclassifier-reset_aesthetics] method to reset all
the aesthetics to their default value. The default values are:

* **style:** "darkgrid"
* **palette:** "GnBu_r_d"
* **title_fontsize:** 20
* **label_fontsize:** 16
* **tick_fontsize:** 12

<br>

## Canvas

Sometimes it's desirable to draw multiple plots side by side in order
to be able to compare them easier. Use the [canvas][atomclassifier-canvas]
method for this. The canvas method is a `@contextmanager`, i.e. it's
used through the `with` command. Plots in a canvas will ignore the
figsize, filename and display parameters. Instead, call these parameters
from the canvas for the final figure. If a variable is assigned to the
canvas (e.g. `with atom.canvas() as fig`), it contains the resulting
matplotlib figure.

For example, we can use a canvas to compare the results of a [XGBoost][]
and [LightGBM][] model on the train and test set. We could also draw the
lines for both models in the same axes, but then the plot would become
too cluttered.

```pycon
>>> atom = ATOMClassifier(X, y)
>>> atom.run(["xgb", "lgb"], n_trials=0)

>>> with atom.canvas(2, 2, title="XGBoost vs LightGBM", filename="canvas"):
...     atom.xgb.plot_roc(dataset="both", title="ROC - XGBoost")
...     atom.lgb.plot_roc(dataset="both", title="ROC - LightGBM")
...     atom.xgb.plot_prc(dataset="both", title="PRC - XGBoost")
...     atom.lgb.plot_prc(dataset="both", title="PRC - LightGBM")

```

![canvas](../img/plots/canvas.png)

<br>

## SHAP

The [SHAP](https://github.com/slundberg/shap) (SHapley Additive exPlanations)
python package uses a game theoretic approach to explain the output of
any machine learning model. It connects optimal credit allocation with
local explanations using the classic [Shapley values](https://en.wikipedia.org/wiki/Shapley_value)
from game theory and their related extensions. ATOM implements methods
to plot 7 of SHAP's plotting functions directly from its API. Check the
available shap plots [here][shap-plots].

Calculating the Shapley values is computationally expensive, especially
for model agnostic explainers like [Permutation](https://shap.readthedocs.io/en/latest/generated/shap.explainers.Permutation.html).
To avoid having to recalculate the values for every plot, ATOM stores
the shapley values internally after the first calculation, and access
them when needed again.

Since the plots are not made by ATOM, we can't draw multiple models in
the same figure. Selecting more than one model will raise an exception.
To avoid this, call the plot directly from a model, e.g. `atom.adab.plot_shap_force()`.

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

### Data plots

:: atom.plots:DataPlot
    :: methods:
        toc_only: True
        exclude:
            - canvas
            - reset_aesthetics

### Model plots

:: atom.plots:ModelPlot
    :: methods:
        toc_only: True
        exclude:
            - canvas
            - reset_aesthetics

### Shap plots

:: atom.plots:ShapPlot
    :: methods:
        toc_only: True
        exclude:
            - canvas
            - reset_aesthetics
