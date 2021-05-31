# Plots
-------

ATOM provides many plotting methods to analyze the data or compare the
model performances. Descriptions and examples can be found in the API
section. ATOM uses the packages [matplotlib](https://matplotlib.org/),
[seaborn](https://seaborn.pydata.org/), [shap](https://github.com/slundberg/shap)
and [wordcloud](http://amueller.github.io/word_cloud/) for plotting.

Plots that compare model performances (methods with the `models`
parameter) can be called directly from a trainer, e.g. `atom.plot_roc()`,
or from one of the models, e.g. `atom.LGB.plot_roc()`. If called from
a trainer, it makes the plot for all models in its pipeline. If called
from a specific model, it makes the plot only for that model.

Plots that analyze the dataset (methods without the `models` parameter)
can only be called from atom. The rest of the trainers are supposed
to be used only when the goal is just modelling, not data manipulation.


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
[reset_aesthetics](../../API/ATOM/atomclassifier#reset-aesthetics) method
to reset all the aesthetics to their default value. The default values are:

* style: "darkgrid"
* palette: "GnBu_r_d"
* title_fontsize: 20
* label_fontsize: 16
* tick_fontsize: 12

<br>

## Canvas

Sometimes it is desirable to draw multiple plots side by side in order
to be able to compare them easier. Use the [canvas](../../API/ATOM/atomclassifier/#canvas)
method for this. The canvas method is a `@contextmanager`, i.e. it's
used through the `with` command. Plots in a canvas will ignore the
figsize, filename and display parameters. Instead, call these parameters
from the canvas for the final figure. If a variable is assigned to the
canvas (e.g. `with atom.canvas() as fig`), it contains the resulting
matplotlib figure.

For example, we can use a canvas to compare the results of a [XGBoost](../../API/models/xgb)
and [LightGBM](../../API/models/lgb) model on the train and test set.
We could also draw the lines for both models in the same axes, but
then the plot would become too cluttered.

```python
atom = ATOMClassifier(X, y)
atom.run(["xgb", "lgb"], n_calls=0)

with atom.canvas(2, 2, title="XGBoost vs LightGBM", filename="canvas"):
    atom.xgb.plot_roc(dataset="both", title="ROC - XGBoost")
    atom.lgb.plot_roc(dataset="both", title="ROC - LightGBM")
    atom.xgb.plot_prc(dataset="both", title="PRC - XGBoost")
    atom.lgb.plot_prc(dataset="both", title="PRC - LightGBM")
```
<div align="center">
    <img src="../../img/plots/canvas.png" alt="canvas" width="1000" height="700"/>
</div>

<br>

## SHAP

The [SHAP](https://github.com/slundberg/shap) (SHapley Additive exPlanations)
python package uses a game theoretic approach to explain the output of
any machine learning model. It connects optimal credit allocation with
local explanations using the classic [Shapley values](https://en.wikipedia.org/wiki/Shapley_value)
from game theory and their related extensions. ATOM implements methods to
plot 7 of SHAP's plotting functions directly from its API. The seven
plots are: [bar_plot](../../API/plots/bar_plot), [beeswarm_plot](../../API/plots/beeswarm_plot),
[decision_plot](../../API/plots/decision_plot), [force_plot](../../API/plots/force_plot),
[heatmap_plot](../../API/plots/heatmap_plot), [scatter_plot](../../API/plots/scatter_plot)
and [waterfall_plot](../../API/plots/waterfall_plot).

Since the plots are not made by ATOM, we can't draw multiple models in
the same figure. Selecting more than one model will raise an exception.
To avoid this, call the plot directly from a model, e.g. `atom.xgb.force_plot()`.

!!! info
    You can recognize the SHAP plots by the fact that they end (instead
    of start) with the word `plot`.

<br>

## Available plots

A list of available plots can be found hereunder. Note that not all
plots can be called from every class and that their availability can
depend on the task at hand.

<table>
<tr>
<td><a href="../../API/plots/plot_correlation">plot_correlation</a></td>
<td>Plot the data's correlation matrix.</td>
</tr>

<tr>
<td><a href="../../API/plots/plot_scatter_matrix">plot_scatter_matrix</a></td>
<td>Plot the data's scatter matrix.</td>
</tr>

<tr>
<td><a href="../../API/plots/plot_qq">plot_qq</a></td>
<td>Plot a quantile-quantile plot.</td>
</tr>

<tr>
<td><a href="../../API/plots/plot_distribution">plot_distribution</a></td>
<td>Plot column distributions.</td>
</tr>

<tr>
<td><a href="../../API/plots/plot_wordcloud">plot_wordcloud</a></td>
<td>Plot a wordcloud from the corpus.</td>
</tr>

<tr>
<td><a href="../../API/plots/plot_ngrams">plot_ngrams</a></td>
<td>Plot n-gram frequencies.</td>
</tr>

<tr>
<td><a href="../../API/plots/plot_pipeline">plot_pipeline</a></td>
<td>Plot a diagram of every estimator in atom's pipeline.</td>
</tr>

<tr>
<td><a href="../../API/plots/plot_pca">plot_pca</a></td>
<td>Plot the explained variance ratio vs the number of components.</td>
</tr>

<tr>
<td><a href="../../API/plots/plot_components">plot_components</a></td>
<td>Plot the explained variance ratio per components.</td>
</tr>

<tr>
<td><a href="../../API/plots/plot_rfecv">plot_rfecv</a></td>
<td>Plot the RFECV results.</td>
</tr>

<tr>
<td><a href="../../API/plots/plot_successive_halving">plot_successive_halving</a></td>
<td>Plot of the models" scores per iteration of the successive halving.</td>
</tr>

<tr>
<td><a href="../../API/plots/plot_learning_curve">plot_learning_curve</a></td>
<td>Plot the model's learning curve.</td>
</tr>

<tr>
<td><a href="../../API/plots/plot_results">plot_results</a></td>
<td>Plot a boxplot of the bootstrap results.</td>
</tr>

<tr>
<td><a href="../../API/plots/plot_bo">plot_bo</a></td>
<td>Plot the bayesian optimization scoring.</td>
</tr>

<tr>
<td><a href="../../API/plots/plot_evals">plot_evals</a></td>
<td>Plot evaluation curves for the train and test set.</td>
</tr>

<tr>
<td><a href="../../API/plots/plot_roc">plot_roc</a></td>
<td>Plot the Receiver Operating Characteristics curve.</td>
</tr>

<tr>
<td><a href="../../API/plots/plot_prc">plot_prc</a></td>
<td>Plot the precision-recall curve.</td>
</tr>

<tr>
<td><a href="../../API/plots/plot_permutation_importance">plot_permutation_importance</a></td>
<td>Plot the feature permutation importance of models.</td>
</tr>

<tr>
<td><a href="../../API/plots/plot_feature_importance">plot_feature_importance</a></td>
<td>Plot a tree-based model's feature importance.</td>
</tr>

<tr>
<td><a href="../../API/plots/plot_partial_dependence">plot_partial_dependence</a></td>
<td>Plot the partial dependence of features.</td>
</tr>

<tr>
<td><a href="../../API/plots/plot_errors">plot_errors</a></td>
<td>Plot a model's prediction errors.</td>
</tr>

<tr>
<td><a href="../../API/plots/plot_residuals">plot_residuals</a></td>
<td>Plot a model's residuals.</td>
</tr>

<tr>
<td><a href="../../API/plots/plot_confusion_matrix">plot_confusion_matrix</a></td>
<td>Plot a model's confusion matrix.</td>
</tr>

<tr>
<td><a href="../../API/plots/plot_threshold">plot_threshold</a></td>
<td>Plot metric performances against threshold values.</td>
</tr>

<tr>
<td><a href="../../API/plots/plot_probabilities">plot_probabilities</a></td>
<td>Plot the probability distribution of the classes in the target column.</td>
</tr>

<tr>
<td><a href="../../API/plots/plot_calibration">plot_calibration</a></td>
<td>Plot the calibration curve for a binary classifier.</td>
</tr>

<tr>
<td><a href="../../API/plots/plot_gains">plot_gains</a></td>
<td>Plot the cumulative gains curve.</td>
</tr>

<tr>
<td><a href="../../API/plots/plot_lift">plot_lift</a></td>
<td>Plot the lift curve.</td>
</tr>

<tr>
<td><a href="../../API/plots/bar_plot">bar_plot</a></td>
<td>Plot SHAP's bar plot.</td>
</tr>

<tr>
<td><a href="../../API/plots/beeswarm_plot">beeswarm_plot</a></td>
<td>Plot SHAP's beeswarm plot.</td>
</tr>

<tr>
<td><a href="../../API/plots/decision_plot">decision_plot</a></td>
<td>Plot SHAP's decision plot.</td>
</tr>

<tr>
<td><a href="../../API/plots/force_plot">force_plot</a></td>
<td>Plot SHAP's force plot.</td>
</tr>

<tr>
<td><a href="../../API/plots/heatmap_plot">heatmap_plot</a></td>
<td>Plot SHAP's heatmap plot.</td>
</tr>

<tr>
<td><a href="../../API/plots/scatter_plot">scatter_plot</a></td>
<td>Plot SHAP's scatter plot.</td>
</tr>

<tr>
<td><a href="../../API/plots/waterfall_plot">waterfall_plot</a></td>
<td>Plot SHAP's waterfall plot.</td>
</tr>
</table>
