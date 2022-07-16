# DirectClassifier
-------------------

<div style="font-size:20px">
<em>class</em> atom.training.<strong style="color:#008AB8">DirectClassifier</strong>(models=None,
metric=None, greater_is_better=True, needs_proba=False, needs_threshold=False,
n_calls=0, n_initial_points=5, est_params=None, bo_params=None, n_bootstrap=0,
n_jobs=1, gpu=False, verbose=0, warnings=True, logger=None, experiment=None,
random_state=None)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/training.py#L252">[source]</a>
</span>
</div>

Fit and evaluate the models. The following steps are applied to every model:

1. Hyperparameter tuning is performed using a Bayesian Optimization
   approach (optional).
2. The model is fitted on the training set using the best combination
   of hyperparameters found.
3. The model is evaluated on the test set.
4. The model is trained on various bootstrapped samples of the training
   set and scored again on the test set (optional).

You can [predict](../../../user_guide/predicting), [plot](../../../user_guide/plots)
and call any [model](../../../user_guide/models) from the instance.
Read more in the [user guide](../../../user_guide/training).

<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<strong>models: str, estimator or sequence, optional (default=None)</strong><br>
Models to fit to the data. Allowed inputs are: an acronym from any of
ATOM's predefined models, an <a href="../../ATOM/atommodel">ATOMModel</a>
or a custom estimator as class or instance. If None, all the predefined
models are used. Available predefined models are:
<ul style="line-height:1.2em;margin-top:5px">
<li>"GP" for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html">Gaussian Process</a></li>
<li>"GNB" for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html">Gaussian Naive Bayes</a></li>
<li>"MNB" for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html">Multinomial Naive Bayes</a></li>
<li>"BNB" for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html">Bernoulli Naive Bayes</a></li>
<li>"CatNB" for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.CategoricalNB.html">Categorical Naive Bayes</a></li>
<li>"CNB" for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html">Complement Naive Bayes</a></li>
<li>"Ridge" for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html">Ridge Classification</a></li>
<li>"LR" for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html">Logistic Regression</a></li> 
<li>"LDA" for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html">Linear Discriminant Analysis</a></li>
<li>"QDA" for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html">Quadratic Discriminant Analysis</a></li>
<li>"KNN" for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html">K-Nearest Neighbors</a></li>
<li>"RNN" for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html">Radius Nearest Neighbors</a></li>
<li>"Tree" for a single <a href="https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html">Decision Tree</a></li>
<li>"Bag" for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html">Bagging</a></li>
<li>"ET" for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html">Extra-Trees</a></li>
<li>"RF" for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html">Random Forest</a></li>
<li>"AdaB" for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html">AdaBoost</a></li>
<li>"GBM" for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html">Gradient Boosting Machine</a></li>
<li>"XGB" for <a href="https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier">XGBoost</a> (only available if package is installed)</li>
<li>"LGB" for <a href="https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html">LightGBM</a> (only available if package is installed)</li>
<li>"CatB" for <a href="https://catboost.ai/docs/concepts/python-reference_catboostclassifier.html">CatBoost</a> (only available if package is installed)</li>
<li>"lSVM" for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html">Linear SVM</a></li> 
<li>"kSVM" for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">Kernel SVM</a></li>
<li>"PA" for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html">Passive Aggressive</a></li>
<li>"SGD" for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html">Stochastic Gradient Descent</a></li>
<li>"MLP" for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html">Multi-layer Perceptron</a></li> 
</ul>
<strong>metric: str, func, scorer, sequence or None, optional (default=None)</strong><br>
Metric on which to fit the models. Choose from any of sklearn's
<a href="https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules">SCORERS</a>,
a function with signature <code>metric(y_true, y_pred)</code>,
a scorer object or a sequence of these. If multiple metrics are
selected, only the first is used to optimize the BO. If None, a
default metric is selected:
<ul style="line-height:1.2em;margin-top:5px">
<li>"f1" for binary classification</li>
<li>"f1_weighted" for multiclass classification</li>
<li>"r2" for regression</li>
</ul>
<p>
<strong>greater_is_better: bool or sequence, optional (default=True)</strong><br>
Whether the metric is a score function or a loss function,
i.e. if True, a higher score is better and if False, lower is
better. This parameter is ignored if the metric is a string or
a scorer. If sequence, the n-th value applies to the n-th
metric.
</p>
<p>
<strong>needs_proba: bool or sequence, optional (default=False)</strong><br>
Whether the metric function requires probability estimates out
of a classifier. If True, make sure that every selected model has
a <code>predict_proba</code> method. This parameter is ignored
if the metric is a string or a scorer. If sequence, the n-th
value applies to the n-th metric.
</p>
<p>
<strong>needs_threshold: bool or sequence, optional (default=False)</strong><br>
Whether the metric function takes a continuous decision certainty.
This only works for binary classification using estimators that
have either a <code>decision_function</code> or <code>predict_proba</code>
method. This parameter is ignored if the metric is a string or a
scorer. If sequence, the n-th value applies to the n-th metric.
</p>
<p>
<strong>n_calls: int or sequence, optional (default=0)</strong><br>
Maximum number of iterations of the BO. It includes the random
points of <code>n_initial_points</code>. If 0, skip the BO and
fit the model on its default parameters. If sequence, the n-th
value applies to the n-th model.
</p>
<p>
<strong>n_initial_points: int or sequence, optional (default=5)</strong><br>
Initial number of random tests of the BO before fitting the
surrogate function. If equal to <code>n_calls</code>, the optimizer will
technically be performing a random search. If sequence, the n-th
value applies to the n-th model.
</p>
<p>
<strong>est_params: dict, optional (default=None)</strong><br>
Additional parameters for the estimators. See the corresponding
documentation for the available options. For multiple models,
use the acronyms as key (or 'all' for all models) and a dict
of the parameters as value. Add _fit to the parameter's name
to pass it to the fit method instead of the initializer.
</p>
<strong>bo_params: dict, optional (default=None)</strong><br>
Additional parameters to for the BO. These can include:
<ul style="line-height:1.2em;margin-top:5px">
<li><b>base_estimator: str, optional (default="GP")</b><br>
Base estimator to use in the BO. Choose from:
<ul style="line-height:1.2em;margin-top:5px">
<li>"GP" for Gaussian Process</li>
<li>"RF" for Random Forest</li>
<li>"ET" for Extra-Trees</li>
<li>"GBRT" for Gradient Boosted Regression Trees</li>
</ul></li>
<li><b>max_time: int, optional (default=np.inf)</b><br>Stop the optimization after <code>max_time</code> seconds.</li>
<li><b>delta_x: int or float, optional (default=0)</b><br>Stop the optimization when <code>|x1 - x2| < delta_x</code>.</li>
<li><b>delta_y: int or float, optional (default=0)</b><br>Stop the optimization if the 5 minima are within <code>delta_y</code> (the function is always minimized).</li>
<li><b>cv: int, optional (default=1)</b><br>Number of folds for
the cross-validation. If 1, the training set is randomly split
in a subtrain and validation set.</li>
<li><b>early stopping: int, float or None, optional (default=None)</b><br>Training
will stop if the model didn't improve in last <code>early_stopping</code> rounds. If <1,
fraction of rounds from the total. If None, no early stopping is performed. Only
available for models that allow in-training evaluation.</li>
<li><b>callback: callable or list of callables, optional (default=None)</b><br>Callbacks for the BO.</li>
<li><b>dimensions: dict, list or None, optional (default=None)</b><br>Custom hyperparameter
space for the bayesian optimization. Can be a list to share dimensions across
models or a dict with the model's name as key (or 'all' for all models). If None,
ATOM's predefined dimensions are used.</li>
<li><b>plot: bool, optional (default=False)</b><br>Whether to plot the BO's progress as it runs.
Creates a canvas with two plots: the first plot shows the score of every trial
and the second shows the distance between the last consecutive steps.</li>
<li><b>Additional keyword arguments for skopt's optimizer.</b></li>                
</ul>
<p>
<strong>bootstrap: int or sequence, optional (default=0)</strong><br>
Number of data sets (bootstrapped from the training set) to use in
the bootstrap algorithm. If 0, no bootstrap is performed.
If sequence, the n-th value will apply to the n-th model.
</p>
<strong>n_jobs: int, optional (default=1)</strong><br>
Number of cores to use for parallel processing.
<ul style="line-height:1.2em;margin-top:5px">
<li>If >0: Number of cores to use.</li>
<li>If -1: Use all available cores.</li>
<li>If <-1: Use available_cores - 1 + <code>n_jobs</code>.</li>
</ul>
<p style="margin-top:5px">
Beware that using multiple processes on the same machine may cause
memory issues for large datasets.
</p>
<strong>gpu: bool or str, optional (default=False)</strong><br>
Train models on GPU (instead of CPU). Refer to the
<a href="../../../user_guide/gpu">documentation</a>
to check which estimators are supported.
<ul style="line-height:1.2em;margin-top:5px">
<li>If False: Always use CPU implementation.</li>
<li>If True: Use GPU implementation if possible.</li>
<li>If "force": Force GPU implementation.</li>
</ul>
<strong>verbose: int, optional (default=0)</strong><br>
Verbosity level of the class. Choose from:
<ul style="line-height:1.2em;margin-top:5px">
<li>0 to not print anything.</li>
<li>1 to print basic information.</li>
<li>2 to print detailed information.</li>
</ul>
<strong>warnings: bool or str, optional (default=False)</strong><br>
<ul style="line-height:1.2em;margin-top:5px;margin-bottom:0">
<li>If True: Default warning action (equal to "default").</li>
<li>If False: Suppress all warnings (equal to "ignore").</li>
<li>If str: One of the actions in python's warnings environment.</li>
</ul>
<p style="margin-top:5px">
Changing this parameter affects the <code>PYTHONWARNINGS</code> environment.
<br>ATOM can't manage warnings that go directly from C/C++ code to stdout.</p>
<strong>logger: str, Logger or None, optional (default=None)</strong><br>
<ul style="line-height:1.2em;margin-top:5px">
<li>If None: Doesn't save a logging file.</li>
<li>If str: Name of the log file. Use "auto" for automatic naming.</li>
<li>Else: Python <code>logging.Logger</code> instance.</li>
</ul>
<p>
<strong>experiment: str or None, optional (default=None)</strong><br>
Name of the mlflow experiment to use for tracking. If None,
no mlflow tracking is performed.
</p>
<p>
<strong>random_state: int or None, optional (default=None)</strong><br>
Seed used by the random number generator. If None, the random number
generator is the <code>RandomState</code> instance used by <code>np.random</code>.
</p>
</td>
</tr>
</table>
<br>



## Magic methods

The class contains some magic methods to help you access some of its
elements faster.

* **\__len__:** Returns the length of the dataset.
* **\__contains__:** Checks if the provided item is a column in the dataset.
* **\__getitem__:** Access a model, column or subset of the dataset.

<br>



## Attributes

### Data attributes

The dataset can be accessed at any time through multiple attributes,
e.g. calling `trainer.train` will return the training set. Updating
one of the data attributes will automatically update the rest as well.
Changing the branch will also change the response from these attributes
accordingly.

<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Attributes:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>dataset: pd.DataFrame</strong><br>
Complete dataset in the pipeline.
</p>
<p>
<strong>train: pd.DataFrame</strong><br>
Training set.
</p>
<p>
<strong>test: pd.DataFrame</strong><br>
Test set.
</p>
<p>
<strong>X: pd.DataFrame</strong><br>
Feature set.
</p>
<p>
<strong>y: pd.Series</strong><br>
Target column.
</p>
<p>
<strong>X_train: pd.DataFrame</strong><br>
Training features.
</p>
<p>
<strong>y_train: pd.Series</strong><br>
Training target.
</p>
<p>
<strong>X_test: pd.DataFrame</strong><br>
Test features.
</p>
<p>
<strong>y_test: pd.Series</strong><br>
Test target.
</p>
<p>
<strong>shape: tuple</strong><br>
Dataset's shape: (n_rows x n_columns) or (n_rows, (shape_sample), n_cols)
for datasets with more than two dimensions.
</p>
<p>
<strong>columns: pd.Index</strong><br>
Names of the columns in the dataset.
</p>
<p>
<strong>n_columns: int</strong><br>
Number of columns in the dataset.
</p>
<p>
<strong>features: pd.Index</strong><br>
Names of the features in the dataset.
</p>
<p>
<strong>n_features: int</strong><br>
Number of features in the dataset.
</p>
<p>
<strong>target: str</strong><br>
Name of the target column.
</p>
</td>
</tr>
</table>
<br>


### Utility attributes

<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Attributes:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>models: list</strong><br>
List of models in the pipeline.
</p>
<p>
<strong>metric: str or list</strong><br>
Metric(s) used to fit the models.
</p>
<p>
<strong>errors: dict</strong><br>
Dictionary of the encountered exceptions (if any).
</p>
<p>
<strong>winners: list of str</strong><br>
Model names ordered by performance on the test set (either through the
<code>metric_test</code> or <code>mean_bootstrap</code> attribute).
</p>
<p>
<strong>winner: <a href="../../../user_guide/models">model</a></strong><br>
Model subclass that performed best on the test set (either through the
<code>metric_test</code> or <code>mean_bootstrap</code> attribute).
</p>
<strong>results: pd.DataFrame</strong><br>
Dataframe of the training results. Columns can include:
<ul style="line-height:1.2em;margin-top:5px">
<li><b>metric_bo:</b> Best score achieved during the BO.</li>
<li><b>time_bo:</b> Time spent on the BO.</li>
<li><b>metric_train:</b> Metric score on the training set.</li>
<li><b>metric_test:</b> Metric score on the test set.</li>
<li><b>time_fit:</b> Time spent fitting and evaluating.</li>
<li><b>mean_bootstrap:</b> Mean score of the bootstrap results.</li>
<li><b>std_bootstrap:</b> Standard deviation score of the bootstrap results.</li>
<li><b>time_bootstrap:</b> Time spent on the bootstrap algorithm.</li>
<li><b>time:</b> Total time spent on the whole run.</li>
</ul>
</td>
</tr>
</table>
<br>


### Plot attributes
 
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Attributes:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>style: str</strong><br>
Plotting style. See seaborn's <a href="https://seaborn.pydata.org/tutorial/aesthetics.html#seaborn-figure-styles">documentation</a>.
</p>
<p>
<strong>palette: str</strong><br>
Color palette. See seaborn's <a href="https://seaborn.pydata.org/tutorial/color_palettes.html">documentation</a>.
</p>
<p>
<strong>title_fontsize: int</strong><br>
Fontsize for the plot's title.
</p>
<p>
<strong>label_fontsize: int</strong><br>
Fontsize for labels and legends.
</p>
<p>
<strong>tick_fontsize: int</strong><br>
Fontsize for the ticks along the plot's axes.
</p>
</td>
</tr>
</table>
<br><br>




## Methods

<table style="font-size:16px">
<tr>
<td><a href="#available-models">available_models</a></td>
<td>Give an overview of the available predefined models.</td>
</tr>

<tr>
<td><a href="#canvas">canvas</a></td>
<td>Create a figure with multiple plots.</td>
</tr>

<tr>
<td><a href="#clear">clear</a></td>
<td>Clear attributes from all models.</td>
</tr>

<tr>
<td><a href="#delete">delete</a></td>
<td>Delete models from the trainer.</td>
</tr>

<tr>
<td><a href="#evaluate">evaluate</a></td>
<td>Get all models' scores for the provided metrics.</td>
</tr>

<tr>
<td><a href="#get-class-weight">get_class_weight</a></td>
<td>Return class weights for a balanced dataset.</td>
</tr>

<tr>
<td><a href="#get-params">get_params</a></td>
<td>Get parameters for this estimator.</td>
</tr>

<tr>
<td><a href="#log">log</a></td>
<td>Save information to the logger and print to stdout.</td>
</tr>

<tr>
<td><a href="#merge">merge</a></td>
<td>Merge another trainer into this one.</td>
</tr>

<tr>
<td><a href="#reset-aesthetics">reset_aesthetics</a></td>
<td>Reset the plot aesthetics to their default values.</td>
</tr>

<tr>
<td><a href="#run">run</a></td>
<td>Fit and evaluate the models.</td>
</tr>

<tr>
<td><a href="#save">save</a></td>
<td>Save the instance to a pickle file.</td>
</tr>

<tr>
<td><a href="#set-params">set_params</a></td>
<td>Set the parameters of this estimator.</td>
</tr>

<tr>
<td><a href="#stacking">stacking</a></td>
<td>Add a Stacking instance to the models in the pipeline.</td>
</tr>

<tr>
<td><a href="#voting">voting</a></td>
<td>Add a Voting instance to the models in the pipeline.</td>
</tr>
</table>
<br>


<a name="available-models"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">available_models</strong>()
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/basepredictor.py#L500">[source]</a>
</span>
</div>
Give an overview of the available predefined models.
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Returns:</strong></td>
<td width="80%" class="td_params">
<strong>pd.DataFrame</strong><br>
Information about the predefined models available for the current task.
Columns include:
<ul style="line-height:1.2em;margin-top:5px">
<li><b>acronym:</b> Model's acronym (used to call the model).</li>
<li><b>fullname:</b> Complete name of the model.</li>
<li><b>estimator:</b> The model's underlying estimator.</li>
<li><b>module:</b> The estimator's module.</li>
<li><b>needs_scaling:</b> Whether the model requires feature scaling.</li>
<li><b>accepts_sparse:</b> Whether the model has native support for sparse matrices.</li>
<li><b>supports_gpu:</b> Whether the model has GPU support.</li>
</ul>
</td>
</tr>
</table>
<br />


<a name="canvas"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">canvas</strong>(nrows=1,
ncols=2, title=None, figsize=None, filename=None, display=True)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L427">[source]</a>
</span>
</div>
This `@contextmanager` allows you to draw many plots in one figure.
The default option is to add two plots side by side. See the
[user guide](../../../user_guide/plots/#canvas) for an example.
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>nrows: int, optional (default=1)</strong><br>
Number of plots in length.
</p>
<p>
<strong>ncols: int, optional (default=2)</strong><br>
Number of plots in width.
</p>
<p>
<strong>title: str or None, optional (default=None)</strong><br>
Plot's title. If None, no title is displayed.
</p>
<p>
<strong>figsize: tuple or None, optional (default=None)</strong><br>
Figure's size, format as (x, y). If None, it adapts the size to the
number of plots in the canvas.
</p>
<p>
<strong>filename: str or None, optional (default=None)</strong><br>
Name of the file. Use "auto" for automatic naming.
If None, the figure is not saved.
</p>
<p>
<strong>display: bool, optional (default=True)</strong><br>
Whether to render the plot.
</p>
</td>
</tr>
</table>
<br />


<a name="clear"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">clear</strong>()
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/basepredictor.py#L536">[source]</a>
</span>
</div>
Reset all model attributes to their initial state, deleting potentially
large data arrays. Use this method to free some memory before saving
the class. The cleared attributes per model are:

* [Prediction attributes](../../../user_guide/predicting).
* [Metrics scores](../../../user_guide/training/#metric).
* [Shap values](../../../user_guide/plots/#shap).

<br /><br /><br />


<a name="delete"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">delete</strong>(models=None)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/basepredictor.py#L551">[source]</a>
</span>
</div>
Delete models from the trainer. If all models are removed, the metric
is reset. Use this method to drop unwanted models from the pipeline
or to free some memory before saving. Deleted models are not removed
from any active mlflow experiment.
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<strong>models: str or sequence, optional (default=None)</strong><br>
Models to delete. If None, delete them all.
</td>
</tr>
</table>
<br />


<a name="evaluate"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">evaluate</strong>(metric=None,
dataset="test", threshold=0.5)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/basepredictor.py#L578">[source]</a>
</span>
</div>
Get all the models' scores for the provided metrics.
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>metric: str, func, scorer, sequence or None, optional (default=None)</strong><br>
Metrics to calculate. If None, a selection of the most common
metrics per task are used.
</p>
<p>
<strong>dataset: str, optional (default="test")</strong><br>
Data set on which to calculate the metric. Choose from: "train",
"test" or "holdout".
</p>
<strong>threshold: float, optional (default=0.5)</strong><br>
Threshold between 0 and 1 to convert predicted probabilities
to class labels. Only used when:
<ul style="line-height:1.2em;margin-top:5px">
<li>The task is binary classification.</li>
<li>The model has a <code>predict_proba</code> method.</li>
<li>The metric evaluates predicted target values.</li>
</ul>
</td>
</tr>
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Returns:</strong></td>
<td width="80%" class="td_params">
<strong>pd.DataFrame</strong><br>
Scores of the models.
</td>
</tr>
</table>
<br />


<a name="get-class-weight"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">get_class_weights</strong>(dataset="train")
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/basepredictor.py#L624">[source]</a>
</span>
</div>
Return class weights for a balanced data set. Statistically, the class
weights re-balance the data set so that the sampled data set represents
the target population as closely as possible. The returned weights are
inversely proportional to the class frequencies in the selected data set. 
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<strong>dataset: str, optional (default="train")</strong><br>
Data set from which to get the weights. Choose from: "train", "test" or "dataset".
</tr>
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Returns:</strong></td>
<td width="80%" class="td_params">
<strong>dict</strong><br>
Classes with the corresponding weights.
</td>
</tr>
</table>
<br />


<a name="get-params"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">get_params</strong>(deep=True)
<span style="float:right">
<a href="https://github.com/scikit-learn/scikit-learn/blob/0fb307bf3/sklearn/base.py#L189">[source]</a>
</span>
</div>
Get parameters for this estimator.
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>deep: bool, optional (default=True)</strong><br>
If True, will return the parameters for this estimator and contained
subobjects that are estimators.
</p>
</td>
</tr>
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Returns:</strong></td>
<td width="80%" class="td_params">
<strong>dict</strong><br>
Parameter names mapped to their values.
</td>
</tr>
</table>
<br />


<a name="log"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">log</strong>(msg, level=0)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/basetransformer.py#L590">[source]</a>
</span>
</div>
Write a message to the logger and print it to stdout.
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>msg: str</strong><br>
Message to write to the logger and print to stdout.
</p>
<p>
<strong>level: int, optional (default=0)</strong><br>
Minimum verbosity level to print the message.
</p>
</td>
</tr>
</table>
<br />


<a name="merge"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">merge</strong>(other, suffix="2")
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/basepredictor.py#L659">[source]</a>
</span>
</div>
Merge another trainer into this one. Branches, models, metrics and
attributes of the other trainer are merged into this one. If there
are branches and/or models with the same name, they are merged
adding the `suffix` parameter to their name. The errors and missing
attributes are extended with those of the other instance. It's only
possible to merge two instances if they are initialized with the same
dataset and trained with the same metric.
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>other: trainer</strong><br>
Trainer instance with which to merge.
</p>
<p>
<strong>suffix: str, optional (default="2")</strong><br>
Conflicting branches and models are merged adding <code>suffix</code>
to the end of their names.
</p>
</td>
</tr>
</table>
<br />


<a name="reset-aesthetics"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">reset_aesthetics</strong>()
<span style="float:right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L213">[source]</a>
</span>
</div>
Reset the [plot aesthetics](../../../user_guide/plots/#aesthetics) to their default values.
<br /><br /><br />


<a name="run"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">run</strong>(*arrays)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/training.py#L49">[source]</a>
</span>
</div>
Fit and evaluate the models.
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<strong>*arrays: sequence of indexables</strong><br>
Training and test set (and optionally a holdout set). Allowed formats are:
<ul style="line-height:1.2em;margin-top:5px">
<li>train, test</li>
<li>train, test, holdout</li>
<li>X_train, X_test, y_train, y_test</li>
<li>X_train, X_test, X_holdout, y_train, y_test, y_holdout</li>
<li>(X_train, y_train), (X_test, y_test)</li>
<li>(X_train, y_train), (X_test, y_test), (X_holdout, y_holdout)</li>
</ul>
</td>
</tr>
</table>
<br />


<a name="save"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">save</strong>(filename="auto", save_data=True)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/basetransformer.py#L611">[source]</a>
</span>
</div>
Save the instance to a pickle file. Remember that the class contains
the complete dataset as attribute, so the file can become large for
big datasets! To avoid this, use `save_data=False`.
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>filename: str, optional (default="auto")</strong><br>
Name of the file. Use "auto" for automatic naming.
</p>
<p>
<strong>save_data: bool, optional (default=True)</strong><br>
Whether to save the data as an attribute of the instance. If False,
remember to add the data to <a href="../../ATOM/atomloader">ATOMLoader</a>
when loading the file.
</p>
</td>
</tr>
</table>
<br>


<a name="set-params"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">set_params</strong>(**params)
<span style="float:right">
<a href="https://github.com/scikit-learn/scikit-learn/blob/0fb307bf3/sklearn/base.py#L221">[source]</a>
</span>
</div>
Set the parameters of this estimator.
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<strong>**params: dict</strong><br>
Estimator parameters.
</tr>
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Returns:</strong></td>
<td width="80%" class="td_params">
<strong>DirectClassifier</strong><br>
Estimator instance.
</td>
</tr>
</table>
<br />


<a name="stacking"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">stacking</strong>(name="Stack",
models=None, **kwargs)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/basepredictor.py#L728">[source]</a>
</span>
</div>
Add a [Stacking](../../../user_guide/models/#stacking) model to the pipeline.
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>name: str, optional (default="Stack")</strong><br>
Name of the model. The name is always presided with the
model's acronym: <code>Stack</code>.
</p>
<p>
<strong>models: sequence or None, optional (default=None)</strong><br>
Models that feed the stacking estimator. If None, it selects
all non-ensemble models trained on the current branch.
</p>
<p>
<strong>**kwargs</strong><br>
Additional keyword arguments for sklearn's <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html">StackingClassifier</a>
instance. The <a href="../../../user_guide/models/#predefined-models">predefined model's</a>
acronyms can be used for the <code>final_estimator</code> parameter.
</td>
</tr>
</table>
<br />


<a name="voting"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">voting</strong>(name="Vote",
models=None, **kwargs)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/basepredictor.py#L798">[source]</a>
</span>
</div>
Add a [Voting](../../../user_guide/models/#voting) model to the pipeline.
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>name: str, optional (default="Vote")</strong><br>
Name of the model. The name is always presided with the
model's acronym: <code>Vote</code>.
</p>
<p>
<strong>models: sequence or None, optional (default=None)</strong><br>
Models that feed the voting estimator. If None, it selects
all non-ensemble models trained on the current branch.
</p>
<p>
<strong>**kwargs</strong><br>
Additional keyword arguments for sklearn's <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html">VotingClassifier</a>
instance.
</td>
</tr>
</table>
<br /><br />



## Example

```python
from atom.training import DirectClassifier

trainer = DirectClassifier(["Tree", "RF"], n_calls=25, n_initial_points=10)
trainer.run(train, test)

# Analyze the results
trainer.plot_prc()
print(trainer.results)
```