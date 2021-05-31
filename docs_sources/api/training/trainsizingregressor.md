# TrainSizingRegressor
----------------------

<div style="font-size:20px">
<em>class</em> atom.training.<strong style="color:#008AB8">TrainSizingRegressor</strong>(models=None,
metric=None, greater_is_better=True, needs_proba=False, needs_threshold=False,
train_sizes=5, n_calls=0, n_initial_points=5, est_params=None, bo_params=None,
n_bootstrap=0, n_jobs=1, verbose=0, warnings=True, logger=None, experiment=None,
random_state=None)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/training.py#L425">[source]</a>
</span>
</div>

Fit and evaluate the models in a [train sizing](../../../user_guide/training/#train-sizing)
fashion. The following steps are applied to every model (per iteration):

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
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>models: str, estimator or sequence, optional (default=None)</strong><br>
Models to fit to the data. Allowed inputs are: an acronym from any of
ATOM's predefined models, an <a href="../../ATOM/atommodel">ATOMModel</a>
or a custom estimator as class or instance. If None, all the predefined
models are used. Available predefined models are:
<ul style="line-height:1.2em;margin-top:5px">
<li>"GP" for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html">Gaussian Process</a></li>
<li>"OLS" for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html">Ordinary Least Squares</a></li>
<li>"Ridge" for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html">Ridge Regression</a></li>
<li>"Lasso" for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html">Lasso Regression</a></li>
<li>"EN" for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html">ElasticNet</a></li>
<li>"BR" for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html">Bayesian Ridge</a></li>
<li>"ARD" for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html">Automated Relevance Determination</a></li>
<li>"KNN" for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html">K-Nearest Neighbors</a></li>
<li>"RNN" for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsRegressor.html">Radius Nearest Neighbors</a></li>
<li>"Tree" for a single <a href="https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html">Decision Tree</a></li>
<li>"Bag" for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html">Bagging</a></li>
<li>"ET" for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html">Extra-Trees</a></li>
<li>"RF" for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html">Random Forest</a></li>
<li>"AdaB" for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html">AdaBoost</a></li>
<li>"GBM" for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html">Gradient Boosting Machine</a></li> 
<li>"XGB" for <a href="https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor">XGBoost</a> (only available if package is installed)</li>
<li>"LGB" for <a href="https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html">LightGBM</a> (only available if package is installed)</li>
<li>"CatB" for <a href="https://catboost.ai/docs/concepts/python-reference_catboostregressor.html">CatBoost</a> (only available if package is installed)</li>
<li>"lSVM" for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html">Linear-SVM</a></li> 
<li>"kSVM" for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html">Kernel-SVM</a></li>
<li>"PA" for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveRegressor.html">Passive Aggressive</a></li>
<li>"SGD" for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html">Stochastic Gradient Descent</a></li>
<li>"MLP" for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor">Multi-layer Perceptron</a></li> 
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
<strong>train_sizes: int or sequence, optional (default=5)</strong><br>
Sequence of training set sizes used to run the trainings.
<ul style="line-height:1.2em;margin-top:5px;margin-bottom:0">
<li>If int: Number of equally distributed splits, i.e. for a
value N it's equal to np.linspace(1.0/N, 1.0, N).</li>
<li>If sequence: Fraction of the training set when <=1, else
total number of samples.</li>
</ul>
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
documentation for the available options. For multiple models, use
the acronyms as key and a dictionary of the parameters as value.
Add _fit to the parameter's name to pass it to the fit method instead
of the initializer.
</p>
<strong>bo_params: dict, optional (default=None)</strong><br>
Additional parameters to for the BO. These can include:
<ul style="line-height:1.2em;margin-top:5px">
<li><b>base_estimator: str, optional (default="GP")</b><br>Base estimator to use in the BO.
Choose from:
<ul style="line-height:1.2em;margin-top:5px">
<li>"GP" for Gaussian Process</li>
<li>"RF" for Random Forest</li>
<li>"ET" for Extra-Trees</li>
<li>"GBRT" for Gradient Boosted Regression Trees</li>
</ul></li>
<li><b>max_time: int, optional (default=np.inf)</b><br>Stop the optimization after <code>max_time</code> seconds.</li>
<li><b>delta_x: int or float, optional (default=0)</b><br>Stop the optimization when <code>|x1 - x2| < delta_x</code>.</li>
<li><b>delta_y: int or float, optional (default=0)</b><br>Stop the optimization if the 5 minima are within <code>delta_y</code> (the function is always minimized).</li>
<li><b>cv: int, optional (default=5)</b><br>Number of folds for the cross-validation. If 1, the
training set is randomly split in a subtrain and validation set.</li>
<li><b>early stopping: int, float or None, optional (default=None)</b><br>Training
will stop if the model didn't improve in last <code>early_stopping</code> rounds. If <1,
fraction of rounds from the total. If None, no early stopping is performed. Only
available for models that allow in-training evaluation.</li>
<li><b>callback: callable or list of callables, optional (default=None)</b><br>Callbacks for the BO.</li>
<li><b>dimensions: dict, array or None, optional (default=None)</b><br>Custom hyperparameter
space for the bayesian optimization. Can be an array to share dimensions across
models or a dictionary with the model's name as key. If None, ATOM's predefined dimensions are used.</li>
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
<strong>verbose: int, optional (default=0)</strong><br>
Verbosity level of the class. Possible values are:
<ul style="line-height:1.2em;margin-top:5px">
<li>0 to not print anything.</li>
<li>1 to print basic information.</li>
<li>2 to print detailed information.</li>
</ul>
<strong>warnings: bool or str, optional (default=True)</strong><br>
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
generator is the <code>RandomState</code> instance used by <code>numpy.random</code>.
</p>
</td>
</tr>
</table>
<br><br>



## Attributes

### Data attributes

The dataset can be accessed at any time through multiple attributes,
e.g. calling `trainer.train` will return the training set. Updating
one of the data attributes will automatically update the rest as well.
Changing the branch will also change the response from these attributes
accordingly.

<table style="font-size:16px">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Attributes:</strong></td>
<td width="80%" style="background:white;">
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
<strong>columns: list</strong><br>
Names of the columns in the dataset.
</p>
<p>
<strong>n_columns: int</strong><br>
Number of columns in the dataset.
</p>
<p>
<strong>features: list</strong><br>
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
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Attributes:</strong></td>
<td width="80%" style="background:white;">
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
<strong>winner: <a href="../../../user_guide/models">model</a></strong><br>
Model subclass that performed best on the test set.
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
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Attributes:</strong></td>
<td width="80%" style="background:white;">
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
<td><a href="#canvas">canvas</a></td>
<td>Create a figure with multiple plots.</td>
</tr>

<tr>
<td><a href="#cross-validate">cross_validate</a></td>
<td>Evaluate the winning model using cross-validation.</td>
</tr>

<tr>
<td><a href="#delete">delete</a></td>
<td>Remove a model from the pipeline.</td>
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
<td><a href="#reset-aesthetics">reset_aesthetics</a></td>
<td>Reset the plot aesthetics to their default values.</td>
</tr>

<tr>
<td><a href="#reset-predictions">reset_predictions</a></td>
<td>Clear the prediction attributes from all models.</td>
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
<td><a href="#scoring">scoring</a></td>
<td>Get all the models scoring for provided metrics.</td>
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


<a name="canvas"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">canvas</strong>(nrows=1,
ncols=2, title=None, figsize=None, filename=None, display=True)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L457">[source]</a>
</span>
</div>
This `@contextmanager` allows you to draw many plots in one figure.
The default option is to add two plots side by side. See the
[user guide](../../../user_guide/#canvas) for an example.
<table style="font-size:16px">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
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


<a name="cross-validate"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">cross_validate</strong>(**kwargs)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/basepredictor.py#L431">[source]</a>
</span>
</div>
Evaluate the winning model using cross-validation. This method cross-validates
the whole pipeline on the complete dataset. Use it to assess the robustness of
the model's performance.
<table style="font-size:16px">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>**kwargs</strong><br>
Additional keyword arguments for sklearn's <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html">cross_validate</a>
function. If the scoring method is not specified, it uses
the trainer's metric.
</td>
</tr>
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Returns:</strong></td>
<td width="80%" style="background:white;">
<strong>scores: dict</strong><br>
Return of sklearn's <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html">cross_validate</a>
function.
</td>
</tr>
</table>
<br />


<a name="delete"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">delete</strong>(models=None)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/basepredictor.py#L469">[source]</a>
</span>
</div>
Delete a model from the trainer. If the winning model is
removed, the next best model (through `metric_test` or
`mean_bootstrap`) is selected as winner. If all models are
removed, the metric and training approach are reset. Use
this method to drop unwanted models from the pipeline
or to free some memory before saving. Deleted models are
not removed from any active mlflow experiment.
<table style="font-size:16px">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>models: str or sequence, optional (default=None)</strong><br>
Models to delete. If None, delete them all.
</td>
</tr>
</table>
<br />


<a name="get-class-weight"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">get_class_weights</strong>(dataset="train")
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/basepredictor.py#L390">[source]</a>
</span>
</div>
Return class weights for a balanced data set. Statistically, the class
weights re-balance the data set so that the sampled data set represents
the target population as closely as possible. The returned weights are
inversely proportional to the class frequencies in the selected data set. 
<table style="font-size:16px">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>dataset: str, optional (default="train")</strong><br>
Data set from which to get the weights. Choose between "train", "test" or "dataset".
</tr>
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Returns:</strong></td>
<td width="80%" style="background:white;">
<strong>class_weights: dict</strong><br>
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
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<p>
<strong>deep: bool, optional (default=True)</strong><br>
If True, will return the parameters for this estimator and contained subobjects that are estimators.
</p>
</td>
</tr>
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Returns:</strong></td>
<td width="80%" style="background:white;">
<strong>params: dict</strong><br>
Dictionary of the parameter names mapped to their values.
</td>
</tr>
</table>
<br />


<a name="log"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">log</strong>(msg, level=0)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/basetransformer.py#L348">[source]</a>
</span>
</div>
Write a message to the logger and print it to stdout.
<table style="font-size:16px">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
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


<a name="reset-aesthetics"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">reset_aesthetics</strong>()
<span style="float:right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L211">[source]</a>
</span>
</div>
Reset the [plot aesthetics](../../../user_guide/#aesthetics) to their default values.
<br /><br /><br />


<a name="reset-predictions"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">reset_predictions</strong>()
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/basepredictor.py#L173">[source]</a>
</span>
</div>
Clear the [prediction attributes](../../..user_guide/predicting) from all models.
Use this method to free some memory before saving the trainer.
<br /><br /><br />


<a name="run"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">run</strong>(*arrays)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/training.py#L209">[source]</a>
</span>
</div>
Fit and evaluate the models.
<table style="font-size:16px">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>*arrays: sequence of indexables</strong><br>
Training set and test set. Allowed input formats are:
<ul style="line-height:1.2em;margin-top:5px">
<li>train, test</li>
<li>X_train, X_test, y_train, y_test</li>
<li>(X_train, y_train), (X_test, y_test)</li>
</ul>
</td>
</tr>
</table>
<br />


<a name="save"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">save</strong>(filename="auto", save_data=True)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/basetransformer.py#L369">[source]</a>
</span>
</div>
Save the instance to a pickle file. Remember that the class contains
the complete dataset as attribute, so the file can become large for
big datasets! To avoid this, use `save_data=False`.
<table style="font-size:16px">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
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


<a name="scoring"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">scoring</strong>(metric=None, dataset="test")
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/basepredictor.py#L437">[source]</a>
</span>
</div>
Get all the models scoring for provided metrics.
<table style="font-size:16px">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<p>
<strong>metric: str, func, scorer, sequence or None, optional (default=None)</strong><br>
Metrics to calculate. If None, a selection of the most common
metrics per task are used.
</p>
<p>
<strong>dataset: str, optional (default="test")</strong><br>
Data set on which to calculate the metric. Options are "train" or "test".
</p>
</td>
</tr>
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Returns:</strong></td>
<td width="80%" style="background:white;">
<strong>score: pd.DataFrame</strong><br>
Scoring of the models.
</td>
</tr>
</table>
<br />


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
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>**params: dict</strong><br>
Estimator parameters.
</tr>
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Returns:</strong></td>
<td width="80%" style="background:white;">
<strong>self: TrainSizingRegressor</strong><br>
Estimator instance.
</td>
</tr>
</table>
<br />


<a name="stacking"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">stacking</strong>(models=None,
estimator=None, stack_method="auto", passthrough=False)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/basepredictor.py#L339">[source]</a>
</span>
</div>
Add a [Stacking](../../../user_guide/#stacking) instance to the models in the pipeline.
<table style="font-size:16px">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<p>
<strong>models: sequence or None, optional (default=None)</strong><br>
Models that feed the stacking. If None, it selects all models
depending on the current branch.
</p>
<p>
<strong>estimator: str, callable or None, optional (default=None)</strong><br>
The final estimator, which is used to combine the base
estimators. If str, choose from ATOM's <a href="../../../user_guide/models/#predefined-models">predefined models</a>.
If None, <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html">Ridge</a> is selected.
</p>
<p>
<strong>stack_method: str, optional (default="auto")</strong><br>
Methods called for each base estimator. If "auto", it will try to 
invoke <code>predict_proba</code>, <code>decision_function</code>
or <code>predict</code> in that order.
</p>
<p>
<strong>passthrough: bool, optional (default=False)</strong>
When False, only the predictions of estimators are used
as training data for the final estimator. When True, the
estimator is trained on the predictions as well as the
original training data. The passed dataset is scaled
if any of the models require scaled features and they are
not already.
</p>
</td>
</tr>
</table>
<br />


<a name="voting"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">voting</strong>(models=None, weights=None)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/basepredictor.py#L306">[source]</a>
</span>
</div>
Add a [Voting](../../../user_guide/#voting) instance to the models in the pipeline.
<table style="font-size:16px">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<p>
<strong>models: sequence or None, optional (default=None)</strong><br>
Models that feed the voting. If None, it selects all models
depending on the current branch.
</p>
<p>
<strong>weights: sequence or None, optional (default=None)</strong><br>
Sequence of weights (int or float) to weight the
occurrences of predicted class labels (hard voting)
or class probabilities before averaging (soft voting).
Uses uniform weights if None.
</p>
</td>
</tr>
</table>
<br /><br />



## Example

```python
from atom.training import TrainSizingRegressor

# Run the pipeline
trainer = TrainSizingRegressor("RF", n_calls=5, n_initial_points=3)
trainer.run(train, test)

# Analyze the results
trainer.plot_learning_curve()
```