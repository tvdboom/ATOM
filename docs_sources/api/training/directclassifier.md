# DirectClassifier
-------------------

<pre><em>class</em> atom.training.<strong style="color:#008AB8">DirectClassifier</strong>(models, metric=None, greater_is_better=True, needs_proba=False,
                                     needs_threshold=False, n_calls=0, n_initial_points=5,
                                     est_params={}, bo_params={}, bagging=0, n_jobs=1,
                                     verbose=0, logger=None, random_state=None) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/training.py#L258">[source]</a></div></pre>
Fit and evaluates the models to the data in the pipeline. The following steps are applied:

1. The optimal hyperparameters are selected using a bayesian optimization algorithm.
2. The model is fitted on the training set using the best combinations of hyperparameters found.
3. Using a bagging algorithm, various scores on the test set are calculated.

Just like atom, you can [predict](../../../user_guide/#predicting),
[plot](../../../user_guide/#plots) and call any [model](../../../user_guide/#models)
from the DirectClassifier instance. Read more in the [user guide](../../../user_guide/#training).
<table>
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>models: str or sequence</strong>
<blockquote>
Models to fit to the data. Use a custom estimator or the model's predefined acronyms. Possible values are (case-insensitive):
<ul>
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
<li>"lSVM" for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html">Linear-SVM</a></li> 
<li>"kSVM" for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">Kernel-SVM</a></li>
<li>"PA" for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html">Passive Aggressive</a></li>
<li>"SGD" for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html">Stochastic Gradient Descent</a></li>
<li>"MLP" for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html">Multi-layer Perceptron</a></li> 
</ul>
</blockquote>
<strong>metric: str, callable or sequence, optional (default=None)</strong>
<blockquote>
Metric on which to fit the models. Choose from any of sklearn's predefined
<a href="https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules">SCORERS</a>,
a score (or loss) function with signature metric(y, y_pred, **kwargs), a
scorer object or a sequence of these. If multiple metrics are selected, only
the first will be used to optimize the BO. If None, a default metric is selected:
<ul>
<li>"f1" for binary classification</li>
<li>"f1_weighted" for multiclass classification</li>
<li>"r2" for regression</li>
</ul>
</blockquote>
<strong>greater_is_better: bool or sequence, optional (default=True)</strong>
<blockquote>
Whether the metric is a score function or a loss function,
i.e. if True, a higher score is better and if False, lower is
better. Will be ignored if the metric is a string or a scorer.
If sequence, the n-th value will apply to the n-th metric.
</blockquote>
<strong> needs_proba: bool or sequence, optional (default=False)</strong>
<blockquote>
Whether the metric function requires probability estimates out of a
classifier. If True, make sure that every selected model has a
<code>predict_proba</code> method. Will be ignored if the metric is a
string or a scorer. If sequence, the n-th value will apply to the n-th
metric.
</blockquote>
<strong> needs_threshold: bool or sequence, optional (default=False)</strong>
<blockquote>
Whether the metric function takes a continuous decision certainty.
This only works for binary classification using estimators that
have either a <code>decision_function</code> or <code>predict_proba</code>
method. Will be ignored if the metric is a string or a scorer. If sequence,
the n-th value will apply to the n-th metric.
</blockquote>
<strong>n_calls: int or sequence, optional (default=0)</strong>
<blockquote>
Maximum number of iterations of the BO. It includes the random
points of <code>n_initial_points</code>. If 0, skip the BO and
fit the model on its default parameters. If sequence, the n-th
value will apply to the n-th model.
</blockquote>
<strong>n_initial_points: int or sequence, optional (default=5)</strong>
<blockquote>
Initial number of random tests of the BO before fitting the
surrogate function. If equal to <code>n_calls</code>, the optimizer will
technically be performing a random search. If sequence, the n-th
value will apply to the n-th model.
</blockquote>
<strong>est_params: dict, optional (default={})</strong>
<blockquote>
Additional parameters for the estimators. See the corresponding
documentation for the available options. For multiple models, use
the acronyms as key and a dictionary of the parameters as value.
Add _fit to the parameter's name to pass it to the fit method instead
of the initializer.
</blockquote>
<strong>bo_params: dict, optional (default={})</strong>
<blockquote>
Additional parameters to for the BO. These can include:
<ul>
<li><b>base_estimator: str, optional (default="GP")</b><br>Base estimator to use in the BO.
Choose from:
<ul>
<li>"GP" for Gaussian Process</li>
<li>"RF" for Random Forest</li>
<li>"ET" for Extra-Trees</li>
<li>"GBRT" for Gradient Boosted Regression Trees</li>
</ul></li>
<li><b>max_time: int, optional (default=np.inf)</b><br>Stop the optimization after <code>max_time</code> seconds.</li>
<li><b>delta_x: int or float, optional (default=0)</b><br>Stop the optimization when <code>|x1 - x2| < delta_x</code>.</li>
<li><b>delta_y: int or float, optional (default=0)</b><br>Stop the optimization if the 5 minima are within <code>delta_y</code> (skopt always minimizes the function).</li>
<li><b>cv: int, optional (default=5)</b><br>Number of folds for the cross-validation. If 1, the
training set will be randomly split in a subtrain and validation set.</li>
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
</blockquote>
<strong>bagging: int or sequence, optional (default=0)</strong>
<blockquote>
Number of data sets (bootstrapped from the training set) to use in
the bagging algorithm. If 0, no bagging is performed.
If sequence, the n-th value will apply to the n-th model.
</blockquote>
<strong>n_jobs: int, optional (default=1)</strong>
<blockquote>
Number of cores to use for parallel processing.
<ul>
<li>If >0: Number of cores to use.</li>
<li>If -1: Use all available cores.</li>
<li>If <-1: Use available_cores - 1 + n_jobs.</li>
</ul>
Beware that using multiple processes on the same machine may cause
memory issues for large datasets.
</blockquote>
<strong>verbose: int, optional (default=0)</strong>
<blockquote>
Verbosity level of the class. Possible values are:
<ul>
<li>0 to not print anything.</li>
<li>1 to print basic information.</li>
<li>2 to print detailed information.</li>
</ul>
</blockquote>
<strong>logger: str, Logger or None, optional (default=None)</strong>
<blockquote>
<ul>
<li>If None: Doesn't save a logging file.</li>
<li>If str: Name of the logging file. Use "auto" for default name.</li>
<li>Else: Python <code>logging.Logger</code> instance.</li>
</ul>
The default name consists of the class' name followed by the
timestamp of the logger's creation.
</blockquote>
<strong>random_state: int or None, optional (default=None)</strong>
<blockquote>
Seed used by the random number generator. If None, the random number
generator is the <code>RandomState</code> instance used by <code>numpy.random</code>.
</blockquote>
</td>
</tr>
</table>
<br><br>



## Attributes
-------------

### Data attributes

The dataset can be accessed at any time through multiple attributes, e.g. calling
`trainer.train` will return the training set. The data can also be changed through
these attributes, e.g. `trainer.test = atom.test.drop(0)` will drop the first row
from the test set. Updating one of the data attributes will automatically update the
rest as well.

<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Attributes:</strong></td>
<td width="75%" style="background:white;">
<strong>dataset: pd.DataFrame</strong>
<blockquote>
Complete dataset in the pipeline.
</blockquote>
<strong>train: pd.DataFrame</strong>
<blockquote>
Training set.
</blockquote>
<strong>test: pd.DataFrame</strong>
<blockquote>
Test set.
</blockquote>
<strong>X: pd.DataFrame</strong>
<blockquote>
Feature set.
</blockquote>
<strong>y: pd.Series</strong>
<blockquote>
Target column.
</blockquote>
<strong>X_train: pd.DataFrame</strong>
<blockquote>
Training features.
</blockquote>
<strong>y_train: pd.Series</strong>
<blockquote>
Training target.
</blockquote>
<strong>X_test: pd.DataFrame</strong>
<blockquote>
Test features.
</blockquote>
<strong>y_test: pd.Series</strong>
<blockquote>
Test target.
</blockquote>
<strong>shape: tuple</strong>
<blockquote>
Dataset's shape: (n_rows x n_columns) or
(n_rows, (shape_sample), n_cols) for deep learning datasets.
</blockquote>
<strong>columns: list</strong>
<blockquote>
Names of the columns in the dataset.
</blockquote>
<strong>n_columns: int</strong>
<blockquote>
Number of columns in the dataset.
</blockquote>
<strong>features: list</strong>
<blockquote>
Names of the features in the dataset.
</blockquote>
<strong>n_features: int</strong>
<blockquote>
Number of features in the dataset.
</blockquote>
<strong>target: str</strong>
<blockquote>
Name of the target column.
</blockquote>
</td>
</tr>
</table>
<br>


### Utility attributes

<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Attributes:</strong></td>
<td width="75%" style="background:white;">
<strong>models: list</strong>
<blockquote>
List of models in the pipeline.
</blockquote>
<strong>metric: str or list</strong>
<blockquote>
Metric(s) used to fit the models.
</blockquote>
<strong>errors: dict</strong>
<blockquote>
Dictionary of the encountered exceptions during fitting (if any).
</blockquote>
<strong>winner: <a href="../../../user_guide/#models">model</a></strong>
<blockquote>
Model subclass that performed best on the test set.
</blockquote>
<strong>results: pd.DataFrame</strong>
<blockquote>
Dataframe of the training results with the model acronyms as index. Columns can include:
<ul>
<li><b>metric_bo:</b> Best score achieved during the BO.</li>
<li><b>time_bo:</b> Time spent on the BO.</li>
<li><b>metric_train:</b> Metric score on the training set.</li>
<li><b>metric_test:</b> Metric score on the test set.</li>
<li><b>time_fit:</b> Time spent fitting and evaluating.</li>
<li><b>mean_bagging:</b> Mean score of the bagging's results.</li>
<li><b>std_bagging:</b> Standard deviation score of the bagging's results.</li>
<li><b>time_bagging:</b> Time spent on the bagging algorithm.</li>
<li><b>time:</b> Total time spent on the whole run.</li>
</ul>
</blockquote>
</td>
</tr>
</table>
<br>


### Plot attributes
 
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Attributes:</strong></td>
<td width="75%" style="background:white;">
<strong>style: str</strong>
<blockquote>
Plotting style. See seaborn's <a href="https://seaborn.pydata.org/tutorial/aesthetics.html#seaborn-figure-styles">documentation</a>.
</blockquote>
<strong>palette: str</strong>
<blockquote>
Color palette. See seaborn's <a href="https://seaborn.pydata.org/tutorial/color_palettes.html">documentation</a>.
</blockquote>
<strong>title_fontsize: int</strong>
<blockquote>
Fontsize for the plot's title.
</blockquote>
<strong>label_fontsize: int</strong>
<blockquote>
Fontsize for labels and legends.
</blockquote>
<strong>tick_fontsize: int</strong>
<blockquote>
Fontsize for the ticks along the plot's axes.
</blockquote>
</td></tr>
</table>
<br><br><br>



## Methods
---------

<table width="100%">

<tr>
<td><a href="#calibrate">calibrate</a></td>
<td>Calibrate the winning model.</td>
</tr>

<tr>
<td width="15%"><a href="#canvas">canvas</a></td>
<td>Create a figure with multiple plots.</td>
</tr>

<tr>
<td width="15%"><a href="#delete">delete</a></td>
<td>Remove a model from the pipeline.</td>
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
<td width="15%"><a href="#log">log</a></td>
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
<td>Returns the scores of the models for a specific metric.</td>
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


<a name="calibrate"></a>
<pre><em>method</em> <strong style="color:#008AB8">calibrate</strong>(**kwargs)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basepredictor.py#L322">[source]</a></div></pre>
Applies probability calibration on the winning model. The calibration is performed
using sklearn's [CalibratedClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)
class. The model is trained via cross-validation on a subset of the training data,
using the rest to fit the calibrator. The new classifier will replace the `estimator`
attribute. After calibrating, all prediction attributes of the winning model will reset.
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>**kwargs</strong>
<blockquote>
Additional keyword arguments for the CalibratedClassifierCV instance. Using
cv="prefit" will use the trained model and fit the calibrator on the test
set. Note that doing this will result in data leakage in the test set. Use
this only if you have another, independent set for testing.
</blockquote>
</tr>
</table>
<br />


<a name="canvas"></a>
<pre><em>method</em> <strong style="color:#008AB8">canvas</strong>(nrows=1, ncols=2, title=None, figsize=None, filename=None, display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L448">[source]</a></div></pre>
This `@contextmanager` allows you to draw many plots in one figure. The default
option is to add two plots side by side. See the [user guide](../../../user_guide/#canvas)
 for an example use case.
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>nrows: int, optional (default=1)</strong>
<blockquote>
Number of plots in length.
</blockquote>
<strong>ncols: int, optional (default=2)</strong>
<blockquote>
Number of plots in width.
</blockquote>
<strong>title: str or None, optional (default=None)</strong>
<blockquote>
Plot's title. If None, no title is displayed.
</blockquote>
<strong>figsize: tuple or None, optional (default=None)</strong>
<blockquote>
Figure's size, format as (x, y). If None, adapts size to the number of plots
 in the canvas.
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


<a name="delete"></a>
<pre><em>method</em> <strong style="color:#008AB8">delete</strong>(models=None)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basepredictor.py#L375">[source]</a></div></pre>
Removes a model from the pipeline. If all models in the pipeline are removed,
the metric is reset. Use this method to remove unwanted models or to free
some memory before saving the instance.
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>models: str or sequence, optional (default=None)</strong>
<blockquote>
Name of the models to clear from the pipeline. If None, clear all models.
</blockquote>
</tr>
</table>
<br />


<a name="get-class-weight"></a>
<pre><em>method</em> <strong style="color:#008AB8">get_class_weight</strong>(dataset="train")
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basepredictor.py#L291">[source]</a></div></pre>
Return class weights for a balanced data set. Statistically, the class weights
re-balance the data set so that the sampled data set represents the target
population as closely as reasonably possible. The returned weights are inversely
proportional to class frequencies in the selected data set. 
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>dataset: str, optional (default="train")</strong>
<blockquote>
Data set from which to get the weights. Choose between "train", "test" or "dataset".
</blockquote>
</tr>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Returns:</strong></td>
<td width="75%" style="background:white;">
<strong>class_weights: dict</strong>
<blockquote>
Classes with the corresponding weights.
</blockquote>
</tr>
</table>
<br />


<a name="get-params"></a>
<pre><em>method</em> <strong style="color:#008AB8">get_params</strong>(deep=True) 
<div align="right"><a href="https://github.com/scikit-learn/scikit-learn/blob/0fb307bf3/sklearn/base.py#L189">[source]</a></div></pre>
Get parameters for this estimator.
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>deep: bool, default=True</strong>
<blockquote>
If True, will return the parameters for this estimator and contained subobjects that are estimators.
</blockquote>
</tr>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Returns:</strong></td>
<td width="75%" style="background:white;">
<strong>params: dict</strong>
<blockquote>
Dictionary of the parameter names mapped to their values.
</blockquote>
</tr>
</table>
<br />


<a name="log"></a>
<pre><em>method</em> <strong style="color:#008AB8">log</strong>(msg, level=0)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basetransformer.py#L315">[source]</a></div></pre>
Write a message to the logger and print it to stdout.
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>msg: str</strong>
<blockquote>
Message to write to the logger and print to stdout.
</blockquote>
<strong>level: int, optional (default=0)</strong>
<blockquote>
Minimum verbosity level to print the message.
</blockquote>
</tr>
</table>
<br />


<a name="reset-aesthetics"></a>
<pre><em>method</em> <strong style="color:#008AB8">reset_aesthetics</strong>()
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L200">[source]</a></div></pre>
Reset the [plot aesthetics](../../../user_guide/#aesthetics) to their default values.
<br /><br /><br />


<a name="reset-predictions"></a>
<pre><em>method</em> <strong style="color:#008AB8">reset_predictions</strong>()
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basepredictor.py#L118">[source]</a></div></pre>
Clear the [prediction attributes](../../../user_guide/#predicting) from all models.
Use this method to free some memory before saving the trainer.
<br /><br /><br />


<a name="run"></a>
<pre><em>method</em> <strong style="color:#008AB8">run</strong>(*arrays)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/training.py#L50">[source]</a></div></pre>
Fit and evaluate the models.
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>*arrays: sequence of indexables</strong>
<blockquote>
Training set and test set. Allowed input formats are:
<ul>
<li>train, test</li>
<li>X_train, X_test, y_train, y_test</li>
<li>(X_train, y_train), (X_test, y_test)</li>
</ul>
</blockquote>
</td>
</tr>
</table>
<br />


<a name="save"></a>
<pre><em>method</em> <strong style="color:#008AB8">save</strong>(filename=None, save_data=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basetransformer.py#L336">[source]</a></div></pre>
Save the instance to a pickle file. Remember that the class contains the complete
 dataset as attribute, so the file can become large for big datasets! To avoid this,
 use `save_data=False`.
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>filename: str or None, optional (default=None)</strong>
<blockquote>
Name to save the file with. None or "auto" to save with
 the __name__ of the class.
</blockquote>
<strong>save_data: bool, optional (default=True)</strong>
<blockquote>
Whether to save the data as an attribute of the instance. If False, remember to
 add the data to <a href="../../ATOM/atomloader">ATOMLoader</a> when loading the file.
</blockquote>
</tr>
</table>
<br>


<a name="scoring"></a>
<pre><em>method</em> <strong style="color:#008AB8">scoring</strong>(metric=None, dataset="test", **kwargs)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basepredictor.py#L328">[source]</a></div></pre>
Print all the models' scoring for a specific metric.
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>metric: str or None, optional (default=None)</strong>
<blockquote>
Name of the metric to calculate. Choose from any of sklearn's classification <a href="https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules">SCORERS</a>
 or one of the following custom metrics:
<ul>
<li>"cm" for the confusion matrix.</li>
<li>"tn" for true negatives.</li>
<li>"fp" for false positives.</li>
<li>"fn" for false negatives.</li>
<li>"tp" for true positives.</li>
<li>"lift" for the lift metric.</li>
<li>"fpr" for the false positive rate.</li>
<li>"tpr" for true positive rate.</li>
<li>"sup" for the support metric.</li>
</ul>
If None, returns the models' final results (ignores the <code>dataset</code> parameter).
</blockquote>
<strong>dataset: str, optional (default="test")</strong>
<blockquote>
Additional keyword arguments for the metric function.
</blockquote>
</table>
<br />


<a name="set-params"></a>
<pre><em>method</em> <strong style="color:#008AB8">set_params</strong>(**params) 
<div align="right"><a href="https://github.com/scikit-learn/scikit-learn/blob/0fb307bf3/sklearn/base.py#L221">[source]</a></div></pre>
Set the parameters of this estimator.
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>**params: dict</strong>
<blockquote>
Estimator parameters.
</blockquote>
</tr>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Returns:</strong></td>
<td width="75%" style="background:white;">
<strong>self: DirectClassifier</strong>
<blockquote>
Estimator instance.
</blockquote>
</tr>
</table>
<br />


<a name="stacking"></a>
<pre><em>method</em> <strong style="color:#008AB8">stacking</strong>(models=None, estimator=None, stack_method="auto", passthrough=False)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basepredictor#L241">[source]</a></div></pre>
Add a Stacking instance to the models in the pipeline.
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>models: sequence or None, optional (default=None)</strong>
<blockquote>
Models that feed the stacking.
</blockquote>
<strong>estimator: str, callable or None, optional (default=None)</strong>
<blockquote>
The final estimator, which will be used to combine the base
 estimators. If str, choose from ATOM's <a href="../../../user_guide/#predefined-models">predefined models</a>.
 If None, <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html">Logistic Regression</a> is selected.
</blockquote>
<strong>stack_method: str, optional (default="auto")</strong>
<blockquote>
Methods called for each base estimator. If "auto", it will try to 
 invoke <code>predict_proba</code>, <code>decision_function</code>
 or <code>predict</code> in that order.
</blockquote>
<strong>passthrough: bool, optional (default=False)</strong>
<blockquote>
When False, only the predictions of estimators will be used
 as training data for the final estimator. When True, the
 estimator is trained on the predictions as well as the
 original training data.
</blockquote>
</tr>
</table>
<br />


<a name="voting"></a>
<pre><em>method</em> <strong style="color:#008AB8">voting</strong>(models=None, weights=None)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basepredictor#L208">[source]</a></div></pre>
Add a Voting instance to the models in the pipeline.
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>models: sequence or None, optional (default=None)</strong>
<blockquote>
Models that feed the voting.
</blockquote>
<strong>weights: sequence or None, optional (default=None)</strong>
<blockquote>
Sequence of weights (int or float) to weight the
 occurrences of predicted class labels (hard voting)
 or class probabilities before averaging (soft voting).
 Uses uniform weights if None.
</blockquote>
</tr>
</table>
<br />



## Example
---------
```python
from atom.training import DirectClassifier

# Run the pipeline
trainer = DirectClassifier(["Tree", "RF"], n_calls=5, n_initial_points=3)
trainer.run(train, test)

# Analyze the results
trainer.scoring("auc")
trainer.Tree.plot_bo()
```