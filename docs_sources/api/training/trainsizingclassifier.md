# TrainSizingClassifier
-----------------------

<a name="atom"></a>
<pre><em>class</em> atom.training.<strong style="color:#008AB8">TrainSizingClassifier</strong>(models, metric=None, greater_is_better=True, needs_proba=False,
                                          needs_threshold=False, train_sizes=np.linspace(0.2, 1.0, 5),
                                          n_calls=10, n_initial_points=5, est_params={}, bo_params={},
                                          bagging=None, n_jobs=1, verbose=0, logger=None, random_state=None) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/training.py#L347">[source]</a></div></pre>
<div style="padding-left:3%">
Fit and evaluate the models in a [train sizing](../../../user_guide/#train-sizing)
 fashion. The pipeline applies the following steps per iteration:

1. The optimal hyperparameters are selected using a bayesian optimization algorithm.
2. The model is fitted on the training set using the best combinations of hyperparameters found.
3. Using a bagging algorithm, various scores on the test set are calculated.

Just like `atom`, you can [predict](../../../user_guide/#predicting),
 [plot](../../../user_guide/#plots) and call any [`model`](../../../user_guide/#models)
 from the TrainSizingClassifier instance. Read more in the [user guide](../../../user_guide/#training).
<br />
<table>
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>models: str or iterable</strong>
<blockquote>
Models to fit on the data. Use the predefined acronyms to select the models. Possible values are (case insensitive):
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
<strong>metric: str, callable or iterable, optional (default=None)</strong>
<blockquote>
Metric(s) on which the pipeline fits the models. Choose from any of sklearn's predefined
 <a href="https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules">scorers</a>,
 use a score (or loss) function with signature metric(y, y_pred, **kwargs) or use a
 scorer object. If multiple metrics are selected, only the first will be used to
 optimize the BO. If None, a default metric is selected:
<ul>
<li>"f1" for binary classification</li>
<li>"f1_weighted" for multiclass classification</li>
<li>"r2" for regression</li>
</ul>
</blockquote>
<strong>greater_is_better: bool or iterable, optional (default=True)</strong>
<blockquote>
Whether the metric is a score function or a loss function,
 i.e. if True, a higher score is better and if False, lower is
 better. Will be ignored if the metric is a string or a scorer.
 If iterable, the n-th value will apply to the n-th metric in the
 pipeline.
</blockquote>
<strong> needs_proba: bool or iterable, optional (default=False)</strong>
<blockquote>
Whether the metric function requires probability estimates out of a
 classifier. If True, make sure that every model in the pipeline has
 a `predict_proba` method. Will be ignored if the metric is a string
 or a scorer. If iterable, the n-th value will apply to the n-th metric
 in the pipeline.
</blockquote>
<strong> needs_threshold: bool or iterable, optional (default=False)</strong>
<blockquote>
Whether the metric function takes a continuous decision certainty.
 This only works for binary classification using estimators that
 have either a `decision_function` or `predict_proba` method. Will
 be ignored if the metric is a string or a scorer. If iterable, the
 n-th value will apply to the n-th metric in the pipeline.
</blockquote>
<strong>train_sizes: array-like, optional (default=np.linspace(0.2, 1.0, 5))</strong>
<blockquote>
Sequence of training set sizes used to run the trainings.
<ul>
<li>If <=1: Fraction of the training set.</li>
<li>If >1: Total number of samples.</li>
</ul>
</blockquote>
<strong>n_calls: int or iterable, optional (default=0)</strong>
<blockquote>
Maximum number of iterations of the BO (including `n_initial_points`).
 If 0, skip the BO and fit the model on its default parameters.
 If iterable, the n-th value will apply to the n-th model in the pipeline.
</blockquote>
<strong>n_initial_points: int or iterable, optional (default=5)</strong>
<blockquote>
Initial number of random tests of the BO before fitting the
 surrogate function. If equal to `n_calls`, the optimizer will
 technically be performing a random search. If iterable, the n-th
 value will apply to the n-th model in the pipeline.
</blockquote>
<strong>est_params: dict, optional (default={}})</strong>
<blockquote>
Additional parameters for the estimators. See the corresponding
 documentation for the available options. For multiple models, use
 the acronyms as key and a dictionary of the parameters as value.
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
<li><b>max_time: int, optional (default=np.inf)</b><br>Stop the optimization after `max_time` seconds.</li>
<li><b>delta_x: int or float, optional (default=0)</b><br>Stop the optimization when `|x1 - x2| < delta_x`.</li>
<li><b>delta_y: int or float, optional (default=0)</b><br>Stop the optimization if the 5 minima are within `delta_y`.</li>
<li><b>cv: int, optional (default=5)</b><br>Number of folds for the cross-validation. If 1, the
 training set will be randomly split in a subtrain and validation set.</li>
<li><b>early stopping: int, float or None, optional (default=None)</b><br>Training
 will stop if the model didn't improve in last `early_stopping` rounds. If <1,
 fraction of rounds from the total. If None, no early stopping is performed. Only
 available for models that allow in-training evaluation.</li>
<li><b>callback: callable or list of callables, optional (default=None)</b><br>Callbacks for the BO.</li>
<li><b>dimensions: dict, array or None, optional (default=None)</b><br>Custom hyperparameter
 space for the bayesian optimization. Can be an array to share dimensions across
 models or a dictionary with the model's name as key. If None, ATOM's predefined dimensions are used.</li>
<li><b>plot_bo: bool, optional (default=False)</b><br>Whether to plot the BO's progress as it runs.
 Creates a canvas with two plots: the first plot shows the score of every trial
 and the second shows the distance between the last consecutive steps. Don't
 forget to call `%matplotlib` at the start of the cell if you are using an interactive
 notebook!</li>
<li><b>Additional keyword argument for skopt's optimizer.</b></li>                
</ul>
</blockquote>
<strong>bagging: int, iterable or None, optional (default=None)</strong>
<blockquote>
Number of data sets (bootstrapped from the training set) to use in
 the bagging algorithm. If None or 0, no bagging is performed.
 If iterable, the n-th value will apply to the n-th model in the pipeline.
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
<strong>logger: bool, str, class or None, optional (default=None)</strong>
<blockquote>
<ul>
<li>If None: Doesn't save a logging file.</li>
<li>If bool: True for logging file with default name. False for no logger.</li>
<li>If str: Name of the logging file. "auto" to create an automatic name.</li>
<li>If class: python `Logger` object.</li>
</ul>
</blockquote>
<strong>random_state: int or None, optional (default=None)</strong>
<blockquote>
Seed used by the random number generator. If None, the random number
 generator is the `RandomState` instance used by `numpy.random`.
</blockquote>
</td>
</tr>
</table>
</div>
<br><br>



## Attributes
-------------

### Data attributes

The dataset can be accessed at any time through multiple properties, e.g. calling
 `trainer.train` will return the training set. The data can also be changed through
 these properties, e.g. `trainer.test = atom.test.drop(0)` will drop the first row
 from the test set. This will also update the other data attributes.

<a name="atom"></a>
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
Dataset's shape in the form (rows x columns).
</blockquote>
<strong>columns: list</strong>
<blockquote>
List of columns in the dataset.
</blockquote>
<strong>target: str</strong>
<blockquote>
Name of the target column.
</blockquote>
<strong>classes: pd.DataFrame</strong>
<blockquote>
Dataframe of the number of rows per target class in the train, test and complete dataset.
</blockquote>
<strong>n_classes: int</strong>
<blockquote>
Number of unique classes in the target column.
</blockquote>
</td></tr>
</table>
<br>


### Utility attributes

<a name="atom"></a>
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
Dictionary of the encountered exceptions (if any).
</blockquote>
<strong>winner: [`model`](../../../user_guide/#models)</strong>
<blockquote>
Model subclass that performed best on the test set.
</blockquote>
<strong>results: pd.DataFrame</strong>
<blockquote>
Dataframe of the training results with the fraction of the training set and the model
 acronyms as indices. Columns can include:
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
 
<a name="atom"></a>
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
<td width="15%"><a href="#clear">clear</a></td>
<td>Remove a model from the pipeline.</td>
</tr>

<tr>
<td width="15%"><a href="#get-class-weight">get_class_weight</a></td>
<td>Return class weights for a balanced data set.</td>
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

</table>
<br>

<a name="calibrate"></a>
<pre><em>method</em> <strong style="color:#008AB8">calibrate</strong>(\*\*kwargs)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basepredictor.py#L116">[source]</a></div></pre>
<div style="padding-left:3%">
Applies probability calibration on the winning model. The calibration is done with the
 [CalibratedClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)
 class from sklearn. The model will be trained via cross-validation on a subset
 of the training data, using the rest to fit the calibrator. The new classifier will
 replace the `estimator` attribute. After calibrating, all prediction attributes of
 the winning model will reset.
<br /><br />
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>**kwargs</strong>
<blockquote>
Additional keyword arguments for the [CalibratedClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)
 instance. Using cv="prefit" will use the trained model and fit the calibrator on the
 test set. Note that doing this will result in data leakage in the test set. Use this
 only if you have another, independent set for testing.
</blockquote>
</tr>
</table>
</div>
<br />

<a name="clear"></a>
<pre><em>method</em> <strong style="color:#008AB8">clear</strong>(models="all")
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basepredictor.py#L204">[source]</a></div></pre>
<div style="padding-left:3%">
Removes all traces of a model in the pipeline (except for the `errors`
 attribute). If all models in the pipeline are removed, the metric is reset.
 Use this method to remove unwanted models from the pipeline or to clear
 memory before saving the instance. 
<br /><br />
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>models: str or iterable, optional (default="all")</strong>
<blockquote>
Model(s) to clear from the pipeline. If "all", clear all models.
</blockquote>
</tr>
</table>
</div>
<br />


<a name="get-class-weight"></a>
<pre><em>method</em> <strong style="color:#008AB8">get_class_weight</strong>(dataset="train")
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basepredictor.py#L204">[source]</a></div></pre>
<div style="padding-left:3%">
Return class weights for a balanced data set. Statistically, the class weights
 re-balance the data set so that the sampled data set represents the target
 population as closely as reasonably possible. The returned weights are inversely
 proportional to class frequencies in the selected data set. 
<br /><br />
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>dataset: str, optional (default="train")</strong>
<blockquote>
Data set from which to get the weights. Choose between "train", "test" or "dataset".
</blockquote>
</tr>
</table>
</div>
<br />


<a name="get-params"></a>
<pre><em>method</em> <strong style="color:#008AB8">get_params</strong>(deep=True) 
<div align="right"><a href="https://github.com/scikit-learn/scikit-learn/blob/0fb307bf3/sklearn/base.py#L189">[source]</a></div></pre>
<div style="padding-left:3%">
Get parameters for this estimator.
<br><br>
</div>
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
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basetransformer.py#L196">[source]</a></div></pre>
<div style="padding-left:3%">
Write a message to the logger and print it to stdout.
<br /><br />
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
Minimum verbosity level in order to print the message.
</blockquote>
</tr>
</table>
</div>
<br />


<a name="run"></a>
<pre><em>method</em> <strong style="color:#008AB8">run</strong>(\*arrays)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/training.py#L189">[source]</a></div></pre>
<div style="padding-left:3%">
Fit and evaluate the models.
<br><br>
</div>
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
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basetransformer.py#L220">[source]</a></div></pre>
<div style="padding-left:3%">
Save the instance to a pickle file. Remember that the class contains the complete
 dataset as property, so the file can become large for big datasets! To avoid this,
 use `save_data=False`.
<br><br>
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>filename: str or None, optional (default=None)</strong>
<blockquote>
Name to save the file with. If None or "auto", use default name (TrainSizingClassifier).
</blockquote>
<strong>save_data: bool, optional (default=True)</strong>
<blockquote>
Whether to save the data as an attribute of the instance. If False, remember to
 update the data immediately after loading the pickle using the dataset's `@setter`.
</blockquote>
</tr>
</table>
</div>
<br>

<a name="scoring"></a>
<pre><em>method</em> <strong style="color:#008AB8">scoring</strong>(metric=None, dataset="test")
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basepredictor#L152">[source]</a></div></pre>
<div style="padding-left:3%">
Returns the scores of the models for a specific metric. If a model
 returns `XXX`, it means the metric failed for that specific model. This
 can happen if either the metric is unavailable for the task or if the
 model does not have a `predict_proba` method while the metric requires it.
<br /><br />
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>metric: str or None, optional (default=None)</strong>
<blockquote>
Name of the metric to calculate. Choose from any of sklearn's [SCORERS](https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules)
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
If None, returns the models" final results (ignores the `dataset` parameter).
</blockquote>
<strong>dataset: str, optional (default="test")</strong>
<blockquote>
Data set on which to calculate the metric. Options are "train" or "test".
</blockquote>
</tr>
</table>
</div>
<br />

<a name="set-params"></a>
<pre><em>method</em> <strong style="color:#008AB8">set_params</strong>(**params) 
<div align="right"><a href="https://github.com/scikit-learn/scikit-learn/blob/0fb307bf3/sklearn/base.py#L221">[source]</a></div></pre>
<div style="padding-left:3%">
Set the parameters of this estimator.
<br><br>
</div>
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>\*\*params: dict</strong>
<blockquote>
Estimator parameters.
</blockquote>
</tr>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Returns:</strong></td>
<td width="75%" style="background:white;">
<strong>self: TrainSizingClassifier</strong>
<blockquote>
Estimator instance.
</blockquote>
</tr>
</table>
<br />



## Example
---------
```python
from atom.training import TrainSizingClassifier

# Run the pipeline
trainer = TrainSizingClassifier("RF", n_calls=5, n_initial_points=3)
trainer.run(train, test)

# Analyze the results
trainer.plot_learning_curve()
```