# Trainerclassifier
------------------

<pre><em>class</em> atom.training.<strong style="color:#008AB8">Trainerclassifier</strong>(models, metric=None, greater_is_better=True, needs_proba=False,
                                      needs_threshold=False, n_calls=10, n_random_points=5, bo_kwargs={},
                                      bagging=None, n_jobs=1, verbose=0, logger=None, random_state=None) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L2155">[source]</a></div></pre>

The Trainerclassifier fits and evaluates the models. The pipeline applies the following steps:

1. The optimal hyperparameters are selected using a Bayesian Optimization (BO) algorithm.
2. The model is fitted on the complete training set using the best combinations of hyperparameters found.
3. Using a bagging algorithm, various scores on the test set are calculated.

Just like an ATOM instance, you can call the [prediction methods](../../ATOM/Trainerclassifier/#prediction-methods),
 [plots](../../ATOM/Trainerclassifier/#plots) and
 [model subclasses](../../ATOM/Trainerclassifier/#model-subclasses) from the instance.
 Read more in the [user guide](../../../user_guide/#model-fitting-and-evaluation).

<table>
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>models: str or sequence</strong>
<blockquote>
List of models to fit on the data. Use the predefined acronyms to select the models. Possible values are (case insensitive):
<ul>
<li>'GNB' for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html">Gaussian Naive Bayes</a> (no hyperparameter tuning)</li>
<li>'MNB' for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html">Multinomial Naive Bayes</a></li>
<li>'BNB' for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html">Bernoulli Naive Bayes</a></li>
<li>'GP' for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html">Gaussian Process</a> (no hyperparameter tuning)</li>
<li>'Ridge' for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html">Ridge Linear Classification</a></li>
<li>'LR' for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html">Logistic Regression</a></li> 
<li>'LDA' for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html">Linear Discriminant Analysis</a></li>
<li>'QDA' for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html">Quadratic Discriminant Analysis</a></li>
<li>'KNN' for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html"> K-Nearest Neighbors</a></li>
<li>'Tree' for a single <a href="https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html">Decision Tree</a></li>
<li>'Bag' for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html">Bagging</a> (uses a decision tree as base estimator)</li>
<li>'ET' for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html">Extra-Trees</a></li>
<li>'RF' for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html">Random Forest</a></li>
<li>'AdaB' for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html">AdaBoost</a> (uses a decision tree as base estimator)</li>
<li>'GBM' for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html">Gradient Boosting Machine</a></li>
<li>'XGB' for <a href="https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier">XGBoost</a> (only available if package is installed)</li>
<li>'LGB' for <a href="https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html">LightGBM</a> (only available if package is installed)</li>
<li>'CatB' for <a href="https://catboost.ai/docs/concepts/python-reference_catboostclassifier.html">CatBoost</a> (only available if package is installed)</li>
<li>'lSVM' for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html">Linear Support Vector Machine</a> (uses a one-vs-rest strategy for multiclass classification)</li> 
<li>'kSVM' for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">Kernel Support Vector Machine</a> (uses a one-vs-one strategy for multiclass classification)</li>
<li>'PA' for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html">Passive Aggressive</a></li>
<li>'SGD' for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html">Stochastic Gradient Descent</a></li>
<li>'MLP' for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html">Multilayer Perceptron</a> (can have between one and three hidden layers)</li> 
</ul>
</blockquote>
<strong>metric: str, callable or sequence, optional (default=None)</strong>
<blockquote>
Metric(s) on which the pipeline fits the models. Choose from any of
the <a href="https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules">scorers</a> predefined by sklearn, use a score (or loss)
function with signature metric(y, y_pred, **kwargs) or use a
scorer object. If multiple metrics are selected, only the first will
be used to optimize the BO. If None, a default metric is selected:
<ul>
<li>'f1' for binary classification</li>
<li>'f1_weighted' for multiclass classification</li>
<li>'r2' for regression</li>
</ul>
</blockquote>
<strong>greater_is_better: bool or sequence, optional (default=True)</strong>
<blockquote>
Whether the metric is a score function or a loss function,
 i.e. if True, a higher score is better and if False, lower is
 better. Will be ignored if the metric is a string or a scorer.
 If sequence, the n-th value will apply to the n-th metric in the
 pipeline.
</blockquote>
<strong> needs_proba: bool or sequence, optional (default=False)</strong>
<blockquote>
Whether the metric function requires probability estimates out of a
 classifier. If True, make sure that every model in the pipeline has
 a `predict_proba` method. Will be ignored if the metric is a string
 or a scorer. If sequence, the n-th value will apply to the n-th metric
 in the pipeline.
</blockquote>
<strong> needs_threshold: bool or sequence, optional (default=False)</strong>
<blockquote>
Whether the metric function takes a continuous decision certainty.
 This only works for binary classification using estimators that
 have either a `decision_function` or `predict_proba` method. Will
 be ignored if the metric is a string or a scorer. If sequence, the
 n-th value will apply to the n-th metric in the pipeline.
</blockquote>
<strong>n_calls: int or sequence, optional (default=0)</strong>
<blockquote>
Maximum number of iterations of the BO (including `random starts`).
 If 0, skip the BO and fit the model on its default Parameters.
 If sequence, the n-th value will apply to the n-th model in the pipeline.
</blockquote>
<strong>n_random_starts: int or sequence, optional (default=5)</strong>
<blockquote>
Initial number of random tests of the BO before fitting the
 surrogate function. If equal to `n_calls`, the optimizer will
 technically be performing a random search. If sequence, the n-th
 value will apply to the n-th model in the pipeline.
</blockquote>
<strong>bo_kwargs: dict, optional (default={})</strong>
<blockquote>
Dictionary of extra keyword arguments for the BO. These can include:
<ul>
<li><b>base_estimator: str</b><br>Base estimator to use in the BO.
 Choose from: 'GP', 'RF', 'ET' or 'GBRT'.</li>
<li><b>max_time: int</b><br>Maximum allowed time for the BO (in seconds).</li>
<li><b>delta_x: int or float</b><br>Maximum distance between two consecutive points.</li>
<li><b>delta_x: int or float</b><br>Maximum score between two consecutive points.</li>
<li><b>cv: int</b><br>Number of folds for the cross-validation. If 1, the
 training set will be randomly split in a subtrain and validation set.</li>
<li><b>callback: callable or list</b><br>Callbacks for the BO.</li>
<li><b>dimensions: dict or array</b><br>Custom hyperparameter space for the
 bayesian optimization. Can be an array (only if there is 1 model in the
 pipeline) or a dictionary with the model's name as key.</li>
<li><b>plot_bo: bool</b><br>Whether to plot the BO's progress as it runs.
 Creates a canvas with two plots: the first plot shows the score of every trial
 and the second shows the distance between the last consecutive steps. Don't
 forget to call `%matplotlib` at the start of the cell if you are using jupyter
 notebook!</li>
<li><b>Any other parameter for the bayesian optimization function.</b></li>                
</ul>
</blockquote>
<strong>bagging: int or None, optional (default=None)</strong>
<blockquote>
Number of data sets (bootstrapped from the training set) to use in
 the bagging algorithm. If None or 0, no bagging is performed.
 If sequence, the n-th value will apply to the n-th model in the pipeline.
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
<li>If bool: True for logging file with default name, False for no logger.</li>
<li>If str: Name of the logging file. 'auto' to create an automatic name.</li>
<li>If class: python Logger object.</li>
</ul>
</blockquote>
<strong>random_state: int or None, optional (default=None)</strong>
<blockquote>
Seed used by the random number generator. If None, the random number
 generator is the RandomState instance used by np.random.
</blockquote>
</td>
</tr>
</table>
<br>



## Attributes
-------------

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
Metric(s) used to fit the models in the pipeline.
</blockquote>

<strong>errors: dict</strong>
<blockquote>
Dictionary of the encountered exceptions (if any).
</blockquote>

<strong>winner: model subclass</strong>
<blockquote>
Model subclass that performed best on the test set. If multi-metric, only the first
 metric is checked.
</blockquote>

<strong>results: pd.DataFrame</strong>
<blockquote>
Dataframe of the training results with the model acronyms as index. Columns can include:
<ul>
<li><b>name:</b> Name of the model.</li>
<li><b>score_bo:</b> Best score achieved during the BO.</li>
<li><b>time_bo:</b> Time spent on the BO.</li>
<li><b>score_train:</b> Score on the training set.</li>
<li><b>score_test:</b> Score on the test set.</li>
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



## Methods
---------

<table width="100%">

<tr>
<td><a href="#Trainerclassifier-calibrate">calibrate</a></td>
<td>Calibrate the winning model.</td>
</tr>

<tr>
<td width="15%"><a href="#Trainerclassifier-clear">clear</a></td>
<td>Remove a model from the pipeline.</td>
</tr>

<tr>
<td><a href="#Trainerclassifier-get-params">get_params</a></td>
<td>Get parameters for this estimator.</td>
</tr>

<tr>
<td width="15%"><a href="#Trainerclassifier-log">log</a></td>
<td>Save information to the logger and print to stdout.</td>
</tr>

<tr>
<td><a href="#Trainerclassifier-run">run</a></td>
<td>Fit and evaluate the models.</td>
</tr>

<tr>
<td><a href="#Trainerclassifier-save">save</a></td>
<td>Save the instance to a pickle file.</td>
</tr>

<tr>
<td><a href="#Trainerclassifier-scoring">scoring</a></td>
<td>Print the scoring of the models for a specific metric.</td>
</tr>

<tr>
<td><a href="#Trainerclassifier-set-params">set_params</a></td>
<td>Set the parameters of this estimator.</td>
</tr>

</table>
<br>

<a name="Trainerclassifier-calibrate"></a>
<pre><em>function</em> Trainerclassifier.<strong style="color:#008AB8">calibrate</strong>(\*\*kwargs)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L616">[source]</a></div></pre>
<div style="padding-left:3%">
Applies probability calibration on the winning model. The calibration is done with the
 [CalibratedClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)
 class from sklearn. The model will be trained via cross-validation on a subset
 of the training data, using the rest to fit the calibrator. The new classifier will
 replace the `model` attribute.
<br /><br />
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>**kwargs</strong>
<blockquote>
Additional keyword arguments for the [CalibratedClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)
 instance. Using cv='prefit' will use the trained model and fit the calibrator on the
 test set. Note that doing this will result in data leakage in the test set. Use this
 only if you have another, independent set for testing.
</blockquote>
</tr>
</table>
</div>
<br />

<a name="Trainerclassifier-clear"></a>
<pre><em>function</em> Trainerclassifier.<strong style="color:#008AB8">clear</strong>(models='all')
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L570">[source]</a></div></pre>
<div style="padding-left:3%">
Removes all traces of a model in the pipeline (except for the errors attribute).
 This includes the models and results attributes, and the model subclass.
 If all models in the pipeline are removed, the metric is reset. Use this method 
 to remove unwanted models from the pipeline or to clear memory before saving. 
<br /><br />
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>models: str, or sequence, optional (default='all')</strong>
<blockquote>
Name of the models to clear from the pipeline. If 'all', clear all models.
</blockquote>
</tr>
</table>
</div>
<br />


<a name="Trainerclassifier-get-params"></a>
<pre><em>function</em> Trainerclassifier.<strong style="color:#008AB8">get_params</strong>(deep=True) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L2155">[source]</a></div></pre>
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


<a name="Trainerclassifier-log"></a>
<pre><em>function</em> Trainerclassifier.<strong style="color:#008AB8">log</strong>(msg, level=0)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L696">[source]</a></div></pre>
<div style="padding-left:3%">
Save information to the ATOM logger and print it to stdout.
<br /><br />
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>msg: str</strong>
<blockquote>
Message to save to the logger and print to stdout.
</blockquote>
<strong>level: int, optional (default=0)</strong>
<blockquote>
Minimum verbosity level in order to print the message.
</blockquote>
</tr>
</table>
</div>
<br />


<a name="Trainerclassifier-run"></a>
<pre><em>function</em> Trainerclassifier.<strong style="color:#008AB8">run</strong>(\*arrays)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L2155">[source]</a></div></pre>
<div style="padding-left:3%">
Fit and evaluate the models.
<br><br>
</div>
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>*arrays: array-like</strong>
<blockquote>
Either a train and test set or X_train, X_test, y_train, y_test.
</blockquote>
</td>
</tr>
</table>
<br />


<a name="Trainerclassifier-save"></a>
<pre><em>function</em> Trainerclassifier.<strong style="color:#008AB8">save</strong>(filename=None, save_data=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L696">[source]</a></div></pre>
<div style="padding-left:3%">
Save the instance to a pickle file. Remember that the class contains the complete
 dataset as property, so the file can become large for big datasets! To avoid this,
 use `save_data=False`.

<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>filename: str or None, optional (default=None)</strong>
<blockquote>
Name to save the file with. If None or 'auto', use default name (Trainerclassifier).
</blockquote>
<strong>save_data: bool, optional (default=True)</strong>
<blockquote>
Whether to save the data as an attribute of the instance. If False, remember to
 update the data immediately after loading the pickle using the dataset's @setter.
</blockquote>
</tr>
</table>
</div>
<br>

<a name="Trainerclassifier-scoring"></a>
<pre><em>function</em> Trainerclassifier.<strong style="color:#008AB8">scoring</strong>(metric=None)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L616">[source]</a></div></pre>
<div style="padding-left:3%">
Print the scoring of the models for a specific metric. If a model
 shows a `XXX`, it means the metric failed for that specific model. This
 can happen if either the metric is unavailable for classification tasks or if the
 model does not have a `predict_proba` method while the metric requires it.
<br /><br />
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>metric: str or None, optional (default=None)</strong>
<blockquote>
Name of the metric to calculate. If None, returns the metric used to fit the pipeline.
If string, choose from any of sklearn's [SCORERS](https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules)
 or one of the following custom metrics:
<ul>
<li>'cm' or 'confusion_matrix' for an array of the confusion matrix.</li>
<li>'tn' for true negatives.</li>
<li>'fp' for false positives.</li>
<li>'fn' for false negatives.</li>
<li>'lift' for the lift metric.</li>
<li>'fpr' for the false positive rate.</li>
<li>'tpr' for true positive rate.</li>
<li>'sup' for the support metric.</li>
</ul>
</blockquote>
</tr>
</table>
</div>
<br />

<a name="Trainerclassifier-set-params"></a>
<pre><em>function</em> Trainerclassifier.<strong style="color:#008AB8">set_params</strong>(**params) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L2155">[source]</a></div></pre>
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
<strong>self: Trainerclassifier</strong>
<blockquote>
Estimator instance.
</blockquote>
</tr>
</table>
<br />



## Example
---------
```python
from atom.training import Trainerclassifier

# Run the pipeline
trainer = Trainerclassifier(['Tree', 'RF'], n_calls=5, n_random_starts=3)
trainer.run(train, test)

# Analyze the results
trainer.scoring('auc')
trainer.Tree.plot_bo()
```