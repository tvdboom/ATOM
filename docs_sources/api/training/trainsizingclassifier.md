
## Pipeline

The pipeline method is where the models are fitted to the data and their
 performance is evaluated according to the selected metric. For every model, the
 pipeline applies the following steps:

1. The optimal hyperparameters are selected using a Bayesian Optimization (BO)
 algorithm with gaussian process as kernel. The resulting score of each step of
 the BO is either computed by cross-validation on the complete training set or
 by randomly splitting the training set every iteration into a (sub) training
 set and a validation set. This process can create some data leakage but
 ensures a maximal use of the provided data. The test set, however, does not
 contain any leakage and will be used to determine the final score of every model.
 Note that, if the dataset is relatively small, the best score on the BO can
 consistently be lower than the final score on the test set (despite the
 leakage) due to the considerable fewer instances on which it is trained.
<div></div>
 
2. Once the best hyperparameters are found, the model is trained again, now
 using the complete training set. After this, predictions are made on the test set.
<div></div>

3. You can choose to evaluate the robustness of each model's applying a bagging
 algorithm, i.e. the model will be trained multiple times on a bootstrapped
 training set, returning a distribution of its performance on the test set.

A couple of things to take into account:

* The metric implementation follows [sklearn's API](https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values).
  This means that the implementation always tries to maximize the scorer, i.e.
  loss functions will be made negative.
* If an exception is encountered while fitting a model, the
  pipeline will automatically jump to the next model and save the
  exception in the `errors` attribute.
* When showing the final results, a `!!` indicates the highest
  score and a `~` indicates that the model is possibly overfitting
  (training set has a score at least 20% higher than the test set).
* The winning model subclass will be attached to the `winner` attribute.
</br>


There are three methods to call for the pipeline.

* The `pipeline` method fits the models directly to the dataset.
 
* If you want to compare similar models, you can use the `successive_halving`
 method when running the pipeline. This technique fits N models to
 1/N of the data. The best half are selected to go to the next iteration where
 the process is repeated. This continues until only one model remains, which is
 fitted on the complete dataset. Beware that a model's performance can depend
 greatly on the amount of data on which it is trained. For this reason we
 recommend only to use this technique with similar models, e.g. only using
 tree-based models.

* The `train_sizing` method fits the models on subsets of the training data.
 This can be used to examine the optimum size of the dataset needed for a
 satisfying performance.
</br>

<table width="100%">
<tr>
<td><a href="#atom-pipeline">pipeline</a></td>
<td>Fit the models to the data in a direct fashion.</td>
</tr>

<tr>
<td><a href="#atom-successive-halving">successive_halving</a></td>
<td>Fit the models to the data in a successive halving fashion.</td>
</tr>

<tr>
<td><a href="#atom-train-sizing">train_sizing</a></td>
<td>Fit the models to the data in a train sizing fashion.</td>
</tr>

</table>


<a name="atom-pipeline"></a>
<pre><em>function</em> atom.training.<strong style="color:#008AB8">pipeline</strong>(models,
                            metric=None,
                            greater_is_better=True,
                            needs_proba=False,
                            needs_threshold=False,
                            n_calls=10,
                            n_random_points=5,
                            bo_kwargs={},
                            bagging=None) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L2155">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
<br /><br />
<table width="100%">
<tr>
<td width="13%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="73%" style="background:white;">
<strong>models: string or sequence</strong>
<blockquote>
List of models to fit on the data. Use the predefined acronyms to select the models. Possible values are (case insensitive):
<ul>
<li>'GNB' for [Gaussian Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)<br>Only for classification tasks. No hyperparameter tuning.</li>
<li>'MNB' for [Multinomial Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)<br>Only for classification tasks.</li>
<li>'BNB' for [Bernoulli Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html)<br>Only for classification tasks.</li>
<li>'GP' for Gaussian Process [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html)<br>No hyperparameter tuning.</li>
<li>'OLS' for [Ordinary Least Squares](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)<br>Only for regression tasks. No hyperparameter tuning.</li>
<li>'Ridge' for Ridge Linear [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)<br>Only for regression tasks.</li>
<li>'Lasso' for [Lasso Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)<br>Only for regression tasks.</li>
<li>'EN' for [ElasticNet Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)<br>Only for regression tasks.</li>
<li>'BR' for [Bayesian Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html)<br>Only for regression tasks. Uses ridge regularization.</li>
<li>'LR' for [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)<br>Only for classification tasks.</li> 
<li>'LDA' for [Linear Discriminant Analysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)<br>Only for classification tasks.</li>
<li>'QDA' for [Quadratic Discriminant Analysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html)<br>Only for classification tasks.</li>
<li>'KNN' for K-Nearest Neighbors [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)</li>
<li>'Tree' for a single Decision Tree [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)</li>
<li>'Bag' for Bagging [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html)<br>Uses a decision tree as base estimator.</li>
<li>'ET' for Extra-Trees [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html)</li>
<li>'RF' for Random Forest [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)</li>
<li>'AdaB' for AdaBoost [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html)<br>Uses a decision tree as base estimator.</li>
<li>'GBM' for Gradient Boosting Machine [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)</li> 
<li>'XGB' for XGBoost [classifier](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier)/[regressor](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor)<br>Only available if package is installed.</li>
<li>'LGB' for LightGBM [classifier](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html)/[regressor](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html)<br>Only available if package is installed.</li>
<li>'CatB' for CatBoost [classifier](https://catboost.ai/docs/concepts/python-reference_catboostclassifier.html)/[regressor](https://catboost.ai/docs/concepts/python-reference_catboostregressor.html)<br>Only available if package is installed.</li>
<li>'lSVM' for Linear Support Vector Machine [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html)<br>Uses a one-vs-rest strategy for multiclass classification tasks.</li> 
<li>'kSVM' for Kernel (non-linear) Support Vector Machine [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)<br>Uses a one-vs-one strategy for multiclass classification tasks.</li>
<li>'PA' for Passive Aggressive [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveRegressor.html)</li>
<li>'SGD' for Stochastic Gradient Descent [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html)</li>
<li>'MLP' for Multilayer Perceptron [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor)<br>Can have between one and three hidden layers.</li> 
</ul>
</blockquote>
<strong>metric: string or callable, optional (default=None)</strong>
<blockquote>
Metric on which the pipeline fits the models. Choose from any of
sklearn's predefined [scorers](https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules), use a score (or loss)
function with signature metric(y, y_pred, **kwargs) or use a
scorer object. If None, ATOM will try to use any metric it already has in the
pipeline. If it hasn't got any, a default metric per task is selected:
<ul>
<li>'f1' for binary classification</li>
<li>'f1_weighted' for multiclass classification</li>
<li>'r2' for regression</li>
</ul>
</blockquote>
<strong>greater_is_better: bool, optional (default=True)</strong>
<blockquote>
Whether the metric is a score function or a loss function,
i.e. if True, a higher score is better and if False, lower is
better. Will be ignored if the metric is a string or a scorer.
</blockquote>
<strong> needs_proba: bool, optional (default=False)</strong>
<blockquote>
Whether the metric function requires probability estimates out of a
 classifier. If True, make sure that every model in the pipeline has
 a `predict_proba` method! Will be ignored if the metric is a string
 or a scorer.
</blockquote>
<strong> needs_threshold: bool, optional (default=False)</strong>
<blockquote>
Whether the metric function takes a continuous decision certainty. This only
 works for binary classification using estimators that have either a
 `decision_function` or `predict_proba` method. Will be ignored if the metric
 is a string or a scorer.
</blockquote>
<strong>n_calls: int or sequence, optional (default=0)</strong>
<blockquote>
Maximum number of iterations of the BO (including `n_random starts`).
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
<li><b>max_time: int</b></br>Maximum allowed time for the BO (in seconds).</li>
<li><b>delta_x: int or float</b></br>Maximum distance between two consecutive points.</li>
<li><b>delta_x: int or float</b></br>Maximum score between two consecutive points.</li>
<li><b>cv: int</b></br>Number of folds for the cross-validation. If 1, the
 training set will be randomly split in a subtrain and validation set.</li>
<li><b>callback: callable or list of callables</b></br>Callbacks for the BO.</li>
<li><b>dimensions: dict or array</b></br>Custom hyperparameter space for the
 bayesian optimization. Can be an array (only if there is 1 model in the
 pipeline) or a dictionary with the model's name as key.</li>
<li><b>plot_bo: bool</b></br>Whether to plot the BO's progress as it runs.
 Creates a canvas with two plots: the first plot shows the score of every trial
 and the second shows the distance between the last consecutive steps. Don't
 forget to call `%matplotlib` at the start of the cell if you are using jupyter
 notebook!</li>
<li><b>Any other parameter for the <a href="https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html">bayesian optimization function</a>.</b></li>                
</ul>
</blockquote>
<strong>bagging: int or None, optional (default=None)</strong>
<blockquote>
Number of data sets (bootstrapped from the training set) to use in the bagging
 algorithm. If None or 0, no bagging is performed.
</blockquote>
</tr>
</table>
</div>
<br />


<a name="atom-successive-halving"></a>
<pre><em>function</em> atom.training.<strong style="color:#008AB8">successive_halving</strong>(models,
                                      metric=None,
                                      greater_is_better=True,
                                      needs_proba=False,
                                      needs_threshold=False,
                                      skip_iter=0,
                                      n_calls=0,
                                      n_random_starts=5,
                                      bo_kwargs={},
                                      bagging=None) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L2172">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
<br /><br />
<table width="100%">
<tr>
<td width="13%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="73%" style="background:white;">
<strong>models: string or sequence</strong>
<blockquote>
List of models to fit on the data. Use the predefined acronyms to select the models. Possible values are (case insensitive):
<ul>
<li>'GNB' for [Gaussian Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)<br>Only for classification tasks. No hyperparameter tuning.</li>
<li>'MNB' for [Multinomial Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)<br>Only for classification tasks.</li>
<li>'BNB' for [Bernoulli Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html)<br>Only for classification tasks.</li>
<li>'GP' for Gaussian Process [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html)<br>No hyperparameter tuning.</li>
<li>'OLS' for [Ordinary Least Squares](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)<br>Only for regression tasks. No hyperparameter tuning.</li>
<li>'Ridge' for Ridge Linear [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)<br>Only for regression tasks.</li>
<li>'Lasso' for [Lasso Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)<br>Only for regression tasks.</li>
<li>'EN' for [ElasticNet Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)<br>Only for regression tasks.</li>
<li>'BR' for [Bayesian Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html)<br>Only for regression tasks. Uses ridge regularization.</li>
<li>'LR' for [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)<br>Only for classification tasks.</li> 
<li>'LDA' for [Linear Discriminant Analysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)<br>Only for classification tasks.</li>
<li>'QDA' for [Quadratic Discriminant Analysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html)<br>Only for classification tasks.</li>
<li>'KNN' for K-Nearest Neighbors [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)</li>
<li>'Tree' for a single Decision Tree [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)</li>
<li>'Bag' for Bagging [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html)<br>Uses a decision tree as base estimator.</li>
<li>'ET' for Extra-Trees [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html)</li>
<li>'RF' for Random Forest [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)</li>
<li>'AdaB' for AdaBoost [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html)<br>Uses a decision tree as base estimator.</li>
<li>'GBM' for Gradient Boosting Machine [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)</li> 
<li>'XGB' for XGBoost [classifier](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier)/[regressor](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor)<br>Only available if package is installed.</li>
<li>'LGB' for LightGBM [classifier](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html)/[regressor](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html)<br>Only available if package is installed.</li>
<li>'CatB' for CatBoost [classifier](https://catboost.ai/docs/concepts/python-reference_catboostclassifier.html)/[regressor](https://catboost.ai/docs/concepts/python-reference_catboostregressor.html)<br>Only available if package is installed.</li>
<li>'lSVM' for Linear Support Vector Machine [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html)<br>Uses a one-vs-rest strategy for multiclass classification tasks.</li> 
<li>'kSVM' for Kernel (non-linear) Support Vector Machine [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)<br>Uses a one-vs-one strategy for multiclass classification tasks.</li>
<li>'PA' for Passive Aggressive [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveRegressor.html)</li>
<li>'SGD' for Stochastic Gradient Descent [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html)</li>
<li>'MLP' for Multilayer Perceptron [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor)<br>Can have between one and three hidden layers.</li> 
</ul>
</blockquote>
<strong>metric: string or callable, optional (default=None)</strong>
<blockquote>
Metric on which the pipeline fits the models. Choose from any of
sklearn's predefined [scorers](https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules), use a score (or loss)
function with signature metric(y, y_pred, **kwargs) or use a
scorer object. If None, ATOM will try to use any metric it already has in the
pipeline. If it hasn't got any, a default metric per task is selected:
<ul>
<li>'f1' for binary classification</li>
<li>'f1_weighted' for multiclas classification</li>
<li>'r2' for regression</li>
</ul>
</blockquote>
<strong>greater_is_better: bool, optional (default=True)</strong>
<blockquote>
Whether the metric is a score function or a loss function,
i.e. if True, a higher score is better and if False, lower is
better. Will be ignored if the metric is a string or a scorer.
</blockquote>
<strong> needs_proba: bool, optional (default=False)</strong>
<blockquote>
Whether the metric function requires probability estimates out of a
 classifier. If True, make sure that every model in the pipeline has
 a `predict_proba` method! Will be ignored if the metric is a string
 or a scorer.
</blockquote>
<strong> needs_threshold: bool, optional (default=False)</strong>
<blockquote>
Whether the metric function takes a continuous decision certainty. This only
 works for binary classification using estimators that have either a
 `decision_function` or `predict_proba` method. Will be ignored if the metric
 is a string or a scorer.
</blockquote>
<strong>skip_iter: int, optional (default=0)</strong>
<blockquote>
Skip last `skip_iter` iterations of the successive halving.
</blockquote>
<strong>n_calls: int or sequence, optional (default=0)</strong>
<blockquote>
Maximum number of iterations of the BO (including `n_random starts`).
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
<li><b>max_time: int</b></br>Maximum allowed time for the BO (in seconds).</li>
<li><b>delta_x: int or float</b></br>Maximum distance between two consecutive points.</li>
<li><b>delta_x: int or float</b></br>Maximum score between two consecutive points.</li>
<li><b>cv: int</b></br>Number of folds for the cross-validation. If 1, the
 training set will be randomly split in a subtrain and validation set.</li>
<li><b>callback: callable or list of callables</b></br>Callbacks for the BO.</li>
<li><b>dimensions: dict or array</b></br>Custom hyperparameter space for the
 bayesian optimization. Can be an array (only if there is 1 model in the
 pipeline) or a dictionary with the model's name as key.</li>
<li><b>plot_bo: bool</b></br>Whether to plot the BO's progress as it runs.
 Creates a canvas with two plots: the first plot shows the score of every trial
 and the second shows the distance between the last consecutive steps. Don't
 forget to call `%matplotlib` at the start of the cell if you are using jupyter
 notebook!</li>
<li><b>Any other parameter for the <a href="https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html">bayesian optimization function</a>.</b></li>                
</ul>
</blockquote>
<strong>bagging: int or None, optional (default=None)</strong>
<blockquote>
Number of data sets (bootstrapped from the training set) to use in the bagging
 algorithm. If None or 0, no bagging is performed.
</blockquote>
</tr>
</table>
</div>
<br />


<a name="atom-train-sizing"></a>
<pre><em>function</em> atom.training.<strong style="color:#008AB8">train_sizing</strong>(models,
                                metric=None,
                                greater_is_better=True,
                                needs_proba=False,
                                needs_threshold=False,
                                train_sizes=np.linspcae(0.2, 1.0, 5),
                                n_calls=0,
                                n_random_starts=5,
                                bo_kwargs={},
                                bagging=None) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L2201">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
<br /><br />
<table width="100%">
<tr>
<td width="13%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="73%" style="background:white;">
<strong>models: string or sequence</strong>
<blockquote>
List of models to fit on the data. Use the predefined acronyms to select the models. Possible values are (case insensitive):
<ul>
<li>'GNB' for [Gaussian Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)<br>Only for classification tasks. No hyperparameter tuning.</li>
<li>'MNB' for [Multinomial Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)<br>Only for classification tasks.</li>
<li>'BNB' for [Bernoulli Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html)<br>Only for classification tasks.</li>
<li>'GP' for Gaussian Process [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html)<br>No hyperparameter tuning.</li>
<li>'OLS' for [Ordinary Least Squares](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)<br>Only for regression tasks. No hyperparameter tuning.</li>
<li>'Ridge' for Ridge Linear [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)<br>Only for regression tasks.</li>
<li>'Lasso' for [Lasso Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)<br>Only for regression tasks.</li>
<li>'EN' for [ElasticNet Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)<br>Only for regression tasks.</li>
<li>'BR' for [Bayesian Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html)<br>Only for regression tasks. Uses ridge regularization.</li>
<li>'LR' for [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)<br>Only for classification tasks.</li> 
<li>'LDA' for [Linear Discriminant Analysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)<br>Only for classification tasks.</li>
<li>'QDA' for [Quadratic Discriminant Analysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html)<br>Only for classification tasks.</li>
<li>'KNN' for K-Nearest Neighbors [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)</li>
<li>'Tree' for a single Decision Tree [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)</li>
<li>'Bag' for Bagging [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html)<br>Uses a decision tree as base estimator.</li>
<li>'ET' for Extra-Trees [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html)</li>
<li>'RF' for Random Forest [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)</li>
<li>'AdaB' for AdaBoost [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html)<br>Uses a decision tree as base estimator.</li>
<li>'GBM' for Gradient Boosting Machine [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)</li> 
<li>'XGB' for XGBoost [classifier](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier)/[regressor](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor)<br>Only available if package is installed.</li>
<li>'LGB' for LightGBM [classifier](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html)/[regressor](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html)<br>Only available if package is installed.</li>
<li>'CatB' for CatBoost [classifier](https://catboost.ai/docs/concepts/python-reference_catboostclassifier.html)/[regressor](https://catboost.ai/docs/concepts/python-reference_catboostregressor.html)<br>Only available if package is installed.</li>
<li>'lSVM' for Linear Support Vector Machine [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html)<br>Uses a one-vs-rest strategy for multiclass classification tasks.</li> 
<li>'kSVM' for Kernel (non-linear) Support Vector Machine [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)<br>Uses a one-vs-one strategy for multiclass classification tasks.</li>
<li>'PA' for Passive Aggressive [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveRegressor.html)</li>
<li>'SGD' for Stochastic Gradient Descent [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html)</li>
<li>'MLP' for Multilayer Perceptron [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor)<br>Can have between one and three hidden layers.</li> 
</ul>
</blockquote>
<strong>metric: string or callable, optional (default=None)</strong>
<blockquote>
Metric on which the pipeline fits the models. Choose from any of
sklearn's predefined [scorers](https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules), use a score (or loss)
function with signature metric(y, y_pred, **kwargs) or use a
scorer object. If None, ATOM will try to use any metric it already has in the
pipeline. If it hasn't got any, a default metric per task is selected:
<ul>
<li>'f1' for binary classification</li>
<li>'f1_weighted' for multiclas classification</li>
<li>'r2' for regression</li>
</ul>
</blockquote>
<strong>greater_is_better: bool, optional (default=True)</strong>
<blockquote>
Whether the metric is a score function or a loss function,
i.e. if True, a higher score is better and if False, lower is
better. Will be ignored if the metric is a string or a scorer.
</blockquote>
<strong> needs_proba: bool, optional (default=False)</strong>
<blockquote>
Whether the metric function requires probability estimates out of a
 classifier. If True, make sure that every model in the pipeline has
 a `predict_proba` method! Will be ignored if the metric is a string
 or a scorer.
</blockquote>
<strong> needs_threshold: bool, optional (default=False)</strong>
<blockquote>
Whether the metric function takes a continuous decision certainty. This only
 works for binary classification using estimators that have either a
 `decision_function` or `predict_proba` method. Will be ignored if the metric
 is a string or a scorer.
</blockquote>
<strong>train_sizes: sequence, optional (default=np.linspace(0.2, 1.0, 5))</strong>
<blockquote>
Relative or absolute numbers of training examples that will be used
 to generate the learning curve. If the dtype is float, it is
 regarded as a fraction of the maximum size of the training set.
 Otherwise it is interpreted as absolute sizes of the training sets.
</blockquote>
<strong>n_calls: int or sequence, optional (default=0)</strong>
<blockquote>
Maximum number of iterations of the BO (including `n_random starts`).
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
<li><b>max_time: int</b></br>Maximum allowed time for the BO (in seconds).</li>
<li><b>delta_x: int or float</b></br>Maximum distance between two consecutive points.</li>
<li><b>delta_x: int or float</b></br>Maximum score between two consecutive points.</li>
<li><b>cv: int</b></br>Number of folds for the cross-validation. If 1, the
 training set will be randomly split in a subtrain and validation set.</li>
<li><b>callback: callable or list of callables</b></br>Callbacks for the BO.</li>
<li><b>dimensions: dict or array</b></br>Custom hyperparameter space for the
 bayesian optimization. Can be an array (only if there is 1 model in the
 pipeline) or a dictionary with the model's name as key.</li>
<li><b>plot_bo: bool</b></br>Whether to plot the BO's progress as it runs.
 Creates a canvas with two plots: the first plot shows the score of every trial
 and the second shows the distance between the last consecutive steps. Don't
 forget to call `%matplotlib` at the start of the cell if you are using jupyter
 notebook!</li>
<li><b>Any other parameter for the <a href="https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html">bayesian optimization function</a>.</b></li>                
</ul>
</blockquote>
<strong>bagging: int or None, optional (default=None)</strong>
<blockquote>
Number of data sets (bootstrapped from the training set) to use in the bagging
 algorithm. If None or 0, no bagging is performed.
</blockquote>
</tr>
</table>
</div>
<br />