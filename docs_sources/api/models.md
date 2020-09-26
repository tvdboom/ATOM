# Models
--------

After fitting, every model class is attached to the `training` instance as an
 attribute. We refer to these "subclasses" as `models` (see the
 [nomenclature](../../user_guide/#nomenclature)). The classes contain a variety of
 attributes and methods to help you understand how the underlying estimator performed.
 They can be accessed using the models' [acronyms](../../user_guide/#models), e.g.
 `atom.LGB` to access LightGBM's `model`. The available models and their corresponding
 acronyms are: 

* 'GP' for Gaussian Process
* 'GNB' for Gaussian Naive Bayes
* 'MNB' for Multinomial Naive Bayes
* 'BNB' for Bernoulli Naive Bayes
* 'OLS' for Ordinary Least Squares
* 'Ridge' for Ridge classification/regression
* 'Lasso' for Lasso regression
* 'EN' for Elastic Net regression
* 'BR' for Bayesian Regression
* 'LR' for Logistic Regression
* 'LDA' for Linear Discriminant Analysis
* 'QDA' for Quadratic Discriminant Analysis
* 'KNN' for K-Nearest Neighbors
* 'Tree' for Decision Tree
* 'Bag' for Bagging
* 'ET' for Extra-Trees
* 'RF' for Random Forest
* 'AdaB' for AdaBoost
* 'GBM' for Gradient Boosting Machine
* 'XGB' for XGBoost
* 'LGB' for LightGBM
* 'CatB' for CatBoost
* 'lSVM' for Linear-SVM
* 'kSVM' for Kernel-SVM
* 'PA' for Passive Aggressive
* 'SGD' for Stochastic Gradient Descent
* 'MLP' for Multilayer Perceptron

<br><br>

!!! tip
    You can also use lowercase to call the `models`, e.g. `atom.lgb.plot_roc()`.

!!! warning
    The `models` should not be initialized by the user! Only use them through the
    `training` instances.


<br><br>
## Attributes
-------------

### Data attributes

You can use the same data attributes as the `training` instances to check the
 dataset that was used to fit a particular model. These can differ from each other
 if the model needs scaled features and the data wasn't already scaled. Note that,
 unlike with the `training` instances, the data can not be updated from the `models`
 (i.e. the data attributes have no `@setter`).

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
<strong>categories: list</strong>
<blockquote>
Sorted list of the unique categories in the target column.
</blockquote>
<strong>n_categories: int</strong>
<blockquote>
Number of unique categories in the target column.
</blockquote>
</td></tr>
</table>
<br>


### Utility attributes

<a name="atom"></a>
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Attributes:</strong></td>
<td width="75%" style="background:white;">
<strong>bo: pd.DataFrame</strong>
<blockquote>
Dataframe containing the information of every step taken by the BO. Columns include:
<ul>
<li>'params': Parameters used in the model.</li>
<li>'model': Model used for this iteration (fitted on last cross-validation).</li>
<li>'score': Score of the chosen metric. List of scores for multi-metric.</li>
<li>'time_iteration': Time spent on this iteration.</li>
<li>'time': Total ime spent since the start of the BO.</li>
</ul>
</blockquote>
<strong>best_params: dict</strong>
<blockquote>
Dictionary of the best combination of hyperparameters found by the BO.
</blockquote>
<strong>estimator: class</strong>
<blockquote>
Estimator instance with the best combination of hyperparameters fitted on the complete training set.
</blockquote>
<strong>time_bo: str</strong>
<blockquote>
Time it took to run the bayesian optimization algorithm.
</blockquote>
<strong>metric_bo: float or list</strong>
<blockquote>
Best metric score(s) on the BO.
</blockquote>
<strong>time_fit: str</strong>
<blockquote>
Time it took to train the model on the complete training set and calculate the
 metric(s) on the test set.
</blockquote>
<strong>metric_train: float or list</strong>
<blockquote>
Metric score(s) on the training set.
</blockquote>
<strong>metric_test: float or list</strong>
<blockquote>
Metric score(s) on the test set.
</blockquote>
<strong>evals: dict</strong>
<blockquote>
Dictionary of the metric calculated during training. The metric is provided by the model's
 package and is different for every model and every task. Only for models that allow
 in-training evaluation (XGB, LGB, CatB). Available keys:
<ul>
<li>'metric': Name of the metric. </li>
<li>'train': List of scores calculated on the training set.</li>
<li>'test': List of scores calculated on the test set.</li>
</ul>
</blockquote>
<strong>metric_bagging: list</strong>
<blockquote>
Array of the bagging's results.
</blockquote>
<strong>mean_bagging: float</strong>
<blockquote>
Mean of the bagging's results.
</blockquote>
<strong>std_bagging: float</strong>
<blockquote>
Standard deviation of the bagging's results.
</blockquote>
</td>
</tr>
</table>
<br>


### Prediction attributes

The prediction attributes are not calculated until the attribute is called for the
 first time. This mechanism avoids having to calculate attributes that are never
 used, saving time and memory.

<a name="atom"></a>
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Prediction attributes:</strong></td>
<td width="75%" style="background:white;">
<strong>predict_train: np.ndarray</strong>
<blockquote>
Predictions of the model on the training set.
</blockquote>
<strong> predict_test: np.ndarray</strong>
<blockquote>
Predictions of the model on the test set.
</blockquote>
<strong>predict_proba_train: np.ndarray</strong>
<blockquote>
Predicted probabilities of the model on the training set. Only for estimators with
a `predict_proba` method.
</blockquote>
<strong>predict_proba_test: np.ndarray</strong>
<blockquote>
Predicted probabilities of the model on the test set. Only for estimators with
a `predict_proba` method.
</blockquote>
<strong>predict_log_proba_train: np.ndarray</strong>
<blockquote>
Predicted log probabilities of the model on the training set. Only for estimators with
a `predict_proba` method.
</blockquote>
<strong>predict_log_proba_test: np.ndarray</strong>
<blockquote>
Predicted log probabilities of the model on the test set. Only for estimators with
a `predict_proba` method.
</blockquote>
<strong>decision_function_train: np.ndarray</strong>
<blockquote>
Decision function scores on the training set. Only for estimators with
a `decision_function` method.
</blockquote>
<strong>decision_function_test: np.ndarray</strong>
<blockquote>
Decision function scores on the test set. Only for estimators with
a `decision_function` method.
</blockquote>
<strong>score_train: np.float64</strong>
<blockquote>
Model's score on the training set.
</blockquote>
<strong>score_test: np.float64</strong>
<blockquote>
Model's score on the test set.
</blockquote>
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
----------

The majority of the [plots](../../user_guide/#plots) and [prediction methods](../../user_guide/#predicting)
 can be called directly from the `models`, e.g. `atom.xgb.plot_roc()` or `atom.xgb.predict_proba(X)`.
 The remaining utility methods can be found hereunder:
<br><br>

<table>
<tr>
<td width="15%"><a href="#models-calibrate">calibrate</a></td>
<td>Calibrate the model.</td>
</tr>

<tr>
<td width="15%"><a href="#models-reset-prediction-attributes">reset_prediction_attributes</a></td>
<td>Clear all the prediction attributes.</td>
</tr>

<tr>
<td width="15%"><a href="#models-scoring">scoring</a></td>
<td>Get the scoring of a specific metric on the test set.</td>
</tr>

<tr>
<td><a href="#models-save-estimator">save_estimator</a></td>
<td>Save the estimator to a pickle file.</td>
</tr>
</table>
<br>


<a name="models-calibrate"></a>
<pre><em>method</em> <strong style="color:#008AB8">calibrate</strong>(\*\*kwargs)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basemodel.py#L785">[source]</a></div></pre>
<div style="padding-left:3%">
Applies probability calibration on the winning model. The calibration is done
 using the [CalibratedClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)
 class from sklearn. The estimator will be trained via cross-validation on a subset of the
 training data, using the rest to fit the calibrator. The new classifier will replace
 the `estimator` attribute. After calibrating, all prediction attributes will reset.
<br /><br />
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>**kwargs</strong>
<blockquote>
Additional keyword arguments for the CalibratedClassifierCV instance.
Using cv='prefit' will use the trained model and fit the calibrator on
the test set. Note that doing this will result in data leakage in the
test set. Use this only if you have another, independent set for testing.
</blockquote>
</tr>
</table>
</div>
<br />


<a name="models-reset-prediction-attributes"></a>
<pre><em>method</em> <strong style="color:#008AB8">reset_prediction_attributes</strong>()
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basemodel.py#L612">[source]</a></div></pre>
<div style="padding-left:3%">
Clear all the prediction attributes. Use this method to free some memory before saving
 the class.
</div>
<br />


<a name="models-scoring"></a>
<pre><em>method</em> <strong style="color:#008AB8">scoring</strong>(metric=None, dataset='test')
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basemodel.py#L817">[source]</a></div></pre>
<div style="padding-left:3%">
Get the scoring of a specific metric on the test set.
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
<li>'cm' or 'confusion_matrix' for an array of the confusion matrix.</li>
<li>'tn' for true negatives.</li>
<li>'fp' for false positives.</li>
<li>'fn' for false negatives.</li>
<li>'lift' for the lift metric.</li>
<li>'fpr' for the false positive rate.</li>
<li>'tpr' for true positive rate.</li>
<li>'sup' for the support metric.</li>
</ul>
If None, returns the final results for the model (ignores the `dataset` parameter).
</blockquote>
<strong>dataset: str, optional (default='test')</strong>
<blockquote>
Data set on which to calculate the metric. Options are 'train' or 'test'.
</blockquote>
</tr>
</table>
</div>
<br />


<a name="models-save-estimator"></a>
<pre><em>method</em> <strong style="color:#008AB8">save_estimator</strong>(filename=None)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basemodel.py#L903">[source]</a></div></pre>
<div style="padding-left:3%">
Save the estimator to a pickle file.
<br /><br />
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>filename: str or None, optional (default=None)</strong>
<blockquote>
Name of the file to save. If None or 'auto', the default name is used.
</blockquote>
</tr>
</table>
</div>
<br />


