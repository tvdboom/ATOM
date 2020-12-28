# Gradient Boosting Machine (GBM)
---------------------------------

A Gradient Boosting Machine builds an additive model in a forward stage-wise
 fashion; it allows for the optimization of arbitrary differentiable loss
 functions. In each stage `n_classes_` regression trees are fit on the negative
 gradient of the binomial or multinomial deviance loss function. Binary
 classification is a special case where only a single regression tree is induced.

Corresponding estimators are:

* [GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
  for classification tasks.
* [GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
  for regression tasks.

Read more in sklearn's [documentation](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting).


<br><br>
## Hyperparameters
------------------

* By default, the estimator adopts the default parameters provided by its package.
  See the [user guide](../../../user_guide/#parameter-customization) on how to
  customize them.
* For multiclass classification tasks, the `loss` parameter is always set to "deviance".
* The `alpha` parameter is only used when loss = "huber" or "quantile".
* The `random_state` parameter is set equal to that of the trainer.

<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Dimensions:</strong></td>
<td width="75%" style="background:white;">
<strong>learning_rate: float, default=0.1</strong>
<blockquote>
Real(0.01, 1.0, "log-uniform", name="learning_rate")
</blockquote>
<strong>n_estimators: int, default=100</strong>
<blockquote>
Integer(10, 500, name="n_estimators")
</blockquote>
<strong>subsample: float, default=1.0</strong>
<blockquote>
Categorical(np.linspace(0.5, 1.0, 6), name="subsample")
</blockquote>
<strong>criterion: str, default="friedman_mse"</strong>
<blockquote>
Categorical(["friedman_mse", "mae", "mse"], name="criterion")
</blockquote>
<strong>min_samples_split: int, default=2</strong>
<blockquote>
Integer(2, 20, name="min_samples_split")
</blockquote>
<strong>min_samples_leaf: int, default=1</strong>
<blockquote>
Integer(1, 20, name="min_samples_leaf")
</blockquote>
<strong>max_depth: int, default=3</strong>
<blockquote>
Integer(1, 10, name="max_depth")
</blockquote>
<strong>max_features: float or None, default=None</strong>
<blockquote>
Categorical([None, \*np.linspace(0.5, 0.9, 5)], name="max_features")
</blockquote>
<strong>ccp_alpha: float, default=0</strong>
<blockquote>
Real(0, 0.035, name="ccp_alpha")
</blockquote>
<strong>loss: str</strong>
<blockquote>
<ul>
<li>binary classifier: default="deviance"<br>
 Categorical(["deviance", "exponential"], name="loss")</li>
<li>regressor: default="ls"<br>
 Categorical(["ls", "lad", "huber", "quantile"], name="loss")</li>
</ul>
</blockquote>
<strong>alpha: float, default=0.9</strong>
<blockquote>
Categorical(np.linspace(0.5, 0.9, 5), name="alpha")
</blockquote>
</td></tr>
</table>



<br><br><br>
## Attributes
-------------

### Data attributes

The dataset can be accessed at any time through multiple attributes, e.g. calling
`trainer.train` will return the training set. The data can differ from the trainer's
dataset if the model needs scaled features and the data wasn't scaled already.

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
Dataset's shape: (n_rows x n_columns).
</blockquote>
<strong>columns: list</strong>
<blockquote>
List of columns in the dataset.
</blockquote>
<strong>features: list</strong>
<blockquote>
List of features in the dataset.
</blockquote>
<strong>target: str</strong>
<blockquote>
Name of the target column.
</blockquote>
<strong>classes: pd.DataFrame</strong>
<blockquote>
Number of rows per target class in the train, test and complete dataset (only if classifier).
</blockquote>
<strong>n_classes: int</strong>
<blockquote>
Number of unique classes in the target column (only if classifier).
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
<li>"params": Parameters used in the model.</li>
<li>"estimator": Estimator used for this iteration (fitted on last cross-validation).</li>
<li>"score": Score of the chosen metric. List of scores for multi-metric.</li>
<li>"time_iteration": Time spent on this iteration.</li>
<li>"time": Total time spent since the start of the BO.</li>
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
<strong>metric_bagging: list</strong>
<blockquote>
Bagging's results with shape=(bagging,) for single-metric runs and shape=(metric, bagging) for multi-metric runs.
</blockquote>
<strong>mean_bagging: float or list</strong>
<blockquote>
Mean of the bagging's results. List of values for multi-metric runs.
</blockquote>
<strong>std_bagging: float or list</strong>
<blockquote>
Standard deviation of the bagging's results. List of values for multi-metric runs.
</blockquote>
<strong>results: pd.DataFrame</strong>
<blockquote>
Dataframe of the training results with the model acronym as index. Columns can include:
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
Predicted probabilities of the model on the training set (only if classifier).
</blockquote>
<strong>predict_proba_test: np.ndarray</strong>
<blockquote>
Predicted probabilities of the model on the test set (only if classifier).
</blockquote>
<strong>predict_log_proba_train: np.ndarray</strong>
<blockquote>
Predicted log probabilities of the model on the training set (only if classifier).
</blockquote>
<strong>predict_log_proba_test: np.ndarray</strong>
<blockquote>
Predicted log probabilities of the model on the test set (only if classifier).
</blockquote>
<strong>decision_function_train: np.ndarray</strong>
<blockquote>
Decision function scores on the training set (only if classifier).
</blockquote>
<strong>decision_function_test: np.ndarray</strong>
<blockquote>
Decision function scores on the test set (only if classifier).
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
<br><br><br>


## Methods
----------

The majority of the [plots](../../../user_guide/#plots) and [prediction methods](../../../user_guide/#predicting)
 can be called directly from the models, e.g. `atom.gbm.plot_permutation_importance()`
 or `atom.gbm.predict(X)`. The remaining utility methods can be found hereunder:
<br><br>

<table>
<tr>
<td width="15%"><a href="#calibrate">calibrate</a></td>
<td>Calibrate the model.</td>
</tr>

<tr>
<td width="15%"><a href="#delete">delete</a></td>
<td>Delete the model from the trainer.</td>
</tr>

<tr>
<td width="15%"><a href="#reset-predictions">reset_predictions</a></td>
<td>Clear all the prediction attributes.</td>
</tr>

<tr>
<td width="15%"><a href="#scoring">scoring</a></td>
<td>Get the score for a specific metric.</td>
</tr>

<tr>
<td><a href="#save-estimator">save_estimator</a></td>
<td>Save the estimator to a pickle file.</td>
</tr>
</table>
<br>


<a name="calibrate"></a>
<pre><em>method</em> <strong style="color:#008AB8">calibrate</strong>(**kwargs)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/modeloptimizer.py#L685">[source]</a></div></pre>
Applies probability calibration on the estimator. The calibration is done using the
 [CalibratedClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)
 class from sklearn. The calibrator will be trained via cross-validation on a subset
 of the training data, using the rest to fit the calibrator. The new classifier will
 replace the `estimator` attribute. After calibrating, all prediction attributes will
 reset. Only if classifier.
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>**kwargs</strong>
<blockquote>
Additional keyword arguments for the CalibratedClassifierCV instance.
Using cv="prefit" will use the trained model and fit the calibrator on
the test set. Note that doing this will result in data leakage in the
test set. Use this only if you have another, independent set for testing.
</blockquote>
</tr>
</table>
<br />


<a name="delete"></a>
<pre><em>method</em> <strong style="color:#008AB8">delete</strong>()
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basemodel.py#L308">[source]</a></div></pre>
Delete the model from the trainer.
<br /><br /><br />


<a name="reset-predictions"></a>
<pre><em>method</em> <strong style="color:#008AB8">reset_predictions</strong>()
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basemodel.py#L167">[source]</a></div></pre>
Clear all the [prediction attributes](../../../user_guide/#predicting).
 Use this method to free some memory before saving the model.
<br /><br /><br />


<a name="scoring"></a>
<pre><em>method</em> <strong style="color:#008AB8">scoring</strong>(metric=None, dataset="test", **kwargs)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basemodel.py#L313">[source]</a></div></pre>
Get the model's score for a specific metric.
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>metric: str or None, optional (default=None)</strong>
<blockquote>
Name of the metric to calculate. Choose from any of sklearn's <a href="https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules">SCORERS</a>
 or one of the following custom metrics (only if classifier):
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
If None, returns the final results for this model (ignores the <code>dataset</code> parameter).
</blockquote>
<strong>dataset: str, optional (default="test")</strong>
<blockquote>
Data set on which to calculate the metric. Options are "train" or "test".
</blockquote>
<strong>**kwargs</strong>
<blockquote>
Additional keyword arguments for the metric function.
</blockquote>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Returns:</strong></td>
<td width="75%" style="background:white;">
<strong>score: np.float64</strong>
<blockquote>
Model's scoring on the selected metric.
</blockquote>
</tr>
</table>
<br />


<a name="save-estimator"></a>
<pre><em>method</em> <strong style="color:#008AB8">save_estimator</strong>(filename=None)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/modeloptimizer.py#L720">[source]</a></div></pre>
Save the estimator to a pickle file.
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>filename: str or None, optional (default=None)</strong>
<blockquote>
Name of the file to save. If None or "auto", the estimator's __name__ is used.
</blockquote>
</tr>
</table>
<br />


## Example
----------

```python
from atom import ATOMRegressor

atom = ATOMRegressor(X, y)
atom.run(models="GBM")
```