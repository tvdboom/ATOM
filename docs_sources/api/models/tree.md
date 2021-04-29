# Decision Tree (Tree)
----------------------

A single decision tree classifier/regressor.

Corresponding estimators are:

* [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
  for classification tasks.
* [DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
  for regression tasks.

Read more in sklearn's [documentation](https://scikit-learn.org/stable/modules/tree.html).


<br><br>
## Hyperparameters

* By default, the estimator adopts the default parameters provided by
  its package. See the [user guide](../../../user_guide/#parameter-customization)
  on how to customize them.
* The `random_state` parameter is set equal to that of the trainer.

<table style="font-size:16px">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Dimensions:</strong></td>
<td width="80%" style="background:white;">
<strong>criterion: str</strong><br>
<ul style="line-height:1.5em;margin-top:5px;margin-bottom:0">
<li>classifier: default="gini"<br>
Categorical(["gini", "entropy"], name="criterion")</li>
<li>regressor: default="mse"<br>
Categorical(["mse", "mae", "friedman_mse"], name="criterion")</li>
</ul>
<p>
<strong>splitter: str, default="best"</strong><br>
Categorical(["best", "random"], name="splitter")
</p>
<p>
<strong>max_depth: int or None, default=None</strong><br>
Categorical([None, *list(range(1, 10))], name="max_depth")
</p>
<p>
<strong>min_samples_split: int, default=2</strong><br>
Integer(2, 20, name="min_samples_split")
</p>
<p>
<strong>min_samples_leaf: int, default=1</strong><br>
Integer(1, 20, name="min_samples_leaf")
</p>
<p>
<strong>max_features: float or None, default=None</strong><br>
Categorical([None, *np.linspace(0.5, 0.9, 5)], name="max_features")
</p>
<p>
<strong>ccp_alpha: float, default=0.0</strong><br>
Real(0, 0.035, name="ccp_alpha")
</p>
</td>
</tr>
</table>



<br><br>
## Attributes

### Data attributes

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
<strong>bo: pd.DataFrame</strong><br>
Information of every step taken by the BO. Columns include:
<ul style="line-height:1.2em;margin-top:5px">
<li><b>params</b>: Parameters used in the model.</li>
<li><b>estimator</b>: Estimator used for this iteration (fitted on last cross-validation).</li>
<li><b>score</b>: Score of the chosen metric. List of scores for multi-metric.</li>
<li><b>time_iteration</b>: Time spent on this iteration.</li>
<li><b>time</b>: Total time spent since the start of the BO.</li>
</ul>
<p>
<strong>best_params: dict</strong><br>
Dictionary of the best combination of hyperparameters found by the BO.
</p>
<p>
<strong>estimator: class</strong><br>
Estimator instance with the best combination of hyperparameters fitted
on the complete training set.
</p>
<p>
<strong>time_bo: str</strong><br>
Time it took to run the bayesian optimization algorithm.
</p>
<p>
<strong>metric_bo: float or list</strong><br>
Best metric score(s) on the BO.
</p>
<p>
<strong>time_fit: str</strong><br>
Time it took to train the model on the complete training set and
calculate the metric(s) on the test set.
</p>
<p>
<strong>metric_train: float or list</strong><br>
Metric score(s) on the training set.
</p>
<p>
<strong>metric_test: float or list</strong><br>
Metric score(s) on the test set.
</p>
<p>
<strong>metric_bagging: list</strong><br>
Bagging's results with shape=(bagging,) for single-metric runs and
shape=(metric, bagging) for multi-metric runs.
</p>
<p>
<strong>mean_bagging: float or list</strong><br>
Mean of the bagging's results. List of values for multi-metric runs.
</p>
<p>
<strong>std_bagging: float or list</strong><br>
Standard deviation of the bagging's results. List of values for multi-metric runs.
</p>
<strong>results: pd.Series</strong><br>
Training results. Columns include:
<ul style="line-height:1.2em;margin-top:5px">
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
</td>
</tr>
</table>
<br>


### Prediction attributes

The prediction attributes are not calculated until the attribute is
called for the first time. This mechanism avoids having to calculate
attributes that are never used, saving time and memory.

<table style="font-size:16px">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Prediction attributes:</strong></td>
<td width="80%" style="background:white;">
<p>
<strong>predict_train: np.ndarray</strong><br>
Predictions of the model on the training set.
</p>
<p>
<strong> predict_test: np.ndarray</strong><br>
Predictions of the model on the test set.
</p>
<p>
<strong>predict_proba_train: np.ndarray</strong><br>
Predicted probabilities of the model on the training set (only if classifier).
</p>
<p>
<strong>predict_proba_test: np.ndarray</strong><br>
Predicted probabilities of the model on the test set (only if classifier).
</p>
<p>
<strong>predict_log_proba_train: np.ndarray</strong><br>
Predicted log probabilities of the model on the training set (only if classifier).
</p>
<p>
<strong>predict_log_proba_test: np.ndarray</strong><br>
Predicted log probabilities of the model on the test set (only if classifier).
</p>
<p>
<strong>score_train: np.float64</strong><br>
Model's score on the training set.
</p>
<p>
<strong>score_test: np.float64</strong><br>
Model's score on the test set.
</p>
</td>
</tr>
</table>



<br><br>
## Methods

The majority of the [plots](../../../user_guide/plots) and [prediction methods](../../..user_guide/predicting)
can be called directly from the models, e.g. `atom.tree.plot_permutation_importance()`
or `atom.tree.predict(X)`. The remaining utility methods can be found hereunder.

<table style="font-size:16px">
<tr>
<td><a href="#calibrate">calibrate</a></td>
<td>Calibrate the model.</td>
</tr>

<tr>
<td><a href="#delete">delete</a></td>
<td>Delete the model from the trainer.</td>
</tr>

<tr>
<td><a href="#rename">rename</a></td>
<td>Change the model's tag.</td>
</tr>

<tr>
<td><a href="#reset-predictions">reset_predictions</a></td>
<td>Clear all the prediction attributes.</td>
</tr>

<tr>
<td><a href="#scoring">scoring</a></td>
<td>Get the score for a specific metric.</td>
</tr>

<tr>
<td><a href="#save-estimator">save_estimator</a></td>
<td>Save the estimator to a pickle file.</td>
</tr>
</table>
<br>


<a name="calibrate"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">calibrate</strong>(**kwargs)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/modeloptimizer.py#L529">[source]</a>
</span>
</div>
Applies probability calibration on the estimator. The calibration is
performed using sklearn's [CalibratedClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)
class. The calibrator is trained via cross-validation on a subset of the
training data, using the rest to fit the calibrator. The new classifier
will replace the `estimator` attribute. After calibrating, all prediction
attributes of the winning model will reset. Only if classifier.
<table style="font-size:16px">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>**kwargs</strong><br>
Additional keyword arguments for the CalibratedClassifierCV instance.
Using cv="prefit" will use the trained model and fit the calibrator
on the test set. Note that doing this will result in data leakage in
the test set. Use this only if you have another, independent set for
testing.
</td>
</tr>
</table>
<br />


<a name="delete"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">delete</strong>()
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/basemodel.py#L335">[source]</a>
</span>
</div>
Delete the model from the trainer. If it's the winning model, the next
best model (through `metric_test` or `mean_bagging`) is selected as
winner. If it's the last model in the trainer, the metric and training
approach are reset. Use this method to drop unwanted models from
the pipeline or to free some memory before saving. The model is not
removed from any active mlflow experiment.
<br /><br /><br />


<a name="rename"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">rename</strong>(name=None)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/modeloptimizer.py#L556">[source]</a>
</span>
</div>
Change the model's tag. The acronym always stays at the beginning
of the model's name.
<table style="font-size:16px">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>name: str or None, optional (default=None)</strong><br>
New tag for the model. If None, the tag is removed.
</table>
<br />


<a name="reset-predictions"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">reset_predictions</strong>()
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/basemodel.py#L200">[source]</a>
</span>
</div>
Clear the [prediction attributes](../../..user_guide/predicting) from all models.
Use this method to free some memory before saving the trainer.
<br /><br /><br />


<a name="scoring"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">scoring</strong>
(metric=None, dataset="test")
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/basemodel.py#L323">[source]</a>
</span>
</div>
Get the model's scoring for provided metrics.
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
<strong>score: pd.Series</strong><br>
Model's scoring.
</td>
</tr>
</table>
<br />


<a name="save-estimator"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">save_estimator</strong>(filename="auto")
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/modeloptimizer.py#L594">[source]</a>
</span>
</div>
Save the estimator to a pickle file.
<table style="font-size:16px">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>filename: str, optional (default="auto")</strong><br>
Name of the file. Use "auto" for automatic naming.
</td>
</tr>
</table>
<br />



## Example

```python
from atom import ATOMRegressor

atom = ATOMRegressor(X, y)
atom.run(models="Tree", metric="MSLE")
```