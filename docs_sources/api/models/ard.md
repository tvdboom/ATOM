# Automatic Relevance Determination (ARD)
-----------------------------------------

Automatic Relevance Determination is very similar to [Bayesian Ridge](../br), but
 can lead to sparser coefficients. Fit the weights of a regression model, using an
 ARD prior. The weights of the regression model are assumed to be in Gaussian
 distributions.

Corresponding estimators are:

* [ARDRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html)
  for regression tasks.

Read more in sklearn's [documentation](https://scikit-learn.org/stable/modules/linear_model.html#automatic-relevance-determination-ard).


<br><br>
## Hyperparameters
------------------

* By default, the estimator adopts the default parameters provided by its package.
  See the [user guide](../../../user_guide/#parameter-customization) on how to
  customize them.

<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Dimensions:</strong></td>
<td width="75%" style="background:white;">
<strong>n_iter: float, default=300</strong>
<blockquote>
Integer(100, 1000, name="n_iter")
</blockquote>
<strong>alpha_1: float, default=1e-6</strong>
<blockquote>
Categorical([1e-8, 1e-6, 1e-4, 1e-2], name="alpha_1")
</blockquote>
<strong>alpha_2: float, default=1e-6</strong>
<blockquote>
Categorical([1e-8, 1e-6, 1e-4, 1e-2], name="alpha_2")
</blockquote>
<strong>lambda_1: float, default=1e-6</strong>
<blockquote>
Categorical([1e-8, 1e-6, 1e-4, 1e-2], name="lambda_1")
</blockquote>
<strong>lambda_2: float, default=1e-6</strong>
<blockquote>
Categorical([1e-8, 1e-6, 1e-4, 1e-2], name="lambda_2")
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
<li>"params": Parameters used in the estimator.</li>
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
 can be called directly from the model, e.g. `atom.ard.plot_permutation_importance()`
 or `atom.ard.predict(X)`. The remaining utility methods can be found hereunder:
<br><br>

<table>
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


<a name="reset-predictions"></a>
<pre><em>method</em> <strong style="color:#008AB8">reset_predictions</strong>()
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basemodel.py#L168">[source]</a></div></pre>
Clear all the [prediction attributes](../../../user_guide/#predicting).
 Use this method to free some memory before saving the model.
<br /><br /><br />


<a name="scoring"></a>
<pre><em>method</em> <strong style="color:#008AB8">scoring</strong>(metric=None, dataset="test", **kwargs)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basemodel.py#L320">[source]</a></div></pre>
Get the model's score for a specific metric.
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>metric: str or None, optional (default=None)</strong>
<blockquote>
Name of the metric to calculate. Choose from any of sklearn's <a href="https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules"></a>.
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
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basemodel.py#L417">[source]</a></div></pre>
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
atom.run(models="ARD", n_calls=20, n_initial_points=7, bagging=5)
```