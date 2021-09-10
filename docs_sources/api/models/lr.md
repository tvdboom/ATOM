# Logistic regression (LR)
--------------------------

Logistic regression, despite its name, is a linear model for
classification rather than regression. Logistic regression is also
known in the literature as logit regression, maximum-entropy
classification (MaxEnt) or the log-linear classifier. In this model,
the probabilities describing the possible outcomes of a single trial
are modeled using a logistic function.

Corresponding estimators are:

* [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
  for classification tasks.

Read more in sklearn's [documentation](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression).


<br><br>
## Hyperparameters

* By default, the estimator adopts the default parameters provided by
  its package. See the [user guide](../../../user_guide/training/#parameter-customization)
  on how to customize them.
* The `penalty` parameter is always set to "l2" when penalty = "none" and solver = "liblinear".
* The `penalty` parameter is always set to "l2" when penalty = "l1" and solver != "liblinear" or "saga".
* The `penalty` parameter is always set to "l2" when penalty = "elasticnet" and solver != "saga".
* The `C` parameter is not used when penalty = "none".
* The `l1_ratio` parameter is only used when penalty = "elasticnet".
* The `n_jobs` and `random_state` parameters are set equal to those of the
 trainer.

<table style="font-size:16px">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Dimensions:</strong></td>
<td width="80%" style="background:white;">
<p>
<strong>penalty: str, default="l2"</strong><br>
Categorical(["none", "l1", "l2", "elasticnet"], name="penalty")
</p>
<p>
<strong>C: float, default=1.0</strong><br>
Real(1e-3, 100, "log-uniform", name="C")
</p>
<p>
<strong>solver: str, default="lbfgs"</strong><br>
Categorical(["lbfgs", "newton-cg", "liblinear", "sag", "saga"], name="solver")
</p>
<p>
<strong>max_iter: int, default=100</strong><br>
Integer(100, 1000, name="max_iter")
</p>
<p>
<strong>l1_ratio: float, default=0.5</strong><br>
Categorical(np.linspace(0.1, 0.9, 9), name="l1_ratio")
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
<strong>metric_bootstrap: list</strong><br>
Bootstrap results with shape=(n_bootstrap,) for single-metric runs and
shape=(metric, n_bootstrap) for multi-metric runs.
</p>
<p>
<strong>mean_bootstrap: float or list</strong><br>
Mean of the bootstrap results. List of values for multi-metric runs.
</p>
<p>
<strong>std_bootstrap: float or list</strong><br>
Standard deviation of the bootstrap results. List of values for multi-metric runs.
</p>
<strong>results: pd.Series</strong><br>
Training results. Columns include:
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
Predicted probabilities of the model on the training set.
</p>
<p>
<strong>predict_proba_test: np.ndarray</strong><br>
Predicted probabilities of the model on the test set.
</p>
<p>
<strong>predict_log_proba_train: np.ndarray</strong><br>
Predicted log probabilities of the model on the training set.
</p>
<p>
<strong>predict_log_proba_test: np.ndarray</strong><br>
Predicted log probabilities of the model on the test set.
</p>
<p>
<strong>decision_function_train: np.ndarray</strong><br>
Decision function scores on the training set.
</p>
<p>
<strong>decision_function_test: np.ndarray</strong><br>
Decision function scores on the test set.
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

The majority of the [plots](../../../user_guide/plots) and [prediction methods](../../../user_guide/predicting)
can be called directly from the model, e.g. `atom.lr.plot_permutation_importance()`
or `atom.lr.predict(X)`. The remaining utility methods can be found hereunder.

<table style="font-size:16px">
<tr>
<td><a href="#calibrate">calibrate</a></td>
<td>Calibrate the model.</td>
</tr>

<tr>
<td><a href="#cross-validate">cross_validate</a></td>
<td>Evaluate the model using cross-validation.</td>
</tr>

<tr>
<td><a href="#delete">delete</a></td>
<td>Delete the model from the trainer.</td>
</tr>

<tr>
<td><a href="#export-pipeline">export_pipeline</a></td>
<td>Export the model's pipeline to a sklearn-like Pipeline object.</td>
</tr>

<tr>
<td><a href="#full-train">full_train</a></td>
<td>Get the estimator trained on the complete dataset.</td>
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
<td><a href="#evaluate">evaluate</a></td>
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
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/modeloptimizer.py#L589">[source]</a>
</span>
</div>
Applies probability calibration on the estimator. The
estimator is trained via cross-validation on a subset of the
training data, using the rest to fit the calibrator. The new
classifier will replace the `estimator` attribute and is
logged to any active mlflow experiment. Since the estimator
changed, all the model's prediction attributes are reset.
<table style="font-size:16px">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>**kwargs</strong><br>
Additional keyword arguments for sklearn's <a href="https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html">CalibratedClassifierCV</a>.
Using cv="prefit" will use the trained model and fit the calibrator
on the test set. Use this only if you have another, independent set
for testing.
</td>
</tr>
</table>
<br />


<a name="cross-validate"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">cross_validate</strong>(**kwargs)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/modeloptimizer.py#L676">[source]</a>
</span>
</div>
Evaluate the model using cross-validation. This method cross-validates the
whole pipeline on the complete dataset. Use it to assess the robustness of
the solution's performance.
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
<em>method</em> <strong style="color:#008AB8">delete</strong>()
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/basemodel.py#L377">[source]</a>
</span>
</div>
Delete the model from the trainer. If it's the winning model, the next
best model (through `metric_test` or `mean_bootstrap`) is selected as
winner. If it's the last model in the trainer, the metric and training
approach are reset. Use this method to drop unwanted models from
the pipeline or to free some memory before saving. The model is not
removed from any active mlflow experiment.
<br /><br /><br />


<a name="export-pipeline"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">export_pipeline</strong>(pipeline=None, verbose=None)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/modeloptimizer.py#L630">[source]</a>
</span>
</div>
Export the model's pipeline to a sklearn-like object. If the model
used feature scaling, the Scaler is added before the model. The
returned pipeline is already fitted on the training set.

!!! note
    ATOM's Pipeline class behaves exactly the same as a sklearn <a href="https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html">Pipeline</a>,
    and additionally, it's compatible with transformers that drop samples
    and transformers that change the target column.

!!! warning
    Due to incompatibilities with sklearn's API, the exported pipeline always
    fits/transforms on the entire dataset provided. Beware that this can
    cause errors if the transformers were fitted on a subset of the data.

<table style="font-size:16px">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>pipeline: bool, sequence or None, optional (default=None)</strong><br>
Transformers to use on the data before predicting.
<ul style="line-height:1.2em;margin-top:5px">
<li>If None: Only transformers that are applied on the whole dataset are used.</li>
<li>If False: Don't use any transformers.</li>
<li>If True: Use all transformers in the pipeline.</li>
<li>If sequence: Transformers to use, selected by their index in the pipeline.</li>
</ul>
<p>
<strong>verbose: int or None, optional (default=None)</strong><br>
Verbosity level of the transformers in the pipeline.
If None, it leaves them to their original verbosity.
</p>
</td>
</tr>
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Returns:</strong></td>
<td width="80%" style="background:white;">
<strong>pipeline: Pipeline</strong><br>
Current branch as a sklearn-like Pipeline object.
</td>
</tr>
</table>
<br />


<a name="full-train"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">full_train</strong>()
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/modeloptimizer.py#L777">[source]</a>
</span>
</div>
Get the estimator trained on the complete dataset. In some cases it
might be desirable to use all the available data to train a final
model after the right hyperparameters are found. Note that this
means that the model can not be evaluated.
<table style="font-size:16px">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Returns:</strong></td>
<td width="80%" style="background:white;">
<strong>est: estimator</strong><br>
Model estimator trained on the full dataset.
</td>
</tr>
</table>
<br />


<a name="rename"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">rename</strong>(name=None)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/modeloptimizer.py#L740">[source]</a>
</span>
</div>
Change the model's tag. The acronym always stays at the beginning
of the model's name. If the model is being tracked by mlflow, the
name of the corresponding run is also changed.
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
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/basemodel.py#L270">[source]</a>
</span>
</div>
Clear the [prediction attributes](../../../user_guide/predicting) from all models.
Use this method to free some memory before saving the trainer.
<br /><br /><br />


<a name="evaluate"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">evaluate</strong>
(metric=None, dataset="test")
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/basemodel.py#L382">[source]</a>
</span>
</div>
Get the model's score for the provided metrics.
<table style="font-size:16px">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>metric: str or None, optional (default=None)</strong><br>
Name of the metric to calculate. If None, returns the models' final
results (ignoring the <code>dataset</code> parameter). Choose from any
of sklearn's classification <a href="https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules">SCORERS</a>
or one of the following custom metrics:
<ul style="line-height:1.2em;margin-top:5px">
<li>"cm" for the confusion matrix.</li>
<li>"tn" for true negatives.</li>
<li>"fp" for false positives.</li>
<li>"fn" for false negatives.</li>
<li>"tp" for true positives.</li>
<li>"fpr" for the false positive rate.</li>
<li>"tpr" for the true positive rate.</li>
<li>"fnr" for the false negative rate.</li>
<li>"tnr" for the true negative rate.</li>
<li>"sup" for the support metric.</li>
<li>"lift" for the lift metric.</li>
</ul>
<p>
<strong>dataset: str, optional (default="test")</strong><br>
Data set on which to calculate the metric. Options are "train" or "test".
</p>
</td>
</tr>
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Returns:</strong></td>
<td width="80%" style="background:white;">
<strong>score: float or np.ndarray</strong><br>
Model's score for the selected metric.
</td>
</tr>
</table>
<br />


<a name="save-estimator"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">save_estimator</strong>(filename="auto")
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/modeloptimizer.py#L805">[source]</a>
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
from atom import ATOMClassifier

atom = ATOMClassifier(X, y)
atom.run(models="LR")
```