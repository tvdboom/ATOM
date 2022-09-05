# AdaBoost
----------

:: atom.models:AdaBoost
    :: tags
    :: description
    :: see also

<br>

## Example

:: examples

<br><br>

## Hyperparameters

* By default, the estimator adopts the default parameters provided by
  its package. See the [user guide](../../../user_guide/training/#parameter-customization)
  on how to customize them.
* The `algorithm` parameter is only used with AdaBoostClassifier.
* The `loss` parameter is only used with AdaBoostRegressor.
* The `random_state` parameter is set equal to that of the parent.

<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Dimensions:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>n_estimators: int, default=50</strong><br>
Integer(10, 500, name="n_estimators")
</p>
<p>
<strong>learning_rate: float, default=1.0</strong><br>
Real(0.01, 10, "log-uniform", name="learning_rate")
</p>
<p>
<strong>algorithm: str, default="SAMME.R"</strong><br>
Categorical(["SAMME.R", "SAMME"], name="algorithm")
</p>
<p>
<strong>loss: str, default="linear"</strong><br>
Categorical(["linear", "square", "exponential"], name="loss")
</p>
</td>
</tr>
</table>


<br><br>

## Attributes

### Data attributes

:: table:
    - attributes:
        from_docstring: False
        include:
            - atom.branch:Branch.pipeline
            - atom.branch:Branch.mapping
            - atom.branch:Branch.dataset
            - atom.branch:Branch.train
            - atom.branch:Branch.test
            - atom.branch:Branch.X
            - atom.branch:Branch.y
            - atom.branch:Branch.X_train
            - atom.branch:Branch.y_train
            - atom.branch:Branch.X_test
            - atom.branch:Branch.y_test
            - atom.branch:Branch.shape
            - atom.branch:Branch.columns
            - atom.branch:Branch.n_columns
            - atom.branch:Branch.features
            - atom.branch:Branch.n_features
            - atom.branch:Branch.target

<br>

### Utility attributes

<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Attributes:</strong></td>
<td width="80%" class="td_params">
<strong>bo: pd.DataFrame</strong><br>
Information of every step taken by the BO. Columns include:
<ul style="line-height:1.2em;margin-top:5px">
<li><b>call</b>: Name of the call.</li>
<li><b>params</b>: Parameters used in the model.</li>
<li><b>estimator</b>: Estimator used for this iteration (fitted on last cross-validation).</li>
<li><b>score</b>: Score of the chosen metric. List of scores for multi-metric.</li>
<li><b>time</b>: Time spent on this iteration.</li>
<li><b>total_time</b>: Total time spent since the start of the BO.</li>
</ul>
<p>
<strong>best_call: str</strong><br>
Name of the best call in the BO.
</p>
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
<strong>metric_bootstrap: np.array</strong><br>
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

:: table:
    - attributes:
        from_docstring: False
        include:
            - atom.basemodel:BaseModel.decision_function_train
            - atom.basemodel:BaseModel.decision_function_test
            - atom.basemodel:BaseModel.decision_function_holdout
            - atom.basemodel:BaseModel.predict_train
            - atom.basemodel:BaseModel.predict_test
            - atom.basemodel:BaseModel.predict_holdout
            - atom.basemodel:BaseModel.predict_log_proba_train
            - atom.basemodel:BaseModel.predict_log_proba_test
            - atom.basemodel:BaseModel.predict_log_proba_holdout
            - atom.basemodel:BaseModel.predict_proba_train
            - atom.basemodel:BaseModel.predict_proba_test
            - atom.basemodel:BaseModel.predict_proba_holdout
            - atom.basemodel:BaseModel.score_train
            - atom.basemodel:BaseModel.score_test
            - atom.basemodel:BaseModel.score_holdout


<br><br>

## Methods

The [model plots][] and [prediction methods][] can be called directly
from the model, e.g. `atom.adab.plot_permutation_importance()` or
`atom.adab.predict(X)`. The remaining utility methods can be found
hereunder.

:: methods:
    toc_only: False
    include:
        - calibrate
        - clear
        - create_app
        - create_dashboard
        - cross_validate
        - delete
        - evaluate
        - export_pipeline
        - full_train
        - inverse_transform
        - rename
        - save_estimator
        - transform
