# Logging & Tracking
--------------------

## Logging

To start logging your experiments, fill the `logger` parameter in the
trainer's initializer with the name/path to store the logging file.
If automatic naming is used, the file is saved using the \__name__ of
the class followed by the timestamp of the logger's creation, e.g.
`ATOMClassifier_11May21_20h11m03s`. The logging file contains method
calls, all printed messages to stdout with maximum verbosity, and any
exception raised during running.

<br>

## Tracking

ATOM uses [mlflow tracking](https://www.mlflow.org/docs/latest/tracking.html)
as a backend API and UI for logging the models in its pipeline. Start
tracking your experiments assigning a name to the `experiment` parameter
in the trainer's initializer. Every model is tracked using a separate run.
When no backend is configured, the data is stored locally at `./mlruns`.
To configure the backend, use [mlflow.set_tracking_uri](https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.set_tracking_uri)
in your notebook or IDE before initializing the trainer. This does not
affect the currently active run (if one exists), but takes effect for
successive runs. When using ATOM on Databricks, the uri should include
the complete path to the storage, e.g. `/Users/username@domain.com/experiment_name`.
The following elements are tracked:

**Tags**<br>
The runs are automatically tagged with the model's full name, the branch
from which the model was trained, and the time it took to fit the model.

**Parameters**<br>
All parameters used by the estimator at initialization are tracked (only if
the estimator has a `get_params` method). Note that additional parameters
passed to the fit method are not.

**Model**<br>
The model's estimator is stored as artifact. The estimator has to be
compatible with the [mlflow.sklearn](https://www.mlflow.org/docs/latest/python_api/mlflow.sklearn.html),
module. This option can be switched off using the trainer's `log_model`
attribute, e.g. `atom.log_model = False`.

**Hyperparameter tuning**<br>
If hyperparameter tuning is performed, every call of the BO is tracked as a
nested run in the model's main run. This option can be switched off using
the trainer's `log_bo` attribute, e.g. `atom.log_bo = False`.

**Metrics**<br>
All metric results are tracked, not only during training, but also if the
[evaluate](../../API/ATOM/atomclassifier/#evaluate) method is called at a
later point. Metrics calculated during in-training evaluation are also
logged (only for [XGB](../../API/models/xgb), [LGB](../../API/models/lgb)
and [CatB](../../API/models/catb)).

**Dataset**<br>
The train and test sets used to fit and evaluate the model can be stored
as .csv files to the run's artifacts. This option can be switched on using
the trainer's `log_data` attribute, e.g. `atom.log_data = True`.

**Pipeline**<br>
The model's pipeline (returned from the [export_pipeline](../../API/ATOM/atomclassifier/#export-pipeline)
method) can be stored as an artifact using the trainer's `log_pipeline`
attribute, e.g. `atom.log_pipeline = True`.

**Plots**<br>
Plots are stored as .png artifacts in all runs corresponding to the models
that are showed in the plot. If the `filename` parameter is specified, they
are stored under that name, else the plot's name is used. This option can be
switched off using the trainer's `log_plots` attribute, e.g. `atom.log_plots = False`.
