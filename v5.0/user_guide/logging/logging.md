# Logging & Tracking
--------------------

## Logging

To start logging your experiments, fill the [`logger`][atomclassifier-logger]
parameter with the name or path to store the logging file. If automatic
naming is used, the file is saved using the \__name__ of the class
followed by the timestamp of the logger's creation, e.g.
`ATOMClassifier_11May21_20h11m03s`. The logging file contains method
calls, all printed messages to stdout with maximum verbosity, and any
exception raised during running.

<br>

## Tracking

ATOM uses [mlflow tracking](https://www.mlflow.org/docs/latest/tracking.html)
as a backend API and UI for logging the models in its pipeline. Start
tracking your experiments assigning a name to the [`experiment`]
[atomclassifier-experiment] parameter. Every model is tracked using a
separate run. When no backend is configured, the data is stored locally
at `./mlruns`. To configure the backend, use [mlflow.set_tracking_uri](https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.set_tracking_uri)
in your notebook or IDE before initializing atom. This does not affect
the currently active run (if one exists), but takes effect for successive
runs.

!!! info
    When using ATOM on [Databricks](https://databricks.com/), the
    experiment's name should include the complete path to the storage,
    e.g. `/Users/username@domain.com/experiment_name`.

The following elements are tracked:

**Tags**<br>
The runs are automatically tagged with the model's full name, the branch
from which the model was trained, and the time it took to fit the model.

**Parameters**<br>
All parameters used by the estimator at initialization are tracked (only
if the estimator has a `get_params` method). Additional parameters passed
to the fit method are not tracked.

**Model**<br>
The model's estimator is stored as artifact. The estimator has to be
compatible with the [mlflow.sklearn](https://www.mlflow.org/docs/latest/python_api/mlflow.sklearn.html),
module. This option can be switched off using atom's [`log_model`]
[atomclassifier-log_model] attribute, e.g. `#!python atom.log_model = False`.

**Hyperparameter tuning**<br>
If hyperparameter tuning is performed, every call of the BO is tracked
as a nested run in the model's main run. This option can be switched
off using atom's [`log_ht`][atomclassifier-log_ht] attribute, e.g.
`#!python atom.log_ht = False`.

**Metrics**<br>
All metric results are tracked, not only during training, but also when
the [evaluate][atomclassifier-evaluate] method is called at a later point.
Metrics calculated during in-training validation are also logged.

**Dataset**<br>
The train and test sets used to fit and evaluate the model can be stored
as .csv files to the run's artifacts. This option can be switched on
using atom's [`log_data`][atomclassifier-log_data] attribute, e.g.
`#!python atom.log_data = True`.

**Pipeline**<br>
The model's pipeline (returned from the [export_pipeline][atomclassifier-export_pipeline]
method) can be stored as an artifact using atom's [`log_pipeline`][atomclassifier-log_pipeline]
attribute, e.g. `#!python atom.log_pipeline = True`.

**Plots**<br>
By default, plots are stored as `.png` artifacts in all runs corresponding
to the models that are showed in the plot. If the `filename` parameter is
specified, they are stored under that name, else the plot's name is used.
This option can be switched off using atom's [`log_plots`][atomclassifier-log_plots]
attribute, e.g. `#!python atom.log_plots = False`.
