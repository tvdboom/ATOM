# Logging & Tracking
--------------------

## Logging

To start logging your experiments, fill the [`logger`][atomclassifier-logger]
parameter with the name or path to store the logging file. If automatic
naming is used, the file is saved using the \__name__ of the class
followed by the timestamp of the logger's creation, e.g.
`ATOMClassifier_11May21_20h11m03s`. The logging file contains method
calls, all printed messages to stdout with maximum verbosity, and any
exception raised during running. Additionally, the logging entries of
external libraries are redirected to the same file handler.

<br>

## Tracking

ATOM uses [MLflow Tracking](https://www.mlflow.org/docs/latest/tracking.html)
as a backend API and UI for logging models, parameters, pipelines, data
and plots. Start tracking your experiments assigning a name to the
[`experiment`][atomclassifier-experiment] parameter. Every model is
tracked using a separate run. When no backend is configured, the data is
stored locally at `./mlruns`. To configure the backend, use
[mlflow.set_tracking_uri](https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.set_tracking_uri)
in your notebook or IDE before initializing atom. This does not affect
the currently active run (if one exists), but takes effect for successive
runs. Run `mlflow ui` on your terminal to open MLflow's Tracking UI and 
view it at http://localhost:5000.

!!! note
    When using ATOM on [Databricks](https://databricks.com/), the
    experiment's name should include the complete path to the storage,
    e.g. `/Users/username@domain.com/experiment_name`.


**Example**

```pycon
>>> from atom import ATOMClassifier
>>> from sklearn.datasets import load_breast_cancer

>>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

>>> atom = ATOMClassifier(X, y, experiment="breast_cancer")
>>> atom.run(models=["LR", "RF", "LGB"], n_trials=(0, 0, 10))
```

![MLflow](../img/mlflow.png)

<br>

### DAGsHub integration

ATOM has a build-in integration with [DAGsHub](https://dagshub.com/), a
web platform based on open source tools, optimized for data science and
oriented towards the open source community. To store your mlflow experiments
in a DAGsHub repo, type `dagshub:<experiment_name>` in the `experiment`
parameter (instead of just the experiment's name). If the repo does not
already exist, a new public repo is created.

!!! info
    If you are logged into your DAGsHub account when initializing atom
    with a dagshub experiment, a page on your web browser is automatically
    opened to give access permissions. If not, read [here](https://dagshub.com/docs/integration_guide/mlflow_tracking/#3-set-up-your-credentials)
    how to set up your DAGsHub credentials.

**Example**

```pycon
>>> from atom import ATOMClassifier
>>> from sklearn.datasets import load_breast_cancer

>>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

>>> atom = ATOMClassifier(X, y, experiment="dagshub:breast_cancer")
>>> atom.run(models=["LR", "RF"])
```

![DAGsHub](../img/dagshub.png)

<br>

### Tracked elements

**Tags**<br>
The runs are automatically tagged with the model's full name, the [branch][branches]
from which the model was trained, and the time it took to fit the model.
Add additional custom tags through the [`ht_params`][directclassifier-ht_params]
parameter, e.g. 
`#!python atom.run(["LR", "RF"], ht_params={"tags": {"tag1": 1}})`.

**Parameters**<br>
All parameters used by the estimator at initialization are tracked. Additional
parameters passed to the fit method are **not** tracked.

**Model**<br>
The model's estimator is stored as artifact. The estimator has to be
compatible with the [mlflow.sklearn](https://www.mlflow.org/docs/latest/python_api/mlflow.sklearn.html),
module. This option can be switched off using atom's [`log_model`]
[atomclassifier-log_model] attribute, e.g. `#!python atom.log_model = False`.

**Hyperparameter tuning**<br>
If [hyperparameter tuning][] is performed, every trial is tracked as a nested
run in the model's main run. This option can be switched off using atom's
[`log_ht`][atomclassifier-log_ht] attribute, e.g. `#!python atom.log_ht = False`.
The data and pipeline options are never stored within nested runs.

**Metrics**<br>
All metric results are tracked, not only during training, but also when
the [evaluate][atomclassifier-evaluate] method is called at a later point.
Metrics calculated during [in-training validation][] are also stored.

**Dataset**<br>
The train and test sets used to fit and evaluate the model can be stored
as `.csv` files to the run's artifacts. This option can be switched on
using atom's [`log_data`][atomclassifier-log_data] attribute, e.g.
`#!python atom.log_data = True`.

**Pipeline**<br>
The model's pipeline (returned from the [export_pipeline][atomclassifier-export_pipeline]
method) can be stored as an artifact. This option can be switched on
using atom's [`log_pipeline`][atomclassifier-log_pipeline] attribute,
e.g. `#!python atom.log_pipeline = True`.

**Plots**<br>
By default, plots are stored as `.html` artifacts in all runs corresponding
to the models that are showed in the plot. If the `filename` parameter is
specified, they are stored under that name, else the method's name is used.
This option can be switched off using atom's [`log_plots`][atomclassifier-log_plots]
attribute, e.g. `#!python atom.log_plots = False`.
