<p align="center">
	<img src="https://github.com/tvdboom/ATOM/blob/master/images/logo.png?raw=true" alt="ATOM" title="ATOM" height="170" width="600"/>
</p>

<br>

# Automated Tool for Optimized Modelling
#### A Python package for fast exploration and experimentation of machine learning pipelines
----------------------------------------

Author: [tvdboom](https://github.com/tvdboom) &nbsp;&nbsp;&nbsp;&nbsp; Email: m.524687@gmail.com &nbsp;&nbsp;&nbsp;&nbsp; Documentation: [https://tvdboom.github.io/ATOM/](https://tvdboom.github.io/ATOM/)


#### Repository:
[![Project Status: Active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Conda Recipe](https://img.shields.io/badge/recipe-atom--ml-green.svg)](https://anaconda.org/conda-forge/atom-ml)
[![Python 3.6|3.7|3.8|3.9](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue)](https://www.python.org/downloads/release/python-390/)
[![License: MIT](https://img.shields.io/github/license/tvdboom/ATOM)](https://opensource.org/licenses/MIT)
[![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/atom-ml.svg)](https://anaconda.org/conda-forge/atom-ml)


#### Release info:
[![PyPI version](https://img.shields.io/pypi/v/atom-ml)](https://pypi.org/project/atom-ml/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/atom-ml.svg)](https://anaconda.org/conda-forge/atom-ml)
[![Downloads](https://pepy.tech/badge/atom-ml)](https://pepy.tech/project/atom-ml)


#### Build status:
[![Build Status](https://github.com/tvdboom/ATOM/workflows/ATOM/badge.svg)](https://github.com/tvdboom/ATOM/actions)
[![Azure Pipelines](https://dev.azure.com/conda-forge/feedstock-builds/_apis/build/status/atom-ml-feedstock?branchName=master)](https://dev.azure.com/conda-forge/feedstock-builds/_build/latest?definitionId=10822&branchName=master)
[![codecov](https://codecov.io/gh/tvdboom/ATOM/branch/master/graph/badge.svg)](https://codecov.io/gh/tvdboom/ATOM)


#### Code analysis:
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/tvdboom/ATOM.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/tvdboom/ATOM/context:python)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/tvdboom/ATOM.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/tvdboom/ATOM/alerts/)


<br><br>



Description  
-----------

There is no magic formula in data science that can tell us which type
of machine learning estimator in combination with which pipeline will
perform best for a given raw dataset. Different models are better
suited for different types of data and different types of problems. At
best, you can follow some [rough guide](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)
on how to approach problems with regard to which model to try on your
data, but these are often more confusing than helpful.

During the exploration phase of a project, the data scientist tries
to find the optimal pipeline for his specific use case. This usually
involves applying standard data cleaning steps, creating or selecting
useful features, trying out different models, etc. Testing many
pipelines require many lines of code. These are usually written in one
notebook, which becomes very long and cluttered, or in multiple
notebooks, which makes it harder to compare the results and keep an
overview.

On top of that, refactoring the code for every test can be quite boring
and time-consuming. How many times have you conducted the same action
to pre-process the raw dataset? How many times have you copied code
from an old repository to re-use it in the new use case? Although best
practices tell us to start with a simple model and build up to more
complicated ones, many data scientists just use the model best known to
them in order to avoid the aforementioned issues. This can result in
poor performance (because the model is just not the right one for the
task) or in inefficient management of time and computing resources
(because a simpler/faster model could have achieved a similar
performance).

ATOM is here to help solve these common issues. The package acts as a
wrapper of the whole machine learning pipeline, helping the data
scientist to rapidly perform data analysis, pipeline evaluations and 
model comparisons, all while keeping the code short and simple. Avoid
endless imports and documentation lookups. Avoid rewriting the code for
the same plots over and over again. With just a few lines of code, it's
now possible to perform basic data cleaning steps, select relevant
features and compare the performance of multiple models on a given
dataset. ATOM should be able to help you provide quick insights on
which pipeline performs best for the task at hand and provide an
indication of the feasibility of the ML solution.

It is important to realize that ATOM is not here to replace all the
work a data scientist has to do before getting his model into
production. ATOM doesn't spit out production-ready models just by
tuning some parameters in its API. After helping you determine the
right pipeline, you will most probably need to fine-tune it using
use-case specific features and data cleaning steps in order to
achieve maximum performance.


Example steps taken by ATOM's pipeline:

1. Data Cleaning
	* Handle missing values
	* Encode categorical features
    * Detect and remove outliers
	* Balance the training set
2. Feature engineering
    * Create new non-linear features
	* Remove multi-collinear features
	* Remove features with too low variance
	* Select the most promising features
3. Train and validate multiple models
	* Select hyperparameters using a Bayesian Optimization approach
	* Train and test the models on the provided data
	* Assess the robustness of the output using a bagging algorithm
4. Analyze the results
    * Get the model scores on various metrics
    * Make plots to compare the model performances


<br/><br/>

<div align="center">
    <img src="img/diagram.jpg" alt="diagram" height="300" width="1000"/>
    <figcaption style="padding:0px 0px 0px 500px">Figure 1. Diagram of the possible steps taken by ATOM.</figcaption>
</div>

<br><br>


Installation
------------

| NOTE: Since atom was already taken, download the package under the name `atom-ml`! |
| --- |


Install ATOM's newest release easily via `pip`:

    $ pip install -U atom-ml


or via `conda`:

    $ conda install -c conda-forge atom-ml


<br><br>


Usage  
-----

Call the `ATOMClassifier` or `ATOMRegressor` class and provide the data you want to use:  

    from sklearn.datasets import load_breast_cancer
    from atom import ATOMClassifier
    
    X, y = load_breast_cancer(return_X_y)
    atom = ATOMClassifier(X, y, logger="auto", n_jobs=2, verbose=2)

ATOM has multiple data cleaning methods to help you prepare the data for modelling:

    atom.impute(strat_num="knn", strat_cat="most_frequent", min_frac_rows=0.1)  
    atom.encode(strategy="LeaveOneOut", max_onehot=8, frac_to_other=0.05)  
    atom.feature_selection(strategy="PCA", n_features=12)

Run the pipeline with the models you want to compare:

    atom.run(
        models=["LR", "LDA", "XGB", "lSVM"],
        metric="f1",
        n_calls=25,
        n_initial_points=10,
        bagging=4,
    )

Make plots to analyze the results: 

	atom.plot_results(figsize=(9, 6), filename="bagging_results.png")  
	atom.lda.plot_confusion_matrix(normalize=True, filename="cm.png")

<br><br>



Release history
---------------

### Version 4.3.1

* Added the [status](https://tvdboom.github.io/ATOM/API/ATOM/atomclassifier/#status)
  method to save an overview of atom's branches and models to the logger.
* Fixed a bug where the winning model wasn't displayed correctly.
* Improved documentation.


### Version 4.3.0

* Possibility to [add](https://tvdboom.github.io/ATOM/API/ATOM/atomclassifier/#add) custom transformers to the pipeline.
* The [export_pipeline](https://tvdboom.github.io/ATOM/API/ATOM/atomclassifier/#export-pipeline) utility method exports atom's current pipeline to a sklearn object.
* Use [AutoML](https://tvdboom.github.io/ATOM/user_guide/#automl) to automate the search for an optimized pipeline.
* New magic methods makes atom behave similarly to sklearn's [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html).
* All [training approaches](https://tvdboom.github.io/ATOM/user_guide/#training) can now be combined in the same atom instance.
* New [plot_scatter_matrix](https://tvdboom.github.io/ATOM/API/plots/plot_scatter_matrix), [plot_distribution](https://tvdboom.github.io/ATOM/API/plots/plot_distribution) and [plot_qq](https://tvdboom.github.io/ATOM/API/plots/plot_qq) for data inspection. 
* Complete rework of all the [shap plots](https://tvdboom.github.io/ATOM/user_guide#shap) to be consistent with their new API.
* Improvements for the [Scaler](https://tvdboom.github.io/ATOM/API/data_cleaning/scaler) and [Pruner](https://tvdboom.github.io/ATOM/API/data_cleaning/pruner) classes.
* The acronym for custom models now defaults to the capital letters in the class' \_\_name__.
* Possibility to apply transformations on only a subset of the columns.
* Plots and methods now accept `winner` as model name.
* Fixed a bug where custom metrics didn't show the correct name.
* Fixed a bug where timers were not displayed correctly.
* Further compatibility with deep learning datasets.
* Large refactoring for performance optimization.
* Cleaner output of messages to the logger.
* Plots no longer show a default title.
* Added the <a href="https://tvdboom.github.io/ATOM/examples/automl.html" target="_blank">AutoML</a> example notebook.
* Minor bug fixes.


### Version 4.2.1

* Bug fix where there was memory leakage in [successive halving](https://tvdboom.github.io/ATOM/user_guide/#successive-halving)
  and [train sizing](https://tvdboom.github.io/ATOM/user_guide/#train-sizing) pipelines.
* The [XGBoost](https://xgboost.readthedocs.io/en/latest/python/python_api.html),
  [LightGBM](https://lightgbm.readthedocs.io/en/latest/) and
  [CatBoost](https://catboost.ai/) packages can now be installed through the installer's
  extras_require under the name `models`, e.g. `pip install -U atom-ml[models]`.
* Improved documentation.


### Version 4.2.0

* Possibility to add custom models to the pipeline using [ATOMModel](https://tvdboom.github.io/ATOM/API/ATOM/atommodel).
* Compatibility with [deep learning](https://tvdboom.github.io/ATOM/user_guide/#deep-learning) models.
* New branch system for different data pipelines. Read more in the [user guide](https://tvdboom.github.io/ATOM/user_guide/#branches).
* Use the [canvas](https://tvdboom.github.io/ATOM/API/ATOM/atomclassifier/#canvas) contextmanager to draw multiple plots in one figure.
* New [voting](https://tvdboom.github.io/ATOM/user_guide/#voting) and [stacking](https://tvdboom.github.io/ATOM/user_guide/#stacking) ensemble techniques.
* New [get_class_weight](https://tvdboom.github.io/ATOM/API/ATOM/atomclassifier/#get-class-weight) utility method.
* Added the `sample_weight` parameter to the [score](https://tvdboom.github.io/ATOM/API/predicting/score) method.
* New Sequential Feature Selection strategy for the [FeatureSelector](https://tvdboom.github.io/ATOM/API/feature_engineering/feature_selector).
* New ways to initialize the data in the trainers.
* The `n_rows` parameter in [ATOMLoader](https://tvdboom.github.io/ATOM/API/ATOM/atomloader) is deprecated in
  favour of the new data input formats.
* The `test_size` parameter now also allows integer values.
* Renamed categories to classes to be consistent with sklearn's API.
* The class property now returns a pd.DataFrame of the number of rows per target class
  in the train, test and complete dataset.
* Possibility to add custom parameters to an estimator's fit method through `est_params`.
* [Successive halving](https://tvdboom.github.io/ATOM/user_guide/#successive-halving)
  and [train sizing](https://tvdboom.github.io/ATOM/user_guide/#train-sizing)
  now both allow subsequent runs from atom without losing previous information.
* Bug fix where ATOMLoader wouldn't encode the target column during transformation.
* Added the <a href="https://tvdboom.github.io/ATOM/examples/deep_learning.html" target="_blank">Deep learning</a>, 
  <a href="https://tvdboom.github.io/ATOM/examples/ensembles.html" target="_blank">Ensembles</a> and
  <a href="https://tvdboom.github.io/ATOM/examples/utilities.html" target="_blank">Utilities</a> example notebooks.
* Compatibility with [python 3.9](https://www.python.org/downloads/release/python-390/).


### Version 4.1.0

* Added the `est_params` parameter to customize the parameters passed to every model's
  estimator.
* Following skopt's API, the `n_random_starts` parameter is deprecated in favour of
 `n_initial_points`.
* The [Balancer](https://tvdboom.github.io/ATOM/API/data_cleaning/balancer) class now allows you to use any of the
  strategies from [imblearn](https://imbalanced-learn.readthedocs.io/en/stable/index.html).
* New utility attributes to inspect the dataset.
* Four new models: [CatNB](https://tvdboom.github.io/ATOM/API/models/catnb), [CNB](https://tvdboom.github.io/ATOM/API/models/cnb),
  [ARD](https://tvdboom.github.io/ATOM/API/models/ard) and [RNN](https://tvdboom.github.io/ATOM/API/models/rnn).
* Added the models section to the documentation.
* Small changes in log outputs.
* Bug fixes and performance improvements.


### Version 4.0.1

* Bug fix where the DFS strategy in [FeatureGenerator](https://tvdboom.github.io/ATOM/API/feature_engineering/feature_generator)
  was not deterministic for a fixed random state.
* Bug fix where subsequent runs with the same metric failed.
* Added the [license](https://tvdboom.github.io/ATOM/license) file to the package's installer.
* Typo fixes in documentation.


### Version 4.0.0

* Bayesian optimization package changed from [GpyOpt](http://sheffieldml.github.io/GPyOpt/)
  to [skopt](https://scikit-optimize.github.io/stable/).
* Complete revision of the model's hyperparameters.
* Four [SHAP plots](https://tvdboom.github.io/ATOM/user_guide/#shap) can now be called directly from an ATOM pipeline.
* Two new plots for regression tasks.
* New [plot_pipeline](https://tvdboom.github.io/ATOM/API/plots/plot_pipeline) and `pipeline` attribute to access all transformers. 
* Possibility to determine transformer parameters per method.
* New [calibration](https://tvdboom.github.io/ATOM/API/ATOM/atomclassifier/#calibrate) method and [plot](https://tvdboom.github.io/ATOM/API/plots/plot_calibration).
* Metrics can now be added as scorers or functions with signature metric(y, y_pred, **kwargs).
* Implementation of [multi-metric](https://tvdboom.github.io/ATOM/user_guide/#metric) runs.
* Possibility to choose which metric to plot.
* Early stopping for models that allow in-training evaluation.
* Added the [ATOMLoader](https://tvdboom.github.io/ATOM/API/ATOM/atomloader) function to load saved atom instances
  and directly apply all data transformations.
* The "remove" strategy in the data cleaning parameters is deprecated in favour of "drop".
* Implemented the DFS strategy in [FeatureGenerator](https://tvdboom.github.io/ATOM/API/feature_engineering/feature_generator).
* All training classes now inherit from BaseEstimator.
* Added multiple new example notebooks.
* Tests coverage up to 100%.
* Completely new documentation page.
* Bug fixes and performance improvements.



<br><br>

Documentation
-----------------
  
For further information, please see the project's [documentation](https://tvdboom.github.io/ATOM).
