<p align="center">
	<img src="https://github.com/tvdboom/ATOM/blob/master/images/logo.png?raw=true" alt="ATOM" title="ATOM" height="170" width="600"/>
</p>


## Automated Tool for Optimized Modelling

Author: tvdboom &nbsp;&nbsp;&nbsp;&nbsp; Email: m.524687@gmail.com &nbsp;&nbsp;&nbsp;&nbsp; Documentation: [https://tvdboom.github.io/ATOM/](https://tvdboom.github.io/ATOM/)


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
-----------------

There is no magic formula in data science that can tell us which type of
machine learning algorithm will perform best for a specific use-case.
Different models are better suited for different types of data and
different problems. At best, you can follow some
[rough guide](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)
on how to approach problems with regard to which model to try on your
data, but these are often more confusing than helpful. Best practices
tell us to start with a simple model (e.g. linear regression) and build
up to more complicated models (e.g. linear regression -> random forest
-> multi-layer perceptron) if you are not satisfied with the results.
Unfortunately, different models require different data cleaning steps,
different type/amount of features, tuning a new set of hyperparameters,
etc. Refactoring the code for this purpose can be quite boring and
time-consuming. Because of this, many data scientists end up just using
the model best known to them and fine-tuning this particular model
without ever trying different ones. This can result in poor performance
(because the model is just not the right one for the task) or in poor
time management (because you could have achieved a similar performance
with a simpler/faster model).

ATOM is made to help you solve these issues. With just a few lines of code,
you can perform basic data cleaning steps, select relevant features and
compare the performance of multiple models on a given dataset. ATOM should
be able to provide quick insights on which algorithms perform best for the
task at hand and provide an indication of the feasibility of the ML solution.

It is important to realize that ATOM is not here to replace all the work a
data scientist has to do before getting his model into production. ATOM
doesn't spit out production-ready models just by tuning some parameters in
its API. After helping you to determine the right model, you will most
probably need to fine-tune it using use-case specific features and data
cleaning steps in order to achieve maximum performance.

So, this sounds a bit like AutoML, how is ATOM different than 
[auto-sklearn](https://automl.github.io/auto-sklearn/master/) or
[TPOT](http://epistasislab.github.io/tpot/)? Well, ATOM does AutoML in
the sense that it helps you find the best model for a specific task, but
contrary to the aforementioned packages, it does not actively search for
the best model. It simply runs all of them and let you pick the one that
you think suites the task best. AutoML packages are often black boxes: if
you provide data, it will magically return a working model. Although it
works great, they often produce complicated pipelines with low explainability.
This is hard to sell to the business. In this, ATOM excels. Every step of
the pipeline is accounted for, and using the provided plotting methods,
it’s easy to demonstrate why a model is a better or worse choice than the other.

Example steps taken by ATOM's pipeline:

1. Data Cleaning
	* Handle missing values
	* Encode categorical features
    * Remove outliers
	* Balance the dataset
2. Feature engineering
    * Create new non-linear features
	* Remove multi-collinear features
	* Remove features with too low variance
	* Select the most promising features based on a statistical test
3. Train and validate multiple models
	* Select hyperparameters using a Bayesian Optimization approach
	* Train and test the models on the provided data
	* Perform bagging to assess the robustness of the output
4. Analyze the results
    * Get the model scores on various metrics
    * Make plots to compare the model performances


<br/><br/>

<p align="center">
	<img src="https://github.com/tvdboom/ATOM/blob/master/images/diagram.jpg?raw=true" alt="diagram" title="diagram" width="900" height="300" />
</p>

<br><br>


Installation
-----------------

| NOTE: Since atom was already taken, download the package under the name `atom-ml`! |
| --- |


Install ATOM's newest release easily via `pip`:

    $ pip install -U atom-ml


or via `conda`:

    $ conda install -c conda-forge atom-ml


<br><br>


Usage  
-----------------

Call the `ATOMClassifier` or `ATOMRegressor` class and provide the data you want to use:  

    from sklearn.datasets import load_breast_cancer
    from atom import ATOMClassifier
    
    X, y = load_breast_cancer(return_X_y)
    atom = ATOMClassifier(X, y, logger="auto", n_jobs=2, verbose=2)

ATOM has multiple data cleaning methods to help you prepare the data for modelling:

    atom.impute(strat_num="knn", strat_cat="most_frequent",  min_frac_rows=0.1)  
    atom.encode(strategy="Target", max_onehot=8, frac_to_other=0.05)  
    atom.feature_selection(strategy="PCA", n_features=12)

Run the pipeline with the models you want to compare:

    atom.run(
        models=["LR", "LDA", "XGB", "lSVM"],
        metric="f1",
        n_calls=25,
        n_initial_points=10,
        bagging=4
    )

Make plots to analyze the results: 

	atom.plot_results(figsize=(9, 6), filename="bagging_results.png")  
	atom.LDA.plot_confusion_matrix(normalize=True, filename="cm.png")

<br><br>



Release history
-----------------

### Version 4.2.1 - 29 December 2020

* Bug fix where there was memory leakage in [successive halving](https://tvdboom.github.io/ATOM/user_guide/#successive-halving)
  and [train sizing](https://tvdboom.github.io/ATOM/user_guide/#train-sizing) pipelines.
* The [XGBoost](https://xgboost.readthedocs.io/en/latest/python/python_api.html),
  [LightGBM](https://lightgbm.readthedocs.io/en/latest/) and
  [CatBoost](https://catboost.ai/) packages can now be installed through the installer's
  extras_require under the name `models`, e.g. `pip install -U atom-ml[models]`.
* Improved documentation.


### Version 4.2.0 - 28 December 2020

* Possibility to add custom models to the pipeline using [ATOMModel](https://tvdboom.github.io/ATOM/API/ATOM/atommodel).
* Compatibility with [deep learning](https://tvdboom.github.io/ATOM/user_guide/#deep-learning) models.
* New branch system for different data pipelines. Read more in the [user guide](https://tvdboom.github.io/ATOM/user_guide/#data-pipelines).
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


### Version 4.1.0 - 16 October 2020

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


### Version 4.0.1 - 29 September 2020

* Bug fix where the DFS strategy in [FeatureGenerator](https://tvdboom.github.io/ATOM/API/feature_engineering/feature_generator)
  was not deterministic for a fixed random state.
* Bug fix where subsequent runs with the same metric failed.
* Added the [license](https://tvdboom.github.io/ATOM/license) file to the package's installer.
* Typo fixes in documentation.


### Version 4.0.0 - 28 September 2020

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
  
For further information about ATOM, please see the project's [documentation](https://tvdboom.github.io/ATOM).
