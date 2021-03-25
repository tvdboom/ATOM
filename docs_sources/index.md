<div align="center">
    <img src="img/logo.png" alt="ATOM" height="170" width="600"/>
</div>
<br><br>

# Automated Tool for Optimized Modelling
----------------------------------------
#### A Python package for fast exploration of machine learning pipelines

<br><br>

During the exploration phase of a machine learning project, a data
scientist tries to find the optimal pipeline for his specific use case.
This usually involves applying standard data cleaning steps, creating
or selecting useful features, trying out different models, etc. Testing
multiple pipelines requires many lines of code, and writing it all in
the same notebook often makes it long and cluttered. On the other hand,
using multiple notebooks makes it harder to compare the results and to
keep an overview. On top of that, refactoring the code for every test
can be time-consuming. How many times have you conducted the same action
to pre-process a raw dataset? How many times have you copy-and-pasted
code from an old repository to re-use it in a new use case?

ATOM is here to help solve these common issues. The package acts as
a wrapper of the whole machine learning pipeline, helping the data
scientist to rapidly find a good model for his problem. Avoid
endless imports and documentation lookups. Avoid rewriting the same
code over and over again. With just a few lines of code, it's now
possible to perform basic data cleaning steps, select relevant
features and compare the performance of multiple models on a given
dataset, providing quick insights on which pipeline performs best
for the task at hand.

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

<br><br><br><br>



# Release history
-----------------

### Version 4.4.0

* The [drop](./API/ATOM/atomclassifier/#drop) method now allows the user
  to drop columns as part of the pipeline.
* It is now possible to add data transformations as function to the pipeline
  through the [apply](./API/ATOM/atomclassifier/#apply) method.
* Added the [status](./API/ATOM/atomclassifier/#status) method to save an
  overview of atom's branches and models to the logger.
* Improved the output messages for the [Imputer](./API/data_cleaning/imputer) class.
* The dataset's columns can now be called directly from atom.
* The [distribution](./API/ATOM/atomclassifier/#distribution) and
  [plot_distribution](./API/plots/plot_distribution) methods now ignore missing
  values instead of raising an exception.
* Fixed a bug where transformations failed when columns were added after
  initializing the pipeline.
* Fixed a bug where the [Cleaner](./API/data_cleaning/cleaner) class didn't drop
  columns with only missing values for `minimum_cardinality=True`.
* Fixed a bug where the winning model wasn't displayed correctly.
* Refactored the way transformers are added or removed from predicting methods.
* Improved documentation.


### Version 4.3.0

* Possibility to [add](./API/ATOM/atomclassifier/#add) custom transformers to the pipeline.
* The [export_pipeline](./API/ATOM/atomclassifier/#export-pipeline) utility method exports atom's current pipeline to a sklearn object.
* Use [AutoML](./user_guide/#automl) to automate the search for an optimized pipeline.
* New magic methods makes atom behave similarly to sklearn's [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html).
* All [training approaches](./user_guide/#training) can now be combined in the same atom instance.
* New [plot_scatter_matrix](./API/plots/plot_scatter_matrix), [plot_distribution](./API/plots/plot_distribution) and [plot_qq](./API/plots/plot_qq) for data inspection.
* Complete rework of all the [shap plots](./user_guide#shap) to be consistent with their new API.
* Improvements for the [Scaler](./API/data_cleaning/scaler) and [Pruner](./API/data_cleaning/pruner) classes.
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

* Bug fix where there was memory leakage in [successive halving](./user_guide/#successive-halving)
  and [train sizing](./user_guide/#train-sizing) pipelines.
* The [XGBoost](https://xgboost.readthedocs.io/en/latest/python/python_api.html),
  [LightGBM](https://lightgbm.readthedocs.io/en/latest/) and
  [CatBoost](https://catboost.ai/) packages can now be installed through the installer's
  extras_require under the name `models`, e.g. `pip install -U atom-ml[models]`.
* Improved documentation.


### Version 4.2.0

* Possibility to add custom models to the pipeline using [ATOMModel](./API/ATOM/atommodel).
* Compatibility with [deep learning](./user_guide/#deep-learning) models.
* New branch system for different data pipelines. Read more in the [user guide](./user_guide/#branches).
* Use the [canvas](./API/ATOM/atomclassifier/#canvas) contextmanager to draw multiple plots in one figure.
* New [voting](./user_guide/#voting) and [stacking](./user_guide/#stacking) ensemble techniques.
* New [get_class_weight](./API/ATOM/atomclassifier/#get-class-weight) utility method.
* New Sequential Feature Selection strategy for the [FeatureSelector](./API/feature_engineering/feature_selector).
* Added the `sample_weight` parameter to the [score](./API/predicting/score) method.
* New ways to initialize the data in the `training` instances.
* The `n_rows` parameter in [ATOMLoader](./API/ATOM/atomloader) is deprecated in
  favour of the new data input formats.
* The `test_size` parameter now also allows integer values.
* Renamed categories to classes to be consistent with sklearn's API.
* The class property now returns a pd.DataFrame of the number of rows per target class
  in the train, test and complete dataset.
* Possibility to add custom parameters to an estimator's fit method through `est_params`.
* [Successive halving](./user_guide/#successive-halving) and [train sizing](./user_guide/#train-sizing)
  now both allow subsequent runs from atom without losing previous information.
* Bug fix where ATOMLoader wouldn't encode the target column during transformation.
* Added the <a href="./examples/deep_learning.html" target="_blank">Deep learning</a>, 
  <a href="./examples/ensembles.html" target="_blank">Ensembles</a> and
  <a href="./examples/utilities.html" target="_blank">Utilities</a> example notebooks.
* Compatibility with [python 3.9](https://www.python.org/downloads/release/python-390/).


### Version 4.1.0

* Added the `est_params` parameter to customize the parameters passed to every model's
  estimator.
* Following skopt's API, the `n_random_starts` parameter is deprecated in favour of
 `n_initial_points`.
* The [Balancer](./API/data_cleaning/balancer) class now allows you to use any of the
  strategies from [imblearn](https://imbalanced-learn.readthedocs.io/en/stable/index.html).
* New utility attributes to inspect the dataset.
* Four new models: [CatNB](./API/models/catnb), [CNB](./API/models/cnb),
  [ARD](./API/models/ard) and [RNN](./API/models/rnn).
* Added the models section to the documentation.
* Small changes in log outputs.
* Bug fixes and performance improvements.


### Version 4.0.1

* Bug fix where the DFS strategy in [FeatureGenerator](./API/feature_engineering/feature_generator)
  was not deterministic for a fixed random state.
* Bug fix where subsequent runs with the same metric failed.
* Added the [license](./license) file to the package's installer.
* Typo fixes in documentation.


### Version 4.0.0

* Bayesian optimization package changed from [GpyOpt](http://sheffieldml.github.io/GPyOpt/)
  to [skopt](https://scikit-optimize.github.io/stable/).
* Complete revision of the model's hyperparameters.
* Four [SHAP plots](./user_guide/#shap) can now be called directly from an ATOM pipeline.
* Two new plots for regression tasks.
* New [plot_pipeline](./API/plots/plot_pipeline) and `pipeline` attribute to access all transformers. 
* Possibility to determine transformer parameters per method.
* New [calibration](./API/ATOM/atomclassifier/#calibrate) method and [plot](./API/plots/plot_calibration).
* Metrics can now be added as scorers or functions with signature metric(y, y_pred, **kwargs).
* Implementation of [multi-metric](./user_guide/#metric) runs.
* Possibility to choose which metric to plot.
* Early stopping for models that allow in-training evaluation.
* Added the [ATOMLoader](./API/ATOM/atomloader) function to load saved atom instances
  and directly apply all data transformations.
* The "remove" strategy in the data cleaning parameters is deprecated in favour of "drop".
* Implemented the DFS strategy in [FeatureGenerator](./API/feature_engineering/feature_generator).
* All training classes now inherit from BaseEstimator.
* Added multiple new example notebooks.
* Tests coverage up to 100%.
* Completely new documentation page.
* Bug fixes and performance improvements.


<br><br><br>

# Content
---------

- [Getting started](./getting_started)
- [User guide](./user_guide)
- API
    - ATOM
        - [ATOMClassifier](./API/ATOM/atomclassifier)
        - [ATOMRegressor](./API/ATOM/atomregressor)
        - [ATOMLoader](./API/ATOM/atomloader)
        - [ATOMModel](./API/ATOM/atommodel)
    - Data cleaning
        - [Scaler](./API/data_cleaning/scaler)
        - [Cleaner](./API/data_cleaning/cleaner)
        - [Imputer](./API/data_cleaning/imputer)
        - [Encoder](./API/data_cleaning/encoder)
        - [Pruner](./API/data_cleaning/pruner)
        - [Balancer](./API/data_cleaning/balancer)
    - Feature engineering
        - [FeatureGenerator](./API/feature_engineering/feature_generator)
        - [FeatureSelector](./API/feature_engineering/feature_selector)
    - Training
        - Direct
            - [DirectClassifier](./API/training/directclassifier)
            - [DirectRegressor](./API/training/directregressor)
        - SuccessiveHalving
            - [SuccessiveHalvingClassifier](./API/training/successivehalvingclassifier)
            - [SuccessiveHalvingClassifier](./API/training/successivehalvingregressor)
        - TrainSizing
            - [TrainSizingClassifier](./API/training/trainsizingclassifier)
            - [TrainSizingRegressor](./API/training/trainsizingregressor)
    - Models
        - [Gaussian Process](./API/models/gp)
        - [Gaussian Naive Bayes](./API/models/gnb)
        - [Multinomial Naive Bayes](./API/models/mnb)
        - [Bernoulli Naive Bayes](./API/models/bnb)
        - [Categorical Naive Bayes](./API/models/catnb)
        - [Complement Naive Bayes](./API/models/cnb)
        - [Ordinary Least Squares](./API/models/ols)
        - [Ridge](./API/models/ridge)
        - [Lasso](./API/models/lasso)
        - [Elastic Net](./API/models/en)
        - [Bayesian Ridge](./API/models/br)
        - [Automated Relevance Determination](./API/models/ard)
        - [Logistic Regression](./API/models/lr)
        - [Linear Discriminant Analysis](./API/models/lda)
        - [Quadratic Discriminant Analysis](./API/models/qda)
        - [K-Nearest Neighbors](./API/models/knn)
        - [Radius Nearest Neighbors](./API/models/rnn)
        - [Decision Tree](./API/models/tree)
        - [Bagging](./API/models/bag)
        - [Extra-Trees](./API/models/et)
        - [Random Forest](./API/models/rf)
        - [AdaBoost](./API/models/adab)
        - [Gradient Boosting Machine](./API/models/gbm)
        - [XGBoost](./API/models/xgb)
        - [LightGBM](./API/models/lgb)
        - [CatBoost](./API/models/catb)
        - [Linear-SVM](./API/models/lsvm)
        - [Kernel-SVM](./API/models/ksvm)
        - [Passive Aggressive](./API/models/pa)
        - [Stochastic Gradient Descent](./API/models/sgd)
        - [Multi-layer Perceptron](./API/models/mlp)
    - Predicting
          - [transform](./API/predicting/transform)
          - [predict](./API/predicting/predict)
          - [predict_proba](./API/predicting/predict_proba)
          - [predict_log_proba](./API/predicting/predict_log_proba)
          - [decision_function](./API/predicting/decision_function)
          - [score](./API/predicting/score)
    - Plots
        - [plot_correlation](./API/plots/plot_correlation)
        - [plot_scatter_matrix](./API/plots/plot_scatter_matrix)
        - [plot_distribution](./API/plots/plot_distribution)
        - [plot_qq](./API/plots/plot_qq)
        - [plot_pipeline](./API/plots/plot_pipeline)
        - [plot_pca](./API/plots/plot_pca)
        - [plot_components](./API/plots/plot_components)
        - [plot_rfecv](./API/plots/plot_rfecv)
        - [plot_successive_halving](./API/plots/plot_successive_halving)
        - [plot_learning_curve](./API/plots/plot_learning_curve)
        - [plot_results](./API/plots/plot_results)
        - [plot_bo](./API/plots/plot_bo)
        - [plot_evals](./API/plots/plot_evals)
        - [plot_roc](./API/plots/plot_roc)
        - [plot_prc](./API/plots/plot_prc)
        - [plot_permutation_importance](./API/plots/plot_permutation_importance)
        - [plot_feature_importance](./API/plots/plot_feature_importance)
        - [plot_partial_dependence](./API/plots/plot_partial_dependence)
        - [plot_errors](./API/plots/plot_errors)
        - [plot_residuals](./API/plots/plot_residuals)
        - [plot_confusion_matrix](./API/plots/plot_confusion_matrix)
        - [plot_threshold](./API/plots/plot_threshold)
        - [plot_probabilities](./API/plots/plot_probabilities)
        - [plot_calibration](./API/plots/plot_calibration)
        - [plot_gains](./API/plots/plot_gains)
        - [plot_lift](./API/plots/plot_lift)
        - [bar_plot](./API/plots/bar_plot)
        - [beeswarm_plot](./API/plots/beeswarm_plot)
        - [decision_plot](./API/plots/decision_plot)
        - [force_plot](./API/plots/force_plot)
        - [heatmap_plot](./API/plots/heatmap_plot)
        - [scatter_plot](./API/plots/scatter_plot)
        - [waterfall_plot](./API/plots/waterfall_plot)
- Examples
    - <a href="./examples/automl.html" target="_blank">AutoML</a>
    - <a href="./examples/binary_classification.html" target="_blank">Binary classification</a>
    - <a href="./examples/calibration.html" target="_blank">Calibration</a>
    - <a href="./examples/deep_learning.html" target="_blank">Deep learning</a>
    - <a href="./examples/early_stopping.html" target="_blank">Early stopping</a>
    - <a href="./examples/ensembles.html" target="_blank">Ensembles</a>
    - <a href="./examples/feature_engineering.html" target="_blank">Feature engineering</a>
    - <a href="./examples/imbalanced_datasets.html" target="_blank">Imbalanced datasets</a>
    - <a href="./examples/multiclass_classification.html" target="_blank">Multiclass classification</a>
    - <a href="./examples/multi_metric.html" target="_blank">Multi-metric runs</a>
    - <a href="./examples/regression.html" target="_blank">Regression</a>
    - <a href="./examples/successive_halving.html" target="_blank">Successive halving</a>
    - <a href="./examples/train_sizing.html" target="_blank">Train sizing</a>
    - <a href="./examples/utilities.html" target="_blank">Utilities</a>
- [FAQ](./faq)
- [Dependencies](./dependencies)
- [License](./license)
