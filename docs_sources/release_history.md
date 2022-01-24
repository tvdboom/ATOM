# Release history
-----------------

<a name="v4110"></a>
### Version 4.11.0

* Full support for parse matrices. Read more in the [user guide](../user_guide/data_management/#sparse-matrices).
* The [shrink](../API/ATOM/atomclassifier/#shrink) method also handles
  sparse features and can convert dense datasets to sparse format.
* Added three new linear models: [Lars](../API/models/lars), [Huber](../API/models/huber)
  and [Perc](../API/models/perc).
* Custom dimensions for the BO can now be shared across models using the
  key 'all' in `bo_params["dimensions"]`.
* Assign hyperparameters to tune with the predefined dimensions through
  `bo_params["dimensions"]`.
* It's now possible to tune a custom number of layers for the [MLP](../API/models/mlp)
  model.
* If multiple BO calls share the best score, the one with the shortest
  training time is selected as winner (instead of the first).
* Fixed a bug where the BO could fail when custom dimensions where defined
  and `n_initial_points=1`.
* Fixed a bug where [FeatureSelector](../API/feature_engineering/feature_selector)
  could fail after repeated calls to fit.
* Fixed a bug where [FeatureGenerator](../API/feature_engineering/feature_generator)
  didn't pass the correct data indices to its output.
* Performance improvements for the custom pipeline.
* Minor documentation fixes.


<a name="v4100"></a>
### Version 4.10.0

* Added the `holdout` data set to have an extra way of assessing a
  model's performance on a completely independent dataset. Read more
  in the [user_guide](../user_guide/data_management/#data-sets).
* Complete rework of the [ensemble](../user_guide/models/#ensembles) models.
* Support for dataframe indexing. Read more in the [user guide](../user_guide/data_management/#indexing).
* New [plot_parshap](../API/plots/plot_parshap) plot to detect overfitting
  features.
* The new [dashboard](../API/models/gnb/#dashboard) method makes analyzing
  the models even easier using a dashboard app.
* The [plot_feature_importance](../API/plots/plot_feature_importance)
  plot now also accepts estimators with coefficients.
* Added the [transform](../API/models/gnb/#transform) method for models.
* Added the `threshold` parameter to the [evaluate](../API/ATOM/atomclassifier/#evaluate)
  method.
* The `reset_predictions` method is deprecated in favour of the new
  [clear](../API/ATOM/atomclassifier/#clear) method.
* Refactor of the model's [full_train](../API/models/gnb/#full-train) method.
* The [merge](../API/training/directclassifier/#merge) method is available
  for all trainers.
* Improvements in the trainer's pipeline.
* Training scores are now also saved to the mlflow run.
* Trying to change the data in a branch after fitting a model with it now
  raises an exception.
* Fixed a bug where the columns of array inputs were not ordered correctly.
* Fixed a bug where branches did not correctly act case-insensitive.
* Fixed a bug where the [export_pipeline](../API/models/gnb/#export-pipeline)
  method for models would not export the transformers in the correct branch.


<a name="v491"></a>
### Version 4.9.1

* Changed the default cross-validation for [hyperparameter tuning](../user_guide/training/#hyperparameter-tuning)
  from 5 to 1 to avoid errors with [deep learning models](../user_guide/models/#deep-learning).
* Added clearer exception messages when a model's run failed.
* Fixed a bug where custom dimensions didn't show during
  [hyperparameter tuning](../user_guide/training/#hyperparameter-tuning).
* Documentation improvements.


<a name="v490"></a>
### Version 4.9.0

* Drop support of [Python 3.6](https://www.python.org/downloads/release/python-360/).
* Added the [HistGBM](../API/models/hgbm) model.
* Improved print layout for [hyperparameter tuning](../user_guide/training/#hyperparameter-tuning).
* The new [available_models](../API/ATOM/atomclassifier/#available-models)
  method returns an overview of the available predefined models.
* The [calibrate](../API/models/gnb/#calibrate) and [cross_validate](../API/models/gnb/#cross-validate)
  methods can no longer be accessed from the trainers.
* The `pipeline` parameter for the [prediction methods](../user_guide/predicting)
  is deprecated.
* Improved visualization of the [plot_rfecv](../API/plots/plot_rfecv),
  [plot_successive_halving](../API/plots/plot_successive_halving) and
  [plot_learning_curve](../API/plots/plot_learning_curve) methods.
* Sparse matrices are now accepted as input.
* Duplicate BO calls are no longer calculated.
* Improvement in performance of the [RNN](../API/models/rnn) model.
* Refactor of the model's `bo` attribute.
* Predefined hyperparameters have been updated to be consistent with sklearn's API.
* Fixed a bug where custom scalers were ignored by the models.
* Fixed a bug where the BO of certain models would crash with custom hyperparameters.
* Fixed a bug where duplicate column names could be generated from a custom transformer.
* Documentation improvements.


<a name="v480"></a>
### Version 4.8.0

* The [Encoder](../API/data_cleaning/encoder) class now directly handles
  unknown categories encountered during fitting.
* The [Balancer](../API/data_cleaning/balancer) and [Encoder](../API/data_cleaning/encoder)
  classes now accept custom estimators for the `strategy` parameter.
* The new [merge](../API/ATOM/atomclassifier/#merge) method enables the
  user to merge multiple atom instances into one.
* The dtype shrinking is moved from atom's initializers to the
  [shrink](../API/ATOM/atomclassifier/#shrink) method.
* ATOM's [custom pipeline](../API/ATOM/ATOMClassifier/#export-pipeline) now
  handles transformers fitted on a subset of the dataset.
* The `column` parameter in the [distribution](../API/ATOM/atomclassifier/#distribution)
  method is renamed to `columns` for continuity of the API.
* The `mae` criterion for the [GBM](../API/models/gbm) model hyperparameter
  tuning is deprecated to be consistent with sklearn's API.
* Branches are now case-insensitive.
* Renaming a branch using an existing name now raises an exception.
* Fixed a bug where columns of type `category` broke the [Imputer](../API/data_cleaning/imputer)
  class.
* Fixed a bug where predictions of the [Stacking](../user_guide/models/#stacking)
  ensemble crashed for branches with multiple transformers.
* The tables in the documentation now adapt to dark mode.


<a name="v473"></a>
### Version 4.7.3

* Fixed a bug where the conda-forge recipe couldn't install properly.


<a name="v472"></a>
### Version 4.7.2

* Fixed a bug where the pipeline failed for custom transformers that
  returned sparse matrices.
* Package requirements files are added to the installer.


<a name="v471"></a>
### Version 4.7.1

* Fixed a bug where the pip installer failed.
* Fixed a bug where categorical columns also selected datetime columns.


<a name="v470"></a>
### Version 4.7.0

* Launched our new [slack](https://join.slack.com/t/atom-alm7229/shared_invite/zt-upd8uc0z-LL63MzBWxFf5tVWOGCBY5g) channel!
* The new [FeatureExtractor](../API/feature_engineering/feature_extractor) class
  extracts useful features from datetime columns.
* The new [plot_det](../API/plots/plot_det) method plots a binary classifier's
  detection error tradeoff curve. 
* The [partial dependence plot](../API/plots/plot_partial_dependence) is
  able to draw Individual Conditional Expectation (ICE) lines.
* The full traceback of exceptions encountered during training are now
  saved to the logger.
* [ATOMClassifier](../API/ATOM/atomclassifier) and [ATOMRegressor](../API/ATOM/atomregressor)
  now convert the dtypes of the input data to the minimal allowed type
  for memory efficiency.
* The scoring method is renamed to [evaluate](../API/ATOM/atomclassifier/#evaluate)
  to clarify its purpose.
* The `column` parameter in the [apply](../API/ATOM/atomclassifier/#apply) method
  is renamed to `columns` for continuity of the API.
* Minor documentation improvements.


<a name="v460"></a>
### Version 4.6.0

* Added the [full_train](../API/models/gnb/#full-train) method to retrieve
  an estimator trained on the complete dataset.
* The [score](../API/predicting/score) method is now also able to calculate
  custom metrics on new data.
* Refactor of the [Imputer](../API/data_cleaning/imputer) class. 
* Refactor of the [Encoder](../API/data_cleaning/encoder) class to avoid errors
  for unknown classes and allow the input of missing values.
* The [clean](../API/ATOM/atomclassifier/#clean) method no longer automatically
  encodes the target column for regression tasks.
* Creating a branch using a models' acronym as name now raises an exception.
* Fixed a bug where [CatBoost](../API/models/catb) failed when `early_stopping` < 1.
* Fixed a bug where created pipelines had duplicated names.


<a name="v450"></a>
### Version 4.5.0

* Support of NLP pipelines. Read more in the [user guide](../user_guide/nlp).
* Integration of [mlflow](https://www.mlflow.org/) to track all models in the
  pipeline. Read more in the [user guide](../user_guide/logging/#tracking).
* The new [Gauss](../API/data_cleaning/gauss) class transforms features to
  a more Gaussian-like distribution.
* New [cross_validate](../API/ATOM/atomclassifier/#cross-validate) method to
  evaluate the robustness of a pipeline using cross_validation.
* New [reset](../API/ATOM/atomclassifier/#reset) method to go back to atom's
  initial state.
* Added the [Dummy](../API/models/dummy) model to compare other models with a
  simple baseline.
* New [plot_wordcloud](../API/plots/plot_wordcloud) and [plot_ngrams](../API/plots/plot_ngrams)
  methods for text visualization.
* Plots now can return the figure object when `display=None`.
* The [Pruner](../API/data_cleaning/pruner) class can now able to drop outliers
  based on the selection of multiple strategies.
* The new `shuffle` parameter in atom's initializer determines whether to
  shuffle the dataset.
* The trainers no longer require you to specify a model using the `models`
  parameter. If left to default, all [predefined models](../user_guide/models/#predefined-models)
  for that task are used.
* The [apply](../API/ATOM/atomclassifier/#apply) method now accepts args and
  kwargs for the function.
* Refactor of the [evaluate](../API/ATOM/atomclassifier/#evaluate) method.
* Refactor of the [export_pipeline](../API/ATOM/atomclassifier/#export-pipeline) method.
* The parameters in the [Cleaner](../API/data_cleaning/pruner) class have
  been refactored to better describe their function.
* The `train_sizes` parameter in [train_sizing](../API/ATOM/atomclassifier/#train-sizing)
  now accepts integer values to automatically create equally distributed
  splits in the training set.
* Refactor of [plot_pipeline](../API/plots/plot_pipeline) to show models in the
  diagram as well.
* Refactor of the `bagging` parameter to the (more appropriate) name `n_bootstrap`.
* New option to exclude columns from a transformer adding `!` before their name.
* Fixed a bug where the [Pruner](../API/data_cleaning/pruner) class failed if
  there were categorical columns in the dataset.
* Completely reworked documentation website.


<a name="v440"></a>
### Version 4.4.0

* The [drop](../API/ATOM/atomclassifier/#drop) method now allows the user
  to drop columns as part of the pipeline.
* New [apply](../API/ATOM/atomclassifier/#apply) method to perform data transformations
  as function to the pipeline
* Added the [status](../API/ATOM/atomclassifier/#status) method to save an
  overview of atom's branches and models to the logger.
* Improved the output messages for the [Imputer](../API/data_cleaning/imputer) class.
* The dataset's columns can now be called directly from atom.
* The [distribution](../API/ATOM/atomclassifier/#distribution) and [plot_distribution](../API/plots/plot_distribution)
  methods now ignore missing values.
* Fixed a bug where transformations could fail when columns were added to the
  dataset after initializing the pipeline.
* Fixed a bug where the [Cleaner](../API/data_cleaning/cleaner) class didn't drop
  columns consisting entirely of missing values when `drop_min_cardinality=True`.
* Fixed a bug where the winning model wasn't displayed correctly.
* Refactored the way transformers are added or removed from predicting methods.
* Improved documentation.


<a name="v430"></a>
### Version 4.3.0

* Possibility to [add](../API/ATOM/atomclassifier/#add) custom transformers to the pipeline.
* The [export_pipeline](../API/ATOM/atomclassifier/#export-pipeline) utility method exports atom's current pipeline to a sklearn object.
* Use [AutoML](../user_guide/data_management/#automl) to automate the search for an optimized pipeline.
* New magic methods makes atom behave similarly to sklearn's [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html).
* All [training approaches](../user_guide/training) can now be combined in the same atom instance.
* New [plot_scatter_matrix](../API/plots/plot_scatter_matrix), [plot_distribution](../API/plots/plot_distribution) and [plot_qq](../API/plots/plot_qq) plots for data inspection.
* Complete rework of all the [shap plots](../user_guide/plots/#shap) to be consistent with their new API.
* Improvements for the [Scaler](../API/data_cleaning/scaler) and [Pruner](../API/data_cleaning/pruner) classes.
* The acronym for custom models now defaults to the capital letters in the class' \_\_name__.
* Possibility to apply transformations on only a subset of the columns.
* Plots and methods now accept `winner` as model name.
* Fixed a bug where custom metrics didn't show the correct name.
* Fixed a bug where timers were not displayed correctly.
* Further compatibility with deep learning datasets.
* Large refactoring for performance optimization.
* Cleaner output of messages to the logger.
* Plots no longer show a default title.
* Added the [AutoML](../examples/automl) example notebook.
* Minor bug fixes.


<a name="v421"></a>
### Version 4.2.1

* Bug fix where there was memory leakage in [successive halving](../user_guide/training/#successive-halving)
  and [train sizing](../user_guide/training/#train-sizing) pipelines.
* The [XGBoost](https://xgboost.readthedocs.io/en/latest/python/python_api.html),
  [LightGBM](https://lightgbm.readthedocs.io/en/latest/) and
  [CatBoost](https://catboost.ai/) packages can now be installed through the installer's
  extras_require under the name `models`, e.g. `pip install -U atom-ml[models]`.
* Improved documentation.


<a name="v420"></a>
### Version 4.2.0

* Possibility to add custom models to the pipeline using [ATOMModel](../API/ATOM/atommodel).
* Compatibility with [deep learning](../user_guide/models/#deep-learning) models.
* New branch system for different data pipelines. Read more in the [user guide](../user_guide/data_management/#branches).
* Use the [canvas](../API/ATOM/atomclassifier/#canvas) contextmanager to draw multiple plots in one figure.
* New [voting](../user_guide/models/#voting) and [stacking](../user_guide/models/#stacking) ensemble techniques.
* New [get_class_weight](../API/ATOM/atomclassifier/#get-class-weight) utility method.
* New Sequential Feature Selection strategy for the [FeatureSelector](../API/feature_engineering/feature_selector).
* Added the `sample_weight` parameter to the [score](../API/predicting/score) method.
* New ways to initialize the data in the `training` instances.
* The `n_rows` parameter in [ATOMLoader](../API/ATOM/atomloader) is deprecated in
  favour of the new input formats.
* The `test_size` parameter now also allows integer values.
* Renamed categories to classes to be consistent with sklearn's API.
* The class property now returns a pd.DataFrame of the number of rows per target class
  in the train, test and complete dataset.
* Possibility to add custom parameters to an estimator's fit method through `est_params`.
* The [successive halving](../user_guide/training/#successive-halving) and [train sizing](../user_guide/training/#train-sizing)
  approaches now both allow subsequent runs from atom without losing the
  information from previous runs.
* Bug fix where ATOMLoader wouldn't encode the target column during transformation.
* Added the [Deep learning](../examples/deep_learning), 
  [Ensembles](../examples/ensembles) and
  [Utilities](../examples/utilities) example notebooks.
* Compatibility with [python 3.9](https://www.python.org/downloads/release/python-390/).


<a name="v410"></a>
### Version 4.1.0

* New `est_params` parameter to customize the parameters in every model's estimator.
* Following skopt's API, the `n_random_starts` parameter to specify the number
  of random trials is deprecated in favour of `n_initial_points`.
* The [Balancer](../API/data_cleaning/balancer) class now allows you to use any of the
  strategies from [imblearn](https://imbalanced-learn.readthedocs.io/en/stable/index.html).
* New utility attributes to inspect the dataset.
* Four new models: [CatNB](../API/models/catnb), [CNB](../API/models/cnb),
  [ARD](../API/models/ard) and [RNN](../API/models/rnn).
* Added the models section to the documentation.
* Small changes in log outputs.
* Bug fixes and performance improvements.


<a name="v401"></a>
### Version 4.0.1

* Bug fix where the [FeatureGenerator](../API/feature_engineering/feature_generator)
  was not deterministic for a fixed random state.
* Bug fix where subsequent runs with the same metric failed.
* Added the [license](../license) file to the package's installer.
* Typo fixes in documentation.


<a name="v400"></a>
### Version 4.0.0

* Bayesian optimization package changed from [GpyOpt](http://sheffieldml.github.io/GPyOpt/)
  to [skopt](https://scikit-optimize.github.io/stable/).
* Complete revision of the model's hyperparameters.
* Four [SHAP plots](../user_guide/plots/#shap) can now be called directly from an ATOM pipeline.
* Two new plots for regression tasks.
* New [plot_pipeline](../API/plots/plot_pipeline) and `pipeline` attribute to access all transformers. 
* Possibility to determine transformer parameters per method.
* New [calibration](../API/ATOM/atomclassifier/#calibrate) method and [plot](../API/plots/plot_calibration).
* Metrics can now be added as scorers or functions with signature metric(y, y_pred, **kwargs).
* Implementation of [multi-metric](../user_guide/training/#metric) runs.
* Possibility to choose which metric to plot.
* Early stopping for models that allow in-training evaluation.
* Added the [ATOMLoader](../API/ATOM/atomloader) function to load any saved pickle instance.
* The "remove" strategy in the data cleaning parameters is deprecated in favour of "drop".
* Implemented the DFS strategy in [FeatureGenerator](../API/feature_engineering/feature_generator).
* All training classes now inherit from BaseEstimator.
* Added multiple new example notebooks.
* Tests coverage up to 100%.
* Completely new documentation page.
* Bug fixes and performance improvements.