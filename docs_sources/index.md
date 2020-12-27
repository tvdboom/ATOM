<div align="center">
    <img src="img/logo.png" alt="ATOM" height="170" width="600"/>
</div>
<br><br>

# Automated Tool for Optimized Modelling
----------------------------------------


There is no magic formula in data science that can tell us which type of machine
 learning algorithm will perform best for a specific use-case. Different models
 are better suited for different types of data and different problems. At best,
 you can follow some [rough guide](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)
 on how to approach problems with regard to which model to try on your data, but
 these are often more confusing than helpful. Best practices tell
 us to start with a simple model (e.g. linear regression) and build up to more
 complicated models (e.g. logistic regression -> random forest -> multi-layer perceptron)
 if you are not satisfied with the results. Unfortunately, different models require
 different data cleaning steps, different type/amount of features, tuning a new set
 of hyperparameters, etc. Refactoring the code for this purpose can be quite boring
 and time-consuming. Because of this, many data scientists end up just using the model
 best known to them and fine-tuning this particular model without ever trying different
 ones. This can result in poor performance (because the model is just not the right one
 for the task) or in poor time management (because you could have achieved a similar
 performance with a simpler/faster model).

ATOM is here to help us solve these issues. With just a few lines of code, you can
 perform basic data cleaning steps, select relevant features and compare the performance
 of multiple models on a given dataset. ATOM should be able to provide quick insights
 on which algorithms perform best for the task at hand and provide an indication of
 the feasibility of the ML solution.

It is important to realize that ATOM is not here to replace all the work a data
 scientist has to do before getting his model into production. ATOM doesn't spit out
 production-ready models just by tuning some parameters in its API. After helping you
 to determine the right model, you will most probably need to fine-tune it using
 use-case specific features and data cleaning steps in order to achieve maximum performance.

So, this sounds a bit like AutoML, how is ATOM different than 
 [auto-sklearn](https://automl.github.io/auto-sklearn/master/) or [TPOT](http://epistasislab.github.io/tpot/)?
 Well, ATOM does AutoML in the sense that it helps you find the best model for a
 specific task, but contrary to the aforementioned packages, it does not actively
 search for the best model. It simply runs all of them and let you pick the one that
 you think suites you best. AutoML packages are often black boxes: if you provide
 data, it will magically return a working model. Although it works great, they often
 produce complicated pipelines with low explainability, hard to sell to the business.
 In this, ATOM excels. Every step of the pipeline is accounted for, and using the
 provided plotting methods, it’s easy to demonstrate why a model is better/worse than
 the other.

!!!note
    A data scientist with domain knowledge can outperform ATOM if he applies
    usecase-specific feature engineering or data cleaning steps! 


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

<div align="center">
    <img src="img/diagram.jpg" alt="diagram" height="300" width="1000"/>
</div>

<br><br><br><br>



# Release history
-----------------

### Version 4.2.0 - Coming soon

* Possibility to add custom models to the pipeline using [ATOMModel](./API/ATOM/atommodel).
* Compatibility with [deep learning](./user_guide/#deep-learning) models.
* New branch system for different data pipelines. Read more in the [user guide](./user_guide/#data-pipelines).
* Use the [canvas](./API/ATOM/atomclassifier/#canvas) contextmanager to draw multiple plots in one figure.
* New [Voting](./user_guide/#voting) and [Stacking](./user_guide/#stacking) ensemble techniques.
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
* [Successive halving](./user_guide/#successive-halving) and [Train sizing](./user_guide/#train-sizing)
  now both allow subsequent runs from atom without losing previous information.
* Bug fix where ATOMLoader wouldn't encode the target column during transformation.
* Added the <a href="./examples/deep_learning.html" target="_blank">Deep learning</a>, 
  <a href="./examples/ensembles.html" target="_blank">Ensembles</a> and
  <a href="./examples/utilities.html" target="_blank">Utilities</a> example notebooks.
* Compatibility with [python 3.9](https://www.python.org/downloads/release/python-390/).


### Version 4.1.0 - 16 October 2020

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


### Version 4.0.1 - 29 September 2020

* Bug fix where the DFS strategy in [FeatureGenerator](./API/feature_engineering/feature_generator)
  was not deterministic for a fixed random state.
* Bug fix where subsequent runs with the same metric failed.
* Added the [license](./license) file to the package's installer.
* Typo fixes in documentation.


### Version 4.0.0 - 28 September 2020

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
        - [Outliers](./API/data_cleaning/outliers)
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
        - [force_plot](./API/plots/force_plot)
        - [dependence_plot](./API/plots/dependence_plot)
        - [summary_plot](./API/plots/summary_plot)
        - [decision_plot](./API/plots/decision_plot)
        - [waterfall_plot](./API/plots/waterfall_plot)
- Examples
    - <a href="./examples/binary_classification.html" target="_blank">Binary classification</a>
    - <a href="./examples/multiclass_classification.html" target="_blank">Multiclass classification</a>
    - <a href="./examples/regression.html" target="_blank">Regression</a>
    - <a href="./examples/successive_halving.html" target="_blank">Successive halving</a>
    - <a href="./examples/train_sizing.html" target="_blank">Train sizing</a>
    - <a href="./examples/deep_learning.html" target="_blank">Deep learning</a>
    - <a href="./examples/imbalanced_datasets.html" target="_blank">Imbalanced datasets</a>
    - <a href="./examples/feature_engineering.html" target="_blank">Feature engineering</a>
    - <a href="./examples/multi_metric.html" target="_blank">Multi-metric runs</a>
    - <a href="./examples/early_stopping.html" target="_blank">Early stopping</a>
    - <a href="./examples/ensembles.html" target="_blank">Ensembles</a>
    - <a href="./examples/calibration.html" target="_blank">Calibration</a>
    - <a href="./examples/utilities.html" target="_blank">Utilities</a>
- [Dependencies](./dependencies)
- [License](./license)

