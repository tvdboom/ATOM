# Introduction
--------------

There is no magic formula in data science that can tell us which type of machine
 learning algorithm will perform best for a specific use-case. Different models
 are better suited for different types of data and different problems. At best,
 you can follow some [rough guide](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)
 on how to approach problems with regard to which model to try on your data, but
 these are often more confusing than helpful. Best practices tell
 us to start with a simple model (e.g. linear regression) and build up to more
 complicated models (e.g. logistic regression -> random forest -> multilayer perceptron)
 if you are not satisfied with the results. Unfortunately, different models require
 different data cleaning steps, different type/amount of features, tuning a new set
 of hyperparameters, etc. Refactoring the code for this purpose can be quite boring
 and time consuming. Because of this, many data scientists end up just using the model
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


<br><br>
# Nomenclature
--------------

In this documentation we will consistently use terms to refer to certain concepts
 related to the ATOM package.

* **ATOM**: Refers to this package.
* **task**: Refers to one of the three supervised machine learning approaches that ATOM supports:
    - [binary classification](https://en.wikipedia.org/wiki/Binary_classification)
    - [multiclass classification](https://en.wikipedia.org/wiki/Multiclass_classification)
    - [regression](https://en.wikipedia.org/wiki/Regression_analysis)
* **category**: Refers to one of the unique values in a column, i.e. a binary classifier has 2 categories in the target column.
* **missing values**: Refers to `None`, `NaN` and `inf` values.
* **categorical columns**: Refers to all columns with dtype.kind not in `ifu`.
* `atom`: Refers to an [ATOMClassifier](../API/ATOM/atomclassifier) or
 [ATOMRegressor](../API/ATOM/atomregressor) instance (note that all examples
 use it as variable name for the instance).
* `model`: Refers to an instance of one of the [models](#models) available in ATOM.
* `training`: Refers to an instance of one of the classes that train and evaluate the
 models. The classes are:
    - [ATOMClassifier](../API/ATOM/atomclassifier)
    - [ATOMRegressor](../API/ATOM/atomregressor)
    - [TrainerClassifier](../API/training/trainerclassifier)
    - [TrainerRegressor](../API/training/trainerregressor)
    - [SuccessiveHalvingClassifier](../API/training/successivehalvingclassifier)
    - [SuccessiveHavingRegressor](../API/training/successivehalvingregressor)
    - [TrainSizingClassifier](../API/training/trainsizingclassifier)
    - [TrainSizingRegressor](../API/training/trainsizingregressor)

!!!note
    Note that `atom` instances are also `training` instances!


<br><br>
# First steps
-------------

You can quickly install atom using `pip` or `conda`, see the [installation guide](../getting_started/#installation).
 ATOM contains a variety of classes to perform data cleaning, feature engineering,
 model training and much more. The easiest way to use all these classes on the same
 dataset is through one of the main classes:

* [ATOMClassifier](../API/ATOM/atomclassifier) for binary or multiclass classification tasks.
* [ATOMRegressor](../API/ATOM/atomregressor) for regression tasks.

These two classes are convenient wrappers for all the possibilities this package
 provides. Like a [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html),
 they assemble several steps that can be cross-validated together while setting
 different parameters. There are some important differences with sklearn's API though:
 
1. `atom` is initialized with the data you want to manipulate. This data can be accessed
 at any moment through `atom`'s [data properties](../API/ATOM/atomclassifier/#data-properties).
2. The classes in ATOM's API are reached through `atom`'s methods. For example, calling
 the [encode](../API/ATOM/atomclassifier/#atomclassifier-encode) method, will initialize
 an [Encoder](../API/data_cleaning/encoder) instance, fit it on the training set and
 transform the whole dataset.
3. The transformations are applied immediately after calling the method (there is no
 fit method). This approach gives the user a clearer overview and more control over
 every step in the pipeline.
4. The pipeline does not have to end with an estimator. ATOM can be used just for data
 cleaning or feature engineering.

Let's get started with an example!

First, initialize `atom` and provide it the data you want to use.

    atom = ATOMClassifier(X, y)

Apply data cleaning methods through the class. For example, calling the
 [impute](../API/ATOM/atomclassifier/#atomclassifier-impute) method will
 handle all missing values in the dataset.

    atom.impute(strat_num='median', strat_cat='most_frequent', min_frac_rows=0.1)

Select the best hyperparameters and fit a Random Forest and AdaBoost model.

    atom.run(['RF', 'AdaB'], metric='accuracy', n_calls=25, n_initial_points=10)

Analyze the results:

    atom.feature_importances(show=10, filename='feature_importance_plot')
    atom.plot_prc(title='Precision-recall curve comparison plot')


<br><br>
# Data cleaning
---------------

More often than not, you need to do some data cleaning before fitting your dataset
 to a model.  Usually, this involves importing different libraries and writing many
 lines of code. Since ATOM is all about fast exploration and experimentation, it
 provides various data cleaning classes to apply the most common transformations fast
 and easy.

<br>

### Scaling the feature set

Standardization of a dataset is a common requirement for many machine learning
 estimators: they might behave badly if the individual features do not more or less
 look like standard normally distributed data (e.g. Gaussian with 0 mean and unit
 variance). The [Scaler](API/data_cleaning/scaler.md) class scales data to mean=0 and
 std=1. It can be accessed from `atom` through the [scale](../API/ATOM/atomclassifier/#atomclassifier-scale)
 method. 

<br>

### Standard data cleaning

There are many data cleaning steps that are useful to perform on any dataset before
 modelling. These are general rules that apply on every use-case and every task. The
 [StandardCleaner](API/data_cleaning/standard_cleaner.md) class is a convenient tool
 to apply such steps. It is automatically called when initializing `atom`. Use the
 class' parameters to choose which transformations to perform. The available steps are:

* Remove columns with prohibited data types.
* Strip categorical features from white spaces.
* Remove categorical columns with maximal cardinality.
* Remove columns with minimum cardinality.
* Remove rows with missing values in the target column.
* Label-encode the target column.

<br> 

### Imputing missing values

For various reasons, many real world datasets contain missing values, often encoded
 as blanks, NaNs or other placeholders. Such datasets however are incompatible with
 ATOM's models which assume that all values in an array are numerical, and that all
 have and hold meaning. The [Imputer](API/data_cleaning/imputer.md) class handles
 missing values in the dataset by either dropping or imputing the value. It can be
 accessed from `atom` through the [impute](../API/ATOM/atomclassifier/#atomclassifier-impute)
 method.

!!!tip
    Use `atom`'s [missing](../API/ATOM/atomclassifier/#utility-properties) property
    for an overview of the missing values in the dataset.

<br>

### Encoding categorical features

Many datasets will contain categorical features. Their variables are typically stored
 as text values which represent various traits. Some examples include color (“Red”,
 “Yellow”, “Blue”), size (“Small”, “Medium”, “Large”) or geographic designations
 (city or country). Regardless of what the value is used for, the challenge is
 determining how to use this data in the analysis. ATOM's models don't support
 direct manipulation of this kind of data. Use the [Encoder](API/data_cleaning/encoder.md)
 class to encode categorical features to numerical values. It can be accessed from
 `atom` through the [encode](../API/ATOM/atomclassifier/#atomclassifier-encode) method.

!!!tip
    Use `atom`'s [categorical](../API/ATOM/atomclassifier/#utility-properties) property
    for a list of the categorical columns in the dataset.

<br> 

### Handling outliers

When modeling, it is important to clean the data sample to ensure that the observations
 best represent the problem. Sometimes a dataset can contain extreme values that are
 outside the range of what is expected and unlike the other data. These are called
 outliers. Often, machine learning modeling and model skill in general can be improved
 by understanding and even removing these outlier values. The [Outliers](API/data_cleaning/outliers.md) 
 class can drop or impute outliers in the dataset. It can be accessed from `atom`
 through the [outliers](../API/ATOM/atomclassifier/#atomclassifier-outliers) method.

<br> 

### Balancing the data

One of the common issues found in datasets that are used for classification is
 imbalanced classes. Data imbalance usually reflects an unequal distribution of
 classes within a dataset. For example, in a credit card fraud detection dataset,
 most of the transactions are non-fraud and a very few cases are fraud. This leaves
 us with a very unbalanced ratio of fraud vs non-fraud cases. The
 [Balancer](API/data_cleaning/balancer.md) class can oversample the minority category
 or undersample the majority category. It can be accessed from `atom` through the
 [balance](../API/ATOM/atomclassifier/#atomclassifier-balance) method.




<br><br>
# Feature engineering
---------------------

<cite>
<div align="center">
"Applied machine learning" is basically feature engineering. ~ Andrew Ng.
</div>
</cite>


<br>
Feature engineering is the process of creating new features from the existing ones,
 in order to capture relationships with the target column that the first set of
 features didn't had on their own. This process is very important to improve the
 performance of machine learning algorithms. Although feature engineering works best
 when the data scientist applies use-case specific transformations, there are ways to
 do this in an automated manner, without prior domain knowledge. One of the problems
 of creating new features without human expert intervention, is that many of the newly
 created features can be useless, i.e. they do not help the algorithm to make better
 predictions. Even worse, having useless features can drop your performance. To avoid
 this, we perform feature selection, a process in which we select the relevant features 
 in the dataset. See [here](examples/feature_engineering/feature_engineering.md) an example.


<br>

### Generating new features

The [FeatureGenerator](API/feature_engineering/feature_generator.md) class creates
 new non-linear features based on the original feature set. It can be accessed from
 `atom` through the [feature_generation](../API/ATOM/atomclassifier/#atomclassifier-feature-generation)
 method. You can choose between two strategies: Deep Feature Synthesis and Genetic
 Feature Generation.


**Deep Feature Synthesis**<br>
Deep feature synthesis (DFS) applies the selected operators on the features in
 the dataset. For example, if the operator is 'log', it will create the new feature
 `LOG(old_feature)` and if the operator is 'mul', it will create the new feature
 `old_feature_1 x old_feature_2`. The operators can be chosen through the `operators`
 parameter. Available options are:
<ul>
<li><b>add: </b>Sum two features together.</li>
<li><b>sub: </b>Subtract two features from each other.</li>
<li><b>mul: </b>Multiply two features with each other.</li>
<li><b>div: </b>Divide two features with each other.</li>
<li><b>srqt: </b>Take the square root of a feature.</li>
<li><b>log: </b>Take the logarithm of a feature.</li>
<li><b>sin: </b>Calculate the sine of a feature.</li>
<li><b>cos: </b>Calculate the cosine of a feature.</li>
<li><b>tan: </b>Calculate the tangent of a feature.</li>
</ul>

ATOM's implementation of DFS uses the [featuretools](https://www.featuretools.com/) package.

!!! tip
    DFS can create many new features and not all of them will be useful. Use
    [FeatureSelector](./API/feature_engineering/feature_selector.md) to reduce
    the number of features!

!!! warning
    Using the div, log or sqrt operators can return new features with `inf` or
    `NaN` values. Check the warnings that may pop up or use `atom`'s
    [missing](/API/ATOM/atomclassifier/#properties) property.

!!! warning
    When using DFS with `n_jobs>1`, make sure to protect your code with `if __name__
    == "__main__"`. Featuretools uses [dask](https://dask.org/), which uses python
    multiprocessing for parallelization. The spawn method on multiprocessing starts
    a new python process, which requires it to import the \__main__ module before it
    can do its task.

<br>

**Genetic Feature Generation**<br>
Genetic feature generation (GFG) uses [genetic programming](https://en.wikipedia.org/wiki/Genetic_programming),
 a branch of evolutionary programming, to determine which features are successful and
 create new ones based on those. Where DFS' method can be seen as some kind of "brute
 force" for feature engineering, GFG tries to improve its features with every
 generation of the algorithm. GFG uses the same operators as DFS, but instead of only
 applying the transformations once, it evolves them further, creating complicated
 non-linear combinations of features with many transformations. The new features are
 given the name `Feature N` for the N-th feature. You can access the genetic feature's
 fitness and description (how they are calculated) through the `genetic_features`
 attribute.

ATOM uses the [SymbolicTransformer](https://gplearn.readthedocs.io/en/stable/reference.html#symbolic-transformer)
 class from the [gplearn](https://gplearn.readthedocs.io/en/stable/index.html) package
 for the genetic algorithm. Read more about this implementation
 [here](https://gplearn.readthedocs.io/en/stable/intro.html#transformer).

!!!warning
    GFG can be slow for very large populations!

<br>

### Selecting useful features

The [FeatureSelector](API/feature_engineering/feature_selector.md) class provides
 tooling to select the relevant features from a dataset. It can be accessed from `atom`
 through the [feature_selection](../API/ATOM/atomclassifier/#atomclassifier-feature-selection)
 method. The following strategies are implemented: univariate, PCA, SFM, RFE and RFECV.


**Univariate**<br>
Univariate feature selection works by selecting the best features based on univariate statistical F-test.
 The test is provided via the `solver` parameter. It takes any function taking two arrays (X, y),
 and returning arrays (scores, p-values).

Read more in sklearn's [documentation](https://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection).


**Principal Components Analysis**<br>
Applying PCA will reduce the dimensionality of the dataset by maximizing the variance of each dimension.
 The new features will be called Component 0, Component 1, etc... The dataset will be
 scaled before applying the transformation (if it wasn't already).

Read more in sklearn's [documentation](https://scikit-learn.org/stable/modules/decomposition.html#pca).


**Selection from model**<br>
SFM uses an estimator with `feature_importances_` or `coef_` attributes to select the
 best features in a dataset based on importance weights. The estimator is provided
 through the `solver` parameter and can be already fitted. ATOM allows you to use one
 its pre-defined [models](#models), e.g. `solver='RF'`. If you didn't call the
 FeatureSeletor through `atom`, don't forget to indicate the estimator's task adding
 `_class` or `_reg` after the name, e.g. `RF_class` to use a random forest classifier.

Read more in sklearn's [documentation](https://scikit-learn.org/stable/modules/feature_selection.html#feature-selection-using-selectfrommodel).


**Recursive feature elimination**<br>
Select features by recursively considering smaller and smaller sets of features.
 First, the estimator is trained on the initial set of features and the importance
 of each feature is obtained either through a `coef_` attribute or through a
 `feature_importances_` attribute. Then, the least important features are pruned from
 current set of features. That procedure is recursively repeated on the pruned set
 until the desired number of features to select is eventually reached. Note that, since
 RFE needs to fit the model again every iteration, this method can be fairly slow.
  
RFECV applies the same algorithm as RFE but uses a cross-validated metric (under the
 scoring parameter, see [RFECV](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html#sklearn.feature_selection.RFECV))
 to assess every step's performance. Also, where RFE returns the number of features selected
 by `n_features`, RFECV returns the number of features that achieved the optimal score
 on the specified metric. Note that this is not always equal to the amount specified
 by `n_features`.

Read more in sklearn's [documentation](https://scikit-learn.org/stable/modules/feature_selection.html#recursive-feature-elimination).


**Removing features with low variance**<br>
Variance is the expectation of the squared deviation of a random variable from its mean.
 Features with low variance have many values repeated, which means the model will not
 learn much from them. [FeatureSelector](API/feature_engineering/feature_selector.md)
 removes all features where the same value is repeated in at least `max_frac_repeated`
 fraction of the rows. The default option is to remove a feature if all values in it
 are the same.
 
Read more in sklearn's [documentation](https://scikit-learn.org/stable/modules/feature_selection.html#removing-features-with-low-variance).


**Removing features with multi-collinearity**<br>
Two features that are highly correlated are redundant, i.e. two will not contribute more
 to the model than only one of them. [FeatureSelector](API/feature_engineering/feature_selector.md)
 will drop a feature that has a Pearson correlation coefficient larger than
 `max_correlation` with another feature. A correlation of 1 means the two columns
 are equal. A dataframe of the removed features and their correlation values can
 be accessed through the `collinear` attribute.

<br>

!!! tip
    Use the [plot_feature_importance](API/plots/plot_feature_importance.md) method to
    examine how much a specific feature contributes to the final predictions. If the
    model doesn't have a `feature_importances_` attribute, use 
    [plot_permutation_importance](API/plots/plot_permutation_importance.md) instead.

!!!warning
    The RFE and RFECV strategies don't work when the solver is a 
    [CatBoost](https://catboost.ai/) model due to incompatibility of the APIs.


<br><br>
# Training
----------

Six classes: TrainerClassifier and reg, SuccessiveHalvingClassifier and reg,
 and TrainSizingClassifier and reg. Accessed from an ATOM instance through run,
 successive_halving and train_sizing.

<br>

A couple of things to take into account:

* The metric implementation follows [sklearn's API](https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values).
  This means that the implementation always tries to maximize the scorer, i.e.
  loss functions will be made negative.
* If an exception is encountered while fitting a model, the
  pipeline will automatically jump to the next model and save the
  exception in the `errors` attribute.
* When showing the final results, a `!!` indicates the highest
  score and a `~` indicates that the model is possibly overfitting
  (training set has a score at least 20% higher than the test set).
* The winning model subclass will be attached to the `winner` attribute.


1. The optimal hyperparameters are selected using a Bayesian Optimization (BO)
 algorithm. The resulting score of each step of the BO is either computed by
 cross-validation on the complete training set or by randomly splitting the training set every iteration into a (sub) training
 set and a validation set. This process can create some data leakage but
 ensures a maximal use of the provided data. The test set, however, does not
 contain any leakage and will be used to determine the final score of every model.
 Note that, if the dataset is relatively small, the best score on the BO can
 consistently be lower than the final score on the test set (despite the
 leakage) due to the considerable fewer instances on which it is trained.
<div></div>
 
2. Once the best hyperparameters are found, the model is trained again, now
 using the complete training set. After this, predictions are made on the test set.
<div></div>

3. You can choose to evaluate the robustness of each model's applying a bagging
 algorithm, i.e. the model will be trained multiple times on a bootstrapped
 training set, returning a distribution of its performance on the test set.


<br>

### Metric

**Scorers**

**Custom metric acronyms**

**Multi-metric runs**

<br>

### Hyperparameter tuning


<br>

### Bagging


<br>

### Early stopping



<br>

### Successive halving


<br>

### Train sizing




<br><br>
# Models
--------

### Properties


### Methods



<br><br>
# Predicting
------------

After running a successful pipeline, it is possible you would like to apply all
 used transformations onto new data, or make predictions using one of the trained
 models. Just like a sklearn estimator, you can call the prediction methods from a
 fitted `training` instance, e.g. `atom.predict(X)`. Calling the method without
 specifying a model will use the winning model in the pipeline (under attribute
 `winner`). To use a different model, simply call the method from a `model`,
 e.g. `atom.KNN.predict(X)`.

If called from `atom`, the prediction methods will transform the provided data through
 all the transformers in the pipeline before making the predictions. By default, this
 excludes outlier handling and balancing the dataset since these steps should only
 be applied on the training set. Use the method's kwargs to select which transformations
 to use in every call.

The available prediction methods are a selection of the most common methods for
 estimators in sklearn's API:

* [transform](../API/predicting/transform)
* [predict](../API/predicting/predict)
* [predict_proba](../API/predicting/predict_proba)
* [predict_log_proba](../API/predicting/predict_log_proba)
* [decision_function](../API/predicting/decision_function)
* [score](../API/predicting/score)


Except for transform, the prediction methods can be calculated on the train and test
 set. You can access them through the [properties](#properties) of the
 `models`, e.g. `atom.mnb.predict_proba_train` or ` atom.mnb.predict_proba_test`.
 Keep in mind that the results of these properties are calculated only once, the
 first time the property is called. Once calculated, they are stored as private
 attributes of the instance. This mechanism avoids having to calculate attributes
 that are never used (saving time and memory), but allows using the same property
 without having to calculate it again. Many of the [plots](#plots) benefit from this
 approach.



<br><br>
# Plots
-------

After fitting the models to the data, it's time to analyze the results. ATOM provides
 many plotting methods to compare the model performances. Descriptions and examples
 can be found in the API section. ATOM uses the packages [matplotlib](https://matplotlib.org/),
 [seaborn](https://seaborn.pydata.org/) and [shap](https://github.com/slundberg/shap)
 for plotting.

The plot methods can be called from a `training` directly, e.g. `atom.plot_roc()`,
 or from one of the `models`, e.g. `atom.LGB.plot_roc()`. If called from `training`,
 it will make the plot for all models in the pipeline. This can be useful to compare
 the results of multiple models. If called from a `model`, it will make the plot for
 only that model. Use this option if you want information just for that specific
 model or to make a plot less crowded.

<br>

### Parameters

Apart from the plot-specific parameters they may have, all plots have four parameters
 in common:

* The `title` parameter allows you to add a custom title to the plot.
* The `figsize` parameter adjust the plot's size.
* The `filename` parameter is used to save the plot.
* The `display` parameter determines whether the plot is rendered.

<br>

### Aesthetics

The plot aesthetics can be customized using the properties described hereunder, e.g.
 `atom.style = 'white'`.

<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Properties:</strong></td>
<td width="75%" style="background:white;">
<strong>style: str, optional (default='darkgrid')</strong>
<blockquote>
Seaborn plotting style. See the <a href="https://seaborn.pydata.org/tutorial/aesthetics.html#seaborn-figure-styles">documentation</a>.
</blockquote>

<strong>palette: str, optional (default='GnBu_r_d')</strong>
<blockquote>
Seaborn color palette. See the <a href="https://seaborn.pydata.org/tutorial/color_palettes.html">documentation</a>.
</blockquote>

<strong>title_fontsize: int, optional (default=20)</strong>
<blockquote>
Fontsize for the plot's title.
</blockquote>

<strong>label_fontsize: int, optional (default=16)</strong>
<blockquote>
Fontsize for labels and legends.
</blockquote>

<strong>tick_fontsize: int, optional (default=12)</strong>
<blockquote>
Fontsize for the ticks along the plot's axes.
</blockquote>

</td></tr>
</table>


<br>

### SHAP

The [SHAP](https://github.com/slundberg/shap) (SHapley Additive exPlanations) python
 package uses a game theoretic approach to explain the output of any machine
 learning model. It connects optimal credit allocation with local explanations
 using the classic Shapley values from game theory and their related extensions.
 ATOM implements methods to plot 4 of shap's plotting functions directly from its
 API. The explainer will be chosen automatically based on the model's type. For 
 kernelExplainer, the data used to estimate the expected values is the complete
 training set when <100 rows, else its summarized with a set of 10 weighted K-means,
 each weighted by the number of points they represent.
 The four plots are: [force_plot](../API/plots/force_plot),
 [dependence_plot](../API/plots/dependence_plot), [summary_plot](../API/plots/summary_plot)
 and [decision_plot](../API/plots/decision_plot).

Since the plots are not made by ATOM, we can't draw multiple models in the same figure.
 Selecting more than one model will raise an exception. To avoid this, call the plot
 from a `model`, e.g. `atom.xgb.force_plot()`.

!!!note
    You can recognize the SHAP plots by the fact that they end (instead of start)
    with `plot`.

<br>

### Available plots

A list of available plots can be find hereunder. Note that not all plots can be
 called from every class and that their availability can depend on the task at hand.

* [plot_correlation](../API/plots/plot_correlation)
* [plot_pipeline](../API/plots/plot_pipeline)
* [plot_pca](../API/plots/plot_pca)
* [plot_components](../API/plots/plot_components)
* [plot_rfecv](../API/plots/plot_rfecv)
* [plot_successive_halving](../API/plots/plot_successive_halving)
* [plot_learning_curve](../API/plots/plot_learning_curve)
* [plot_bagging](../API/plots/plot_bagging)
* [plot_bo](../API/plots/plot_bo)
* [plot_evals](../API/plots/plot_evals)
* [plot_roc](../API/plots/plot_roc)
* [plot_prc](../API/plots/plot_prc)
* [plot_permutation_importance](../API/plots/plot_permutation_importance)
* [plot_feature_importance](../API/plots/plot_feature_importance)
* [plot_partial_dependence](../API/plots/plot_partial_dependence)
* [plot_errors](../API/plots/plot_errors)
* [plot_residuals](../API/plots/plot_residuals)
* [plot_confusion_matrix](../API/plots/plot_confusion_matrix)
* [plot_threshold](../API/plots/plot_threshold)
* [plot_probabilities](../API/plots/plot_probabilities)
* [plot_calibration](../API/plots/plot_calibration)
* [plot_gain](../API/plots/plot_gain)
* [plot_lift](../API/plots/plot_lift)
* [force_plot](../API/plots/force_plot)
* [dependence_plot](../API/plots/dependence_plot)
* [summary_plot](../API/plots/summary_plot)
* [decision_plot](../API/plots/decision_plot)
