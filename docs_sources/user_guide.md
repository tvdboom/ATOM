# Introduction
--------------

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


<br><br>
# Nomenclature
--------------

In this documentation we will consistently use terms to refer to certain
concepts related to this package.

* **atom**: Instance of the [ATOMClassifier](../API/ATOM/atomclassifier) or
  [ATOMRegressor](../API/ATOM/atomregressor) classes (note that the examples
  use it as the default variable name).
* **ATOM**: Refers to this package.
* **branch**: Collection of estimators in the pipeline fitted to a specific dataset,
  see the [Data Pipelines](#data-pipelines) section.
* **BO**: Bayesian optimization algorithm used for hyperparameter optimization.
* **categorical columns**: Refers to all columns with dtype.kind not in `ifu`.
* **class**: Unique value in a column, e.g. a binary classifier has 2 classes in the target column.
* **estimator**: An object which manages the estimation and decoding of an algorithm. The
  algorithm is estimated as a deterministic function of a set of parameters, a dataset
  and a random state.
* **missing values**: Values in the `missing` attribute.
* **model**: Instance of a [model](#models) in the pipeline.
* **pipeline**: All the content in atom for a specific branch.
* **predictor**: An estimator implementing a `predict` method. This encompasses all
  classifiers and regressors.
* **scorer**: A non-estimator callable object which evaluates an estimator on given test
  data, returning a number. Unlike evaluation metrics, a greater returned number must
  correspond with a better score. See sklearn's [documentation](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter).
* **sequence**: A one-dimensional array of variable type `list`, `tuple`, `np.ndarray` or `pd.Series`.
* **target**: Name of the dependent variable, passed as y to an estimator's fit method.
* **task**: One of the three supervised machine learning approaches that ATOM supports:
    - [binary classification](https://en.wikipedia.org/wiki/Binary_classification)
    - [multiclass classification](https://en.wikipedia.org/wiki/Multiclass_classification)
    - [regression](https://en.wikipedia.org/wiki/Regression_analysis)
* **trainer**: Instance of a class that train and evaluate the models (implement a
  `run` method). The following classes are considered trainers:
    - [ATOMClassifier](../API/ATOM/atomclassifier)
    - [ATOMRegressor](../API/ATOM/atomregressor)
    - [DirectClassifier](../API/training/directclassifier)
    - [DirectRegressor](../API/training/directregressor)
    - [SuccessiveHalvingClassifier](../API/training/successivehalvingclassifier)
    - [SuccessiveHavingRegressor](../API/training/successivehalvingregressor)
    - [TrainSizingClassifier](../API/training/trainsizingclassifier)
    - [TrainSizingRegressor](../API/training/trainsizingregressor)
* **transformer**: An estimator implementing a `transform` method. This encompasses all
  data cleaning and feature engineering classes.


<br><br>
# First steps
-------------

You can quickly install atom using `pip` or `conda`, see the [installation guide](../getting_started/#installation).
ATOM contains a variety of classes to perform data cleaning, feature
engineering, model training, plotting and much more. The easiest way
to use all these classes on the same dataset is through one of the
main classes:

* [ATOMClassifier](../API/ATOM/atomclassifier) for binary or multiclass classification tasks.
* [ATOMRegressor](../API/ATOM/atomregressor) for regression tasks.

These two classes are convenient wrappers for all the possibilities this
package provides. Like a sklearn [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html),
they assemble several steps that can be cross-validated together while setting
different parameters. There are some important differences with sklearn's API:

1. atom is initialized with the data you want to manipulate. This data can be
   accessed at any moment through atom's [data attributes](../API/ATOM/atomclassifier/#data-attributes).
2. The classes in ATOM's API are reached through atom's methods. For example,
   calling the [encode](../API/ATOM/atomclassifier/#encode) method will
   initialize an [Encoder](../API/data_cleaning/encoder) instance, fit it on
   the training set and transform the whole dataset.
3. The transformations are applied immediately after calling the method
   (there is no fit method). This approach gives the user a clearer overview
   and more control over every step in the pipeline.
4. The pipeline does not have to end with an estimator. ATOM can be just for data
   cleaning or feature engineering purposes only.

Let's get started with an example!

First, initialize atom and provide it the data you want to use. You can
either input a dataset and let ATOM split the train and test set or provide
a train and test set already split. Note that if a dataframe is provided,
the indices will be reset by atom.

```python
atom = ATOMClassifier(X, y, test_size=0.25)
```

Apply data cleaning methods through the class. For example, calling the
[impute](../API/ATOM/atomclassifier/#impute) method will handle all
missing values in the dataset.

```python
atom.impute(strat_num="median", strat_cat="most_frequent", min_frac_rows=0.1)
```

Select the best hyperparameters and fit a Random Forest and AdaBoost model.

```python
atom.run(["RF", "AdaB"], metric="accuracy", n_calls=25, n_initial_points=10)
```

Analyze the results:

```python
atom.feature_importances(show=10, filename="feature_importance_plot")
atom.plot_prc(title="Precision-recall curve comparison plot")
```




<br><br>
# Data pipelines
----------------

It may happen that you want to compare how a model performs on different
datasets. For example, on one dataset balanced with an undersampling
strategy and the other with an oversampling strategy. For this, atom has
data pipelines.

Data pipelines manage separate paths atom's dataset can take. The paths
are called branches and can be accessed through the `branch` attribute.
Calling it will show the branches in the pipeline. The current branch is
indicated with `!`. A branch contains a specific dataset, and the
transformers it took to arrive to that dataset from the one atom
initialized with. Accessing data attributes such as `atom.dataset` will
return the data in the current branch. Use the `pipeline` attribute to
see the estimators in the branch. All data cleaning, feature engineering
and trainers called will use the dataset in the current branch. This
means that models are trained and validated on the data in that branch.
Don't change the data in a branch after fitting a model, this can cause
unexpected model behaviour. Instead, create a new branch for every unique
model pipeline.

By default, atom starts with one branch called "master". To start a new
branch, set a new name to the property, e.g. `atom.branch = "new_branch"`.
This will start a new branch from the current one. To create a branch
from any other branch type "\_from\_" between the new name and the branch
from which to split, e.g. `atom.branch = "branch2_from_branch1"` will
create branch "branch2" from branch "branch1". To switch between existing
branches, just type the name of the desired branch, e.g. `atom.branch = "master"`
to go back to the main branch. Note that every branch contains a unique copy
of the whole dataset! Creating many branches can cause memory issues for
large datasets.

You can delete a branch either deleting the attribute, e.g. `del atom.branch`,
or using the delete method, e.g. `atom.branch.delete()`. A branch can only be
deleted if no models were trained on its dataset. Use `atom.branch.status()`
to print a list of the transformers and models in the branch.

See the <a href="../examples/imbalanced_datasets.html" target="_blank">Imbalanced datasets</a>
or <a href="../examples/feature_engineering.html" target="_blank">Feature engineering</a>
examples for branching use cases.

!!!warning
    Always create a new branch if you want to change the dataset after fitting
    a model! Not doing this can cause unexpected model behaviour.



<br><br>
# Data cleaning
---------------

More often than not, you need to do some data cleaning before fitting your
dataset to a model.  Usually, this involves importing different libraries
and writing many lines of code. Since ATOM is all about fast exploration 
and experimentation, it provides various data cleaning classes to apply 
the most common transformations fast and easy.

<br>

### Scaling the feature set

Standardization of a dataset is a common requirement for many machine
learning estimators: they might behave badly if the individual features
do not more or less look like standard normally distributed data (e.g.
Gaussian with 0 mean and unit variance). The [Scaler](API/data_cleaning/scaler.md)
class scales data to mean=0 and std=1. It can be accessed from atom 
through the [scale](../API/ATOM/atomclassifier/#scale) method. 

<br>

### Standard data cleaning

There are many data cleaning steps that are useful to perform on any
dataset before modelling. These are general rules that apply almost
on every use-case and every task. The [Cleaner](API/data_cleaning/cleaner.md)
class is a convenient tool to apply such steps. It can be accessed
from atom through the [clean](../API/ATOM/atomclassifier/#clean)
method. Use the class' parameters to choose which transformations to
perform. The available steps are:

* Remove columns with prohibited data types.
* Strip categorical features from white spaces.
* Remove categorical columns with maximal cardinality.
* Remove columns with minimum cardinality.
* Remove rows with missing values in the target column.
* Encode the target column.

<br>

### Imputing missing values

For various reasons, many real world datasets contain missing values,
often encoded as blanks, NaNs or other placeholders. Such datasets
however are incompatible with ATOM's models which assume that all
values in an array are numerical, and that all have and hold meaning.
The [Imputer](API/data_cleaning/imputer.md) class handles missing
values in the dataset by either dropping or imputing the value. It can 
be accessed from atom through the [impute](../API/ATOM/atomclassifier/#impute)
method.

!!!tip
    Use atom's [missing](../API/ATOM/atomclassifier/#data-attributes) attribute
    for an overview of the missing values in the dataset.

<br>

### Encoding categorical features

Many datasets will contain categorical features. Their variables are
typically stored as text values which represent various traits. Some 
examples include color (“Red”, “Yellow”, “Blue”), size (“Small”,
“Medium”, “Large”) or geographic designations (city or country).
Regardless of what the value is used for, the challenge is determining
how to use this data in the analysis. ATOM's models don't support
direct manipulation of this kind of data. Use the [Encoder](API/data_cleaning/encoder.md)
class to encode categorical features to numerical values. It can be
accessed from atom through the [encode](../API/ATOM/atomclassifier/#encode) 
method.

!!!tip
    Use atom's [categorical](../API/ATOM/atomclassifier/#data-attributes) attribute
    for a list of the categorical columns in the dataset.

<br> 

### Handling outliers

When modeling, it is important to clean the data sample to ensure that
the observations best represent the problem. Sometimes a dataset can
contain extreme values that are outside the range of what is expected
and unlike the other data. These are called outliers. Often, machine
learning modeling and model skill in general can be improved by 
understanding and even removing these outlier values. The [Outliers](API/data_cleaning/outliers.md) 
class can drop or impute outliers in the dataset. It can be accessed
from atom through the [outliers](../API/ATOM/atomclassifier/#outliers)
method.

<br> 

### Balancing the data

One of the common issues found in datasets that are used for
classification is imbalanced classes. Data imbalance usually reflects
an unequal distribution of classes within a dataset. For example, in
a credit card fraud detection dataset, most of the transactions are
non-fraud, and a very few cases are fraud. This leaves us with a very
unbalanced ratio of fraud vs non-fraud cases. The [Balancer](API/data_cleaning/balancer.md)
class can oversample the minority class or undersample the majority
class. It can be accessed from atom through the [balance](../API/ATOM/atomclassifier/#balance)
method.




<br><br>
# Feature engineering
---------------------

<cite>
<div align="center">
"Applied machine learning" is basically feature engineering. ~ Andrew Ng.
</div>
</cite>


<br>
Feature engineering is the process of creating new features from the
existing ones, in order to capture relationships with the target
column that the first set of features didn't had on their own. This
process is very important to improve the performance of machine learning
algorithms. Although feature engineering works best when the data 
scientist applies use-case specific transformations, there are ways to
do this in an automated manner, without prior domain knowledge. One of
the problems of creating new features without human expert intervention,
is that many of the newly created features can be useless, i.e. they do
not help the algorithm to make better predictions. Even worse, having
useless features can drop your performance. To avoid this, we perform
feature selection, a process in which we select the relevant features 
in the dataset. See the <a href="../examples/feature_engineering.html" target="_blank">Feature engineering</a> example.


<br>

### Generating new features

The [FeatureGenerator](API/feature_engineering/feature_generator.md)
class creates new non-linear features based on the original feature
set. It can be accessed from atom through the [feature_generation](../API/ATOM/atomclassifier/#feature-generation)
method. You can choose between two strategies: Deep Feature Synthesis
and Genetic Feature Generation.


**Deep Feature Synthesis**<br>
Deep feature synthesis (DFS) applies the selected operators on the
features in the dataset. For example, if the operator is "log",
it will create the new feature `LOG(old_feature)` and if the
operator is "mul", it will create the new feature `old_feature_1 x old_feature_2`.
The operators can be chosen through the `operators` parameter.
Available options are:
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
    DFS can create many new features and not all of them will be useful.
    Use [FeatureSelector](./API/feature_engineering/feature_selector.md)
    to reduce the number of features!

!!! warning
    Using the div, log or sqrt operators can return new features with
    `inf` or `NaN` values. Check the warnings that may pop up or use
    atom's [missing](/API/ATOM/atomclassifier/#properties) property.

!!! warning
    When using DFS with `n_jobs>1`, make sure to protect your code with
    `if __name__ == "__main__"`. Featuretools uses [dask](https://dask.org/),
    which uses python multiprocessing for parallelization. The spawn
    method on multiprocessing starts a new python process, which requires
    it to import the \__main__ module before it can do its task.

<br>

**Genetic Feature Generation**<br>
Genetic feature generation (GFG) uses [genetic programming](https://en.wikipedia.org/wiki/Genetic_programming),
a branch of evolutionary programming, to determine which features
are successful and create new ones based on those. Where DFS can be
seen as some kind of "brute force" for feature engineering, GFG tries
to improve its features with every generation of the algorithm. GFG
uses the same operators as DFS, but instead of only applying the
transformations once, it evolves them further, creating complicated
non-linear combinations of features with many transformations. The
new features are given the name `Feature N` for the N-th feature. You
can access the genetic feature's fitness and description (how they are
calculated) through the `genetic_features` attribute.

ATOM uses the [SymbolicTransformer](https://gplearn.readthedocs.io/en/stable/reference.html#symbolic-transformer)
 class from the [gplearn](https://gplearn.readthedocs.io/en/stable/index.html)
 package for the genetic algorithm. Read more about this implementation
 [here](https://gplearn.readthedocs.io/en/stable/intro.html#transformer).

!!!warning
    GFG can be slow for very large populations!

<br>

### Selecting useful features

The [FeatureSelector](API/feature_engineering/feature_selector.md) class
provides tooling to select the relevant features from a dataset. It can
be accessed from atom through the [feature_selection](../API/ATOM/atomclassifier/#feature-selection)
method. The following strategies are implemented: univariate, PCA, SFM,
RFE and RFECV.


**Univariate**<br>
Univariate feature selection works by selecting the best features based
on univariate statistical F-test. The test is provided via the `solver`
parameter. It takes any function taking two arrays (X, y), and returning
arrays (scores, p-values).

Read more in sklearn's [documentation](https://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection).


**Principal Components Analysis**<br>
Applying PCA will reduce the dimensionality of the dataset by maximizing
the variance of each dimension. The new features will be called Component
0, Component 1, etc... The dataset will be scaled before applying the
transformation (if it wasn't already).

Read more in sklearn's [documentation](https://scikit-learn.org/stable/modules/decomposition.html#pca).


**Selection from model**<br>
SFM uses an estimator with `feature_importances_` or `coef_` attributes
to select the best features in a dataset based on importance weights.
The estimator is provided through the `solver` parameter and can be
already fitted. ATOM allows you to use one its pre-defined [models](#models),
e.g. `solver="RF"`. If you didn't call the FeatureSelector through atom,
don't forget to indicate the estimator's task adding `_class` or `_reg`
after the name, e.g. `RF_class` to use a random forest classifier.

Read more in sklearn's [documentation](https://scikit-learn.org/stable/modules/feature_selection.html#feature-selection-using-selectfrommodel).


**Recursive feature elimination**<br>
Select features by recursively considering smaller and smaller sets of
features. First, the estimator is trained on the initial set of features
and the importance of each feature is obtained either through a `coef_`
attribute or through a `feature_importances_` attribute. Then, the least
important features are pruned from current set of features. That procedure
is recursively repeated on the pruned set until the desired number of
features to select is eventually reached. Note that, since RFE needs to
fit the model again every iteration, this method can be fairly slow.

RFECV applies the same algorithm as RFE but uses a cross-validated metric
(under the scoring parameter, see [RFECV](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html#sklearn.feature_selection.RFECV))
to assess every step's performance. Also, where RFE returns the number
of features selected by `n_features`, RFECV returns the number of
features that achieved the optimal score on the specified metric. Note
that this is not always equal to the amount specified by `n_features`.

Read more in sklearn's [documentation](https://scikit-learn.org/stable/modules/feature_selection.html#recursive-feature-elimination).


**Removing features with low variance**<br>
Variance is the expectation of the squared deviation of a random
variable from its mean. Features with low variance have many values
repeated, which means the model will not learn much from them.
[FeatureSelector](API/feature_engineering/feature_selector.md) removes
all features where the same value is repeated in at least
`max_frac_repeated` fraction of the rows. The default option is to
remove a feature if all values in it are the same.
 
Read more in sklearn's [documentation](https://scikit-learn.org/stable/modules/feature_selection.html#removing-features-with-low-variance).


**Removing features with multi-collinearity**<br>
Two features that are highly correlated are redundant, i.e. two will
not contribute more to the model than only one of them.
[FeatureSelector](API/feature_engineering/feature_selector.md) will
drop a feature that has a Pearson correlation coefficient larger than
`max_correlation` with another feature. A correlation of 1 means the
two columns are equal. A dataframe of the removed features and their
correlation values can be accessed through the `collinear` attribute.

<br>

!!! tip
    Use the [plot_feature_importance](API/plots/plot_feature_importance.md)
    method to examine how much a specific feature contributes to the
    final predictions. If the model doesn't have a `feature_importances_`
    attribute, use [plot_permutation_importance](API/plots/plot_permutation_importance.md) instead.

!!!warning
    The RFE and RFECV strategies don't work when the solver is a 
    [CatBoost](https://catboost.ai/) model due to incompatibility
    of the APIs.



<br><br>
# Models
--------

### Predefined models

ATOM provides 31 models for classification and regression tasks that
can be used to fit the data in the pipeline. After fitting, every model
class is attached to the trainer as an attribute. We refer to these
"subclasses" as models (see the [nomenclature](#nomenclature)). The
classes contain a variety of attributes and methods to help you
understand how the underlying estimator performed. They can be accessed
using their acronyms, e.g. `atom.LGB` to access the LightGBM's model.
The available models and their corresponding acronyms are: 

* "GP" for [Gaussian Process](../API/models/gp)
* "GNB" for [Gaussian Naive Bayes](../API/models/gnb)
* "MNB" for [Multinomial Naive Bayes](../API/models/mnb)
* "BNB" for [Bernoulli Naive Bayes](../API/models/bnb)
* "CatNB" for [Categorical Naive Bayes](../API/models/catnb)
* "CNB" for [Complement Naive Bayes](../API/models/cnb)
* "OLS" for [Ordinary Least Squares](../API/models/ols)
* "Ridge" for [Ridge Classification/Regression](../API/models/ridge)
* "Lasso" for [Lasso Regression](../API/models/lasso)
* "EN" for [Elastic Net](../API/models/en)
* "BR" for [Bayesian Ridge](../API/models/br)
* "ARD" for [Automated Relevance Determination](../API/models/ard)
* "LR" for [Logistic Regression](../API/models/lr)
* "LDA" for [Linear Discriminant Analysis](../API/models/lda)
* "QDA" for [Quadratic Discriminant Analysis](../API/models/qda)
* "KNN" for [K-Nearest Neighbors](../API/models/knn)
* "RNN" for [Radius Nearest Neighbors](../API/models/rnn)
* "Tree" for [Decision Tree](../API/models/tree)
* "Bag" for [Bagging](../API/models/bag)
* "ET" for [Extra-Trees](../API/models/et)
* "RF" for [Random Forest](../API/models/rf)
* "AdaB" for [AdaBoost](../API/models/adab)
* "GBM" for [Gradient Boosting Machine](../API/models/gbm)
* "XGB" for [XGBoost](../API/models/xgb)
* "LGB" for [LightGBM](../API/models/lgb)
* "CatB" for [CatBoost](../API/models/catb)
* "lSVM" for [Linear-SVM](../API/models/lsvm)
* "kSVM" for [Kernel-SVM](../API/models/ksvm)
* "PA" for [Passive Aggressive](../API/models/pa)
* "SGD" for [Stochastic Gradient Descent](../API/models/sgd)
* "MLP" for [Multi-layer Perceptron](../API/models/mlp)

!!! tip
    You can also use lowercase to call the models, e.g. `atom.lgb`.

!!! warning
    The models should not be initialized by the user! Only use them through the
    trainers.


<br>

### Custom models

It is also possible to use your own models in ATOM's pipeline. For
example, imagine we want to use sklearn's [Lars](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lars.html)
estimator (note that is not included in ATOM's [predefined models](#predefined-models)).
There are two ways to achieve this:

Using [ATOMModel](../API/ATOM/atommodel) (recommended). With this
approach you can pass the required model characteristics to the pipeline.

```python
from sklearn.linear_model import Lars
from atom import ATOMRegressor, ATOMModel

model = ATOMModel(models=Lars, fullname="Lars Regression", needs_scaling=True, type="linear")

atom = ATOMRegressor(X, y)
atom.run(model)
```

Using the estimator's class or an instance of the class. This approach will
also call [ATOMModel](../API/ATOM/atommodel) under the hood, but it will
leave its parameters to their default values.

```python
from sklearn.linear_model import Lars
from atom import ATOMRegressor, ATOMModel

atom = ATOMRegressor(X, y)
atom.run(models=Lars)
```

Additional things to take into account:

* Custom models are not restricted to sklearn estimators, but they should
  follow sklearn's API, i.e. have a fit and predict method.
* [Parameter customization](#parameter-customization) (for the initializer)
  is only possible for custom models which provide an estimator's class
  or an instance that has a `set_params()` method, i.e. it's a child class
  of [BaseEstimator](https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html).
* [Hyperparameter optimization](#hyperparameter-optimization) for custom
  models is ignored unless appropriate dimensions are provided through
  `bo_params`.
* If the estimator has a `n_jobs` and/or `random_state` parameter that is
  left to its default value, it will automatically adopt the values from
  the trainer it's called from.

<br>


### Deep learning

Deep learning models can be used through ATOM's [custom models](#custom-models)
as long as they follow sklearn's API. For example, models implemented
with the Keras package should use the sklearn wrappers
[KerasClassifier](https://www.tensorflow.org/api_docs/python/tf/keras/wrappers/scikit_learn/KerasClassifier)
or [kerasRegressor](https://www.tensorflow.org/api_docs/python/tf/keras/wrappers/scikit_learn/KerasRegressor).

Many deep learning models, for example in computer vision and natural
language processing, use datasets with more than 2 dimensions, e.g.
image data can have shape (n_samples, length, width, rgb). These data
structures are not intended to store in a two-dimensional pandas
dataframe. Since ATOM requires a dataframe as instance for the dataset,
multidimensional data sets are stored in a single column called "Features"
where every row contains one (multidimensional) sample. See the <a href="../examples/deep_learning.html" target="_blank">Deep learning</a>
example. Note that, because of this, the [data cleaning](#data-cleaning),
[feature engineering](#feature-engineering) and some of the [plotting](#plots)
methods are unavailable for deep learning datasets.





<br><br>
# Training
----------

The training phase is where the models are fitted and evaluated. After
this, the models are attached to the trainer and you can use the
[plotting](#plots) and [predicting](#predicting) methods.  The pipeline
applies the following steps iteratively for all models:

1. The [optimal hyperparameters](#hyperparameter-optimization) are selected.
2. The model is trained on the training set and evaluated on the test set.
3. The [bagging](#bagging) algorithm is applied.

There are three approaches to run the training.

* Direct training:
    - [DirectClassifier](../API/training/directclassifier)
    - [DirectRegressor](../API/training/directregressor)
* Training via [successive halving](#successive-halving):
    - [SuccessiveHalvingClassifier](../API/training/successivehalvingclassifier)
    - [SuccessiveHavingRegressor](../API/training/successivehalvingregressor)
* Training via [train sizing](#train-sizing):
    - [TrainSizingClassifier](../API/training/trainsizingclassifier)
    - [TrainSizingRegressor](../API/training/trainsizingregressor)

The direct fashion repeats the aforementioned steps only once, while the
other two approaches repeats them more than once. Every approach can be
directly called from atom through the [run](../API/ATOM/atomclassifier/#run),
[successive_halving](../API/ATOM/atomclassifier/#successive-halving)
and [train_sizing](../API/ATOM/atomclassifier/#train-sizing) methods
respectively. Every approach should be called from an independent instance
of atom. Subsequent runs from different approaches will raise an exception.
You can, however, rerun the same approach multiple times. In that case,
the results are combined. Note that if you rerun the same model, only
the last subclass is saved.

For example, in this case, only the Ridge and the last fitted Lasso
regressor (the one using bagging) are kept in the pipeline. The first
Lasso model will be overwritten and all its information will be lost.
Note that reruns are only allowed if the same metric is used. Leaving
the `metric` parameter empty after the first run will automatically use
the one in the pipeline.

```python
atom = ATOMRegressor(X, y)
atom.run("Ridge", metric="MSE")
atom.run("Lasso")
atom.run("Lasso", bagging=5)
```

Models are called through their [acronyms](#models), e.g. `atom.run(models="RF")`
will train a [Random Forest](../API/models/rf). If you want to run
the same model multiple times, add a tag after the acronym to
differentiate them.

```Python
atom.run(models=["RF1", "RF2"], est_params={"RF1": {"n_estimators": 100}, "RF2": {"n_estimators": 200}}) 
```

For example, this pipeline will fit two Random Forest models, one
with 100 and the other with 200 decision trees. The models can be
accessed through `atom.rf1` and `atom.rf2`. Use tagged models to
test how the same model performs when fitted with different
parameters or on different data sets. See the <a href="../examples/imbalanced_datasets.html" target="_blank">Imbalanced datasets</a>
 example.


Additional things to take into account:

* If an exception is encountered while fitting an estimator, the
  pipeline will automatically jump to the next model and save the
  exception in the `errors` attribute. Note that in that case,
  there will be no model for that estimator.
* When showing the final results, a `!` indicates the highest score
  and a `~` indicates that the model is possibly overfitting (training
  set has a score at least 20% higher than the test set).
* The winning model (the one with the highest `mean_bagging` or
  `metric_test`) will be attached to the `winner` attribute.

<br>

### Metric

ATOM uses sklearn's [SCORERS](https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules)
for model selection and evaluation. A scorer consists of a metric
function and some parameters that define the scorer's properties such
as it's a score or loss function or if the function needs probability
estimates or rounded predictions (see [make_scorer](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html)).
ATOM lets you define the scorer for the pipeline in three ways:

* The `metric` parameter is one of sklearn's predefined scorers
  (as string).
* The `metric` parameter is a score (or loss) function with signature
  metric(y, y_pred, **kwargs). In this case, use the `greater_is_better`,
  `needs_proba` and `needs_threshold` parameters to specify the scorer's
  properties.
* The `metric` parameter is a scorer object.


Note that all scorers follow the convention that higher return values
are better than lower return values. Thus metrics which measure the
distance between the model and the data (i.e. loss functions), like
`max_error` or `mean_squared_error`, will return the negated value of
the metric.


**Custom scorer acronyms**<br>
Since some of sklearn's scorers have quite long names and ATOM is all
about <s>lazy</s>fast experimentation, the package provides acronyms
for some of the most commonly used ones. These acronyms are case
insensitive can be used for the `metric` parameter instead of the
scorer's full name, e.g. `atom.run("LR", metric="BA")` will use
`balanced_accuracy`. The available acronyms are:

* "AP" for "average_precision"
* "BA" for "balanced_accuracy"
* "AUC" for "roc_auc"
* "EV" for "explained_variance"
* "ME" for "max_error"
* "MAE" for "neg_mean_absolute_error"
* "MSE" for "neg_mean_squared_error"
* "RMSE" for "neg_root_mean_squared_error"
* "MSLE" for "neg_mean_squared_log_error"
* "MEDAE" for "neg_median_absolute_error"
* "POISSON" for "neg_mean_poisson_deviance"
* "GAMMA" for "neg_mean_gamma_deviance"


**Multi-metric runs**<br>
Sometimes it is useful to measure the performance of the models in more
than one way. ATOM lets you run the pipeline with multiple metrics at
the same time. To do so, provide the `metric` parameter with a list of
desired metrics, e.g. `atom.run("LDA", metric=["r2", "mse"])`. If you
provide metric functions, don't forget to also provide lists to the
`greater_is_better`, `needs_proba` and `needs_threshold` parameters,
where the n-th value in the list corresponds to the n-th function. If
you leave them as a single value, that value will apply to every
provided metric.

When fitting multi-metric runs, the resulting scores will return a list
of metrics. For example, if you provided three metrics to the pipeline,
`atom.knn.metric_bo` could return [0.8734, 0.6672, 0.9001]. It is also
important to note that only the first metric of a multi-metric run is
used to evaluate every step of the bayesian optimization and to select
the winning model.

!!!tip
    Some plots let you choose which of the metrics to show using the
    `metric` parameter.

<br>

### Parameter customization

By default, the parameters every estimator uses are the same default
parameters they get from their respective packages. To select different
ones, use `est_params`. There are two ways to add custom parameters to
the models: adding them directly to the dictionary as key-value pairs
or through multiple dicts with the model names as keys.

Adding the parameters directly to `est_params` will share them across
all models in the pipeline. In this example, both the XGBoost and the
LightGBM model will use n_estimators=200. Make sure all the models do 
have the specified parameters or an exception will be raised!

```Python
atom.run(["XGB", "LGB"], est_params={"n_estimators": 200})
```

To specify parameters per model, use the model name as key and a dict
of the parameters as value. In this example, the XGBoost model will
use n_estimators=200 and the Multi-layer Perceptron will use one hidden
layer with 75 neurons.

```Python
atom.run(["XGB", "MLP"], est_params={"XGB": {"n_estimators": 200}, "MLP": {"hidden_layer_sizes": (75,)}})
```

Some estimators allow you to pass extra parameters to the fit method
(besides X and y). This can be done adding `_fit` at the end of the
parameter. For example, to change XGBoost's verbosity, we can run:

```Python
atom.run("XGB", est_params={"verbose_fit": True}
```

!!!note
    If a parameter is specified through `est_params`, it will be
    ignored by the bayesian optimization! 


<br>

### Hyperparameter optimization

In order to achieve maximum performance, we need to tune an estimator's
hyperparameters before training it. ATOM provides [hyperparameter tuning](https://en.wikipedia.org/wiki/Hyperparameter_optimization)
using a [bayesian optimization](https://en.wikipedia.org/wiki/Bayesian_optimization#:~:text=Bayesian%20optimization%20is%20a%20sequential,expensive%2Dto%2Devaluate%20functions.)
(BO) approach implemented by [skopt](https://scikit-optimize.github.io/stable/).
The BO is optimized on the first metric provided with the `metric`
parameter. Each step is either computed by cross-validation on the
complete training set or by randomly splitting the training set every
iteration into a (sub) training set and a validation set. This process
can create some data leakage but ensures maximal use of the provided
data. The test set, however, does not contain any leakage and will be
used to determine the final score of every model. Note that, if the
dataset is relatively small, the BO's best score can consistently be 
lower than the final score on the test set (despite the leakage) due
to the considerable fewer instances on which it is trained.

There are many possibilities to tune the BO to your liking. Use
`n_calls` and `n_initial_points` to determine the number of iterations
that are performed randomly at the start (exploration) and the number
of iterations spent optimizing (exploitation). If `n_calls` is equal to
`n_initial_points`, every iteration of the BO will select its
hyperparameters randomly. This means the algorithm is technically
performing a [random search](https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf).

!!!note
    The `n_calls` parameter includes the iterations in `n_initial_points`,
    i.e. calling `atom.run("LR", n_calls=20, n_intial_points=10)` will run
    20 iterations of which the first 10 are random.

!!!note
    If `n_initial_points=1`, the first trial will be equal to the
    estimator's default parameters.

Other settings can be changed through the `bo_params` parameter, a
dictionary where every key-value combination can be used to further
customize the BO.

By default, the hyperparameters and corresponding dimensions per model
are predefined by ATOM. Use the `dimensions` key to use custom ones.
Just like with `est_params`, you can share the same dimensions across
models or use a dictionary with the model names as keys to specify the
dimensions for every individual model. Note that the provided search
space dimensions must be compliant with skopt's API.

```Python
atom.run("LR", n_calls=10, bo_params={"dimensions": [Integer(100, 1000, name="max_iter")]})
```

The majority of skopt's callbacks to stop the optimizer early can be
accessed through `bo_params`. You can include other callbacks using
the `callbacks` key.

```Python
atom.run("LR", n_calls=10, bo_params={"max_time": 1000, "callbacks": custom_callback()})
```

You can also include other parameters for the optimizer as key-value pairs.

```Python
atom.run("LR", n_calls=10, bo_params={"acq_func": "EI"})
```


<br>

### Bagging

After fitting the estimator, you can asses the robustness of the model
using [bootstrap aggregating](https://en.wikipedia.org/wiki/Bootstrap_aggregating)
(bagging). This technique creates several new data sets selecting random 
samples from the training set (with replacement) and evaluates them on 
the test set. This way we get a distribution of the performance of the
model. The number of sets can be chosen through the `bagging` parameter.

!!!tip
    Use the [plot_results](../API/plots/plot_results) method to plot
    the bagging scores in a boxplot.


<br>

### Early stopping

[XGBoost](../API/models/xgb), [LightGBM](../API/models/lgb) and [CatBoost](../API/models/catb)
allow in-training evaluation. This means that the estimator is evaluated
after every round of the training, and that the training is stopped
early if it didn't improve in the last `early_stopping` rounds. This
can save the pipeline much time that would otherwise be wasted on an
estimator that is unlikely to improve further. Note that this technique
will be applied both during the BO and at the final fit on the complete
training set.

There are two ways to apply early stopping on these models:

* Through the `early_stopping` key in `bo_params`. This approach applies
  early stopping to all models in the pipeline and allows the input of a
  fraction of the total number of rounds.
* Filling the `early_stopping_rounds` parameter directly in `est_params`.
  Don't forget to add `_fit` to the parameter to call it from the fit method.

After fitting, the model will get the `evals` attribute, a dictionary of
the train and test performances per round (also if early stopping wasn't
applied). Click <a href="../examples/early_stopping.html" target="_blank">here</a>
for an example using early stopping.

!!!tip
    Use the [plot_evals](../API/plots/plot_evals) method to plot the
    in-training evaluation on the train and test set.


<br>

### Successive halving

Successive halving is a bandit-based algorithm that fits N models to
1/N of the data. The best half are selected to go to the next iteration
where the process is repeated. This continues until only one model
remains, which is fitted on the complete dataset. Beware that a model's
performance can depend greatly on the amount of data on which it is
trained. For this reason, we recommend only to use this technique with
similar models, e.g. only using tree-based models.

Use successive halving through the [SuccessiveHalvingClassifier](../API/training/successivehalvingclassifier)/[SuccessiveHalvingRegressor](../API/training/successivehalvingregressor)
classes or from atom via the [successive_halving](../API/ATOM/atomclassifier/#successive-halving)
method. After running the pipeline, the `results` attribute will be
multi-index, where the first index indicates the number of models (N)
in the iteration and the second the model's acronym. Note that only
the last trained model from every class (the one trained on the largest
data set) is stored under its normal attribute.

Click <a href="../examples/successive_halving.html" target="_blank">here</a> for a
successive halving example.

!!!tip
    Use the [plot_successive_halving](../API/plots/plot_successive_halving)
    method to see every model's performance per iteration of the
    successive halving.

<br>

### Train sizing

When training models, there is usually a trade-off between model
performance and computation time that is regulated by the number of
samples in the training set. Train sizing can be used to create
insights in this trade-off and help determine the optimal size of
the training set, fitting the models multiple times, ever increasing
the number of samples in the training set.

Use train sizing through the [TrainSizingClassifier](../API/training/trainsizingclassifier)/[TrainSizingRegressor](../API/training/trainsizingregressor)
classes or from atom via the [train_sizing](../API/ATOM/atomclassifier/#train-sizing)
method. The number of iterations and the number of samples per training
can be specified with the `train_sizes` parameter. After running the
pipeline, the `results` attribute will be multi-index, where the first
index indicates the fraction of rows in the training set and the second
the model's acronym. Note that only the last trained model from every
class (the one trained on the largest data set) is stored under its
normal attribute.

Click <a href="../examples/train_sizing.html" target="_blank">here</a> for a
train sizing example.

!!!tip
    Use the [plot_learning_curve](../API/plots/plot_learning_curve)
    method to see the model's performance per size of the training set.

<br>

### Voting

The idea behind Voting is to combine the predictions of conceptually
different models to make new predictions. Such a technique can be
useful for a set of equally well performing models in order to balance
out their individual weaknesses. Read more in sklearn's [documentation](https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier).

A Voting model is created from a trainer through the [voting](../API/ATOM/atomclassifier/#voting)
method. The Voting model will be added automatically to the list of
models in the pipeline, under the `Vote` acronym. Although similar,
this model is different from the VotingClassifier and VotingRegressor
estimators from sklearn. Remember that the model will be added to the
plots if the `models` parameter is not specified. Plots that require
a data set will use the one in the current branch. Plots that require
an estimator object will raise an exception.

The Voting class has the same prediction attributes and prediction
methods as other models. The `predict_proba`, `predict_log_proba`,
`decision_function` and `score` methods return the average predictions
(soft voting) over the models in the instance. Note that these methods
will raise an exception if not all estimators in the Voting instance
have the specified method. The `predict` method returns the majority
vote (hard voting). The `scoring` method also returns the average
scoring for the selected metric over the models.

Click <a href="../examples/ensembles.html" target="_blank">here</a> for
a voting example.

!!!warning
    Although it is possible to include models from different branches
    in the same Voting instance, this is highly discouraged. Data sets
    from different branches with unequal shape can result in unexpected
    errors for plots and prediction methods.


<br>

### Stacking

Stacking is a method for combining estimators to reduce their biases.
More precisely, the predictions of each individual estimator are
stacked together and used as input to a final estimator to compute the
prediction. Read more in sklearn's [documentation](https://scikit-learn.org/stable/modules/ensemble.html#stacked-generalization).

A Stacking model is created from a trainer through the [stacking](../API/ATOM/atomclassifier/#stacking)
method. The Stacking model will be added automatically to the list of
models in the pipeline, under the `Stack` acronym. Remember that the
model will be added to the plots if the `models` parameter is not
specified. Plots that require a data set will use the one in the
current branch. The prediction methods, the scoring method and the
plot methods that require an estimator object will use the Voting's
final estimator, under the `estimator` attribute.

Click <a href="../examples/ensembles.html" target="_blank">here</a> for
a stacking example.

!!!warning
    Although it is possible to include models from different branches
    in the same Stacking instance, this is highly discouraged. Data
    sets from different branches with unequal shape can result in
    unexpected errors for plots and prediction methods.


<br><br>
# Predicting
------------

After running a successful pipeline, it is possible you would like to
apply all used transformations onto new data, or make predictions using
one of the trained models. Just like a sklearn estimator, you can call
the prediction methods from a fitted trainer, e.g. `atom.predict(X)`.
Calling the method without specifying a model will use the winning model
in the pipeline (under attribute `winner`). To use a different model,
simply call the method from a model, e.g. `atom.KNN.predict(X)`.

All prediction methods transform the provided data through the data
cleaning and feature engineering transformers before making the
predictions. By default, this excludes outlier handling and balancing
the dataset since these steps should only be applied on the training
set. Use the method's kwargs to select which transformations to use
in every call.

The available prediction methods are a selection of the most common
methods for estimators in sklearn's API:

<table>
<tr>
<td width="15%"><a href="../API/predicting/transform">transform</a></td>
<td>Transform new data through all transformers in a branch.</td>
</tr>

<tr>
<td width="15%"><a href="../API/predicting/predict">predict</a></td>
<td>Transform new data through all transformers in a branch and return class predictions.</td>
</tr>

<tr>
<td width="15%"><a href="../API/predicting/predict_proba">predict_proba</a></td>
<td>Transform new data through all transformers in a branch and return class predictions.</td>
</tr>

<tr>
<td width="15%"><a href="../API/predicting/predict_log_proba">predict_log_proba</a></td>
<td>Transform new data through all transformers in a branch and return class log-probabilities. </td>
</tr>

<tr>
<td width="15%"><a href="../API/predicting/decision_function">decision_function</a></td>
<td>Transform new data through all transformers in a branch and return predicted confidence scores.</td>
</tr>

<tr>
<td width="15%"><a href="../API/predicting/score">score</a></td>
<td>Transform new data through all transformers in a branch and return the model's score.</td>
</tr>
</table>

Except for transform, the prediction methods can be calculated on the train
and test set. You can access them through the model's prediction attributes,
e.g. `atom.mnb.predict_train` or ` atom.mnb.predict_test`. Keep in mind that
the results are not calculated until the attribute is called for the first
time. This mechanism avoids having to calculate attributes that are never
used, saving time and memory.

!!!note
    Many of the [plots](#plots) use the prediction attributes. This can
    considerably increase the size of the class for large datasets. Use
    the `reset_predictions` method if you need to free some memory!



<br><br>
# Plots
-------

After fitting the models to the data, it's time to analyze the results.
ATOM provides many plotting methods to compare the model performances.
Descriptions and examples can be found in the API section. ATOM uses
the packages [matplotlib](https://matplotlib.org/), [seaborn](https://seaborn.pydata.org/)
and [shap](https://github.com/slundberg/shap) for plotting.

The plot methods can be called from a `training` directly, e.g.
`atom.plot_roc()`, or from one of the models, e.g. `atom.LGB.plot_roc()`.
If called from `training`, it will make the plot for all models in
the pipeline. This can be useful to compare the results of multiple
models. If called from a model, it will make the plot for only that
model. Use this option if you want information just for that specific
model or to make a plot less crowded.

<br>

### Parameters

Apart from the plot-specific parameters they may have, all plots have
four parameters in common:

* The `title` parameter allows you to add a custom title to the plot.
* The `figsize` parameter adjust the plot's size.
* The `filename` parameter is used to save the plot.
* The `display` parameter determines whether the plot is rendered.

<br>

### Aesthetics

The plot aesthetics can be customized using the plot attributes, e.g.
`atom.style = "white"`. These attributes can be called from any instance
with plotting methods. Note that the plot attributes are attached to the
class and not the instance. This means that changing the attribute will
also change it for all other instances in the module. ATOM's default
values are:

* style: "darkgrid"
* palette: "GnBu_r_d"
* title_fontsize: 20
* label_fontsize: 16
* tick_fontsize: 12

Use the `reset_aesthetics` method to reset all the aesthetics to their
default value.

<br>


### Canvas

Sometimes it is desirable to draw multiple plots side by side in order
to be able to compare them easier. Use the atom's [canvas](../API/ATOM/atomclassifier/#canvas)
method for this. The canvas method is a `@contextmanager`, i.e. it is
used through the `with` command. Plots in a canvas will ignore the
figsize, filename and display parameters. Instead, call these parameters
from the canvas for the final figure.

For example, we can use a canvas to compare the results of a [XGBoost](../API/models/xgb)
and [LightGBM](../API/models/lgb) model on the train and test set. We
could also draw the lines for both models in the same axes, but then
the plot could become too messy.

```python
atom = ATOMClassifier(X, y)
atom.run(["xgb", "lgb"], n_calls=0)

with atom.canvas(2, 2, title="XGBoost vs LightGBM", filename="canvas"):
    atom.xgb.plot_roc(dataset="both", title="ROC - XGBoost")
    atom.lgb.plot_roc(dataset="both", title="ROC - LightGBM")
    atom.xgb.plot_prc(dataset="both", title="PRC - XGBoost")
    atom.lgb.plot_prc(dataset="both", title="PRC - LightGBM")
```
<div align="center">
    <img src="../../../img/plots/canvas.png" alt="canvas" width="1000" height="700"/>
</div>


<br>


### SHAP

The [SHAP](https://github.com/slundberg/shap) (SHapley Additive exPlanations)
python package uses a game theoretic approach to explain the output of
any machine learning model. It connects optimal credit allocation with
local explanations using the classic Shapley values from game theory
and their related extensions. ATOM implements methods to plot 5 of
shap's plotting functions directly from its API. The explainer will be
chosen automatically based on the model's type. For kernelExplainer,
the data used to estimate the expected values is the complete training
set when <100 rows, else its summarized with a set of 10 weighted
K-means, each weighted by the number of points they represent. The five
plots are: [force_plot](../API/plots/force_plot), [dependence_plot](../API/plots/dependence_plot),
[summary_plot](../API/plots/summary_plot), [decision_plot](../API/plots/decision_plot)
and [waterfall_plot](../API/plots/waterfall_plot).

Since the plots are not made by ATOM, we can't draw multiple models in
the same figure. Selecting more than one model will raise an exception.
To avoid this, call the plot from a model, e.g. `atom.xgb.force_plot()`.

!!!info
    You can recognize the SHAP plots by the fact that they end (instead
    of start) with plot.

<br>

### Available plots

A list of available plots can be find hereunder. Note that not all plots can be
 called from every class and that their availability can depend on the task at hand.

<table>
<tr>
<td width="15%"><a href="../API/plots/plot_correlation">plot_correlation</a></td>
<td>Plot the data's correlation matrix.</td>
</tr>

<tr>
<td width="15%"><a href="../API/plots/plot_pipeline">plot_pipeline</a></td>
<td>Plot a diagram of every estimator in atom's pipeline.</td>
</tr>

<tr>
<td width="15%"><a href="../API/plots/plot_pca">plot_pca</a></td>
<td>Plot the explained variance ratio vs the number of components.</td>
</tr>

<tr>
<td width="15%"><a href="../API/plots/plot_components">plot_components</a></td>
<td>Plot the explained variance ratio per components.</td>
</tr>

<tr>
<td width="15%"><a href="../API/plots/plot_rfecv">plot_rfecv</a></td>
<td>Plot the RFECV results.</td>
</tr>

<tr>
<td width="15%"><a href="../API/plots/plot_successive_halving">plot_successive_halving</a></td>
<td>Plot of the models" scores per iteration of the successive halving.</td>
</tr>

<tr>
<td width="15%"><a href="../API/plots/plot_learning_curve">plot_learning_curve</a></td>
<td>Plot the model's learning curve.</td>
</tr>

<tr>
<td width="15%"><a href="../API/plots/plot_results">plot_results</a></td>
<td>Plot a boxplot of the bagging's results.</td>
</tr>

<tr>
<td width="15%"><a href="../API/plots/plot_bo">plot_bo</a></td>
<td>Plot the bayesian optimization scoring.</td>
</tr>

<tr>
<td width="15%"><a href="../API/plots/plot_evals">plot_evals</a></td>
<td>Plot evaluation curves for the train and test set.</td>
</tr>

<tr>
<td width="15%"><a href="../API/plots/plot_roc">plot_roc</a></td>
<td>Plot the Receiver Operating Characteristics curve.</td>
</tr>

<tr>
<td width="15%"><a href="../API/plots/plot_prc">plot_prc</a></td>
<td>Plot the precision-recall curve.</td>
</tr>

<tr>
<td width="15%"><a href="../API/plots/plot_permutation_importance">plot_permutation_importance</a></td>
<td>Plot the feature permutation importance of models.</td>
</tr>

<tr>
<td width="15%"><a href="../API/plots/plot_feature_importance">plot_feature_importance</a></td>
<td>Plot a tree-based model's feature importance.</td>
</tr>

<tr>
<td width="15%"><a href="../API/plots/plot_partial_dependence">plot_partial_dependence</a></td>
<td>Plot the partial dependence of features.</td>
</tr>

<tr>
<td width="15%"><a href="../API/plots/plot_errors">plot_errors</a></td>
<td>Plot a model's prediction errors.</td>
</tr>

<tr>
<td width="15%"><a href="../API/plots/plot_residuals">plot_residuals</a></td>
<td>Plot a model's residuals.</td>
</tr>

<tr>
<td width="15%"><a href="../API/plots/plot_confusion_matrix">plot_confusion_matrix</a></td>
<td>Plot a model's confusion matrix.</td>
</tr>

<tr>
<td width="15%"><a href="../API/plots/plot_threshold">plot_threshold</a></td>
<td>Plot metric performances against threshold values.</td>
</tr>

<tr>
<td width="15%"><a href="../API/plots/plot_probabilities">plot_probabilities</a></td>
<td>Plot the probability distribution of the classes in the target column.</td>
</tr>

<tr>
<td width="15%"><a href="../API/plots/plot_calibration">plot_calibration</a></td>
<td>Plot the calibration curve for a binary classifier.</td>
</tr>

<tr>
<td width="15%"><a href="../API/plots/plot_gains">plot_gains</a></td>
<td>Plot the cumulative gains curve.</td>
</tr>

<tr>
<td width="15%"><a href="../API/plots/plot_lift">plot_lift</a></td>
<td>Plot the lift curve.</td>
</tr>

<tr>
<td width="15%"><a href="../API/plots/force_plot">force_plot</a></td>
<td>Plot SHAP's force plot.</td>
</tr>

<tr>
<td width="15%"><a href="../API/plots/dependence_plot">dependence_plot</a></td>
<td>Plot SHAP's dependence plot.</td>
</tr>

<tr>
<td width="15%"><a href="../API/plots/summary_plot">summary_plot</a></td>
<td>Plot SHAP's summary plot.</td>
</tr>

<tr>
<td width="15%"><a href="../API/plots/decision_plot">decision_plot</a></td>
<td>Plot SHAP's decision plot.</td>
</tr>

<tr>
<td width="15%"><a href="../API/plots/waterfall_plot">waterfall_plot</a></td>
<td>Plot SHAP's waterfall plot.</td>
</tr>
</table>
