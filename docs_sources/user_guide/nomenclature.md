# Nomenclature
--------------

This documentation consistently uses terms to refer to certain concepts
related to this package. The most frequent terms are described hereunder.

**atom**<br>
Instance of the [ATOMClassifier](../../API/ATOM/atomclassifier) or
 [ATOMRegressor](../../API/ATOM/atomregressor) classes (note that the
examples use it as the default variable name).

**ATOM**<br>
Refers to this package.

**branch**<br>
Collection of transformers fitted to a specific dataset. See
the [branches](../data_pipelines/#branches) section.


**BO**<br>
Bayesian optimization algorithm used for hyperparameter tuning.

**categorical columns**<br>
Refers to all columns of type `object` or `category`.

**class**<br>
Unique value in a column, e.g. a binary classifier has 2 classes in the
target column.

**estimator**<br>
An object which manages the estimation and decoding of an algorithm.
The algorithm is estimated as a deterministic function of a set of
parameters, a dataset and a random state.

**missing values**<br>
All values in the `missing` attribute, as well as `None`, `NaN`, `+inf`
and `-inf`.

**model**<br>
Instance of a [model](../models) in the pipeline.


**outlier**<br>
Sample that contains one or more outlier values. Note that the
[Pruner](../../API/data_cleaning/pruner) class can use a different
definition for outliers depending on the chosen strategy.

**outlier value**<br>
Value that lies further than 3 times the standard deviation away
from the mean of its column, i.e. |z-score| > 3.


**pipeline**<br>
Dataset, transformers and models in a specific branch.

**scorer**<br>
A non-estimator callable object which evaluates an estimator on given
test data, returning a number. Unlike evaluation metrics, a greater
returned number must correspond with a better score. See sklearn's
[documentation](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter).

**sequence**<br>
A one-dimensional array of type `list`, `tuple`, `np.ndarray` or `pd.Series`.

**target**<br>
Name of the dependent variable, passed as y to an estimator's fit method.

**task**<br>
One of the three supervised machine learning approaches that ATOM supports:

<ul style="line-height:1.2em;margin-top:-10px">
<li><a href="https://en.wikipedia.org/wiki/Binary_classification">binary classification</a></li>
<li><a href="https://en.wikipedia.org/wiki/Multiclass_classification">multiclass classification</a></li>
<li><a href="https://en.wikipedia.org/wiki/Regression_analysis">regression</a></li>
</ul>

**trainer**<br>
Instance of a class that trains and evaluates the models (implements a
`run` method). The following classes are considered trainers:

<ul style="line-height:1.2em;margin-top:-10px">
<li><a href="../../API/ATOM/atomclassifier">ATOMClassifier</a></li>
<li><a href="../../API/ATOM/atomregressor">ATOMRegressor</a></li>
<li><a href="../../API/training/directclassifier">DirectClassifier</a></li>
<li><a href="../../API/training/directregressor">DirectRegressor</a></li>
<li><a href="../../API/training/successivehalvingclassifier">SuccessiveHalvingClassifier</a></li>
<li><a href="../../API/training/successivehalvingregressor">SuccessiveHavingRegressor</a></li>
<li><a href="../../API/training/trainsizingclassifier">TrainSizingClassifier</a></li>
<li><a href="../../API/training/trainsizingregressor">TrainSizingRegressor</a></li>
</ul>

**transformer**<br>
An estimator implementing a `transform` method. This encompasses all
 data cleaning and feature engineering classes.
