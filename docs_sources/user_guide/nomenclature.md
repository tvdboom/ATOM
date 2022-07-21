# Nomenclature
--------------

This documentation consistently uses terms to refer to certain concepts
related to this package. The most frequent terms are described hereunder.

<br>

#### dataframe-like
Any type object from which a [pd.DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)
can be created. This includes an [iterable](https://docs.python.org/3/glossary.html#term-iterable),
a [dict](https://docs.python.org/3/library/functions.html#func-dict)
whose values are 1d-arrays, a two-dimensional [list](https://docs.python.org/3/library/functions.html#func-list),
[tuple](https://docs.python.org/3/library/functions.html#func-tuple), [np.ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html) or
[sps.csr_matrix](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html),
and most commonly, a dataframe. This is the standard input format
for any dataset.

<br>

#### ATOM
Refers to this package.

<br>

#### atom
Instance of the [ATOMClassifier](../../API/ATOM/atomclassifier) or
[ATOMRegressor](../../API/ATOM/atomregressor) classes (note that the
examples use it as the default variable name).

<br>

#### branch
Collection of transformers fitted to a specific dataset. See
the [branches](../data_management/#branches) section.

<br>

#### categorical columns
Refers to all columns of type `object` or `category`.

<br>

#### class
Unique value in a column, e.g. a binary classifier has 2 classes in the
target column.

<br>


#### estimator
An object which manages the estimation and decoding of an algorithm.
The algorithm is estimated as a deterministic function of a set of
parameters, a dataset and a random state. Should implement a `fit`
method. Often used interchangeably with <a href="#predictor">predictor</a>
because of user preference.

<br>

#### missing values
All values in the class' `missing` attribute, as well as `None`, `NaN`,
`+inf` and `-inf`.

<br>

#### model
Instance of a [model](../models) in the pipeline. Not to confuse with
[estimator](#estimator)


#### outlier
Sample that contains one or more outlier values. Note that the
[Pruner](../../API/data_cleaning/pruner) class can use a different
definition for outliers depending on the chosen strategy.

<br>

#### outlier value
Value that lies further than 3 times the standard deviation away
from the mean of its column, i.e. |z-score| > 3.

<br>

#### pipeline
Dataset, transformers and models in a specific branch.

<br>

#### predictor
An estimator implementing a `predict` method.

<br>

#### scorer
A non-estimator callable object which evaluates an estimator on given
test data, returning a number. Unlike evaluation metrics, a greater
returned number must correspond with a better score. See sklearn's
[documentation](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter).

<br>

#### sequence
A one-dimensional array of type [list](https://docs.python.org/3/library/functions.html#func-list),
[tuple](https://docs.python.org/3/library/functions.html#func-tuple),
[np.ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html)
or [pd.Series](https://pandas.pydata.org/docs/reference/api/pandas.Series.html).
This is the standard input format for a dataset's target column.

<br>

#### target
Name of the dependent variable, passed as y to an estimator's fit method.

<br>

#### task
One of the three supervised machine learning approaches that ATOM supports:

<ul style="line-height:1.2em;margin-top:-10px">
<li><a href="https://en.wikipedia.org/wiki/Binary_classification">binary classification</a></li>
<li><a href="https://en.wikipedia.org/wiki/Multiclass_classification">multiclass classification</a></li>
<li><a href="https://en.wikipedia.org/wiki/Regression_analysis">regression</a></li>
</ul>

<br>

#### trainer
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

<br>

#### transformer
An estimator implementing a `transform` method. This encompasses all
data cleaning and feature engineering classes.
