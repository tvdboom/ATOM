# Nomenclature
--------------

This documentation consistently uses terms to refer to certain concepts
related to this package. The most frequent terms are described hereunder.

<br>

<strong>ATOM</strong>
<div markdown style="margin: -1em 0 0 1.2em">
Refers to this package.
</div>

[](){#atom}
<strong>atom</strong>
<div markdown style="margin: -1em 0 0 1.2em">
Instance of the [ATOMClassifier][], [ATOMForecaster][] or [ATOMRegressor][]
classes (note that the examples use it as the default variable name).
</div>

[](){#categorical-columns}
<strong>categorical columns</strong>
<div markdown style="margin: -1em 0 0 1.2em">
Refers to all columns of type `object`, `category`, `string` or `boolean`.
</div>

[](){#class}
<strong>class</strong>
<div markdown style="margin: -1em 0 0 1.2em">
Unique value in a column, e.g., a binary classifier has two classes in
the target column.
</div>

[](){#dataframe}
<strong>dataframe</strong>
<div markdown style="margin: -1em 0 0 1.2em">
Two-dimensional, size-mutable, potentially heterogeneous tabular data.
The type is usually [pd.DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html),
but could potentially be any of the dataframe types backed by the
selected [data engine][data-engines].
</div>

[](){#dataframe-like}
<strong>dataframe-like</strong>
<div markdown style="margin: -1em 0 0 1.2em">
Any type object from which a [pd.DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)
can be created. This includes an [iterable](https://docs.python.org/3/glossary.html#term-iterable),
a [dict](https://docs.python.org/3/library/functions.html#func-dict) whose
values are 1d-arrays, a two-dimensional [list](https://docs.python.org/3/library/functions.html#func-list),
[tuple](https://docs.python.org/3/library/functions.html#func-tuple), [np.ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html) or
[sps.csr_matrix](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html),
or any object that follows the [dataframe interchange protocol](https://data-apis.org/dataframe-protocol/latest/index.html).
This is the standard input format for any dataset.

Additionally, you can provide a callable whose output is any of the
aforementioned types. This is useful when the dataset is very large and
you are performing [parallel operations][parallel-execution], since it
can avoid broadcasting a large dataset from the driver to the workers.
</div>

[](){#estimator}
<strong>estimator</strong>
<div markdown style="margin: -1em 0 0 1.2em">
An object which manages the estimation and decoding of an algorithm.
The algorithm is estimated as a deterministic function of a set of
parameters, a dataset and a random state. Should implement a `fit`
method. Often used interchangeably with [predictor][] because of user
preference.
</div>

[](){#missing values}
<strong>missing values</strong>
<div markdown style="margin: -1em 0 0 1.2em">
All values in the [`missing`][atomclassifier-missing] attribute, as
well as `None`, `NaN`, `+inf` and `-inf`.
</div>

[](){#model}
<strong>model</strong>
<div markdown style="margin: -1em 0 0 1.2em">
Instance of a [model][models] in atom. Not to confuse with [estimator][].
</div>

[](){#outliers}
<strong>outliers</strong>
<div markdown style="margin: -1em 0 0 1.2em">
Sample that contains one or more outlier values. Note that the
[Pruner][] class can use a different definition for outliers
depending on the chosen strategy.
</div>

[](){#outlier-value}
<strong>outlier value</strong>
<div markdown style="margin: -1em 0 0 1.2em">
Value that lies further than 3 times the standard deviation away
from the mean of its column, i.e., |z-score| > 3.
</div>

[](){#predictor}
<strong>predictor</strong>
<div markdown style="margin: -1em 0 0 1.2em">
An estimator implementing a `predict` method.
</div>

[](){#scorer}
<strong>scorer</strong>
<div markdown style="margin: -1em 0 0 1.2em">
A non-estimator callable object which evaluates an estimator on given
test data, returning a number. Unlike evaluation metrics, a greater
returned number must correspond with a better score. See sklearn's
[documentation](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter).
</div>

[](){#segment}
<strong>segment</strong>
<div markdown style="margin: -1em 0 0 1.2em">
Subset (segment) of a sequence, whether through slicing or generating a
range of values. When given as a parameter type, it includes both
[range](https://docs.python.org/3/library/functions.html#func-range)
and [slice](https://docs.python.org/3/library/functions.html#slice).
</div>

[](){#sequence}
<strong>sequence</strong>
<div markdown style="margin: -1em 0 0 1.2em">
A one-dimensional, indexable array of type [sequence](https://docs.python.org/3/library/stdtypes.html#sequence-types-list-tuple-range)
(except string), [np.ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html),
[pd.Index](https://pandas.pydata.org/docs/reference/api/pandas.Index.html) or [series][].
This is the standard input format for a dataset's target column.
</div>

[](){#series}
<strong>series</strong>
<div markdown style="margin: -1em 0 0 1.2em">
One-dimensional ndarray with axis labels. The type is usually
[pd.Series](https://pandas.pydata.org/docs/reference/api/pandas.Series.html#pandas.Series),
but could potentially be any of the series types backed by the
selected [data engine][data-engines].
</div>

[](){#target}
<strong>target</strong>
<div markdown style="margin: -1em 0 0 1.2em">
The dependent variable in a supervised learning task. Passed as `y` to
an estimator's fit method.
</div>

[](){#task}
<strong>task</strong>
<div markdown style="margin: -1em 0 0 1.2em">
One of the supervised machine learning approaches that ATOM supports:
<ul style="line-height:1.2em;margin-top:-10px">
<li><a href="https://en.wikipedia.org/wiki/Binary_classification">binary classification</a></li>
<li><a href="https://en.wikipedia.org/wiki/Multiclass_classification">multiclass classification</a></li>
<li><a href="https://scikit-learn.org/stable/modules/multiclass.html#multilabel-classification">multilabel classification</a></li>
<li><a href="https://scikit-learn.org/stable/modules/multiclass.html#multiclass-multioutput-classification">multiclass-multioutput classification</a></li>
<li><a href="https://en.wikipedia.org/wiki/Regression_analysis">regression</a></li>
<li><a href="https://scikit-learn.org/stable/modules/multiclass.html#multioutput-regression">multioutput regression</a></li>
<li><a href="https://en.wikipedia.org/wiki/Forecasting">univariate forecast</a></li>
<li><a href="https://www.sktime.net/en/latest/examples/01_forecasting.html#1.2.4.-Multivariate-forecasting">multivariate forecast</a></li>
</ul>
</div>

[](){#transformer}
<strong>transformer</strong>
<div markdown style="margin: -1em 0 0 1.2em">
An estimator implementing a `transform` method. This encompasses all
data cleaning and feature engineering classes.
</div>
