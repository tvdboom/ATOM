# Nomenclature
--------------

This documentation consistently uses terms to refer to certain concepts
related to this package. The most frequent terms are described hereunder.

<br>

<div id="ATOM"><strong>ATOM</strong></div>
<div markdown style="margin: -1em 0 0 1.2em">
Refers to this package.
</div>

<div id="atom"><strong>atom</strong></div>
<div markdown style="margin: -1em 0 0 1.2em">
Instance of the [ATOMClassifier][] or [ATOMRegressor][] classes (note
that the examples use it as the default variable name).
</div>

<a id="branch"></a><strong>branch</strong>
<div markdown style="margin: -1em 0 0 1.2em">
A [pipeline][], corresponding dataset and models fitted to that dataset.
See the [branches][] section of the user guide.
</div>

<div id="categorical-columns"><strong>categorical columns</strong></div>
<div markdown style="margin: -1em 0 0 1.2em">
Refers to all columns of type `object` or `category`.
</div>

<div id="class"><strong>class</strong></div>
<div markdown style="margin: -1em 0 0 1.2em">
Unique value in a column, e.g. a binary classifier has 2 classes in
the target column.
</div>

<div id="dataframe-like"><strong>dataframe-like</strong></div>
<div markdown style="margin: -1em 0 0 1.2em">
Any type object from which a [pd.DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)
can be created. This includes an [iterable](https://docs.python.org/3/glossary.html#term-iterable),
a [dict](https://docs.python.org/3/library/functions.html#func-dict)
whose values are 1d-arrays, a two-dimensional [list](https://docs.python.org/3/library/functions.html#func-list),
[tuple](https://docs.python.org/3/library/functions.html#func-tuple), [np.ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html) or
[sps.csr_matrix](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html),
and most commonly, a dataframe. This is the standard input format
for any dataset.
</div>

<div id="estimator"><strong>estimator</strong></div>
<div markdown style="margin: -1em 0 0 1.2em">
An object which manages the estimation and decoding of an algorithm.
The algorithm is estimated as a deterministic function of a set of
parameters, a dataset and a random state. Should implement a `fit`
method. Often used interchangeably with [predictor][] because of user
preference.
</div>

<div id="missing values"><strong>missing values</strong></div>
<div markdown style="margin: -1em 0 0 1.2em">
All values in the [`missing`][atomclassifier-missing] attribute, as
well as `None`, `NaN`, `+inf` and `-inf`.
</div>

<div id="model"><strong>model</strong></div>
<div markdown style="margin: -1em 0 0 1.2em">
Instance of a [model][models] in the pipeline. Not to confuse with
[estimator][].
</div>

<div id="outliers"><strong>outliers</strong></div>
<div markdown style="margin: -1em 0 0 1.2em">
Sample that contains one or more outlier values. Note that the
[Pruner][] class can use a different definition for outliers
depending on the chosen strategy.
</div>

<div id="outlier-value"><strong>outlier value</strong></div>
<div markdown style="margin: -1em 0 0 1.2em">
Value that lies further than 3 times the standard deviation away
from the mean of its column, i.e. |z-score| > 3.
</div>

<div id="pipeline"><strong>pipeline</strong></div>
<div markdown style="margin: -1em 0 0 1.2em">
Sequence of transformers in a specific (usually the current) [branch][].
</div>

<div id="predictor"><strong>predictor</strong></div>
<div markdown style="margin: -1em 0 0 1.2em">
An estimator implementing a `predict` method.
</div>

<div id="scorer"><strong>scorer</strong></div>
<div markdown style="margin: -1em 0 0 1.2em">
A non-estimator callable object which evaluates an estimator on given
test data, returning a number. Unlike evaluation metrics, a greater
returned number must correspond with a better score. See sklearn's
[documentation](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter).
</div>

<div id="sequence"><strong>sequence</strong></div>
<div markdown style="margin: -1em 0 0 1.2em">
A one-dimensional array of type [list](https://docs.python.org/3/library/functions.html#func-list),
[tuple](https://docs.python.org/3/library/functions.html#func-tuple),
[np.ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html)
or [pd.Series](https://pandas.pydata.org/docs/reference/api/pandas.Series.html).
This is the standard input format for a dataset's target column.
</div>

<div id="target"><strong>target</strong></div>
<div markdown style="margin: -1em 0 0 1.2em">
The dependent variable in a supervised learning task. Passed as `y` to
an estimator's fit method.
</div>

<div id="task"><strong>task</strong></div>
<div markdown style="margin: -1em 0 0 1.2em">
One of the three supervised machine learning approaches that ATOM supports:
<ul style="line-height:1.2em;margin-top:-10px">
<li><a href="https://en.wikipedia.org/wiki/Binary_classification">binary classification</a></li>
<li><a href="https://en.wikipedia.org/wiki/Multiclass_classification">multiclass classification</a></li>
<li><a href="https://en.wikipedia.org/wiki/Regression_analysis">regression</a></li>
</ul>
</div>

<div id="transformer"><strong>transformer</strong></div>
<div markdown style="margin: -1em 0 0 1.2em">
An estimator implementing a `transform` method. This encompasses all
data cleaning and feature engineering classes.
</div>
