# Data management
-----------------

## Data sets

ATOM is designed to work around one single dataset: the one with which
atom is initialized. This is the dataset you want to explore, transform,
and use for model training and validation. ATOM differentiates three
different data sets:

* The **training set** is usually the largest of the data sets. As the
  name suggests, this set is used to train the pipeline. During
  [hyperparameter tuning][], only the training set is used to fit and
  evaluate the estimator in every call. The training set in the current
  [branch][branches] can be accessed through the `train` attribute. It's
  features and target can be accessed through `X_train` and `y_train`
  respectively.
* The **test set** is used to evaluate the models. The model scores on
  this set give an indication on how the model performs on new data. The
  test set can be accessed through the `test` attribute. It's features
  and target can be accessed through `X_test` and `y_test` respectively.
* The **holdout set** is an optional, separate set that should only be
  used to evaluate the final model's performance. Create this set when
  you are going to use the test set for an intermediate validation step.
  The holdout set is immediately set apart during initialization and is
  not considered part of atom's dataset (the `dataset` attribute only
  returns the train and test sets). The holdout set is left untouched
  until predictions are made on it, i.e., it does not undergo any pipeline
  transformations until the data set is requested for the first time.
  The holdout set is stored in atom's `holdout` attribute. See
  [here][example-holdout-set]an example that shows how to use the holdout
  data set.

The data can be provided in different formats. If the data sets are not
specified beforehand, you can input the features and target separately
or together:

* X
* X, y

Remember to use the `y` parameter to indicate the target column in X when
using the first option. If not specified, the last column in X is used as
target. In both these cases, the size of the sets are defined using the
`test_size` and `holdout_size` parameters. Note that the splits are made
after the subsample of the dataset with the `n_rows` parameter (when not
left to its default value).

If you already have the separate data sets, provide them using one of the
following formats:

* train, test
* train, test, holdout
* X_train, X_test, y_train, y_test
* X_train, X_test, X_holdout, y_train, y_test, y_holdout
* (X_train, y_train), (X_test, y_test)
* (X_train, y_train), (X_test, y_test), (X_holdout, y_holdout)

The input data is always converted internally to a [dataframe][],
if it isn't one already. The column names should always be strings. If
they are not, atom changes their type at initialization. If no column
names are provided, default names are given of the form `X[N-1]`,
where N stands for the n-th feature in the dataset.

<br>

## Indexing

By default, atom resets the dataframe's [index](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html)
after initialization and after every transformation in the pipeline.
To avoid this, specify the `index` parameter. If the dataset has an
'identifier' column, it is useful to use it as index for two reasons:

* An identifier doesn't usually contain any useful information
  on the target column, and should therefore be removed before training.
* [Predictions][predicting] of specific rows can be [accessed through their index][row-and-column-selection].

!!! warning
    Avoid duplicate indices in the dataframe. Having them raises an error
    when initializing atom and may potentially lead to unexpected behavior
    if introduced later.

<br>

## Sparse datasets

If atom is initialized using a scipy sparse matrix, it is converted
internally to a dataframe of sparse columns. Read more about pandas'
sparse data structures [here](https://pandas.pydata.org/pandas-docs/stable/user_guide/sparse.html).
The same conversion takes place when a transformer returns a sparse
matrix, like for example, the [Vectorizer][].

Note that ATOM considers a dataset to be sparse if any of the columns
is sparse. A dataset can only benefit from sparsity when all its
columns are sparse, hence mixing sparse and non-sparse columns is
not recommended and can cause estimators to decrease their training
speed or even crash. Use the [shrink][atomclassifier-shrink] method
to convert dense features to sparse and the [available_models]
[atomclassifier-available_models] method to check which models have
native support for sparse matrices.

Click [here][example-nlp] to see an example that uses sparse data.

!!! warning
    Estimators accelerated using [sklearnex][] don't support sparse 
    datasets.

<br>

## Multioutput tasks

Multioutput is a task where there are more than one target column, i.e.,
the goal is to predict multiple targets at the same time. When providing
a dataframe as target, use the [y][atomclassifier-y] parameter. Providing
`y` without keyword makes ATOM think you are providing `train, test` (see
the [data sets][] section).

### Task types

ATOM recognizes four multioutput tasks.

!!! note
    Combinations of binary and multiclass target columns are treated as
    [multiclass-multioutput][] tasks.


#### Multilabel

Multilabel is a classification task, labeling each sample with `m` labels
from `n_classes` possible classes, where `m` can be 0 to `n_classes` inclusive.
This can be thought of as predicting properties of a sample that are not
mutually exclusive.

For example, prediction of the topics relevant to a text document. The
document may be about one of religion, politics, finance or education,
several of the topic classes or all of the topic classes. The target
column (`atom.y`) could look like this:

```
0                        [politics]
1               [religion, finance]
2    [politics, finance, education]
3                                []
4                         [finance]
5               [finance, religion]
6                         [finance]
7               [religion, finance]
8                       [education]
9     [finance, religion, politics]

Name: target, dtype: object
```

A model can not directly ingest a variable amount of target classes. Use
the [clean][atomclassifier-clean] method to assign a binary output to
each class, for every sample. Positive classes are indicated with 1 and
negative classes with 0. It is thus comparable to running n_classes
binary classification tasks. In our example, the target (`atom.y`) is
converted to:

```
   education  finance  politics  religion
0          0        0         1         0
1          0        1         0         1
2          1        1         1         0
3          0        0         0         0
4          0        1         0         0
5          0        1         0         1
6          0        1         0         0
7          0        1         0         1
8          1        0         0         0
9          0        1         1         1
```

#### Multiclass-multioutput

Multiclass-multioutput (also known as multitask classification) is a
classification task which labels each sample with a set of non-binary
properties. Both the number of properties and the number of classes per
property is greater than 2. A single estimator thus handles several joint
classification tasks. This is both a generalization of the multilabel
classification task, which only considers binary attributes, as well as
a generalization of the multiclass classification task, where only one
property is considered.

For example, classification of the properties "type of fruit" and "colour"
for a set of images of fruit. The property "type of fruit" has the possible
classes: "apple", "pear" and "orange". The property "colour" has the possible
classes: "green", "red", "yellow" and "orange". Each sample is an image of
a fruit, a label is output for both properties and each label is one of the
possible classes of the corresponding property.

#### Multioutput regression

Multioutput regression predicts multiple numerical properties for each
sample. Each property is a numerical variable and the number of properties
to be predicted for each sample is >= 2. Some estimators that support
multioutput regression are faster than just running n_output estimators.

For example, prediction of both wind speed and wind direction, in degrees,
using data obtained at a certain location. Each sample would be data
obtained at one location and both wind speed and direction would be output
for each sample.

#### Multivariate

Multivariate is the multioutput task for forecasting. In this case, we
try to forecast more than one time series at the same time.

Although all forecasting models in ATOM support multivariate tasks, we
differentiate two types of models:

* The "native multivariate" models apply forecasts where every prediction
  of endogeneous (`y`) variables will depend on values of the other target
  columns.
* The rest of the models apply an estimator per column, meaning that forecasts
  will be made per endogeneous variable, and not be affected by other variables.
  To access the column-wise estimators, use the estimator's `forecasters_`
  parameter, which stores the fitted forecasters in a dataframe.

Read more about time series tasks [here][time-series].

### Native multioutput models

Some models have native support for multioutput tasks. This means that
the original estimator is used to make predictions directly on all the
target columns. Examples of such models are [KNearestNeighbors][],
[RandomForest][] and [ExtraTrees][].


### Non-native multioutput models

The majority of the models don't have integrated support for multioutput
tasks. However, it's possible to still use them for such tasks, wrapping
them in a meta-estimator capable of handling multiple target columns. For
non-native multioutput models, ATOM does so automatically. For
[multilabel][] tasks, the meta-estimator is:

* [ClassifierChain][]

And for [multiclass-multioutput][] and [multioutput regression][], the
meta-estimators are respectively:

* [MultioutputClassifier][]
* [MultioutputRegressor][]

!!! warning
    Currently, scikit-learn metrics do not support multiclass-multioutput
    classification tasks. In this case, ATOM calculates the mean of the
    selected metric over every individual target.

!!! tip
    * Set the `native_multilabel` or `native_multioutput` parameter in
    [ATOMModel][] equal to `True` to ignore the meta-estimator for [custom models][].
    * Check out the [multilabel classification][example-multilabel-classification]
    and [multioutput regression][example-multioutput-regression] examples.


<br>

## Branches

You might want to compare how a model performs on a dataset transformed
through multiple pipelines, each using different transformers. For
example, on one pipeline with an undersampling strategy and the other
with an oversampling strategy. To be able to do this, ATOM has a
branching system.

The branching system helps the user to manage multiple data pipelines
within the same atom instance. Branches are created and accessed
through atom's `branch` property. A branch contains a specific pipeline,
the dataset transformed through that pipeline, and all data and utility
attributes that refer to that dataset. Transformers and models called
from atom use the dataset in the current branch, as well as data
attributes such as `atom.dataset`. It's not allowed to change the data
in a branch after fitting a model with it. Instead, create a new branch
for every unique pipeline.

By default, atom starts with one branch called "main". To start a new
branch, set a new name to the property, e.g., `#!python atom.branch = "undersample"`.
This creates a new branch from the current one. To create a branch from any
other branch type "\_from\_" between the new name and the branch from which
to split, e.g., `#!python atom.branch = "oversample_from_main"` creates
branch "oversample" from branch "main", even if the current branch is
"undersample". To switch between existing branches, just type the name of
the desired branch, e.g., `#!python atom.branch = "main"` brings you back
to the main branch. Note that every branch contains a unique copy of the
whole dataset! Creating many branches can cause [memory issues](#memory-considerations)
for large datasets.

See the [Imbalanced datasets][example-imbalanced-datasets] or
[Feature engineering][example-feature-engineering] examples for
branching use cases.

!!! warning
    Always create a new branch if you want to change the dataset after fitting
    a model! Forcing a data change through the data property's `@setter` can
    cause unexpected model behavior and break down the plotting methods.

<br>

<div align="center" markdown>
  ![diagram_branch](../img/diagram_branch.png)
  <figcaption>Figure 1. Diagram of a possible branch system to compare an oversampling with an undersampling pipeline.</figcaption>
</div>

<br>

## Memory considerations

An atom instance stores one copy of the dataset for each branch (this
doesn't include the [holdout set](#data-sets), which is only stored once),
and one copy of the initial dataset with which the instance is initialized.
This copy of the original dataset is necessary to avoid data leakage during
hyperparameter tuning and for some specific methods like [cross_validate][adaboost-cross_validate]
and [reset][atomclassifier-reset]. It's created as soon as there are no
branches in the initial state (usually after calling the first data
transformation). If the dataset is occupying too much memory, consider
using the [shrink][atomclassifier-shrink] method to convert the dtypes to
their smallest possible matching dtype.

When working with large datasets and multiple branches, it becomes impossible
to store all branches in memory at the same time. To avoid out-of-memory errors,
use atom's [`memory`][atomclassifier-memory] parameter. If not `False`, atom
saves the data of inactive branches as well as the original branch at the
specified location (in a directory called `joblib`, the name of the underlying
library managing the caching), maintaining only the current active branch in
memory. This mechanism results in a slight drop in performance because of the
I/O overhead, but can save a lot of memory. Additionally, the memory's location
is also used to cache the output of the `fit` method of transformers in the
pipeline. See [here][example-memory-considerations] an example using the memory
parameter.

Apart from the dataset itself, a model's metric scores and [shap values][shap]
are also stored as attributes of the model to avoid having to recalculate
them every time they are needed. You can delete all these attributes using
the [clear][atomclassifier-clear] method in order to free some memory before
[saving][atomclassifier-save] atom.

<br>

## Data transformations

Performing data transformations is a common requirement of many datasets
before they are ready to be ingested by a model. ATOM provides various
classes to apply [data cleaning][] and [feature engineering][]
transformations to the data. This tooling should be able to help you
apply most of the typically needed transformations to get the data ready
for modeling. For further fine-tuning, it's also possible to transform
the data using custom transformers (see the [add][atomclassifier-add]
method) or through a function (see the [apply][atomclassifier-apply]
method). Remember that all transformations are only applied to the
dataset in the current branch.


## Row and column selection

Many methods in atom contain the `rows` or `columns` parameter to select a
subset of the dataset. Examples are the [evaluate][atomclassifier-evaluate]
and [save_data][atomclassifier-save_data] methods for `rows`, and the
[distribution][atomclassifier-distribution] and [shrink][atomclassifier-shrink]
methods for `columns`. All [data cleaning][] and [feature engineering][] methods
use the `columns` parameter to apply the transformation only to that selection
of columns, and all [prediction methods][predicting] use the `rows` parameter
to make predictions on that selection of rows.

As you can see, these two parameters are very important and shared across
many methods in atom. Rows and columns can be selected in multiple ways.
The check is performed in the order described hereunder:

1. By actual dataset, e.g., `#!python rows=atom.test` is equal to `#!python rows="test"`.
2. By range or slice, e.g., `#!python rows=range(100)` to select the first 100
   rows from the dataset or `#!python rows=slice(20, 100)` to select rows 20 to 99.
3. By exact name, e.g., `#!python rows=["row1", "row2"]` to select rows with
   indices `row1` and `row2` or `#!python columns=["col1", "col2"]` to select
   columns `col1` and `col2`. It's also possible to use the `+` sign to select
   multiple rows or columns, e.g., `#!python columns="col1+col2` is the same
   as `#!python columns=["col1", "col2"]`.
4. By position, e.g., `#!python rows=[0, 1, 2]` to select the first three rows.
5. By name of the data set (only for rows), e.g., `#!python rows="train"` to
   select all rows in the training set, or `#!python rows="test+holdout"` to
   select all rows in the test and holdout sets. Valid data sets are `dataset`,
   `train`, `test` and `holdout`.
5. By dtype (only for columns), e.g., `#!python columns="number"` to select only 
   numerical columns. See pandas' [user guide](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.select_dtypes.html).
6. By regex match, e.g., `#!python columns="mean_.*"` to select all columns
   starting with `mean_`.
7. Excluding instead of including using the `!` sign, e.g. `#!python columns="!col1"`
   to select all columns except `col1`. You can also exclude multiple rows or
   columns like this `#!python columns=["!col1", "!col2"]` or this
   `#!python columns="!col1+!col2"`. It's also possible to exclude data sets
   for row selection, e.g., `#!python columns="!train"` or dtypes for column
   selection, e.g., `#!python columns="!number"`. Note that if a column name
   starts with `!`, the selection of that name will take priority over exclusion.
   Rows and columns can only be included or excluded, and not both at the same
   time. For example, this selection raises an exception `#!python column=["col1", "!col2"]`.

!!! info
    In some [plotting methods][prediction-plots], it's possible to plot separate
    lines for different subsets of the rows. For example, to compare the results
    on the train and test set. For these cases, either provide a sequence to the
    `rows` parameter for every line you want to draw, e.g., `#!python atom.plot_roc(rows=("train", "test"))`,
    or provide a dictionary where the keys are the names of the sets (used in the
    legend) and the values are the corresponding selection of rows, selected using
    any of the aforementioned approaches, e.g, `#!python atom.plot_roc(rows={"0-99": range(100), "100-199": range(100, 200})`.
    Note that for these methods, using `#!python atom.plot_roc(rows="train+test")`,
    only plots one line with the data from both sets. See the
    [advanced plotting example][example-advanced-plotting].