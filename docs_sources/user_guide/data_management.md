# Data management
-----------------

## Data sets

ATOM is designed to work around one single dataset: the one with which
atom is initialized. This is the dataset you want to explore, transform,
and use for model training and validation. ATOM differentiates three
different data sets:

* The **training set** is usually the largest of the data sets. As the
  name suggests, this set is used to train the pipeline. During
  [hyperparameter tuning](../training/#hyperparameter-tuning), only the
  training set is used to fit and evaluate the estimator in every call.
  The training set in the current [branch](#branches) can be accessed
  through the `train` attribute. It's features and target can be
  accessed through `X_train` and `y_train` respectively.
* The **test set** is used to evaluate the models in the pipeline. The
  model scores on this set give an indication on how the model performs
  on new data. The test set can be accessed through the `test` attribute.
  It's features and target can be accessed through `X_test` and `y_test`
  respectively.
* The **holdout set** is an optional, separate set that should only be
  used to evaluate the final model's performance. Create this set when
  you are going to use the test set for an intermediate validation step.
  The holdout set is immediately set apart during initialization and is
  not considered part of atom's dataset (the `dataset` attribute only
  returns the train and test sets). The holdout set is left untouched
  until predictions are made on it, i.e. it does not undergo any pipeline
  transformations. This also means that the holdout set is independent
  of the branches. The holdout set is stored in the trainer's `holdout`
  attribute. It's features and target can not be accessed separately.
  See [here](../../examples/holdout_set) an example that shows how to
  use the holdout data set.

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


<br>

## Indexing

!!! info
    MultiIndex is not supported.



<br>

## Branches

You might want to compare how a model performs on a dataset transformed
through multiple pipelines, each using different transformers. For
example, on one pipeline with an undersampling strategy and the other
with an oversampling strategy. To be able to do this, ATOM has the
branching system.

The branching system helps the user to manage multiple pipelines within
the same atom instance. Every pipeline is stored in a branch, which can
be accessed through the `branch` property. A branch contains a copy of
the dataset, and all transformers and models that are fitted on that
specific dataset. Transformers and models called from atom use the
dataset in the current branch, as well as data attributes such as
`atom.dataset`. Use the branch's \__repr__ to get an overview of the
transformers in the branch. It's not allowed to change the data in a
branch after fitting a model with it. Doing this would cause unexpected
model behaviour and break down the plotting methods. Instead, create a
new branch for every unique pipeline.

By default, atom starts with one branch called "master". To start a new
branch, set a new name to the property, e.g. `atom.branch = "undersample"`.
This will create a new branch from the current one. To create a branch
from any other branch type "\_from\_" between the new name and the branch
from which to split, e.g. `atom.branch = "oversample_from_master"` will
create branch "oversample" from branch "master", even if the current branch
is "undersample". To switch between existing branches, just type the name
of the desired branch, e.g. `atom.branch = "master"` brings you back to the
master branch. Note that every branch contains a unique copy of the whole
dataset! Creating many branches can cause memory issues for large datasets.

See the [Imbalanced datasets](../../examples/imbalanced_datasets) or
[Feature engineering](../../examples/feature_engineering) examples for
branching use cases.

!!! warning
    Always create a new branch if you want to change the dataset after fitting
    a model!

<br>

<div align="center">
    <img src="../../img/diagram_branch.png" alt="diagram_branch" />
    <figcaption>Figure 1. Diagram of a possible branch system to compare an oversampling with an undersampling pipeline.</figcaption>
</div>

The branch class has the following methods.

<table style="font-size:16px;margin-top:5px">
<tr>
<td><a href="#delete">delete</a></td>
<td>Delete the branch from the atom instance.</td>
</tr>

<tr>
<td><a href="#rename">rename</a></td>
<td>Change the name of the branch.</td>
</tr>

<tr>
<td><a href="#status">status</a></td>
<td>Get an overview of the pipeline and models in the branch.</td>
</tr>
</table>
<br>


<a name="delete"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">delete</strong>()
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/branch.py#L106">[source]</a>
</span>
</div>
Delete the branch and all the models in it. Same as executing `del atom.branch`.
<br /><br /><br />


<a name="rename"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">rename</strong>(name)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/branch.py#L128">[source]</a>
</span>
</div>
Change the name of the branch.
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>name: str</strong><br>
New name for the branch. Can not be empty nor equal to an existing branch.
</p>
</td>
</tr>
</table>
<br />


<a name="status"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">status</strong>()
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/branch.py#L152">[source]</a>
</span>
</div>
Get an overview of the pipeline and models in the branch. This method
prints the same information as the \__repr__ and also saves it to the
logger.
<br /><br /><br />


## Memory considerations

Atom stores one copy of the dataframe in each branch. Note that there
are always at least two branches in the instance: master (or another
user defined branch) and one additional branch that stores the
dataframe with which the class was initialized. This internal branch
is called `og` (original) and can not be accessed by the user.

Apart from the dataset itself, the model's [predictions](./predicting)
(e.g. `predict_proba_train`), metric scores and [shap values](./plots/#shap)
are also stored as attributes of the model to avoid having to recalculate
them every time they are needed. All this data can occupy a considerable
amount memory for large datasets. You can delete all these attributes
using the [clear](../../API/ATOM/atomclassifier/#clear) method in order
to free some memory before [saving](../../API/ATOM/atomclassifier/#save)
the class.

!!! note
    Sparse matrices fed to atom or an [exported pipeline](../../API/ATOM/atomclassifier/#export-pipeline)
    are converted internally to the full array. Note that this can fill
    large spaces in memory.

<br>

## Data transformations

Performing data transformations is a common requirement of many
datasets before they are ready to be ingested by a model. ATOM
provides various classes to apply [data cleaning](../data_cleaning)
and [feature engineering](../feature_engineering) transformations
to the data. This tooling should be able to help you apply most
of the typically needed transformations to get the data ready for
modelling. For further fine-tuning, it's also possible to transform
the data using custom transformers (see the [add](../../API/ATOM/atomclassifier/#add)
method) or through a function (see the [apply](../../API/ATOM/atomclassifier/#apply)
method). Remember that all transformations are only applied to the
dataset in the current branch.

<br>

## AutoML

Automated machine learning (AutoML) automates the selection,
composition and parameterization of machine learning pipelines.
Automating the machine learning process makes it more user-friendly
and often provides faster, more accurate outputs than hand-coded
algorithms. ATOM uses the [TPOT](http://epistasislab.github.io/tpot/)
package for AutoML optimization. TPOT uses a genetic algorithm to
intelligently explore thousands of possible pipelines in order to
find the best one for your data. Such an algorithm can be started
through the [automl](../../API/ATOM/atomclassifier/#automl) method.
The resulting data transformers and final estimator are merged with
atom's pipeline (check the `pipeline` and `models` attributes after
the method finishes running).

!!! warning
    AutoML algorithms aren't intended to run for only a few minutes. If left
    to its default parameters, the method can take a very long time to finish!
