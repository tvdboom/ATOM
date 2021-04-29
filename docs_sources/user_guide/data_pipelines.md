# Data pipelines
----------------

During the exploration phase, you might want to compare how a model
performs on a dataset processed using different techniques. For
example, on one dataset balanced with an undersampling strategy and
the other with an oversampling strategy. For this, atom has the
branching system.

<br>

## Branches

The branching system manages separate paths the dataset can take. Every
path is called a branch and can be accessed through the `branch`
attribute. A branch contains a dataset, and all transformers and models
that are fitted on that specific dataset. Accessing data attributes such
as `atom.dataset` will return the data in the current branch. Use the
branch's \__repr__ to get an overview of the transformers in the branch.
All data cleaning, feature engineering and models use the dataset in the
current branch. Don't change the data in a branch after fitting a model,
this can cause unexpected model behaviour. Instead, create a new branch
for every unique pipeline.

By default, atom starts with one branch called "master". To start a new
branch, set a new name to the property, e.g. `atom.branch = "undersample"`.
This will create a new branch from the current one. To create a branch
from any other branch type "\_from\_" between the new name and the branch
from which to split, e.g. `atom.branch = "oversample_from_master"` will
create branch "oversample" from branch "master", even if the current branch
is "undersample". To switch between existing branches, just type the name
of the desired branch, e.g. `atom.branch = "master"` brings you back to the
main branch. Note that every branch contains a unique copy of the whole
dataset! Creating many branches can cause memory issues for large datasets.

<br>

<div align="center">
    <img src="../../img/diagram_branch.png" alt="diagram_branch"/>
    <figcaption>Figure 1. Diagram of a possible branch system to compare an oversampling with an undersampling pipeline.</figcaption>
</div>

You can delete a branch either deleting the attribute, e.g. `del atom.branch`,
or using the delete method, e.g. `atom.branch.delete()`. A branch can only be
deleted if no models were trained on its dataset. Use `atom.branch.status()`
for an overview of the transformers and models in the branch.

See the [Imbalanced datasets](../../examples/imbalanced_datasets) or
[Feature engineering](../../examples/feature_engineering) examples for
branching use cases.

!!! warning
    Always create a new branch if you want to change the dataset after fitting
    a model! Not doing so can cause unexpected model behaviour.


<br>

## Data transformations

Performing data transformations is a common requirement of many datasets
before they are ready to be ingested by a model. ATOM provides various
classes to apply [data cleaning](../data_cleaning) and
[feature engineering](../feature_engineering) transformations to the data.
This tooling should be able to help you apply most of the typically needed
transformations to get the data ready for modelling. For further
fine-tuning, it's also possible to transform the data using
custom transformers (see the [add](../../API/ATOM/atomclassifier/#add) method)
or through a function (see the [apply](../../API/ATOM/atomclassifier/#apply)
method). Remember that all transformations are only applied to the dataset
in the current branch.

<br>

## AutoML

Automated machine learning (AutoML) automates the selection, composition
and parameterization of machine learning pipelines. Automating the machine
learning process makes it more user-friendly and often provides faster, more
accurate outputs than hand-coded algorithms. ATOM uses the [TPOT](http://epistasislab.github.io/tpot/)
package for AutoML optimization. TPOT uses a genetic algorithm to intelligently
explore thousands of possible pipelines in order to find the best one for your
data. Such an algorithm can be started through the [automl](../../API/ATOM/atomclassifier/#automl)
method. The resulting data transformers and final estimator are merged with atom's
pipeline (check the `pipeline` and `models` attributes after the method
finishes running).

!!! warning
    AutoML algorithms aren't intended to run for only a few minutes. If left
    to its default parameters, the method can take a very long time to finish!
