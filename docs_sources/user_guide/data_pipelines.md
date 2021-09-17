# Data pipelines
----------------

During the exploration phase, you might want to compare how a model
performs on a dataset processed using different transformers. For
example, on one dataset balanced with an undersampling strategy and
the other with an oversampling strategy. For this, atom has the
branching system.

<br>

## Branches

The branching system helps manage multiple pipelines within the same
atom instance. Every pipeline is stored in a branch, which can be
accessed through the `branch` attribute. A branch contains a copy of
the dataset, and all transformers and models that are fitted on that
specific dataset. Transformers and models called from atom use the
dataset in the current branch, as well as data attributes such as
`atom.dataset`. Also  Use the branch's \__repr__ to get an overview
of the transformers in the branch. Don't change the data in a branch
after fitting a model, this can cause unexpected model behaviour.
Instead, create a new branch for every unique pipeline.

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
    a model! Not doing so can cause unexpected model behaviour.

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
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/branch.py#L137">[source]</a>
</span>
</div>
Delete the branch and all the models in it. Same as executing `del atom.branch`.
<br /><br /><br />


<a name="rename"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">rename</strong>(name)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/branch.py#L137">[source]</a>
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
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/branch.py#L137">[source]</a>
</span>
</div>
Get an overview of the pipeline and models in the branch. This method
prints the same information as the \__repr__ and also saves it to the
logger.
<br /><br /><br />



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
