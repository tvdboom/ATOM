# ATOMClassifier
----------------

:: atom.api:ATOMClassifier
    :: signature
    :: description
    :: table:
        - parameters
    :: see also

<br>

## Example

:: examples

<br>

## Magic methods

The class contains some magic methods to help you access some of its
elements faster. Note that methods that apply on the pipeline can return
different results per branch.

* **\__repr__:** Prints an overview of atom's branches, models, metric and errors.
* **\__len__:** Returns the length of the dataset.
* **\__iter__:** Iterate over the pipeline's transformers.
* **\__contains__:** Checks if the provided item is a column in the dataset.
* **\__getitem__:** Access a branch, model, column or subset of the dataset.

<br>

## Attributes

### Data attributes

The data attributes are used to access the dataset and its properties.
Updating the dataset will automatically update the response of these
attributes accordingly.

:: table:
    - attributes:
        from_docstring: False
        include:
            - atom.branch:Branch.pipeline
            - atom.branch:Branch.mapping
            - atom.branch:Branch.dataset
            - atom.branch:Branch.train
            - atom.branch:Branch.test
            - atom.branch:Branch.X
            - atom.branch:Branch.y
            - atom.branch:Branch.X_train
            - atom.branch:Branch.y_train
            - atom.branch:Branch.X_test
            - atom.branch:Branch.y_test
            - atom.branch:Branch.shape
            - atom.branch:Branch.columns
            - atom.branch:Branch.n_columns
            - atom.branch:Branch.features
            - atom.branch:Branch.n_features
            - atom.branch:Branch.target
            - scaled
            - duplicates
            - nans
            - n_nans
            - numerical
            - n_numerical
            - categorical
            - n_categorical
            - outliers
            - n_outliers
            - classes
            - n_classes

<br>

### Utility attributes

The utility attributes are used to access information about the models
in the instance after [training][].

:: table:
    - attributes:
        from_docstring: False
        include:
            - models
            - metric
            - errors
            - winners
            - winner

<br>

### Tracking attributes

The tracking attributes are used to customize what elements of the
experiment are tracked. Read more in the [user guide][tracking].

:: table:
    - attributes:
        from_docstring: False
        include:
            - log_bo
            - log_model
            - log_plots
            - log_data
            - log_pipeline

<br>

### Plot attributes

The plot attributes are used to customize the plot's aesthetics. Read
more in the [user guide][aesthetics].

:: table:
    - attributes:
        from_docstring: False
        include:
            - style
            - palette
            - title_fontsize
            - label_fontsize
            - tick_fontsize

<br>

## Utility methods

The class contains a variety of utility methods to handle the data and
manage the pipeline.

:: methods:
    toc_only: False
    include:
        - add
        - apply
        - automl
        - available_models
        - canvas
        - clear
        - delete
        - distribution
        - evaluate
        - export_pipeline
        - get_class_weight
        - inverse_transform
        - log
        - merge
        - report
        - reset
        - reset_aesthetics
        - save
        - save_data
        - shrink
        - stacking
        - stats
        - status
        - transform
        - voting

<br>

## Data cleaning

The data cleaning methods can help you scale the data, handle missing
values, categorical columns, outliers and unbalanced datasets. All
attributes of the data cleaning classes are attached to atom after
running. Read more in the [user guide][data-cleaning].

!!! tip
    Use the [report][atomclassifier-report] method to examine the data and
    help you determine suitable parameters for the data cleaning methods.

:: methods:
    toc_only: False
    include:
        - balance
        - clean
        - discretize
        - encode
        - impute
        - normalize
        - prune
        - scale

<br>

## NLP

The Natural Language Processing (NLP) transformers help to convert raw
text to meaningful numeric values, ready to be ingested by a model. All
transformations are applied only on the column in the dataset called
`corpus`. Read more in the [user guide][nlp].

:: methods:
    toc_only: False
    include:
        - textclean
        - textnormalize
        - tokenize
        - vectorize


<a name="textnormalize"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">textnormalize</strong>(stopwords=True,
custom_stopwords=None, stem=False, lemmatize=True)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/nlp.py#L1357">[source]</a>
</span>
</div>
Convert words to a more uniform standard. The transformations
are applied on the column named `corpus`, in the same order the
parameters are presented. If there is no column with that name,
an exception is raised. If the provided documents are strings,
words are separated by spaces. See the [TextNormalizer](../nlp/textnormalizer.md)
class for a description of the parameters.
<br /><br /><br />


<a name="tokenize"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">tokenize</strong>(bigram_freq=None,
trigram_freq=None, quadgram_freq=None)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/nlp.py#L1322">[source]</a>
</span>
</div>
Convert documents into sequences of words. Additionally, create
n-grams (represented by words united with underscores, e.g.
"New_York") based on their frequency in the corpus. The
transformations are applied on the column named `corpus`. If
there is no column with that name, an exception is raised. See
the [Tokenizer](../nlp/tokenizer.md) class for a description
of the parameters.
<br /><br /><br />


<a name="vectorize"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">vectorize</strong>(strategy="bow",
return_sparse=True, **kwargs)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/nlp.py#L1390">[source]</a>
</span>
</div>
Transform the corpus into meaningful vectors of numbers. The
transformation is applied on the column named `corpus`. If there
is no column with that name, an exception is raised. The transformed
columns are named after the word they are embedding (if the column is
already present in the provided dataset, `_[strategy]` is added behind
the name). See the [Vectorizer](../nlp/vectorizer.md) class for a
description of the parameters.
<br /><br /><br />



## Feature engineering

To further pre-process the data, it's possible to extract features
from datetime columns, create new non-linear features transforming
the existing ones, group similar features or, if the dataset is too
large, remove features. Read more in the [user guide][feature-engineering].

:: methods:
    toc_only: False
    include:
        - feature_extraction
        - feature_generation
        - feature_grouping
        - feature_selection

<br>


## Training

The training methods are where the models are fitted to the data and
their performance is evaluated according to the selected metric. There
are three methods to call the three different training approaches. All
relevant attributes and methods from the training classes are attached
to atom for convenience. These include the errors, winner and results
attributes, as well as the [models](../../../user_guide/models),
and the [prediction](../../../user_guide/predicting) and
[plotting](../../../user_guide/plots) methods.

<table style="font-size:16px;margin-top:5px">
<tr>
<td><a href="#run">run</a></td>
<td>Fit the models to the data in a direct fashion.</td>
</tr>

<tr>
<td><a href="#successive-halving">successive_halving</a></td>
<td>Fit the models to the data in a successive halving fashion.</td>
</tr>

<tr>
<td><a href="#train-sizing">train_sizing</a></td>
<td>Fit the models to the data in a train sizing fashion.</td>
</tr>
</table>
<br>


<a name="run"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">run</strong>(models=None,
metric=None, greater_is_better=True, needs_proba=False, needs_threshold=False,
n_calls=10, n_initial_points=5, est_params=None, bo_params=None, n_bootstrap=0)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L1634">[source]</a>
</span>
</div>
Fit and evaluate the models. The following steps are applied to every model:

1. Hyperparameter tuning is performed using a Bayesian Optimization
   approach (optional).
2. The model is fitted on the training set using the best combination
   of hyperparameters found.
3. The model is evaluated on the test set.
4. The model is trained on various bootstrapped samples of the training
   set and scored again on the test set (optional).

See [DirectClassifier](../training/directclassifier.md) for a description of
the parameters.
<br /><br /><br />


<a name="successive-halving"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">successive_halving</strong>(models=None,
metric=None, greater_is_better=True, needs_proba=False, needs_threshold=False,
skip_runs=0, n_calls=0, n_initial_points=5, est_params=None, bo_params=None,
n_bootstrap=0)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L1673">[source]</a>
</span>
</div>
Fit and evaluate the models in a [successive halving](../../../user_guide/training/#successive-halving)
fashion. The following steps are applied to every model (per iteration):

1. Hyperparameter tuning is performed using a Bayesian Optimization
   approach (optional).
2. The model is fitted on the training set using the best combination
   of hyperparameters found.
3. The model is evaluated on the test set.
4. The model is trained on various bootstrapped samples of the training
   set and scored again on the test set (optional).

See [SuccessiveHalvingClassifier](../training/successivehalvingclassifier.md)
for a description of the parameters.
<br /><br /><br />


<a name="train-sizing"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">train_sizing</strong>(models=None,
metric=None, greater_is_better=True, needs_proba=False, needs_threshold=False,
train_sizes=5, n_calls=0, n_initial_points=5, est_params=None, bo_params=None,
n_bootstrap=0)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L1719">[source]</a>
</span>
</div>
Fit and evaluate the models in a [train sizing](../../../user_guide/training/#train-sizing)
fashion. The following steps are applied to every model (per iteration):

1. Hyperparameter tuning is performed using a Bayesian Optimization
   approach (optional).
2. The model is fitted on the training set using the best combination
   of hyperparameters found.
3. The model is evaluated on the test set.
4. The model is trained on various bootstrapped samples of the training
   set and scored again on the test set (optional).

See [TrainSizingClassifier](../training/trainsizingclassifier.md) for a
description of the parameters.
<br /><br /><br />
