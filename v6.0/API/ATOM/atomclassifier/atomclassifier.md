# ATOMClassifier
----------------

:: atom.api:ATOMClassifier
    :: signature
    :: head
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

* **\__repr__:** Prints an overview of atom's branches, models, and metrics.
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
            - atom.data:Branch.pipeline
            - atom.data:Branch.mapping
            - atom.data:Branch.dataset
            - atom.data:Branch.train
            - atom.data:Branch.test
            - atom.data:Branch.X
            - atom.data:Branch.y
            - holdout
            - atom.data:Branch.X_train
            - atom.data:Branch.y_train
            - atom.data:Branch.X_test
            - atom.data:Branch.y_test
            - atom.data:Branch.shape
            - atom.data:Branch.columns
            - atom.data:Branch.n_columns
            - atom.data:Branch.features
            - atom.data:Branch.n_features
            - atom.data:Branch.target
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
            - ignore
            - missing
            - branch
            - models
            - metric
            - winners
            - winner
            - results

<br>

### Tracking attributes

The tracking attributes are used to customize what elements of the
experiment are tracked. Read more in the [user guide][tracking].

:: table:
    - attributes:
        from_docstring: False
        include:
            - log_ht
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
            - palette
            - title_fontsize
            - label_fontsize
            - tick_fontsize
            - line_width
            - marker_size

<br>

## Utility methods

Next to the [plotting][plots] methods, the class contains a variety
of utility methods to handle the data and manage the pipeline.

:: methods:
    toc_only: False
    include:
        - add
        - apply
        - available_models
        - canvas
        - clear
        - delete
        - distributions
        - eda
        - evaluate
        - export_pipeline
        - get_class_weight
        - get_sample_weight
        - inverse_transform
        - load
        - merge
        - update_layout
        - update_traces
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
    Use the [eda][atomclassifier-eda] method to examine the data and
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

<br>

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
their performance is evaluated against a selected metric. There are
three methods to call the three different training approaches. Read
more in the [user guide][training].

:: methods:
    toc_only: False
    include:
        - run
        - successive_halving
        - train_sizing
