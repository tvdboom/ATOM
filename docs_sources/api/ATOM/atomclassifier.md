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
            - scaled
            - duplicates
            - nans
            - n_nans
            - numerical
            - n_numerical
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
running.

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


<a name="discretize"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">discretize</strong>(strategy="quantile",
bins=5, labels=None)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L1127">[source]</a>
</span>
</div>
Bin continuous data into intervals. For each feature, the bin edges are
computed during fit and, together with the number of bins, they will
define the intervals. Ignores numerical columns. See
[Discretizer](../data_cleaning/discretizer.md) for a description of the parameters.
<br /><br /><br />


<a name="encode"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">encode</strong>(strategy="LeaveOneOut",
max_onehot=10, ordinal=None, rare_to_value=None, value="rare")
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L1151">[source]</a>
</span>
</div>
Perform encoding of categorical features. The encoding type depends
on the number of unique values in the column:
<ul style="line-height:1.2em;margin-top:5px">
<li>If n_unique=2 or ordinal feature, use Label-encoding.</li>
<li>If 2 < n_unique <= max_onehot, use OneHot-encoding.</li>
<li>If n_unique > max_onehot, use `strategy`-encoding.</li>
</ul>
Missing values are propagated to the output column. Unknown classes
encountered during transforming are converted to `np.NaN`. The class
is also capable of replacing classes with low occurrences with the
value `other` in order to prevent too high cardinality. See
[Encoder](../data_cleaning/encoder.md) for a description of the parameters.

!!! note
    This method only encodes the categorical features. It does not encode
    the target column! Use the [clean](#clean) method for that.

<br /><br /><br />


<a name="impute"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">impute</strong>(strat_num="drop",
strat_cat="drop", max_nan_rows=None, max_nan_cols=None, missing=None)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L1093">[source]</a>
</span>
</div>
Impute or remove missing values according to the selected strategy.
Also removes rows and columns with too many missing values. The
imputer is fitted only on the training set to avoid data leakage.
Use the `missing` attribute to customize what are considered "missing
values". See [Imputer](../data_cleaning/imputer.md) for a description
of the parameters. Note that since the Imputer can remove rows from
both the train and test set, the size of the sets may change after
the transformation.
<br /><br /><br />


<a name="normalize"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">normalize</strong>(strategy="yeojohnson", **kwargs)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L1019">[source]</a>
</span>
</div>
Transform the data to follow a Normal/Gaussian distribution. This
transformation is useful for modeling issues related to heteroscedasticity
(non-constant variance), or other situations where normality is desired.
Missing values are disregarded in fit and maintained in transform.
Categorical columns are ignored. The estimator created by the class is
attached to atom. See the See the [Normalizer](../data_cleaning/normalizer.md)
class for a description of the parameters.
<br /><br /><br />


<a name="prune"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">prune</strong>(strategy="zscore",
method="drop", max_sigma=3, include_target=False, **kwargs)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L1194">[source]</a>
</span>
</div>
Prune outliers from the training set. The definition of outlier depends
on the selected strategy and can greatly differ from one each other. 
Ignores categorical columns. The estimators created by the class
are attached to atom. See [Pruner](../data_cleaning/pruner.md) for a
description of the parameters.

!!! note
    This transformation is only applied to the training set in order
    to maintain the original distribution of samples in the test set.

<br /><br /><br />


<a name="scale"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">scale</strong>(strategy="standard", **kwargs)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L999">[source]</a>
</span>
</div>
Applies one of sklearn's scalers. Non-numerical columns are ignored. The
estimator created by the class is attached to atom. See the
[Scaler](../data_cleaning/scaler.md) class for a description of the parameters.
<br /><br /><br />



## NLP

The Natural Language Processing (NLP) transformers help to convert raw
text to meaningful numeric values, ready to be ingested by a model.

<table style="font-size:16px;margin-top:5px">
<tr>
<td><a href="#textclean">textclean</a></td>
<td>Applies standard text cleaning to the corpus.</td>
</tr>

<tr>
<td><a href="#textnormalize">textnormalize</a></td>
<td>Convert words to a more uniform standard.</td>
</tr>

<tr>
<td><a href="#tokenize">tokenize</a></td>
<td>Convert documents into sequences of words</td>
</tr>

<tr>
<td><a href="#vectorize">vectorize</a></td>
<td>Transform the corpus into meaningful vectors of numbers.</td>
</tr>
</table>
<br>


<a name="textclean"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">textclean</strong>(decode=True,
lower_case=True, drop_emails=True, regex_emails=None, drop_url=True,
regex_url=None, drop_html=True, regex_html=None, drop_emojis, regex_emojis=None,
drop_numbers=True, regex_numbers=None, drop_punctuation=True)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/nlp.py#L1269">[source]</a>
</span>
</div>
Applies standard text cleaning to the corpus. Transformations include
normalizing characters and dropping noise from the text (emails, HTML
tags, URLs, etc...). The transformations are applied on the column
named `corpus`, in the same order the parameters are presented. If
there is no column with that name, an exception is raised. See the
[TextCleaner](../nlp/textcleaner.md) class for a description of the
parameters.
<br /><br /><br />


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
the existing ones or, if the dataset is too large, remove features
using one of the provided strategies.

<table style="font-size:16px;margin-top:5px">
<tr>
<td><a href="#feature-extraction">feature_extraction</a></td>
<td>Extract features from datetime columns.</td>
</tr>

<tr>
<td><a href="#feature-generation">feature_generation</a></td>
<td>Create new features from combinations of existing ones.</td>
</tr>

<tr>
<td><a href="#feature-selection">feature_selection</a></td>
<td>Remove features according to the selected strategy.</td>
</tr>
</table>
<br>


<a name="feature-extraction"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">feature_extraction</strong>(features=["day", "month", "year"],
fmt=None, encoding_type="ordinal", drop_columns=True)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L1422">[source]</a>
</span>
</div>
Extract features (hour, day, month, year, etc..) from datetime columns.
Columns of dtype `datetime64` are used as is. Categorical columns that
can be successfully converted to a datetime format (less than 30% NaT
values after conversion) are also used. See the [FeatureExtractor](../feature_engineering/feature_extractor.md) class for a
description of the parameters.
<br /><br /><br />


<a name="feature-generation"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">feature_generation</strong>(strategy="dfs",
n_features=None, operators=None, **kwargs)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L1455">[source]</a>
</span>
</div>
Create new combinations of existing features to capture the non-linear
relations between the original features. See [FeatureGenerator](../feature_engineering/feature_generator.md)
for a description of the parameters. Attributes created by the class
are attached to atom.
<br /><br /><br />


<a name="feature-selection"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">feature_selection</strong>(strategy=None,
solver=None, n_features=None, max_frac_repeated=1., max_correlation=1., **kwargs)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L1489">[source]</a>
</span>
</div>
Remove features according to the selected strategy. Ties between
features with equal scores are broken in an unspecified way.
Additionally, remove multicollinear and low variance features.
See [FeatureSelector](../feature_engineering/feature_selector.md)
for a description of the parameters. Plotting methods and attributes
created by the class are attached to atom.

!!! note
    <ul style="line-height:1.2em;margin-top:5px">
    <li>When strategy="univariate" and solver=None, [f_classif](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html)
        is used as default solver.</li>
    <li>When the strategy requires a model and it's one of ATOM's
        [predefined models](../../../user_guide/models/#predefined-models), the
        algorithm automatically selects the classifier (no need to add `_class`
        to the solver).</li>
    <li>When strategy is not one of univariate or pca, and solver=None, atom
        uses the winning model (if it exists) as solver.</li>
    <li>When strategy is sfs, rfecv or any of the advanced strategies and no
        scoring is specified, atom's metric is used (if it exists) as scoring.</li>

<br /><br />



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
