# Feature engineering
---------------------

Feature engineering is the process of creating new features from the
existing ones, in order to capture relationships with the target
column that the first set of features didn't have on their own. This
process is very important to improve the performance of machine learning
algorithms. Although feature engineering works best when the data 
scientist applies use-case specific transformations, there are ways to
do this in an automated manner, without prior domain knowledge. One of
the problems of creating new features without human expert intervention,
is that many of the newly created features can be useless, i.e. they do
not help the algorithm to make better predictions. Even worse, having
useless features can drop your performance. To avoid this, we perform
feature selection, a process in which we select the relevant features 
in the dataset. See the [Feature engineering](../../examples/feature_engineering)
example.

!!! note
    All of atom's feature engineering methods automatically adopt the relevant
    transformer attributes (`n_jobs`, `verbose`, `logger`, `random_state`) from
    atom. A different choice can be added as parameter to the method call,
    e.g. `atom.feature_selection("PCA", n_features=10, random_state=2)`.

!!! note
    Like the [add](../../API/ATOM/atomclassifier/#add) method, the feature engineering
    methods accept the `columns` parameter to only transform a subset of the
    dataset's features, e.g. `atom.feature_selection("PCA", n_features=10, columns=slice(5, 15))`.

<br>

## Extracting datetime features

Features that contain dates or timestamps can not be directly ingested
by models since they are not strictly numerical. Encoding them as
categorical features is also undesirable since the encoding does not
capture the relationship between the different moments in time. The
[FeatureExtractor](../../API/feature_engineering/feature_extractor)
class creates new datetime features (e.g. day, month, year, hour...).
It can be accessed from atom through the [feature_extraction](../../API/ATOM/atomclassifier/#feature-extraction)
method. The extracted features are named equally to the column they
from which they are transformed, followed by an underscore and the
datetime attribute they create, e.g. `Feature 1_day` for the day
element of `Feature 1`.

Note that many time features have a cyclic pattern, e.g. after Sunday
comes Monday. This means that if we would encode the days of the week
from 0 to 6, we would lose that relation. A common method used to encode
cyclical features is to transform the data into two dimensions using a
sine and cosine transformation:

$$
x_{sin} = sin\left(\frac{2\pi * x}{max(x)}\right)
$$

$$
x_{cos} = cos\left(\frac{2\pi * x}{max(x)}\right)
$$

The resulting features have their names followed by sin or cos, e.g.
`Feature 1_day_sin` and `Feature 1_day_cos`. The datetime attributes
that can be encoded in a cyclic fashion are: microsecond, second,
minute, hour, weekday, day, day_of_year, week, month, quarter. Note
that decision trees based algorithms build their split rules according
to one feature at a time. This means that they will fail to correctly
process cyclic features since the sin/cos values are expected to be
considered as one single coordinate system.

Use the `fmt` parameter to specify your feature's format in case the
column is categorical. The FeatureExtractor class will convert the
column to the datetime dtype before extracting the specified features.
Click [here](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes)
for an overview of the available formats.

<br>

## Generating new features

The [FeatureGenerator](../../API/feature_engineering/feature_generator)
class creates new non-linear features based on the original feature
set. It can be accessed from atom through the [feature_generation](../../API/ATOM/atomclassifier/#feature-generation)
method. You can choose between two strategies: Deep Feature Synthesis
and Genetic Feature Generation.


**Deep Feature Synthesis**<br>
Deep feature synthesis (DFS) applies the selected operators on the
features in the dataset. For example, if the operator is "log",
it will create the new feature `LOG(old_feature)` and if the
operator is "mul", it will create the new feature `old_feature_1 x old_feature_2`.
The operators can be chosen through the `operators` parameter.
Available options are:
<ul>
<li><b>add: </b>Sum two features together.</li>
<li><b>sub: </b>Subtract two features from each other.</li>
<li><b>mul: </b>Multiply two features with each other.</li>
<li><b>div: </b>Divide two features with each other.</li>
<li><b>srqt: </b>Take the square root of a feature.</li>
<li><b>log: </b>Take the logarithm of a feature.</li>
<li><b>sin: </b>Calculate the sine of a feature.</li>
<li><b>cos: </b>Calculate the cosine of a feature.</li>
<li><b>tan: </b>Calculate the tangent of a feature.</li>
</ul>

ATOM's implementation of DFS uses the [featuretools](https://www.featuretools.com/) package.

!!! tip
    DFS can create many new features and not all of them will be useful.
    Use [FeatureSelector](./../../API/feature_engineering/feature_selector)
    to reduce the number of features!

!!! warning
    Using the div, log or sqrt operators can return new features with
    `inf` or `NaN` values. Check the warnings that may pop up or use
    atom's [missing](/../../API/ATOM/atomclassifier/#properties) property.

!!! warning
    When using DFS with `n_jobs>1`, make sure to protect your code with
    `if __name__ == "__main__"`. Featuretools uses [dask](https://dask.org/),
    which uses python multiprocessing for parallelization. The spawn
    method on multiprocessing starts a new python process, which requires
    it to import the \__main__ module before it can do its task.

<br>

**Genetic Feature Generation**<br>
Genetic feature generation (GFG) uses [genetic programming](https://en.wikipedia.org/wiki/Genetic_programming),
a branch of evolutionary programming, to determine which features
are successful and create new ones based on them. Where DFS can be
seen as some kind of "brute force" for feature engineering, GFG tries
to improve its features with every generation of the algorithm. GFG
uses the same operators as DFS, but instead of only applying the
transformations once, it evolves them further, creating complicated
non-linear combinations of features with many transformations. The
new features are given the name `Feature N` for the N-th feature. You
can access the genetic feature's fitness and description (how they are
calculated) through the `genetic_features` attribute.

ATOM uses the [SymbolicTransformer](https://gplearn.readthedocs.io/en/stable/reference.html#symbolic-transformer)
 class from the [gplearn](https://gplearn.readthedocs.io/en/stable/index.html)
 package for the genetic algorithm. Read more about this implementation
 [here](https://gplearn.readthedocs.io/en/stable/intro.html#transformer).

!!! warning
    GFG can be slow for very large populations!

<br>

## Selecting useful features

The [FeatureSelector](../../API/feature_engineering/feature_selector) class
provides tooling to select the relevant features from a dataset. It can
be accessed from atom through the [feature_selection](../../API/ATOM/atomclassifier/#feature-selection)
method. The following strategies are implemented: univariate, PCA, SFM,
RFE, RFECV and SFS.

!!! tip
    Use the [plot_feature_importance](../../API/plots/plot_feature_importance)
    method to examine how much a specific feature contributes to the
    final predictions. If the model doesn't have a `feature_importances_`
    attribute, use [plot_permutation_importance](../../API/plots/plot_permutation_importance) instead.

!!! warning
    The RFE and RFECV strategies don't work when the solver is a 
    [CatBoost](https://catboost.ai/) model due to incompatibility
    of the APIs.

<br>

<a name="univariate"></a>
**Univariate**<br>
Univariate feature selection works by selecting the best features based
on univariate statistical F-test. The test is provided via the `solver`
parameter. It takes any function taking two arrays (X, y), and returning
arrays (scores, p-values). Read more in sklearn's [documentation](https://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection).


<a name="pca"></a>
**Principal Components Analysis**<br>
Applying PCA will reduce the dimensionality of the dataset by maximizing
the variance of each dimension. The new features are called Component
1, Component 2, etc... The data is scaled to mean=0 and std=1 before
fitting the transformer (if it wasn't already). Read more in sklearn's [documentation](https://scikit-learn.org/stable/modules/decomposition.html#pca).


<a name="sfm"></a>
**Selection from model**<br>
SFM uses an estimator with `feature_importances_` or `coef_` attributes
to select the best features in a dataset based on importance weights.
The estimator is provided through the `solver` parameter and can be
already fitted. ATOM allows you to use one its predefined [models](#models),
e.g. `solver="RF"`. If you didn't call the FeatureSelector through atom,
don't forget to indicate the estimator's task adding `_class` or `_reg`
after the name, e.g. `RF_class` to use a random forest classifier. Read
more in sklearn's [documentation](https://scikit-learn.org/stable/modules/feature_selection.html#feature-selection-using-selectfrommodel).


<a name="rfe"></a>
**Recursive feature elimination**<br>
Select features by recursively considering smaller and smaller sets of
features. First, the estimator is trained on the initial set of features,
and the importance of each feature is obtained either through a `coef_`
or through a `feature_importances_` attribute. Then, the least important
features are pruned from current set of features. That procedure is
recursively repeated on the pruned set until the desired number of
features to select is eventually reached. Note that, since RFE needs to
fit the model again every iteration, this method can be fairly slow.

RFECV applies the same algorithm as RFE but uses a cross-validated metric
(under the scoring parameter, see [RFECV](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html#sklearn.feature_selection.RFECV))
to assess every step's performance. Also, where RFE returns the number
of features selected by `n_features`, RFECV returns the number of
features that achieved the optimal score on the specified metric. Note
that this is not always equal to the amount specified by `n_features`.
Read more in sklearn's [documentation](https://scikit-learn.org/stable/modules/feature_selection.html#recursive-feature-elimination).


<a name="sfs"></a>
**Sequential Feature Selection**<br>
Sequential feature selection adds (forward selection) or removes (backward
selection) features to form a feature subset in a greedy fashion. At each
stage, this estimator chooses the best feature to add or remove based on
the cross-validation score of an estimator. Read more in sklearn's [documentation](https://scikit-learn.org/stable/modules/feature_selection.html#sequential-feature-selection).


**Removing features with low variance**<br>
Variance is the expectation of the squared deviation of a random
variable from its mean. Features with low variance have many values
repeated, which means the model will not learn much from them.
[FeatureSelector](../../API/feature_engineering/feature_selector) removes
all features where the same value is repeated in at least
`max_frac_repeated` fraction of the rows. The default option is to
remove a feature if all values in it are the same. Read more in sklearn's [documentation](https://scikit-learn.org/stable/modules/feature_selection.html#removing-features-with-low-variance).


**Removing features with multi-collinearity**<br>
Two features that are highly correlated are redundant, i.e. two will
not contribute more to the model than only one of them.
[FeatureSelector](../../API/feature_engineering/feature_selector) will
drop a feature that has a Pearson correlation coefficient larger than
`max_correlation` with another feature. A correlation of 1 means the
two columns are equal. A dataframe of the removed features and their
correlation values can be accessed through the `collinear` attribute.
