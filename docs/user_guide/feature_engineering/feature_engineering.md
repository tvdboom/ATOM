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
in the dataset. See the [Feature engineering][example-feature-engineering]
example.

!!! note
    All of atom's feature engineering methods automatically adopt the relevant
    transformer attributes (`n_jobs`, `verbose`, `logger`, `random_state`) from
    atom. A different choice can be added as parameter to the method call,
    e.g. `atom.feature_selection("pca", n_features=10, random_state=2)`.

!!! note
    Like the [add][atomclassifier-add] method, the feature engineering
    methods accept the `columns` parameter to only transform a subset of the
    dataset's features, e.g. `atom.feature_selection("pca", n_features=10, columns=slice(5, 15))`.

<br>

## Extracting datetime features

Features that contain dates or timestamps can not be directly ingested
by models since they are not strictly numerical. Encoding them as
categorical features is not an option since the encoding does not
capture the relationship between the different moments in time. The
[FeatureExtractor][] class creates new features extracting datetime
elements (e.g. day, month, year, hour...) from the columns. It can be
accessed from atom through the [feature_extraction][atomclassifier-feature_extraction]
method. The new features are named equally to the column from which
they are extracted, followed by an underscore and the datetime element
they create, e.g. `x0_day` for the day element of `x0`.

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
`x0_day_sin` and `x0_day_cos`. The datetime elements
that can be encoded in a cyclic fashion are: microsecond, second,
minute, hour, weekday, day, day_of_year, month and quarter. Note that
decision trees based algorithms build their split rules according to
one feature at a time. This means that they will fail to correctly
process cyclic features since the sin/cos values are expected to be
considered as one single coordinate system.

Use the `fmt` parameter to specify your feature's format in case the
column is categorical. The FeatureExtractor class will convert the
column to the datetime dtype before extracting the specified features.
Click [here](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes)
for an overview of the available formats.

<br>

## Generating new features

The [FeatureGenerator][] class creates new non-linear features based
on the original feature set. It can be accessed from atom through the
[feature_generation][atomclassifier-feature_generation] method. You
can choose between two strategies: Deep Feature Synthesis and Genetic
Feature Generation.


**Deep Feature Synthesis**<br>
Deep feature synthesis (DFS) applies the selected operators on the
features in the dataset. For example, if the operator is "log", it will
create the new feature `LOG(old_feature)` and if the operator is "mul",
it will create the new feature `old_feature_1 x old_feature_2`. The
operators can be chosen through the `operators` parameter. Choose from:

* **add**: Take the sum of two features.
* **sub:** Subtract two features from each other.</li>
* **mul:** Multiply two features with each other.</li>
* **div:** Divide two features with each other.</li>
* **abs:** Calculate the absolute value of a feature.</li>
* **srqt:** Calculate the square root of a feature.</li>
* **log:** Calculate the natural logarithm of a feature.</li>
* **sin:** Calculate the sine of a feature.</li>
* **cos:** Calculate the cosine of a feature.</li>
* **tan:** Calculate the tangent of a feature.</li>

ATOM's implementation of DFS uses the [featuretools](https://www.featuretools.com/) package.

<br>

**Genetic Feature Generation**<br>
Genetic feature generation (GFG) uses [genetic programming](https://en.wikipedia.org/wiki/Genetic_programming),
a branch of evolutionary programming, to determine which features
are successful and create new ones based on those. Where dfs can be
seen as some kind of "brute force" for feature engineering, gfg tries
to improve its features with every generation of the algorithm. gfg
uses the same operators as dfs, but instead of only applying the
transformations once, it evolves them further, creating nested
structures of combinations of features. The new features are given the
name `feature_n`, where n stands for the n-th feature in the dataset.
You can access the genetic feature's fitness and description (how they
are calculated) through the `genetic_features` attribute.

ATOM uses the [SymbolicTransformer][] class from the [gplearn](https://gplearn.readthedocs.io/en/stable/index.html)
package for the genetic algorithm. Read more about this implementation
[here](https://gplearn.readthedocs.io/en/stable/intro.html#transformer).

<br>

## Grouping similar features

When your dataset contains many similar features corresponding to a
certain natural group or entity, it's possible to replace these
features for a handful of them, that should capture the relations of
the group, in order to lose as little information as possible. To
achieve this, the [FeatureGrouper][] class computes certain statistical
properties that describe the group's distribution, like the mean or the
median, and replaces the columns with the result of these statistical
calculations over every row in the dataset. The goal of this approach
is to reduce the number of columns in the dataset, avoiding the
[curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality).


<br>

## Selecting useful features

The [FeatureSelector][] class provides tooling to select the relevant
features from a dataset. It can be accessed from atom through the
[feature_selection][atomclassifier-feature_selection] method.

<br>

### Standard strategies

<a name="univariate"></a>
**Univariate**<br>
Univariate feature selection works by selecting the best features based
on univariate statistical F-test. The test is provided via the `solver`
parameter. It takes any function taking two arrays (X, y), and returning
arrays (scores, p-values). Read more in sklearn's [documentation](https://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection).

<br style="display: block; margin-top: 2em; content: ' '">

<a name="pca"></a>
**Principal Components Analysis**<br>
Applying PCA reduces the dimensionality of the dataset by maximizing
the variance of each dimension. The new features are called `pca0`,
`pca1`, etc... PCA can be applied in three ways:

* If the data is dense (i.e. not sparse), the estimator used is [PCA][].
  Before fitting the transformer, the data is scaled to mean=0 and std=1
  if it wasn't already. Read more in sklearn's [documentation](https://scikit-learn.org/stable/modules/decomposition.html#pca).
* If the data is [sparse][sparse datasets] (often the case for term-document
  matrices, see [Vectorizer][]), the estimator used is [TruncatedSVD][].
  Read more in sklearn's [documentation](https://scikit-learn.org/stable/modules/decomposition.html#truncated-singular-value-decomposition-and-latent-semantic-analysis).
* If [`engine`][featureselector-engine] is "sklearnex" or "cuml", the estimator
  used is the package's PCA implementation. Sparse data is not supported for
  neither engine.


<br style="display: block; margin-top: 2em; content: ' '">

<a name="sfm"></a>
**Selection from model**<br>
SFM uses an estimator with `feature_importances_` or `coef_` attributes
to select the best features in a dataset based on importance weights.
The estimator is provided through the `solver` parameter and can be
already fitted. ATOM allows you to use one its predefined [models][],
e.g. `solver="RF"`. If you didn't call the FeatureSelector through atom,
don't forget to indicate the estimator's task adding `_class` or `_reg`
after the name, e.g. `RF_class` to use a random forest classifier. Read
more in sklearn's [documentation](https://scikit-learn.org/stable/modules/feature_selection.html#feature-selection-using-selectfrommodel).

<br style="display: block; margin-top: 2em; content: ' '">

<a name="sfs"></a>
**Sequential Feature Selection**<br>
Sequential feature selection adds (forward selection) or removes (backward
selection) features to form a feature subset in a greedy fashion. At each
stage, this estimator chooses the best feature to add or remove based on
the cross-validation score of an estimator. Read more in sklearn's [documentation](https://scikit-learn.org/stable/modules/feature_selection.html#sequential-feature-selection).

<br style="display: block; margin-top: 2em; content: ' '">

<a name="rfe"></a>
**Recursive Feature Elimination**<br>
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

<br>

### Advanced strategies

The following strategies are a collection of nature-inspired optimization
algorithms that maximize an objective function. If not manually specified,
the function calculates the cross-validated score of a model on the data.
Use the `scoring` parameter (not present in description, part of kwargs)
to specify the metric to optimize on.

<br style="display: block; margin-top: 2em; content: ' '">

<a name="pso"></a>
**Particle Swarm Optimization**<br>
Particle Swarm Optimization (PSO) optimizes a problem by having a population
of candidate solutions (particles), and moving them around in the search-space
according to simple mathematical formula over the particle's position and
velocity. Each particle's movement is influenced by its local best known
position, but is also guided toward the best known positions in the search-space,
which are updated as better positions are found by other particles. This is
expected to move the swarm toward the best solutions. Read more [here](https://link.springer.com/chapter/10.1007/978-3-319-13563-2_51).

<br style="display: block; margin-top: 2em; content: ' '">

<a name="hho"></a>
**Harris Hawks Optimization**<br>
Harris Hawks Optimization (HHO) mimics the action and reaction of Hawk's
team collaboration hunting in nature and prey escaping to discover the
solutions of the single-objective problem. Read more [here](https://link.springer.com/chapter/10.1007/978-981-32-9990-0_12).

<br style="display: block; margin-top: 2em; content: ' '">

<a name="gwo"></a>
**Grey Wolf Optimization**<br>
The Grey Wolf Optimizer (GWO) mimics the leadership hierarchy and hunting
mechanism of grey wolves in nature. Four types of grey wolves such as alpha,
beta, delta, and omega are employed for simulating the leadership hierarchy.
In addition, three main steps of hunting, searching for prey, encircling prey,
and attacking prey, are implemented to perform optimization. Read more
[here](https://www.sciencedirect.com/science/article/abs/pii/S0925231215010504?via%3Dihub).

<br style="display: block; margin-top: 2em; content: ' '">

<a name="dfo"></a>
**Dragonfly Optimization**<br>
The Dragonfly Algorithm (DFO) algorithm originates from static and dynamic
swarming behaviours. These two swarming behaviours are very similar to the
two main phases of optimization using meta-heuristics: exploration and
exploitation. Dragonflies create sub swarms and fly over different areas in
a static swarm, which is the main objective of the exploration phase. In
the static swarm, however, dragonflies fly in bigger swarms and along one
direction, which is favourable in the exploitation phase. Read more
[here](https://linkinghub.elsevier.com/retrieve/pii/S0950705120303889).

<br style="display: block; margin-top: 2em; content: ' '">

<a name="genetic"></a>
**Genetic Optimization**<br>
Genetic Optimization is a metaheuristic inspired by the process of natural
selection that belongs to the larger class of evolutionary algorithms.
Genetic algorithms are commonly used to generate high-quality solutions to
optimization and search problems by relying on biologically inspired
operators such as mutation, crossover and selection. Read more
[here](https://ieeexplore.ieee.org/document/953980).


<br>

### Other selection methods

**Removing features with low or high variance**<br>
Variance is the expectation of the squared deviation of a random
variable from its mean. Features with low variance have many values
repeated, which means the model can't learn much from them. In a
similar way, features with very high variance have very few values
repeated, which makes it also difficult for a model to learn from
this feature.

[FeatureSelector][] removes a categorical feature when the maximum
number of occurrences for any value is below `min_repeated` or when
the same value is repeated in at least `max_repeated` fraction of the
rows. The default option is to remove a feature if all values in it are
either different or exactly the same.

<br style="display: block; margin-top: 2em; content: ' '">

**Removing features with multi-collinearity**<br>
Two features that are highly correlated are redundant, i.e. two will
not contribute more to the model than only one of them. [FeatureSelector][]
will drop a feature that has a [Pearson correlation coefficient][pearson]
larger than `max_correlation` with another feature. A correlation of 1
means the two columns are equal. A dataframe of the removed features and
their correlation values can be accessed through the `collinear` attribute.
