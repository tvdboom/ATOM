# ATOM

<pre><strong><em>class</em> atom.<strong style="color:#008AB8">ATOM</strong>(X,
                y=None,
                percentage=100,
                test_size=0.3,
                log=None,
                n_jobs=1,
                warnings=False,
                verbose=0,
                random_state=None,
                verbose=0)</strong><br>
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L113">[source]</a></div></pre>

Main class of the package. The `ATOM` class is a parent class of the `ATOMClassifier`
 and `ATOMRegressor` classes. These will inherit all methods and attributes described
 in this page. Note that contrary to scikit-learn's API, the ATOM object already contains
 the dataset on which we want to perform the analysis. Calling a method will automatically
 apply it on the dataset it contains.

!!! warning
    Don't call the `ATOM` class directly! Use `ATOMClassifier` or `ATOMRegressor`
     depending on the task at hand. Click [here](../getting_started/#usage) for an example.

The class initializer will label-encode the target column if its labels are
not ordered integers. It will also apply some standard data
cleaning steps unto the dataset. These steps include:

  * Transforming the input data into a pd.DataFrame (if it wasn't one already)
   that can be accessed through the class' data attributes.
  * Removing columns with prohibited data types ('datetime64',
   'datetime64[ns]', 'timedelta[ns]').
  * Removing categorical columns with maximal cardinality (the number of
   unique values is equal to the number of instances. Usually the case for
    names, IDs, etc...).
  * Removing columns with minimum cardinality (all values are the same).
  * Removing rows with missing values in the target column.


<table>
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>X: dict, sequence, np.array or pd.DataFrame</strong>
<blockquote>
Dataset containing the features, with shape=(n_samples, n_features).
</blockquote>

<strong>y: string, sequence, np.array or pd.Series, optional (default=None)</strong>
<blockquote>
<ul>
<li>If None: the last column of X is selected as target column</li>
<li>If string: name of the target column in X</li>
<li>Else: data target column with shape=(n_samples,)</li>
</blockquote>

<strong>percentage: int or float, optional (default=100)</strong>
<blockquote>
Percentage of the provided dataset to use.
</blockquote>

<strong>test_size: float, optional (default=0.3)</strong>
<blockquote>
Split fraction of the train and test set.
</blockquote>

<strong>log: string or None, optional (default=None)</strong>
<blockquote>
Name of the logging file. 'auto' for default name with date and time. None to not save any log.
</blockquote>

<strong>n_jobs: int, optional (default=1)</strong>
<blockquote>
Number of cores to use for parallel processing.
<ul>
<li>If -1, use all available cores</li>
<li>If <-1, use available_cores - 1 + n_jobs</li>
Beware that using multiple processes on the same machine may cause
memory issues for large datasets.
</blockquote>

<strong> warnings: bool, optional (default=False)</strong>
<blockquote>
If False, it supresses all warnings.
</blockquote>

<strong>verbose: int, optional (default=0)</strong>
<blockquote>
Verbosity level of the class. Possible values are:
<ul>
<li>0 to not print anything</li>
<li>1 to print minimum information</li>
<li>2 to print average information</li>
<li>3 to print maximum information</li>
</ul>
</blockquote>

<strong>random_state: int or None, optional (default=None)</strong>
<blockquote>
Seed used by the random number generator. If None, the random number
 generator is the RandomState instance used by np.random.
</blockquote>

<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Data attributes:</strong></td>
<td width="75%" style="background:white;">
<strong>dataset: pd.DataFrame</strong>
<blockquote>
Complete dataset in the pipeline.
</blockquote>

<strong>train: pd.DataFrame</strong>
<blockquote>
Training set.
</blockquote>

<strong>test: pd.DataFrame</strong>
<blockquote>
Test set.
</blockquote>

<strong>X: pd.DataFrame</strong>
<blockquote>
Feature set.
</blockquote>

<strong>y: pd.Series</strong>
<blockquote>
Target column.
</blockquote>

<strong>X_train: pd.DataFrame</strong>
<blockquote>
Training features.
</blockquote>

<strong>y_train: pd.Series</strong>
<blockquote>
Training target.
</blockquote>

<strong>X_test: pd.DataFrame</strong>
<blockquote>
Test features.
</blockquote>

<strong>y_test: pd.Series</strong>
<blockquote>
Test target.
</blockquote>
</td></tr>

<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Attributes:</strong></td>
<td width="75%" style="background:white;">

<strong>mapping: dict</strong>
<blockquote>
Dictionary of the target values mapped to their respective encoded integer. Only for classification tasks.
</blockquote>

<strong>errors: dict</strong>
<blockquote>
Dictionary of the encountered exceptions (if any) while fitting the models.
</blockquote>

<strong>winner: callable</strong>
<blockquote>
Model subclass that performed best on the test set.
</blockquote>

<strong>scores: pd.DataFrame</strong>
<blockquote>
Dataframe (or list of dataframes if successive_halving=True) of the results. Columns can include:
<ul>
<li>model: model's name (acronym)</li>
<li>total_time: time spent on this model</li>
<li>score_train: metric score on the training set</li>
<li>score_test: metric score on the test set</li>
<li>fit_time: time spent fitting and predicting</li>
<li>bagging_mean: mean score of the bagging's results</li>
<li>bagging_std: standard deviation score of the bagging's results</li>
<li>bagging_time: time spent on the bagging algorithm</li>
</ul>
</blockquote>
</td>
</tr>
</table>
<br>



## Utilities

The ATOM class contains a variety of methods to help you handle the data and
inspect the pipeline.

<table width="100%">
<tr>
<td width="15%"><a href="#atom-stats">stats</a></td>
<td>Print out a list of basic statistics on the dataset.</td>
</tr>

<tr>
<td><a href="#atom-scale">scale</a></td>
<td>Scale all the features to mean=1 and std=0.</td>
</tr>

<tr>
<td><a href="#atom-update">update</a></td>
<td>Update all data attributes.</td>
</tr>

<tr>
<td><a href="#atom-report">report</a></td>
<td>Get an extensive profile analysis of the data.</td>
</tr>

<tr>
<td><a href="#atom-results">results</a></td>
<td>Print final results for a specific metric.</td>
</tr>

<tr>
<td><a href="#atom-save">save</a></td>
<td>Save the ATOM class to a pickle file.</td>
</tr>
</table>
<br>


<a name="atom-stats"></a>
<pre><em>function</em> atom.ATOM.<strong style="color:#008AB8">stats</strong>()
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L423">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Print out a list of basic statistics on the dataset.
</div>
<br /><br />


<a name="atom-scale"></a>
<pre><em>function</em> atom.ATOM.<strong style="color:#008AB8">scale</strong>()
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L495">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Scale all the features to mean=1 and std=0.
</div>
<br /><br />


<a name="atom-update"></a>
<pre><em>function</em> atom.ATOM.<strong style="color:#008AB8">update</strong>(df='dataset')
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L524">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
If you change any of the class' data attributes in between the pipeline, you
 should call this method to change all other data attributes to their correct
  values. Independent attributes are updated in unison, that is, setting
   df='X_train' will also update X_test, y_train and y_test, or df='train'
    will also update the test set, etc... This means that you can change both
     X_train and y_train and update them with one call of the method.
<br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>df: string, optional(default='dataset')</strong>
<blockquote>
Data attribute that has been changed.
</blockquote>
</td></tr>
</table>
</div>
<br />


<a name="atom-report"></a>
<pre><em>function</em> atom.ATOM.<strong style="color:#008AB8">report</strong>(df='dataset', rows=None, filename=None)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L582">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Get an extensive profile analysis of the data. The report is rendered
 in HTML5 and CSS3 and saved to the `profile` attribute. Note that this method
 can be slow for very large datasets.
Dependency: [pandas-profiling](https://pandas-profiling.github.io/pandas-profiling/docs/).
<br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>df: string, optional(default='dataset')</strong>
<blockquote>
Name of the data class attribute to get the profile from.
</blockquote>
<strong>rows: int or None, optional(default=None)</strong>
<blockquote>
Number of rows to process (randomly picked). None for all rows.
</blockquote>
<strong>filename: str or None, optional (default=None)</strong>
<blockquote>
Name of the file when saved (as .html). None to not save anything.
</blockquote>
</tr>
</table>
</div>
<br />


<a name="atom-results"></a>
<pre><em>function</em> atom.ATOM.<strong style="color:#008AB8">results</strong>(metric=None)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L630">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Print the pipeline's final results for a specific metric. If a model
 shows a `XXX`, it means the metric failed for that specific model. This
 can happen if either the metric is unavailable for the task or if the
 model does not have a `predict_proba` method while the metric needs it.
<br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>metric: string or None, optional (default=None)</strong>
<blockquote>
String of one of sklearn's predefined metrics. If None, the metric
 used to fit the pipeline is selected and the bagging results will
 be showed (if used).
</blockquote>
</tr>
</table>
</div>
<br />


<a name="atom-save"></a>
<pre><em>function</em> atom.ATOM.<strong style="color:#008AB8">save</strong>(filename=None)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L711">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Save the ATOM class to a pickle file. This method is also available for the model
 subclasses, e.g. `atom.XGB.save(filename='ATOM_xgboost')`. In this case, the
 model subclass is saved, instead of the ATOM class.
 
!!! warning
    Remember that the class contains the complete dataset (and variations of
    it). This means the files can become very large for big datasets!

<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>filename: str or None, optional (default=None)</strong>
<blockquote>
Name to save the file with. None to save with default name.
</blockquote>
</tr>
</table>
</div>
<br />



## Data cleaning

Before throwing your data in a model, it is crucial to apply some standard data
cleaning steps. ATOM provides four data cleaning methods to handle missing values,
 categorical columns, outliers and unbalanced datasets. Calling on one of them
 will automatically apply the method on the dataset in the class and update the
 class' data attributes accordingly.

!!! tip
    Use the `report` method to examine the data and help you determine suitable
    parameters for the data cleaning methods.
    

<table width="100%">
<tr>
<td><a href="#atom-impute">impute</a></td>
<td>Handle missing values in the dataset.</td>
</tr>

<tr>
<td><a href="#atom-encode">encode</a></td>
<td>Encode categorical columns.</td>
</tr>

<tr>
<td><a href="#atom-outliers">outliers</a></td>
<td>Remove outliers from the training set.</td>
</tr>

<tr>
<td><a href="#atom-balance">balance</a></td>
<td>Balance the number of instances per target class.</td>
</tr>
</table>
<br>


<a name="atom-impute"></a>
<pre><em>function</em> atom.ATOM.<strong style="color:#008AB8">impute</strong>(strat_num='remove',
                          strat_cat='remove',
                          max_frac_rows=0.5,
                          max_frac_cols=0.5,
                          missing=None) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L729">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Handle missing values according to the selected strategy. Also removes rows and
 columns with too many missing values.
 <br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>strat_num: str, int or float, optional (default='remove')</strong>
<blockquote>
Imputing strategy for numerical columns. Choose from:
<ul>
<li>'remove': remove row if any missing value</li>
<li>'mean': impute with mean of column</li>
<li>'median': impute with median of column</li>
<li>'knn': impute using a K-Nearest Neighbors approach</li>
<li>'most_frequent': impute with most frequent value</li>
<li>int or float: impute with provided numerical value</li>
</ul>
</blockquote>
<strong>strat_cat: str, optional (default='remove')</strong>
<blockquote>
Imputing strategy for categorical columns. Choose from:
<ul>
<li>'remove': remove row if any missing value</li>
<li>'most_frequent': impute with most frequent value</li>
<li>string: impute with provided string</li>
</ul>
</blockquote>
<strong>min_frac_rows: float, optional (default=0.5)</strong>
<blockquote>
Minimum fraction of non missing values in row. If less, the row is removed.
</blockquote>
<strong> min_frac_cols: float, optional (default=0.5)</strong>
<blockquote>
Minimum fraction of non missing values in column. If less, the column is removed.
</blockquote>
<strong> missing: int, float or list, optional (default=None)</strong>
<blockquote>
List of values to impute. None for default list: [None, np.NaN, np.inf, -np.inf, '', '?', 'NA', 'nan', 'inf'].
</blockquote>
</tr>
</table>
</div>
<br />


<a name="atom-encode"></a>
<pre><em>function</em> atom.ATOM.<strong style="color:#008AB8">encode</strong>(max_onehot=10, frac_to_other=0) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L887">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Perform encoding of categorical features. The encoding type depends on the
 number of unique values in the column:
<ul>
<li>label-encoding for n_unique=2</li>
<li>one-hot-encoding for 2 < n_unique <= max_onehot</li>
<li>target-encoding for n_unique > max_onehot</li>
</ul>
It also replaces classes with low occurences with the value 'other' in order
 to prevent too high cardinality.
 <br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>max_onehot: int or None, optional (default=10)</strong>
<blockquote>
Maximum number of unique values in a feature to perform one-hot-encoding.
 If None, it will never perform one-hot-encoding.
</blockquote>
<strong>frac_to_other: float, optional (default=0)</strong>
<blockquote>
Classes with less instances than n_rows * fraction_to_other are replaced with 'other'.
</tr>
</table>
</div>
<br />


<a name="atom-outliers"></a>
<pre><em>function</em> atom.ATOM.<strong style="color:#008AB8">outliers</strong>(max_sigma=3, include_target=False) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L982">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Remove rows from the training set where at least one of the values lies further
 than `max_sigma` * standard_deviation away from the mean of the column.
 <br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>max_sigma: int or float, optional (default=3)</strong>
<blockquote>
Maximum allowed standard deviations from the mean.
</blockquote>
<strong>include_target: bool, optional (default=False)</strong>
<blockquote>
Wether to include the target column when searching for outliers.
</blockquote>
</tr>
</table>
</div>
<br />


<a name="atom-balance"></a>
<pre><em>function</em> atom.ATOM.<strong style="color:#008AB8">balance</strong>(oversample=None, undersample=None, n_neighbors=5) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L1021">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Balance the number of instances per target class in the training set. If both
 oversampling and undersampling are used, they will be applied in that order.
 Only for classification tasks.
 Dependency: [imbalanced-learn](https://imbalanced-learn.readthedocs.io/en/stable/).
 <br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>oversample: float, string or None, optional (default=None)</strong>
<blockquote>
Oversampling strategy using  [ADASYN](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.ADASYN.html).
 Choose from:
<ul>
<li>None: don't oversample</li>
<li>float: fraction of minority/majority classes (only for binary classif.)</li>
<li>'minority': resample only the minority class</li>
<li>'not minority': resample all but minority class</li>
<li>'not majority': resample all but majority class</li>
<li>'all': resample all classes</li>
</ul>
</blockquote>
<strong>undersample: float, string or None, optional (default=None)</strong>
<blockquote>
Undersampling strategy using [NearMiss](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.under_sampling.NearMiss.html).
 Choose from:
<ul>
<li>None: don't oversample</li>
<li>float: fraction of minority/majority classes (only for binary classif.)</li>
<li>'majority': resample only the majority class</li>
<li>'not minority': resample all but minority class</li>
<li>'not majority': resample all but majority class</li>
<li>'all': resample all classes</li>
</ul>
</blockquote>
<strong>n_neighbors: int, optional (default=5)</strong>
<blockquote>
Number of nearest neighbors used for any of the algorithms.
</blockquote>
</tr>
</table>
</div>
<br />



## Feature selection

To further pre-process the data you can create new non-linear features using a
 genetic algorithm or, if your dataset is too large, remove features using one
 of the provided strategies.

<table width="100%">
<tr>
<tr>
<td><a href="#atom-feature-insertion">feature_insertion</a></td>
<td>Use a genetic algorithm to create new combinations of existing features.</td>
</tr>

<tr>
<td><a href="#atom-feature-selection">feature_selection</a></td>
<td>Remove features according to the selected strategy.</td>
</tr>
</table>
<br>



<a name="atom-feature-insertion"></a>
<pre><em>function</em> atom.ATOM.<strong style="color:#008AB8">feature_insertion</strong>(n_features=2, generations=20, population=500) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L1156">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Use a genetic algorithm to create new combinations of existing
features and add them to the original dataset in order to capture
the non-linear relations between the original features. A dataframe
containing the description of the newly generated features and their
scores can be accessed through the `genetic_features` attribute. The
algorithm is implemented using the [Symbolic Transformer](https://gplearn.readthedocs.io/en/stable/reference.html#symbolic-transformer)
 method, which can be accessed through the `genetic_algorithm` attribute. It is
adviced to only use this method when fitting linear models.
Dependency: [gplearn](https://gplearn.readthedocs.io/en/stable/).
 <br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>n_features: int, optional (default=2)</strong>
<blockquote>
Maximum number of newly generated features (no more than 1% of the population).
</blockquote>
<strong>generations: int, optional (default=20)</strong>
<blockquote>
Number of generations to evolve.
</blockquote>
<strong>population: int, optional (default=500)</strong>
<blockquote>
Number of programs in each generation.
</blockquote>
</tr>
</table>
</div>
<br />


<a name="atom-feature-selection"></a>
<pre><em>function</em> atom.ATOM.<strong style="color:#008AB8">feature_selection</strong>(strategy=None,
                                     solver=None,
                                     n_features=None,
                                     max_frac_repeated=1.,
                                     max_correlation=0.98,
                                     **kwargs) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L1285">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Remove features according to the selected strategy. Ties between
features with equal scores will be broken in an unspecified way. Also
removes features with too low variance and finds pairs of collinear features
based on the Pearson correlation coefficient. For each pair above the specified
limit (in terms of absolute value), it removes one of the two.

Note that the RFE and RFECV strategies don't work when the solver is a
 CatBoost model due to incompatibility of the APIs. If the pipeline has already
 ran before running the RFECV, the scoring parameter will be set to the selected
 metric (if scoring=None).
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>strategy: string or None, optional (default=None)</strong>
<blockquote>
Feature selection strategy to use. Choose from:
<ul>
<li>None: do not perform any feature selection algorithm</li>
<li>'univariate': perform a univariate F-test, from sklearn [SelectKBest](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html)</li>
<li>'PCA': perform a principal component analysis, from sklearn [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)</li>
<li>'SFM': select best features from model, from sklearn [SelectFromModel](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html)</li>
<li>'RFE': recursive feature eliminator, from sklearn [RFE](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html)</li>
<li>'RFECV': RFE with cross-validated selection, from sklearn [RFECV](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html)</li>
</ul>
The sklearn objects can be found under the `univariate`, `PCA`, `SFM`,
 `RFE` or `RFECV` attributes of the class.
</blockquote>
<strong>solver: string, callable or None, optional (default=None)</strong>
<blockquote>
Solver or model to use for the feature selection strategy. See the
sklearn documentation for an extended descrition of the choices.
Select None for the default option per strategy (not applicable
for SFM, RFE and RFECV).
<ul>
<li>for 'univariate', choose from:
   <ul>
   <li>'f_classif' (default for classification tasks)</li>
   <li>'f_regression' (default for regression tasks)</li>
   <li>'mutual_info_classif'</li>
   <li>'mutual_info_regression'</li>
   <li>'chi2'</li>
   <li>Any function taking two arrays (X, y), and returning arrays (scores, pvalues).</li>
   </ul>
<li>for 'PCA', choose from:
    <ul>
    <li>'auto' (default)</li>
    <li>'full'</li>
    <li>'arpack'</li>
    <li>'randomized'</li>
    </ul>
<li>for 'SFM': choose a base estimator from which the transformer is built.
               The estimator must have either a `feature_importances_` or
               `coef_` attribute after fitting. You can use a model from the
               [pipeline](#pipeline). No default option.</li>
<li>for 'RFE': choose a supervised learning estimator. The estimator must have
               either a `feature_importances_` or `coef_` attribute after
               fitting. You can use a model from the [pipeline](#pipeline).
               No default option.</li>
<li>for 'RFECV': choose a supervised learning estimator. The estimator must have
               either a `feature_importances_` or `coef_` attribute after
               fitting. You can use a model from the [pipeline](#pipeline).
               No default option.</li>
</ul>
</blockquote>
<strong>n_features: int, float or None, optional (default=None)</strong>
<blockquote>
Number of features to select (except for RFECV, where it's the
 minimum number of features to select).
<ul>
<li>if < 1: fraction of features to select</li>
<li>if >= 1: number of features to select</li>
<li>None to select all, or 1 for the RFECV</li>
</ul>
</blockquote>
<strong>max_frac_repeated: float or None, optional (default=1.)</strong>
<blockquote>
Remove features with the same value in at least this fraction of
 the total rows. The default is to keep all features with non-zero
 variance, i.e. remove the features that have the same value in all
 samples. None to skip this step.
</blockquote>
<strong>max_correlation: float or None, optional (default=0.98)</strong>
<blockquote>
Minimum value of the Pearson correlation cofficient to identify
correlated features. A dataframe of the removed features and their
correlation values can be accessed through the `collinear` attribute.
None to skip this step.
</blockquote>
<strong>\*\*kwargs</strong>
<blockquote>
Any extra parameter for the PCA, SFM, RFE or RFECV. See the sklearn
documentation for the available options.
</blockquote>
</tr>
</table>
</div>
<br />



## Pipeline

The pipeline method is where the models are fitted to the data and their
 performance is evaluated according to the selected metric. For every model, the
 pipeline applies the following steps:

1. The optimal hyperparameters are selectred using a Bayesian Optimization (BO)
 algorithm with gaussian process as kernel. The resulting score of each step of
 the BO is either computed by cross-validation on the complete training set or
 by randomly splitting the training set every iteration into a (sub) training
 set and a validation set. This process can create some data leakage but
 ensures a maximal use of the provided data. The test set, however, does not
 contain any leakage and will be used to determine the final score of every model.
 Note that, if the dataset is relatively small, the best score on the BO can
 consistently be lower than the final score on the test set (despite the
 leakage) due to the considerable fewer instances on which it is trained.
<div></div>
 
2. Once the best hyperparameters are found, the model is trained again, now
 using the complete training set. After this, predictions are made on the test set.
<div></div>

3. You can choose to evaluate the robustness of each model's applying a bagging
 algorithm, i.e. the model will be trained multiple times on a bootstrapped
 training set, returning a distribution of its performance on the test set.

A couple of things to take into account:

* The metric implementation follows [sklearn's API](https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values).
  This means that the implementation always tries to maximize the scorer, i.e.
  loss functions will be made negative.
* If an exception is encountered while fitting a model, the
  pipeline will automatically jump to the next model and save the
  exception in the `errors` attribute.
* When showing the final results, a `!!` indicates the highest
  score and a `~` indicates that the model is possibly overfitting
  (training set has a score at least 20% higher than the test set).
* The winning model subclass will be attached to the `winner` attribute.
</br>


There are three methods to call for the pipeline.

* The `pipeline` method fits the models directly to the dataset.
 
* If you want to compare similar models, you can use the `successive_halving`
 method when running the pipeline. This technique fits N models to
 1/N of the data. The best half are selected to go to the next iteration where
 the process is repeated. This continues until only one model remains, which is
 fitted on the complete dataset. Beware that a model's performance can depend
 greatly on the amount of data on which it is trained. For this reason we
 recommend only to use this technique with similar models, e.g. only using
 tree-based models.

* The `train_sizing` method fits the models on subsets of the training data.
 This can be used to examine the optimum size of the dataset needed for a
 satisfying performance.
</br>

<table width="100%">
<tr>
<td><a href="#atom-pipeline">pipeline</a></td>
<td>Fit the models to the data in a direct fashion.</td>
</tr>

<tr>
<td><a href="#atom-successive-halving">successive_halving</a></td>
<td>Fit the models to the data in a successive halving fashion.</td>
</tr>

<tr>
<td><a href="#atom-train-sizing">train_sizing</a></td>
<td>Fit the models to the data in a train sizing fashion.</td>
</tr>

</table>
<br>


<a name="atom-pipeline"></a>
<pre><em>function</em> atom.ATOM.<strong style="color:#008AB8">pipeline</strong>(models,
                            metric=None,
                            greater_is_better=True,
                            needs_proba=False,
                            max_iter=0,
                            max_time=np.inf,
                            init_points=5,
                            plot_bo=False,
                            cv=3,
                            bagging=None) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L1626">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
<br /><br />
<table width="100%">
<tr>
<td width="13%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="73%" style="background:white;">
<strong>models: string or sequence</strong>
<blockquote>
List of models to fit on the data. Use the predefined acronyms to select the models. Possible values are (case insensitive):
<ul>
<li>'GNB' for [Gaussian Naïve Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)<br>Only for classification tasks. No hyperparameter tuning.</li>
<li>'MNB' for [Multinomial Naïve Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)<br>Only for classification tasks.</li>
<li>'BNB' for [Bernoulli Naïve Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html)<br>Only for classification tasks.</li>
<li>'GP' for Gaussian Process [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html)<br>No hyperparameter tuning.</li>
<li>'OLS' for [Ordinary Least Squares](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)<br>Only for regression tasks. No hyperparameter tuning.</li>
<li>'Ridge' for Ridge Linear [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)<br>Only for regression tasks.</li>
<li>'Lasso' for [Lasso Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)<br>Only for regression tasks.</li>
<li>'EN' for [ElasticNet Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)<br>Only for regression tasks.</li>
<li>'BR' for [Bayesian Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html)<br>Only for regression tasks. Uses ridge regularization.</li>
<li>'LR' for [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)<br>Only for classification tasks.</li> 
<li>'LDA' for [Linear Discriminant Analysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)<br>Only for classification tasks.</li>
<li>'QDA' for [Quadratic Discriminant Analysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html)<br>Only for classification tasks.</li>
<li>'KNN' for K-Nearest Neighbors [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)</li>
<li>'Tree' for a single Decision Tree [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)</li>
<li>'Bag' for Bagging [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html)<br>Uses a decision tree as base estimator.</li>
<li>'ET' for Extra-Trees [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html)</li>
<li>'RF' for Random Forest [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)</li>
<li>'AdaB' for AdaBoost [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html)<br>Uses a decision tree as base estimator.</li>
<li>'GBM' for Gradient Boosting Machine [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)</li> 
<li>'XGB' for XGBoost [classifier](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier)/[regressor](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor)<br>Only available if package is installed.</li>
<li>'LGB' for LightGBM [classifier](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html)/[regressor](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html)<br>Only available if package is installed.</li>
<li>'CatB' for CatBoost [classifier](https://catboost.ai/docs/concepts/python-reference_catboostclassifier.html)/[regressor](https://catboost.ai/docs/concepts/python-reference_catboostregressor.html)<br>Only available if package is installed.</li>
<li>'lSVM' for Linear Support Vector Machine [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html)<br>Uses a one-vs-rest strategy for multiclass classification tasks.</li> 
<li>'kSVM' for Kernel (non-linear) Support Vector Machine [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)<br>Uses a one-vs-one strategy for multiclass classification tasks.</li>
<li>'PA' for Passive Aggressive [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveRegressor.html)</li>
<li>'SGD' for Stochastic Gradient Descent [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html)</li>
<li>'MLP' for Multilayer Perceptron [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor)<br>Can have between one and three hidden layers.</li> 
</ul>
</blockquote>
<strong>metric: string or callable, optional (default=None)</strong>
<blockquote>
Metric on which the pipeline fits the models. Choose from any of
sklearn's predefined [scorers](https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules), use a score (or loss)
function with signature metric(y, y_pred, **kwargs) or use a
scorer object. If None, ATOM will try to use any metric it already has in the
pipeline. If it hasn't got any, a default metric per task is selected:
<ul>
<li>'f1' for binary classification</li>
<li>'f1_weighted' for multiclas classification</li>
<li>'r2' for regression</li>
</ul>
</blockquote>
<strong>greater_is_better: bool, optional (default=True)</strong>
<blockquote>
Wether the metric is a score function or a loss function,
i.e. if True, a higher score is better and if False, lower is
better. Will be ignored if the metric is a string or a scorer.
</blockquote>
<strong> needs_proba: bool, optional (default=False)</strong>
<blockquote>
Whether the metric function requires probability estimates out of a
 classifier. If True, make sure that every model in the pipeline has
 a `predict_proba` method! Will be ignored if the metric is a string
 or a scorer.
</blockquote>
<strong> needs_threshold: bool, optional (default=False)</strong>
<blockquote>
Whether the metric function takes a continuous decision certainty. This only
 works for binary classification using estimators that have either a
 `decision_function` or `predict_proba` method. Will be ignored if the metric
 is a string or a scorer.
</blockquote>
<strong>max_iter: int or sequence, optional (default=0)</strong>
<blockquote>
Maximum number of iterations of the BO. If 0, skip the BO and fit
 the model on its default parameters. If sequence, the n-th value
 will apply to the n-th model in the pipeline.
</blockquote>
<strong>max_time: int, float or sequence, optional (default=np.inf)</strong>
<blockquote>
Maximum time allowed for the BO per model (in seconds). If 0, skip
 the BO and fit the model on its default parameters. If sequence,
 the n-th value will apply to the n-th model in the pipeline.
</blockquote>
<strong>init_points: int or sequence, optional (default=5)</strong>
<blockquote>
Initial number of tests of the BO before fitting the surrogate
 function. If 1, the default models' hyperparameters will be used. If sequence,
 the n-th value will apply to the n-th model in the pipeline.
</blockquote>
<strong>cv: int or sequence, optional (default=3)</strong>
<blockquote>
Strategy to fit and score the model selected after every step of the BO.
<ul>
<li>if 1, randomly split into a train and validation set</li>
<li>if >1, perform a k-fold cross validation on the training set</li>
</ul>
</blockquote>
<strong>plot_bo: bool, optional (default=False)</strong>
<blockquote>
Wether to plot the BO's progress as it runs. Creates a canvas with
two plots: the first plot shows the score of every trial and the
second shows the distance between the last consecutive steps. Don't
forget to call `%matplotlib` at the start of the cell if you are
using jupyter notebook!
</blockquote>
<strong>bagging: int or None, optional (default=None)</strong>
<blockquote>
Number of data sets (bootstrapped from the training set) to use in the bagging
 algorithm. If None or 0, no bagging is performed.
</blockquote>
</tr>
</table>
</div>
<br />


<a name="atom-successive-halving"></a>
<pre><em>function</em> atom.ATOM.<strong style="color:#008AB8">successive_halving</strong>(models,
                                      metric=None,
                                      greater_is_better=True,
                                      needs_proba=False,
                                      skip_iter=0,
                                      max_iter=0,
                                      max_time=np.inf,
                                      init_points=5,
                                      plot_bo=False,
                                      cv=3,
                                      bagging=None) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L1626">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
<br /><br />
<table width="100%">
<tr>
<td width="13%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="73%" style="background:white;">
<strong>models: string or sequence</strong>
<blockquote>
List of models to fit on the data. Use the predefined acronyms to select the models. Possible values are (case insensitive):
<ul>
<li>'GNB' for [Gaussian Naïve Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)<br>Only for classification tasks. No hyperparameter tuning.</li>
<li>'MNB' for [Multinomial Naïve Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)<br>Only for classification tasks.</li>
<li>'BNB' for [Bernoulli Naïve Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html)<br>Only for classification tasks.</li>
<li>'GP' for Gaussian Process [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html)<br>No hyperparameter tuning.</li>
<li>'OLS' for [Ordinary Least Squares](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)<br>Only for regression tasks. No hyperparameter tuning.</li>
<li>'Ridge' for Ridge Linear [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)<br>Only for regression tasks.</li>
<li>'Lasso' for [Lasso Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)<br>Only for regression tasks.</li>
<li>'EN' for [ElasticNet Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)<br>Only for regression tasks.</li>
<li>'BR' for [Bayesian Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html)<br>Only for regression tasks. Uses ridge regularization.</li>
<li>'LR' for [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)<br>Only for classification tasks.</li> 
<li>'LDA' for [Linear Discriminant Analysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)<br>Only for classification tasks.</li>
<li>'QDA' for [Quadratic Discriminant Analysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html)<br>Only for classification tasks.</li>
<li>'KNN' for K-Nearest Neighbors [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)</li>
<li>'Tree' for a single Decision Tree [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)</li>
<li>'Bag' for Bagging [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html)<br>Uses a decision tree as base estimator.</li>
<li>'ET' for Extra-Trees [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html)</li>
<li>'RF' for Random Forest [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)</li>
<li>'AdaB' for AdaBoost [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html)<br>Uses a decision tree as base estimator.</li>
<li>'GBM' for Gradient Boosting Machine [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)</li> 
<li>'XGB' for XGBoost [classifier](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier)/[regressor](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor)<br>Only available if package is installed.</li>
<li>'LGB' for LightGBM [classifier](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html)/[regressor](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html)<br>Only available if package is installed.</li>
<li>'CatB' for CatBoost [classifier](https://catboost.ai/docs/concepts/python-reference_catboostclassifier.html)/[regressor](https://catboost.ai/docs/concepts/python-reference_catboostregressor.html)<br>Only available if package is installed.</li>
<li>'lSVM' for Linear Support Vector Machine [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html)<br>Uses a one-vs-rest strategy for multiclass classification tasks.</li> 
<li>'kSVM' for Kernel (non-linear) Support Vector Machine [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)<br>Uses a one-vs-one strategy for multiclass classification tasks.</li>
<li>'PA' for Passive Aggressive [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveRegressor.html)</li>
<li>'SGD' for Stochastic Gradient Descent [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html)</li>
<li>'MLP' for Multilayer Perceptron [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor)<br>Can have between one and three hidden layers.</li> 
</ul>
</blockquote>
<strong>metric: string or callable, optional (default=None)</strong>
<blockquote>
Metric on which the pipeline fits the models. Choose from any of
sklearn's predefined [scorers](https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules), use a score (or loss)
function with signature metric(y, y_pred, **kwargs) or use a
scorer object. If None, ATOM will try to use any metric it already has in the
pipeline. If it hasn't got any, a default metric per task is selected:
<ul>
<li>'f1' for binary classification</li>
<li>'f1_weighted' for multiclas classification</li>
<li>'r2' for regression</li>
</ul>
</blockquote>
<strong>greater_is_better: bool, optional (default=True)</strong>
<blockquote>
Wether the metric is a score function or a loss function,
i.e. if True, a higher score is better and if False, lower is
better. Will be ignored if the metric is a string or a scorer.
</blockquote>
<strong> needs_proba: bool, optional (default=False)</strong>
<blockquote>
Whether the metric function requires probability estimates out of a
 classifier. If True, make sure that every model in the pipeline has
 a `predict_proba` method! Will be ignored if the metric is a string
 or a scorer.
</blockquote>
<strong> needs_threshold: bool, optional (default=False)</strong>
<blockquote>
Whether the metric function takes a continuous decision certainty. This only
 works for binary classification using estimators that have either a
 `decision_function` or `predict_proba` method. Will be ignored if the metric
 is a string or a scorer.
</blockquote>
<strong>skip_iter: int, optional (default=0)</strong>
<blockquote>
Skip last `skip_iter` iterations of the successive halving. Will be ignored if
 successive_halving=False.
</blockquote>
<strong>max_iter: int or sequence, optional (default=0)</strong>
<blockquote>
Maximum number of iterations of the BO. If 0, skip the BO and fit
 the model on its default parameters. If sequence, the n-th value
 will apply to the n-th model in the pipeline.
</blockquote>
<strong>max_time: int, float or sequence, optional (default=np.inf)</strong>
<blockquote>
Maximum time allowed for the BO per model (in seconds). If 0, skip
 the BO and fit the model on its default parameters. If sequence,
 the n-th value will apply to the n-th model in the pipeline.
</blockquote>
<strong>init_points: int or sequence, optional (default=5)</strong>
<blockquote>
Initial number of tests of the BO before fitting the surrogate
 function. If 1, the default models' hyperparameters will be used. If sequence,
 the n-th value will apply to the n-th model in the pipeline.
</blockquote>
<strong>cv: int or sequence, optional (default=3)</strong>
<blockquote>
Strategy to fit and score the model selected after every step of the BO.
<ul>
<li>if 1, randomly split into a train and validation set</li>
<li>if >1, perform a k-fold cross validation on the training set</li>
</ul>
</blockquote>
<strong>plot_bo: bool, optional (default=False)</strong>
<blockquote>
Wether to plot the BO's progress as it runs. Creates a canvas with
two plots: the first plot shows the score of every trial and the
second shows the distance between the last consecutive steps. Don't
forget to call `%matplotlib` at the start of the cell if you are
using jupyter notebook!
</blockquote>
<strong>bagging: int or None, optional (default=None)</strong>
<blockquote>
Number of data sets (bootstrapped from the training set) to use in the bagging
 algorithm. If None or 0, no bagging is performed.
</blockquote>
</tr>
</table>
</div>
<br />


<a name="atom-train-sizing"></a>
<pre><em>function</em> atom.ATOM.<strong style="color:#008AB8">train_sizing</strong>(models,
                                metric=None,
                                greater_is_better=True,
                                needs_proba=False,
                                train_sizes=np.linspcae(0.1, 1.0, 10),
                                max_iter=0,
                                max_time=np.inf,
                                init_points=5,
                                plot_bo=False,
                                cv=3,
                                bagging=None) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L1626">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
<br /><br />
<table width="100%">
<tr>
<td width="13%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="73%" style="background:white;">
<strong>models: string or sequence</strong>
<blockquote>
List of models to fit on the data. Use the predefined acronyms to select the models. Possible values are (case insensitive):
<ul>
<li>'GNB' for [Gaussian Naïve Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)<br>Only for classification tasks. No hyperparameter tuning.</li>
<li>'MNB' for [Multinomial Naïve Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)<br>Only for classification tasks.</li>
<li>'BNB' for [Bernoulli Naïve Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html)<br>Only for classification tasks.</li>
<li>'GP' for Gaussian Process [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html)<br>No hyperparameter tuning.</li>
<li>'OLS' for [Ordinary Least Squares](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)<br>Only for regression tasks. No hyperparameter tuning.</li>
<li>'Ridge' for Ridge Linear [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)<br>Only for regression tasks.</li>
<li>'Lasso' for [Lasso Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)<br>Only for regression tasks.</li>
<li>'EN' for [ElasticNet Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)<br>Only for regression tasks.</li>
<li>'BR' for [Bayesian Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html)<br>Only for regression tasks. Uses ridge regularization.</li>
<li>'LR' for [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)<br>Only for classification tasks.</li> 
<li>'LDA' for [Linear Discriminant Analysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)<br>Only for classification tasks.</li>
<li>'QDA' for [Quadratic Discriminant Analysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html)<br>Only for classification tasks.</li>
<li>'KNN' for K-Nearest Neighbors [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)</li>
<li>'Tree' for a single Decision Tree [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)</li>
<li>'Bag' for Bagging [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html)<br>Uses a decision tree as base estimator.</li>
<li>'ET' for Extra-Trees [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html)</li>
<li>'RF' for Random Forest [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)</li>
<li>'AdaB' for AdaBoost [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html)<br>Uses a decision tree as base estimator.</li>
<li>'GBM' for Gradient Boosting Machine [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)</li> 
<li>'XGB' for XGBoost [classifier](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier)/[regressor](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor)<br>Only available if package is installed.</li>
<li>'LGB' for LightGBM [classifier](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html)/[regressor](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html)<br>Only available if package is installed.</li>
<li>'CatB' for CatBoost [classifier](https://catboost.ai/docs/concepts/python-reference_catboostclassifier.html)/[regressor](https://catboost.ai/docs/concepts/python-reference_catboostregressor.html)<br>Only available if package is installed.</li>
<li>'lSVM' for Linear Support Vector Machine [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html)<br>Uses a one-vs-rest strategy for multiclass classification tasks.</li> 
<li>'kSVM' for Kernel (non-linear) Support Vector Machine [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)<br>Uses a one-vs-one strategy for multiclass classification tasks.</li>
<li>'PA' for Passive Aggressive [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveRegressor.html)</li>
<li>'SGD' for Stochastic Gradient Descent [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html)</li>
<li>'MLP' for Multilayer Perceptron [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)/[regressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor)<br>Can have between one and three hidden layers.</li> 
</ul>
</blockquote>
<strong>metric: string or callable, optional (default=None)</strong>
<blockquote>
Metric on which the pipeline fits the models. Choose from any of
sklearn's predefined [scorers](https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules), use a score (or loss)
function with signature metric(y, y_pred, **kwargs) or use a
scorer object. If None, ATOM will try to use any metric it already has in the
pipeline. If it hasn't got any, a default metric per task is selected:
<ul>
<li>'f1' for binary classification</li>
<li>'f1_weighted' for multiclas classification</li>
<li>'r2' for regression</li>
</ul>
</blockquote>
<strong>greater_is_better: bool, optional (default=True)</strong>
<blockquote>
Wether the metric is a score function or a loss function,
i.e. if True, a higher score is better and if False, lower is
better. Will be ignored if the metric is a string or a scorer.
</blockquote>
<strong> needs_proba: bool, optional (default=False)</strong>
<blockquote>
Whether the metric function requires probability estimates out of a
 classifier. If True, make sure that every model in the pipeline has
 a `predict_proba` method! Will be ignored if the metric is a string
 or a scorer.
</blockquote>
<strong> needs_threshold: bool, optional (default=False)</strong>
<blockquote>
Whether the metric function takes a continuous decision certainty. This only
 works for binary classification using estimators that have either a
 `decision_function` or `predict_proba` method. Will be ignored if the metric
 is a string or a scorer.
</blockquote>
<strong>train_sizes: sequence, optional (default=np.linspace(0.1, 1.0, 10))</strong>
<blockquote>
Relative or absolute numbers of training examples that will be used
 to generate the learning curve. If the dtype is float, it is
 regarded as a fraction of the maximum size of the training set.
 Otherwise it is interpreted as absolute sizes of the training sets.
</blockquote>
<strong>max_iter: int or sequence, optional (default=0)</strong>
<blockquote>
Maximum number of iterations of the BO. If 0, skip the BO and fit
 the model on its default parameters. If sequence, the n-th value
 will apply to the n-th model in the pipeline.
</blockquote>
<strong>max_time: int, float or sequence, optional (default=np.inf)</strong>
<blockquote>
Maximum time allowed for the BO per model (in seconds). If 0, skip
 the BO and fit the model on its default parameters. If sequence,
 the n-th value will apply to the n-th model in the pipeline.
</blockquote>
<strong>init_points: int or sequence, optional (default=5)</strong>
<blockquote>
Initial number of tests of the BO before fitting the surrogate
 function. If 1, the default models' hyperparameters will be used. If sequence,
 the n-th value will apply to the n-th model in the pipeline.
</blockquote>
<strong>cv: int or sequence, optional (default=3)</strong>
<blockquote>
Strategy to fit and score the model selected after every step of the BO.
<ul>
<li>if 1, randomly split into a train and validation set</li>
<li>if >1, perform a k-fold cross validation on the training set</li>
</ul>
</blockquote>
<strong>plot_bo: bool, optional (default=False)</strong>
<blockquote>
Wether to plot the BO's progress as it runs. Creates a canvas with
two plots: the first plot shows the score of every trial and the
second shows the distance between the last consecutive steps. Don't
forget to call `%matplotlib` at the start of the cell if you are
using jupyter notebook!
</blockquote>
<strong>bagging: int or None, optional (default=None)</strong>
<blockquote>
Number of data sets (bootstrapped from the training set) to use in the bagging
 algorithm. If None or 0, no bagging is performed.
</blockquote>
</tr>
</table>
</div>
<br />



## Model subclass

After running the pipeline method, a class for every selected model is created and
 attached to the main ATOM class as an attribute. These classes can be called upon
 using the models' acronyms, e.g. `atom.LGB`. Lowercase calls are also allowed
 for this attribute, e.g. `atom.lgb`. The model subclasses contain a variety of
 methods and attributes to help you understand how every specific model performed.
 
The majority of the [plots](#plots) can be called directly from the
 subclasses. For example, to plot the ROC for the LightGBM model we could type
 `atom.lgb.plot_ROC()`. You can also save the whole subclass to a pickle file
 using the `save` method, e.g. `atom.rf.save('random_forest')`, or only save 
 the best fitted model with the `save_model` method,
 e.g.`atom.rf.save_model('random_forest_model')`.

You can also call for any of the sklearn pre-defined metrics,
 (e.g. `atom.ET.recall`) or for any of the
 following custom metrics: `tn` (true negatives), `fp` (false positives), `fn` (false
 negatives), `tp` (true positives), `tpr` (true positive rate), `fpr` (false
 positive rate), `sup` (support/predicted positive rate) or `lift`. The rest
 of the available attributes can be found hereunder:


<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>error: string</strong>
<blockquote>
Any exception encountered by the model.
</blockquote>

<strong>best_params: dict</strong>
<blockquote>
Dictionary of the best combination of hyperparameters found by the BO.
</blockquote>

<strong>best_model: callable</strong>
<blockquote>
Model object that performed best on the test set. Not fitted.
</blockquote>

<strong>best_model_fit: callable</strong>
<blockquote>
Model object that performed best on the test set. Fitted to the complete training set.
</blockquote>

<strong>predict_train: list</strong>
<blockquote>
Predictions of the model on the training set.
</blockquote>

<strong> predict_test: list</strong>
<blockquote>
Predictions of the model on the test set.
</blockquote>

<strong>predict_proba_train: list</strong>
<blockquote>
Predict probabilities of the model on the training set. Only for models with
a `predict_proba` method.
</blockquote>

<strong>predict_proba_test: list</strong>
<blockquote>
Predict probabilities of the model on the test set. Only for models with
a `predict_proba` method.
</blockquote>

<strong>decision_function_train: list</strong>
<blockquote>
Decision function scores on the training set. Only for models with
a `decision_function` method.
</blockquote>

<strong>decision_function_test: list</strong>
<blockquote>
Decision function scores on the test set. Only for models with
a `decision_function` method.
</blockquote>

<strong>score_bo: float</strong>
<blockquote>
Best score of the model on the Bayesian Optimization algorithm.
</blockquote>

<strong>score_train: float</strong>
<blockquote>
Metric score of the model on the training set.
</blockquote>

<strong>score_test: float</strong>
<blockquote>
Metric score of the model on the test set.
</blockquote>

<strong>bagging_scores: np.ndarray</strong>
<blockquote>
Array of the bagging's results.
</blockquote>

<strong>permutations: dict</strong>
<blockquote>
Dictionary of the permutation's results (if `plot_permutation_importance` was used).
</blockquote>

<strong>BO: dict</strong>
<blockquote>
Dictionary containing the information of every step taken by the BO. Keys include:
<ul>
<li>'params': Parameters used for the model</li>
<li>'score': Score of the chosen metric</li>
<li>'time': Time spent on this iteration</li>
<li>'total_time': Time spent since the start of the BO</li>
</ul>
</blockquote>
</tr>
</table>
<br>




## Plots

The ATOM class provides a variety of plot methods to analyze the results of the
 pipeline. To use the plots to compare the results of multiple models, you can
 call them directly from the main class using the `models` parameter, e.g.
 `atom.plot_PRC(models=['LDA', 'LGB'])`.
 
 To call the plot for a single model, you can either fill the model in the `models`
 parameter (e.g. `atom.plot_PRC(models='LDA')`) or call the from the model subclass
 (e.g. `atom.LDA.plot_PRC()`). These two examples will render the same plot.
 Note that the latter approach is not available for the [`plot_correlation`](#atom-plot-correlation),
 [`plot_PCA`](#atom-plot-pca), [`plot_components`](#atom-plot-components) and 
 [`plot_RFECV`](#atom-plot-RFECV) methods since they are unrelated to the models
 fitted in the pipeline.

!!! note
    Remember that if successive halving=True, only the last fitted model is saved
    in the model subclass. Avoid plotting models from different iterations
    together since this can lead to misleading insights.

The plots aesthetics can be customized using various [classmethods](#atom-plot-customization).

<table width="100%">
<tr>
<td><a href="#atom-plot-correlation">plot_correlation</a></td>
<td>Correlation matrix plot of the data.</td>
</tr>

<tr>
<td><a href="#atom-plot-PCA">plot_PCA</a></td>
<td>Plot the explained variance ratio vs the number of components.</td>
</tr>

<tr>
<td><a href="#atom-plot-components">plot_components</a></td>
<td>Plot the explained variance ratio per component.</td>
</tr>

<tr>
<td><a href="#atom-plot-RFECV">plot_RFECV</a></td>
<td>Plot the scores obtained by the estimator on the RFECV.</td>
</tr>

<tr>
<td><a href="#atom-plot-bagging">plot_bagging</a></td>
<td>Plot a boxplot of the bagging's results.</td>
</tr>

<tr>
<td><a href="#atom-plot-successive-halving">plot_successive_halving</a></td>
<td>Plot the models' scores per iteration of the successive halving.</td>
</tr>

<tr>
<td><a href="#atom-plot-learning-curve">plot_learning_curve</a></td>
<td>Plot the model's learning curve: score vs training samples.</td>
</tr>

<tr>
<td><a href="#atom-plot-ROC">plot_ROC</a></td>
<td>Plot the Receiver Operating Characteristics curve.</td>
</tr>

<tr>
<td><a href="#atom-plot-PRC">plot_PRC</a></td>
<td>Plot the precision-recall curve.</td>
</tr>

<tr>
<td><a href="#atom-plot-permutation-importance">plot_permutation_importance</a></td>
<td>Plot the feature permutation importance of models.</td>
</tr>

<tr>
<td><a href="#atom-plot-feature-importance">plot_feature_importance</a></td>
<td>Plot the feature permutation importance of models.</td>
</tr>

<tr>
<td><a href="#atom-plot-permutation-importance">plot_permutation_importance</a></td>
<td>Plot tree-based model's normalized feature importances.</td>
</tr>

<tr>
<td><a href="#atom-plot-confusion-matrix">plot_confusion_matrix</a></td>
<td>Plot a model's confusion matrix.</td>
</tr>

<tr>
<td><a href="#atom-plot-threshold">plot_threshold</a></td>
<td>Plot performance metric(s) against threshold values.</td>
</tr>

<tr>
<td><a href="#atom-plot-probabilities">plot_probabilities</a></td>
<td>Plot the probabilities of the different classes of belonging to the target class.</td>
</tr>

<tr>
<td><a href="#atom-plot-calibration">plot_calibration</a></td>
<td>Plot the calibration curve for a binary classifier.</td>
</tr>

<tr>
<td><a href="#atom-plot-gains">plot_gains</a></td>
<td>Plot the cumulative gains curve.</td>
</tr>

<tr>
<td><a href="#atom-plot-lift">plot_lift</a></td>
<td>Plot the lift curve.</td>
</tr>

</table>
<br>

<a name="atom-plot-correlation"></a>
<pre><em>function</em> atom.ATOM.<strong style="color:#008AB8">plot_correlation</strong>(title=None,
                                    figsize=(10, 10),
                                    filename=None,
                                    display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L39">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Correlation matrix plot of the dataset. Ignores non-numeric columns. Can't be called
 from the model subclasses.
<br /><br />
![plot_correlation](img/plots/plot_correlation.png)
<br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>title: string or None, optional (default=None)</strong>
<blockquote>
Plot's title. If None, the default option is used.
</blockquote>
<strong>figsize: tuple, optional (default=(10, 10))</strong>
<blockquote>
Figure's size, format as (x, y).
</blockquote>
<strong>filename: string or None, optional (default=None)</strong>
<blockquote>
Name of the file (to save). If None, the figure is not saved.
</blockquote>
<strong>display: bool, optional (default=True)</strong>
<blockquote>
Wether to render the plot.
</blockquote>
</tr>
</table>
</div>
<br />


<a name="atom-plot-PCA"></a>
<pre><em>function</em> atom.ATOM.<strong style="color:#008AB8">plot_PCA</strong>(title=None,
                            figsize=(10, 6),
                            filename=None,
                            display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L87">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Plot the explained variance ratio vs the number of componenets. Only if a Principal Component Analysis
 was applied on the dataset through the [`feature_selection`](#atom-feature-selection) method.
 Can't be called from the model subclasses.
<br /><br />
![plot_PCA](img/plots/plot_PCA.png)
<br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>title: string or None, optional (default=None)</strong>
<blockquote>
Plot's title. If None, the default option is used.
</blockquote>
<strong>figsize: tuple, optional (default=(10, 6))</strong>
<blockquote>
Figure's size, format as (x, y).
</blockquote>
<strong>filename: string or None, optional (default=None)</strong>
<blockquote>
Name of the file (to save). If None, the figure is not saved.
</blockquote>
<strong>display: bool, optional (default=True)</strong>
<blockquote>
Wether to render the plot.
</blockquote>
</tr>
</table>
</div>
<br />


<a name="atom-plot-components"></a>
<pre><em>function</em> atom.ATOM.<strong style="color:#008AB8">plot_components</strong>(show=None,
                                   title=None,
                                   figsize=None,
                                   filename=None,
                                   display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L87">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Plot the explained variance ratio per components. Only if a Principal Component Analysis
 was applied on the dataset through the [`feature_selection`](#atom-feature-selection) method.
 Can't be called from the model subclasses.
<br /><br />
![plot_components](img/plots/plot_components.png)
<br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>show: int or None, optional (default=None)</strong>
<blockquote>
Number of components to show. If None, the selected number of componenets are plotted.
</blockquote>
<strong>title: string or None, optional (default=None)</strong>
<blockquote>
Plot's title. If None, the default option is used.
</blockquote>
<strong>figsize: tuple, optional (default=None)</strong>
<blockquote>
Figure's size, format as (x, y). If None, adapts size to `show` parameter.
</blockquote>
<strong>filename: string or None, optional (default=None)</strong>
<blockquote>
Name of the file (to save). If None, the figure is not saved.
</blockquote>
<strong>display: bool, optional (default=True)</strong>
<blockquote>
Wether to render the plot.
</blockquote>
</tr>
</table>
</div>
<br />


<a name="atom-plot-RFECV"></a>
<pre><em>function</em> atom.ATOM.<strong style="color:#008AB8">plot_RFECV</strong>(title=None,
                              figsize=(10, 6),
                              filename=None,
                              display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L148">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Plot the scores obtained by the estimator fitted on every subset of
 the data. Only if RFECV was applied on the dataset through the
 [`feature_selection`](#atom-feature-selection) method. Can't be called from
 the model subclasses.
<br /><br />
![plot_RFECV](img/plots/plot_RFECV.png)
<br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>title: string or None, optional (default=None)</strong>
<blockquote>
Plot's title. If None, the default option is used.
</blockquote>
<strong>figsize: tuple, optional (default=(10, 6))</strong>
<blockquote>
Figure's size, format as (x, y).
</blockquote>
<strong>filename: string or None, optional (default=None)</strong>
<blockquote>
Name of the file (to save). If None, the figure is not saved.
</blockquote>
<strong>display: bool, optional (default=True)</strong>
<blockquote>
Wether to render the plot.
</blockquote>
</tr>
</table>
</div>
<br />


<a name="atom-plot-bagging"></a>
<pre><em>function</em> atom.ATOM.<strong style="color:#008AB8">plot_bagging</strong>(models=None,
                                title=None,
                                figsize=None,
                                filename=None,
                                display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L208">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Plot a boxplot of the bagging's results. Only available if the models were
 fitted using bagging>0.
<br /><br />
![plot_bagging](img/plots/plot_bagging.png)
<br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>models: string, sequence or None, optional (default=None)</strong>
<blockquote>
Name of the models to plot. If None, all the models in the pipeline are selected.
</blockquote>
<strong>title: string or None, optional (default=None)</strong>
<blockquote>
Plot's title. If None, the default option is used.
</blockquote>
<strong>figsize: tuple, optional (default=None)</strong>
<blockquote>
Figure's size, format as (x, y). If None, adapts size the to number of models.
</blockquote>
<strong>filename: string or None, optional (default=None)</strong>
<blockquote>
Name of the file (to save). If None, the figure is not saved.
</blockquote>
<strong>display: bool, optional (default=True)</strong>
<blockquote>
Wether to render the plot.
</blockquote>
</tr>
</table>
</div>
<br />


<a name="atom-plot-successive-halving"></a>
<pre><em>function</em> atom.ATOM.<strong style="color:#008AB8">plot_successive_halving</strong>(models=None,
                                           title=None,
                                           figsize=(10, 6),
                                           filename=None,
                                           display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L272">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Plot of the models' scores per iteration of the successive halving. Only
 available if the models were fitted via successive_halving.
<br /><br />
![plot_successive_halving](img/plots/plot_successive_halving.png)
<br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>models: string, sequence or None, optional (default=None)</strong>
<blockquote>
Name of the models to plot. If None, all the models in the pipeline are selected.
</blockquote>
<strong>title: string or None, optional (default=None)</strong>
<blockquote>
Plot's title. If None, the default option is used.
</blockquote>
<strong>figsize: tuple, optional (default=(10, 6))</strong>
<blockquote>
Figure's size, format as (x, y).
</blockquote>
<strong>filename: string or None, optional (default=None)</strong>
<blockquote>
Name of the file (to save). If None, the figure is not saved.
</blockquote>
<strong>display: bool, optional (default=True)</strong>
<blockquote>
Wether to render the plot.
</blockquote>
</tr>
</table>
</div>
<br />


<a name="atom-plot-learning-curve"></a>
<pre><em>function</em> atom.ATOM.<strong style="color:#008AB8">plot_learning_curve</strong>(models=None,
                                       train_sizes=np.linspace(0.1, 1.0, 10),
                                       cv=None, 
                                       title=None,
                                       figsize=(10, 6),
                                       filename=None,
                                       display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L355">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Plot the model's learning curve: score vs number of training samples.
 Only available if the models were fitted via train_sizing.
<br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>models: string, sequence or None, optional (default=None)</strong>
<blockquote>
Name of the models to plot. If None, all the models in the pipeline are selected.
</blockquote>
<strong>title: string or None, optional (default=None)</strong>
<blockquote>
Plot's title. If None, the default option is used.
</blockquote>
<strong>figsize: tuple, optional (default=(10, 6))</strong>
<blockquote>
Figure's size, format as (x, y).
</blockquote>
<strong>filename: string or None, optional (default=None)</strong>
<blockquote>
Name of the file (to save). If None, the figure is not saved.
</blockquote>
<strong>display: bool, optional (default=True)</strong>
<blockquote>
Wether to render the plot.
</blockquote>
</tr>
</table>
</div>
<br />


<a name="atom-plot-ROC"></a>
<pre><em>function</em> atom.ATOM.<strong style="color:#008AB8">plot_ROC</strong>(models=None,
                            title=None,
                            figsize=(10, 6)),
                            filename=None,
                            display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L476">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Plot the Receiver Operating Characteristics curve. The legend shows the Area Under the ROC Curve (AUC) score.
 Only for binary classification tasks.
<br /><br />
![plot_ROC](img/plots/plot_ROC.png)
<br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>models: string, sequence or None, optional (default=None)</strong>
<blockquote>
Name of the models to plot. If None, all the models in the pipeline are selected.
</blockquote>
<strong>title: string or None, optional (default=None)</strong>
<blockquote>
Plot's title. If None, the default option is used.
</blockquote>
<strong>figsize: tuple, optional (default=(10, 6))</strong>
<blockquote>
Figure's size, format as (x, y).
</blockquote>
<strong>filename: string or None, optional (default=None)</strong>
<blockquote>
Name of the file (to save). If None, the figure is not saved.
</blockquote>
<strong>display: bool, optional (default=True)</strong>
<blockquote>
Wether to render the plot.
</blockquote>
</tr>
</table>
</div>
<br />


<a name="atom-plot-PRC"></a>
<pre><em>function</em> atom.ATOM.<strong style="color:#008AB8">plot_PRC</strong>(models=None,
                            title=None,
                            figsize=(10, 6),
                            filename=None,
                            display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L545">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Plot the precision-recall curve. The legend shows the average precision (AP) score.
 Only for binary classification tasks.
<br /><br />
![plot_PRC](img/plots/plot_PRC.png)
<br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>models: string, sequence or None, optional (default=None)</strong>
<blockquote>
Name of the models to plot. If None, all the models in the pipeline are selected.
</blockquote>
<strong>title: string or None, optional (default=None)</strong>
<blockquote>
Plot's title. If None, the default option is used.
</blockquote>
<strong>figsize: tuple, optional (default=(10, 6))</strong>
<blockquote>
Figure's size, format as (x, y).
</blockquote>
<strong>filename: string or None, optional (default=None)</strong>
<blockquote>
Name of the file (to save). If None, the figure is not saved.
</blockquote>
<strong>display: bool, optional (default=True)</strong>
<blockquote>
Wether to render the plot.
</blockquote>
</tr>
</table>
</div>
<br />


<a name="atom-plot-permutation-importance"></a>
<pre><em>function</em> atom.ATOM.<strong style="color:#008AB8">plot_permutation_importance</strong>(models=None,
                                               show=None,
                                               n_repeats=10,
                                               title=None,
                                               figsize=None,
                                               filename=None,
                                               display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L612">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Plot the feature permutation importance of models.
<br /><br />
![plot_permutation_importance](img/plots/plot_permutation_importance.png)
<br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>models: string, sequence or None, optional (default=None)</strong>
<blockquote>
Name of the models to plot. If None, all the models in the pipeline are selected.
</blockquote>
<strong>show: int, optional (default=None)</strong>
<blockquote>
Number of best features to show in the plot. None for all.
</blockquote>
<strong>n_repeats: int, optional (default=10)</strong>
<blockquote>
Number of times to permute each feature.
</blockquote>
<strong>title: string or None, optional (default=None)</strong>
<blockquote>
Plot's title. If None, the default option is used.
</blockquote>
<strong>figsize: tuple, optional (default=None)</strong>
<blockquote>
Figure's size, format as (x, y). If None, adapts size to `show` parameter.
</blockquote>
<strong>filename: string or None, optional (default=None)</strong>
<blockquote>
Name of the file (to save). If None, the figure is not saved.
</blockquote>
<strong>display: bool, optional (default=True)</strong>
<blockquote>
Wether to render the plot.
</blockquote>
</tr>
</table>
</div>
<br />


<a name="atom-plot-feature-importance"></a>
<pre><em>function</em> atom.ATOM.<strong style="color:#008AB8">plot_feature_importance</strong>(models=None,
                                           show=None,
                                           title=None,
                                           figsize=None,
                                           filename=None,
                                           display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L723">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Plot a tree-based model's normalized feature importance.
<br /><br />
![plot_feature_importance](img/plots/plot_feature_importance.png)
<br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>models: string, sequence or None, optional (default=None)</strong>
<blockquote>
Name of the models to plot. If None, all the models in the pipeline are selected.
</blockquote>
<strong>show: int, optional (default=None)</strong>
<blockquote>
Number of best features to show in the plot. None for all.
</blockquote>
<strong>title: string or None, optional (default=None)</strong>
<blockquote>
Plot's title. If None, the default option is used.
</blockquote>
<strong>figsize: tuple, optional (default=None)</strong>
<blockquote>
Figure's size, format as (x, y). If None, adapts size to `show` parameter.
</blockquote>
<strong>filename: string or None, optional (default=None)</strong>
<blockquote>
Name of the file (to save). If None, the figure is not saved.
</blockquote>
<strong>display: bool, optional (default=True)</strong>
<blockquote>
Wether to render the plot.
</blockquote>
</tr>
</table>
</div>
<br />


<a name="atom-plot-confusion-matrix"></a>
<pre><em>function</em> atom.ATOM.<strong style="color:#008AB8">plot_confusion_matrix</strong>(models=None,
                                         normalize=False,
                                         title=None,
                                         figsize=(8, 8),
                                         filename=None,
                                         display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L816">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
<ul>
<li>For 1 model: plot it's confusion matrix in a heatmap.</li>
<li>For >1 models: compare TP, FP, FN and TN in a barplot. Not supported for multiclass classification.</li>
</ul>
![plot_confusion_matrix](img/plots/plot_confusion_matrix.png)
<br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>models: string, sequence or None, optional (default=None)</strong>
<blockquote>
Name of the models to plot. If None, all the models in the pipeline are selected.
</blockquote>
<strong>normalize: bool, optional (default=False)</strong>
<blockquote>
Wether to normalize the matrix.
</blockquote>
<strong>title: string or None, optional (default=None)</strong>
<blockquote>
Plot's title. If None, the default option is used.
</blockquote>
<strong>figsize: tuple, optional (default=(8, 8))</strong>
<blockquote>
Figure's size, format as (x, y).
</blockquote>
<strong>filename: string or None, optional (default=None)</strong>
<blockquote>
Name of the file (to save). If None, the figure is not saved.
</blockquote>
<strong>display: bool, optional (default=True)</strong>
<blockquote>
Wether to render the plot.
</blockquote>
</tr>
</table>
</div>
<br />


<a name="atom-plot-threshold"></a>
<pre><em>function</em> atom.ATOM.<strong style="color:#008AB8">plot_threshold</strong>(models=None,
                                  metric=None,
                                  steps=100,
                                  title=None,
                                  figsize=(10, 6),
                                  filename=None,
                                  display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L943">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Plot performance metric(s) against multiple threshold values.
<br /><br />
![plot_threshold](img/plots/plot_threshold.png)
<br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>models: string, sequence or None, optional (default=None)</strong>
<blockquote>
Name of the models to plot. If None, all the models in the pipeline are selected.
</blockquote>
<strong>metric: string, callable, sequence or None, optional (default=None)</strong>
<blockquote>
Metric(s) to plot. These can be one of the pre-defined sklearn scorers as string,
 a metric function or a sklearn scorer object. If None, the metric used to fit
 the pipeline is used.
</blockquote>
<strong>steps: int, optional (default=100)</strong>
<blockquote>
Number of thresholds measured.
</blockquote>
<strong>title: string or None, optional (default=None)</strong>
<blockquote>
Plot's title. If None, the default option is used.
</blockquote>
<strong>figsize: tuple, optional (default=(10, 6))</strong>
<blockquote>
Figure's size, format as (x, y).
</blockquote>
<strong>filename: string or None, optional (default=None)</strong>
<blockquote>
Name of the file (to save). If None, the figure is not saved.
</blockquote>
<strong>display: bool, optional (default=True)</strong>
<blockquote>
Wether to render the plot.
</blockquote>
</tr>
</table>
</div>
<br />


<a name="atom-plot-probabilities"></a>
<pre><em>function</em> atom.ATOM.<strong style="color:#008AB8">plot_probabilities</strong>(models=None,
                                      target=1,
                                      title=None,
                                      figsize=(10, 6),
                                      filename=None,
                                      display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L1042">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Plot a function of the probability of the classes of being the target class.
<br /><br />
![plot_probabilities](img/plots/plot_probabilities.png)
<br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>models: string, sequence or None, optional (default=None)</strong>
<blockquote>
Name of the models to plot. If None, all the models in the pipeline are selected.
</blockquote>
<strong>target: int or string, optional (default=1)</strong>
<blockquote>
Probability of being that class (as index or name).
</blockquote>
<strong>title: string or None, optional (default=None)</strong>
<blockquote>
Plot's title. If None, the default option is used.
</blockquote>
<strong>figsize: tuple, optional (default=(10, 6))</strong>
<blockquote>
Figure's size, format as (x, y).
</blockquote>
<strong>filename: string or None, optional (default=None)</strong>
<blockquote>
Name of the file (to save). If None, the figure is not saved.
</blockquote>
<strong>display: bool, optional (default=True)</strong>
<blockquote>
Wether to render the plot.
</blockquote>
</tr>
</table>
</div>
<br />


<a name="atom-plot-calibration"></a>
<pre><em>function</em> atom.ATOM.<strong style="color:#008AB8">plot_calibration</strong>(models=None,
                                    n_bins=10,
                                    title=None,
                                    figsize=(10, 10),
                                    filename=None,
                                    display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L1042">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Plot the calibration curve for a binary classifier.

 Well calibrated classifiers are probabilistic classifiers for which the
 output of the predict_proba method can be directly interpreted as a
 confidence level. For instance a well calibrated (binary) classifier
 should classify the samples such that among the samples to which it gave
 a predict_proba value close to 0.8, approx. 80% actually belong to the
 positive class. This figure shows two plots: the calibration curve and a
 distribution of all predicted probabilities of the classifier.
 Code snippets from [https://scikit-learn.org/stable/auto_examples/
calibration/plot_calibration_curve.html](https://scikit-learn.org/stable/auto_examples/
calibration/plot_calibration_curve.html).
<br /><br />
![plot_calibration](img/plots/plot_calibration.png)
<br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>models: string, sequence or None, optional (default=None)</strong>
<blockquote>
Name of the models to plot. If None, all the models in the pipeline are selected.
</blockquote>
<strong>n_bins: int, optional (default=10)</strong>
<blockquote>
Number of bins for the calibration calculation and the histogram.
 Minimum of 5 required.
</blockquote>
<strong>title: string or None, optional (default=None)</strong>
<blockquote>
Plot's title. If None, the default option is used.
</blockquote>
<strong>figsize: tuple, optional (default=(10, 10))</strong>
<blockquote>
Figure's size, format as (x, y).
</blockquote>
<strong>filename: string or None, optional (default=None)</strong>
<blockquote>
Name of the file (to save). If None, the figure is not saved.
</blockquote>
<strong>display: bool, optional (default=True)</strong>
<blockquote>
Wether to render the plot.
</blockquote>
</tr>
</table>
</div>
<br />


<a name="atom-plot-gains"></a>
<pre><em>function</em> atom.ATOM.<strong style="color:#008AB8">plot_gains</strong>(models=None,
                              title=None,
                              figsize=(10, 6),
                              filename=None,
                              display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L1042">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Plot the cumulative gains curve. Only for binary classification.
<br /><br />
![plot_gains](img/plots/plot_gains.png)
<br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>models: string, sequence or None, optional (default=None)</strong>
<blockquote>
Name of the models to plot. If None, all the models in the pipeline are selected.
</blockquote>
<strong>title: string or None, optional (default=None)</strong>
<blockquote>
Plot's title. If None, the default option is used.
</blockquote>
<strong>figsize: tuple, optional (default=(10, 6))</strong>
<blockquote>
Figure's size, format as (x, y).
</blockquote>
<strong>filename: string or None, optional (default=None)</strong>
<blockquote>
Name of the file (to save). If None, the figure is not saved.
</blockquote>
<strong>display: bool, optional (default=True)</strong>
<blockquote>
Wether to render the plot.
</blockquote>
</tr>
</table>
</div>
<br />


<a name="atom-plot-lift"></a>
<pre><em>function</em> atom.ATOM.<strong style="color:#008AB8">plot_lift</strong>(models=None,
                             title=None,
                             figsize=(10, 6),
                             filename=None,
                             display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L1042">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Plot the lift curve. Only for binary classification.
<br /><br />
![plot_lift](img/plots/plot_lift.png)
<br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>models: string, sequence or None, optional (default=None)</strong>
<blockquote>
Name of the models to plot. If None, all the models in the pipeline are selected.
</blockquote>
<strong>title: string or None, optional (default=None)</strong>
<blockquote>
Plot's title. If None, the default option is used.
</blockquote>
<strong>figsize: tuple, optional (default=(10, 6))</strong>
<blockquote>
Figure's size, format as (x, y).
</blockquote>
<strong>filename: string or None, optional (default=None)</strong>
<blockquote>
Name of the file (to save). If None, the figure is not saved.
</blockquote>
<strong>display: bool, optional (default=True)</strong>
<blockquote>
Wether to render the plot.
</blockquote>
</tr>
</table>
</div>
<br />



<a name="atom-plot-customization"></a>
### Plot customization

The plotting aesthetics can be customized with the use of the `@classmethods` described hereunder, e.g. `ATOMClassifier.set_style('white')`.


<table width="100%">
<tr>
<td width="15%"><a href="#atom-set-style">set_style</a></td>
<td>Change the seaborn plotting style.</td>
</tr>

<tr>
<td width="15%"><a href="#atom-set-palette">set_palette</a></td>
<td>Change the seaborn color palette.</td>
</tr>

<tr>
<td width="15%"><a href="#atom-set-title-fontsize">set_title_fontsize</a></td>
<td>Change the fontsize of the plot's title.</td>
</tr>

<tr>
<td width="15%"><a href="#atom-set-label-fontsize">set_label_fontsize</a></td>
<td>Change the fontsize of the plot's labels and legends.</td>
</tr>

<tr>
<td width="15%"><a href="#atom-set-tick-fontsize">set_tick_fontsize</a></td>
<td>Change the fontsize of the plot's ticks.</td>
</tr>
</table>
<br>

<a name="atom-set-style"></a>
<pre><em>classmethod</em> ATOM.<strong style="color:#008AB8">set_style</strong>(style='darkgrid')
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L2343">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Change the plotting style. See the seaborn [documentation](https://seaborn.pydata.org/tutorial/aesthetics.html#seaborn-figure-styles).
<br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>style: string, optional (default='darkgrid')</strong>
<blockquote>
Style to change to. Available options are: 'darkgrid', 'whitegrid', 'dark', 'white', and 'ticks'.
</blockquote>
</tr>
</table>
</div>
<br />

<a name="atom-set-palette"></a>
<pre><em>classmethod</em> ATOM.<strong style="color:#008AB8">set_palette</strong>(palette='GnBu_d')
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L2360">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Change the plotting palette. See the seaborn [documentation](https://seaborn.pydata.org/tutorial/color_palettes.html)
 for the available options.
<br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>palette: string, optional (default='GnBu_d')</strong>
<blockquote>
Palette to change to.
</blockquote>
</tr>
</table>
</div>
<br />

<a name="atom-set-title-fontsize"></a>
<pre><em>classmethod</em> ATOM.<strong style="color:#008AB8">set_title_fontsize</strong>(fontsize=20)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L2377">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Change the fontsize of the plot's title.
<br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>fontsize: int, optional (default=20)</strong>
<blockquote>
Fontsize to change to.
</blockquote>
</tr>
</table>
</div>
<br />

<a name="atom-set-label-fontsize"></a>
<pre><em>classmethod</em> ATOM.<strong style="color:#008AB8">set_label_fontsize</strong>(fontsize=16)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L2392">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Change the fontsize of the plot's labels and legends.
<br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>fontsize: int, optional (default=16)</strong>
<blockquote>
Fontsize to change to.
</blockquote>
</tr>
</table>
</div>
<br />

<a name="atom-set-tick-fontsize"></a>
<pre><em>classmethod</em> ATOM.<strong style="color:#008AB8">set_tick_fontsize</strong>(fontsize=12)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L2407">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Change the fontsize of the plot's ticks.
<br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>fontsize: int, optional (default=12)</strong>
<blockquote>
Fontsize to change to.
</blockquote>
</tr>
</table>
</div>
<br />
