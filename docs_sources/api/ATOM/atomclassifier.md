# ATOMClassifier
-------------------

<pre><em>class </em><strong style="color:#008AB8">ATOMClassifier</strong>(X, y=-1, n_rows=1, test_size=0.3, logger=None,
                     n_jobs=1, warnings=True, verbose=0, random_state=None)</>
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/api.py#L23">[source]</a></div></pre>

ATOMClassifier is the ATOM wrapper for classification tasks. These can include binary
 and multiclass. Use this class to easily apply all data transformations and model
 management of the ATOM package on a given dataset. Note that contrary to
 scikit-learn's API, the ATOMClassifier object already contains the dataset on which
 we want to perform the analysis. Calling a method will automatically apply it on the
 dataset it contains. The class initializer always calls [StandardCleaner](../data_cleaning/standard_cleaner.md).


<table>
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>X: dict, sequence, np.array or pd.DataFrame</strong>
<blockquote>
Dataset containing the features, with shape=(n_samples, n_features).
</blockquote>

<strong>y: int, str, sequence, np.array or pd.Series, optional (default=-1)</strong>
<blockquote>
<ul>
<li>If int: Index of the target column in X. The default value selects the last column.</li>
<li>If string: Name of the target column in X</li>
<li>Else: Data target column with shape=(n_samples,)</li>
</ul>
</blockquote>

<strong>n_rows: int or float, optional (default=1)</strong>
<blockquote>
<ul>
<li>if <=1: Fraction of the data to use.</li>
<li>if >1: Number of rows to use.</li>
</ul>
</blockquote>

<strong>test_size: float, optional (default=0.3)</strong>
<blockquote>
Split fraction for the training and test set.
</blockquote>

<strong>n_jobs: int, optional (default=1)</strong>
<blockquote>
Number of cores to use for parallel processing.
<ul>
<li>If >0: Number of cores to use.</li>
<li>If -1: Use all available cores</li>
<li>If <-1: Use available_cores - 1 + n_jobs</li>
</ul>
Beware that using multiple processes on the same machine may cause
memory issues for large datasets.
</blockquote>

<strong>verbose: int, optional (default=0)</strong>
<blockquote>
Verbosity level of the class. Possible values are:
<ul>
<li>0 to not print anything.</li>
<li>1 to print basic information.</li>
<li>2 to print detailed information.</li>
</ul>
</blockquote>

<strong>warnings: bool or str, optional (default=True)</strong>
<blockquote>
<ul>
<li>If True: Default warning action (equal to 'default' when string).</li>
<li>If False: Suppress all warnings (equal to 'ignore' when string).</li>
<li>If str: One of the possible actions in python's warnings environment.</li>
</ul>
Note that changing this parameter will affect the PYTHONWARNINGS environment.
</blockquote>

<strong>logger: str, class or None, optional (default=None)</strong>
<blockquote>
<ul>
<li>If None: Doesn't save a logging file.</li>
<li>If str: Name of the logging file. 'auto' to create an automatic name.</li>
<li>If class: python Logger object.</li>
</ul>
</blockquote>

<strong>random_state: int or None, optional (default=None)</strong>
<blockquote>
Seed used by the random number generator. If None, the random number
 generator is the RandomState instance used by np.random.
</blockquote>
</td>
</tr>
</table>



# Data properties
-------------------

The dataset within ATOM's pipeline can be accessed at any time through multiple
 properties, e.g. calling `atom.train` will return the training set. The data can also
 be changed through these properties, e.g. `atom.test = atom.test.drop(0)` will
 drop the first row from the test set. This will also update the other data properties.

<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Data properties:</strong></td>
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
</table>



# Attributes
-------------

<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Attributes:</strong></td>
<td width="75%" style="background:white;">

<strong>mapping: dict</strong>
<blockquote>
Dictionary of the target values mapped to their respective encoded integer.
 Only available for classification tasks.
</blockquote>

<strong>profile: ProfileReport</strong>
<blockquote>
Profile created by pandas-profiling after calling the report method.
</blockquote>

<strong>genetic_features: pd.DataFrame</strong>
<blockquote>
Dataframe of the non-linear features created by the <a href="#atomclassifier-feature-generation">feature_generation</a> method.
 Columns include:
<ul>
<li><b>name:</b> Name of the feature (automatically created).</li>
<li><b>description:</b> Operators used to create this feature.</li>
<li><b>fitness:</b> Fitness score.</li>
</ul>
</blockquote>

<strong>errors: dict</strong>
<blockquote>
Dictionary of the encountered exceptions (if any) after calling any of the training methods.
</blockquote>

<strong>winner: callable</strong>
<blockquote>
Model subclass that performed best on the test set.
</blockquote>

<strong>results: pd.DataFrame</strong>
<blockquote>
Dataframe of the training results with model's acronym as index. For <a href="#atomclassifier-successive-halving">successive_halving</a>
 and <a href="#atomclassifier-train-sizing">train_sizing</a>, an extra index level is added to indicate the different runs.
 Columns can include:
<ul>
<li><b>name:</b> Name of the model.</li>
<li><b>score_train:</b> Score on the training set.</li>
<li><b>score_test:</b> Score on the test set.</li>
<li><b>time_fit:</b> Time spent fitting and evaluating.</li>
<li><b>mean_bagging:</b> Mean score of the bagging's results.</li>
<li><b>std_bagging:</b> Standard deviation score of the bagging's results.</li>
<li><b>time_bagging:</b> Time spent on the bagging algorithm.</li>
<li><b>time:</b> Total time spent on the whole run.</li>
</ul>
</blockquote>
</td>
</tr>
</table>



# Utility methods
----------------

The ATOM class contains a variety of methods to help you handle the data and
inspect the pipeline.

<table width="100%">
<tr>
<td width="15%"><a href="#atomclassifier-stats">stats</a></td>
<td>Print out a list of basic statistics on the dataset.</td>
</tr>

<tr>
<td width="15%"><a href="#atomclassifier-log">log</a></td>
<td>Save information to the logger and print to stdout.</td>
</tr>

<tr>
<td><a href="#atomclassifier-report">report</a></td>
<td>Get an extensive profile analysis of the data.</td>
</tr>

<tr>
<td><a href="#atomclassifier-outcome">outcome</a></td>
<td>Print the final outcome of the models for a specific metric.</td>
</tr>


<tr>
<td><a href="#atomclassifier-save">save</a></td>
<td>Save the ATOMClassifier instance to a pickle file.</td>
</tr>
</table>
<br>


<a name="atomclassifier-stats"></a>
<pre><em>function</em> ATOMClassifier.<strong style="color:#008AB8">stats</strong>()
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L414">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Print out a list of basic information on the dataset.
</div>
<br /><br />


<a name="atomclassifier-log"></a>
<pre><em>function</em> ATOMClassifier.<strong style="color:#008AB8">log</strong>(msg, level=0)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L696">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Save information to the ATOM logger and print it to stdout.
<br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>msg: str</strong>
<blockquote>
Message to save to the logger and print to stdout.
</blockquote>
<strong>level: int, optional (default=0)</strong>
<blockquote>
Minimum verbosity level in order to print the message.
</blockquote>
</tr>
</table>
</div>
<br />


<a name="atomclassifier-report"></a>
<pre><em>function</em> ATOMClassifier.<strong style="color:#008AB8">report</strong>(df='dataset', n_rows=None, filename=None)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L570">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Get an extensive profile analysis of the data. The report is rendered
 in HTML5 and CSS3 and saved to the `profile` attribute. Note that this method
 can be slow for n_rows > 10k.
Dependency: [pandas-profiling](https://pandas-profiling.github.io/pandas-profiling/docs/).
<br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>df: str, optional (default='dataset')</strong>
<blockquote>
Name of the data class property to get the profile from.
</blockquote>
<strong>n_rows: int or None, optional (default=None)</strong>
<blockquote>
Number of (randomly picked) rows to process. None for all rows.
</blockquote>
<strong>filename: str or None, optional (default=None)</strong>
<blockquote>
Name of the file when saved (as .html). None to not save anything.
</blockquote>
</tr>
</table>
</div>
<br />


<a name="atomclassifier-outcome"></a>
<pre><em>function</em> ATOMClassifier.<strong style="color:#008AB8">outcome</strong>(metric=None)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L616">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Print the final outcome of the models for a specific metric. If a model
 shows a `XXX`, it means the metric failed for that specific model. This
 can happen if either the metric is unavailable for classification tasks or if the
 model does not have a `predict_proba` method while the metric requires it.
<br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>metric: str or None, optional (default=None)</strong>
<blockquote>
String of one of sklearn's predefined metrics (see [documentation](https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values)).
 If None, the metric used to run the pipeline is selected.
</blockquote>
</tr>
</table>
</div>
<br />


<a name="atomclassifier-save"></a>
<pre><em>function</em> ATOMClassifier.<strong style="color:#008AB8">save</strong>(filename=None)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L696">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Save the instance to a pickle file. Remember that the class contains the complete
 dataset as property. This means the files can become large for big datasets!

<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>filename: str or None, optional (default=None)</strong>
<blockquote>
Name to save the file with. None to save with default name (ATOMClassifier.pkl).
</blockquote>
</tr>
</table>
</div>
<br />



# Data cleaning
--------------

Usually, before throwing your data in a model, you need to apply some data
cleaning steps. ATOM provides four data cleaning methods to handle missing values,
 categorical columns, outliers and unbalanced datasets. Calling on one of them
 will automatically apply the method on the dataset in the pipeline.

!!! tip
    Use the [report](#atomclassifier-report) method to examine the data and help you
    determine suitable parameters for the data cleaning methods.
    

<table width="100%">
<tr>
<td><a href="#atomclassifier-scale">scale</a></td>
<td>Scale all the features to mean=1 and std=0.</td>
</tr>

<tr>
<td><a href="#atomclassifier-impute">impute</a></td>
<td>Handle missing values in the dataset.</td>
</tr>

<tr>
<td><a href="#atomclassifier-encode">encode</a></td>
<td>Encode categorical columns.</td>
</tr>

<tr>
<td><a href="#atomclassifier-outliers">outliers</a></td>
<td>Remove outliers from the training set.</td>
</tr>

<tr>
<td><a href="#atomclassifier-balance">balance</a></td>
<td>Balance the number of rows per category.</td>
</tr>
</table>
<br>


<a name="atomclassifier-scale"></a>
<pre><em>function</em> ATOMClassifier.<strong style="color:#008AB8">scale</strong>()
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L487">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Scale the feature set to mean=1 and std=0. This method calls the [Cleaner](#../data_cleaning/cleaner.md)
 class under the hood and fits it only on the training set to avoid data leakage.
</div>
<br /><br />


<a name="atomclassifier-impute"></a>
<pre><em>function</em> ATOMClassifier.<strong style="color:#008AB8">impute</strong>(strat_num='drop', strat_cat='drop', min_frac_rows=0.5, min_frac_cols=0.5, missing=None) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L711">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Handle missing values according to the selected strategy. Also removes rows and
 columns with too many missing values. The imputer is fitted only on the training set
 to avoid data leakage. See [Imputer](../data_cleaning/imputer.md) for a description
 of the parameters.
</div>
<br />


<a name="atomclassifier-encode"></a>
<pre><em>function</em> ATOMClassifier.<strong style="color:#008AB8">encode</strong>(max_onehot=10, encode_type='LeaveOneOut', frac_to_other=0) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L867">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Perform encoding of categorical features. The encoding type depends on the
 number of unique values in the column:
<ul>
<li>If n_unique=2, use label-encoding.</li>
<li>If 2 < n_unique <= max_onehot, use one-hot-encoding.</li>
<li>If n_unique > max_onehot, use `encode_type`.</li>
</ul>
Also replaces classes with low occurrences with the value 'other' in
order to prevent too high cardinality. Categorical features are defined as
all columns whose dtype.kind not in 'ifu'. The encoder is fitted only on
the training set to avoid data leakage. See [Encoder](../data_cleaning/encoder.md)
for a description of the parameters.
</div>
<br />


<a name="atomclassifier-outliers"></a>
<pre><em>function</em> ATOMClassifier.<strong style="color:#008AB8">outliers</strong>(strategy='drop', max_sigma=3, include_target=False) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L962">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Remove or replace outliers in the training set. Outliers are defined as values that
 lie further than `max_sigma` * standard_deviation away from the mean of the column.
 See [Outliers](../data_cleaning/outliers.md) for a description of the parameters.
</div>
<br />


<a name="atomclassifier-balance"></a>
<pre><em>function</em> ATOMClassifier.<strong style="color:#008AB8">balance</strong>(oversample=None, undersample=None, n_neighbors=5) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L1000">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Balance the number of instances per target category in the training set.
 Using oversample and undersample at the same time or not using any will
 raise an exception. Only the training set is balanced in order to maintain the
 original distribution of target categories in the test set.
 See [Balancer](../data_cleaning/balancer.md) for a description of the parameters.
 Dependency: [imbalanced-learn](https://imbalanced-learn.readthedocs.io/en/stable/).
</div>
<br />



# Feature selection
-------------------

To further pre-process the data you can create new non-linear features using a
 genetic algorithm or, if your dataset is too large, remove features using one
 of the provided strategies.

<table width="100%">
<tr>
<td><a href="#atomclassifier-feature-generation">feature_generation</a></td>
<td>Use a genetic algorithm to create new combinations of existing features.</td>
</tr>

<tr>
<td><a href="#atomclassifier-feature-selection">feature_selection</a></td>
<td>Remove features according to the selected strategy.</td>
</tr>
</table>
<br>



<a name="atomclassifier-feature-generation"></a>
<pre><em>function</em> ATOMClassifier.<strong style="color:#008AB8">feature_generation</strong>(n_features=2, generations=20, population=500) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L1131">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Use a genetic algorithm to create new combinations of existing
features and add them to the original dataset in order to capture
the non-linear relations between the original features. A dataframe
containing the description of the newly generated features and their
scores can be accessed through the `genetic_features` attribute. It is
recommended to only use this method when fitting linear models.
See [FeatureGenerator](../data_cleaning/feature_generator.md) for a description of the parameters.
Dependency: [gplearn](https://gplearn.readthedocs.io/en/stable/).
</div>
<br />


<a name="atomclassifier-feature-selection"></a>
<pre><em>function</em> ATOMClassifier.<strong style="color:#008AB8">feature_selection</strong>(strategy=None, solver=None, n_features=None,
                                     max_frac_repeated=1., max_correlation=1., **kwargs) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L1259">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Remove features according to the selected strategy. Ties between
features with equal scores will be broken in an unspecified way. Also
removes features with too low variance and finds pairs of collinear features
based on the Pearson correlation coefficient. For each pair above the specified
limit (in terms of absolute value), it removes one of the two.

!!! note
    <ul>
    <li>When strategy='univariate' and solver=None, [f_classif](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html)
        will be used as default solver.</li>
    <li>When strategy is one of 'SFM', 'RFE' or 'RFECV' and the solver is one of 
        ATOM's models, the algorithm will automatically select the classifier.</li>
    <li>When strategy is one of 'SFM', 'RFE' or 'RFECV' and solver=None, ATOM will
         use the winning model (if it exists) as solver.</li>
    <li>When strategy='RFECV', ATOM will use the metric in the pipeline (if it exists)
        as the scoring parameter (only if not specified manually).</li>

</div>
<br />



# Training
----------

The training methods are where the models are fitted to the data and their
 performance is evaluated according to the selected metric. ATOMClassifier contains
 three methods to call the training classes from the ATOM package. All relevant
 attributes and methods from the training classes are attached to ATOMClassifier for
 convenience. These include the errors, winner and results attributes, the model
 subclasses, and the predicting and plotting methods described hereunder.


<table width="100%">
<tr>
<td><a href="#atomclassifier-run">run</a></td>
<td>Fit the models to the data in a direct fashion.</td>
</tr>

<tr>
<td><a href="#atomclassifier-successive-halving">successive_halving</a></td>
<td>Fit the models to the data in a successive halving fashion.</td>
</tr>

<tr>
<td><a href="#atomclassifier-train-sizing">train_sizing</a></td>
<td>Fit the models to the data in a train sizing fashion.</td>
</tr>
</table>
<br>


<a name="atomclassifier-run"></a>
<pre><em>function</em> ATOMClassifier.<strong style="color:#008AB8">run</strong>(models, metric=None, greater_is_better=True, needs_proba=False,
                            needs_threshold=False, n_calls=10, n_random_points=5, bo_params={}, bagging=None) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L2155">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Calls a [TrainerClassifier](../training/trainingclassifier.md) instance.
 Using this class through ATOMClassifier allows subsequent runs with different models
 without losing previous information (only the model subclasses are overwritten if the
 same model is rerun).
</div>
<br />


<a name="atomclassifier-successive-halving"></a>
<pre><em>function</em> ATOMClassifier.<strong style="color:#008AB8">successive_halving</strong>(models, metric=None, greater_is_better=True, needs_proba=False,
                                           needs_threshold=False, skip_iter=0, n_calls=0, n_random_starts=5,
                                           bo_params={}, bagging=None) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L2172">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Calls a [SuccessiveHalvingClassifier](../training/successivehalvingclassifier.md) instance.
</div>
<br />


<a name="atomclassifier-train-sizing"></a>
<pre><em>function</em> ATOMClassifier.<strong style="color:#008AB8">train_sizing</strong>(models, metric=None, greater_is_better=True, needs_proba=False,
                                     needs_threshold=False, train_sizes=np.linspace(0.2, 1.0, 5),
                                     n_calls=0, n_random_starts=5, bo_params={}, bagging=None) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L2201">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Calls a [TrainSizingClassifier](../training/trainsizingclassifier.md) instance.
</div>
<br />



# Predicting
------------

Like the majority of estimators in scikit-learn, you can use a fitted instance
 of ATOMClassifier to make predictions onto new data, e.g. `atom.predict_proba(X)`.
 The following methods will apply all selected data pre-processing steps on the
 provided data first, and use the winning model from the pipeline (under attribute
 `winner`) to make the predictions. If you want to use a different model, you can
 call the method from the model subclass, e.g. `atom.LGB.predict(X)`.

<table width="100%">
<tr>
<td><a href="#atomclassifier-transform">transform</a></td>
<td>Transform new data through all the pre-processing data steps.</td>
</tr>

<tr>
<td><a href="#atomclassifier-predict">predict</a></td>
<td>Make predictions on new data.</td>
</tr>

<tr>
<td><a href="#atomclassifier-predict-proba">predict_proba</a></td>
<td>Make probability predictions on new data.</td>
</tr>

<tr>
<td><a href="#atomclassifier-predict-log-proba">predict_log_proba</a></td>
<td>Make logarithmic probability predictions on new data.</td>
</tr>

<tr>
<td><a href="#atomclassifier-decision-function">decision_function</a></td>
<td>Return the decision function of a model on new data.</td>
</tr>

<tr>
<td><a href="#atomclassifier-score">score</a></td>
<td>Return the score of a model on new data.</td>
</tr>

</table>
<br>


<a name="atomclassifier-transform"></a>
<pre><em>function</em> ATOMClassifier.<strong style="color:#008AB8">transform</strong>(X, standard_cleaner=True, scale=True, impute=True, encode=True, outliers=False,
                                  balance=False, feature_generation=True, feature_selection=True, verbose=None) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L711">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Transform new data through all the pre-processing steps. The outliers and balancer
steps are not included in the default steps since they should only be applied
on the training set.
 <br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>X: dict, sequence, np.array or pd.DataFrame</strong>
<blockquote>
Data containing the features, with shape=(n_samples, n_features).
</blockquote>
<strong>standard_cleaner: bool, optional (default=True)</strong>
<blockquote>
Whether to apply the standard cleaning step in the transformer.
</blockquote>
<strong>scale: bool, optional (default=True)</strong>
<blockquote>
Whether to apply the scaler step in the transformer.
</blockquote>
<strong>impute: bool, optional (default=True)</strong>
<blockquote>
Whether to apply the imputer step in the transformer.
</blockquote>
<strong>encode: bool, optional (default=True)</strong>
<blockquote>
Whether to apply the encoder step in the transformer.
</blockquote>
<strong>outliers: bool, optional (default=False)</strong>
<blockquote>
Whether to apply the outliers step in the transformer.
</blockquote>
<strong>balance: bool, optional (default=False)</strong>
<blockquote>
Whether to apply the balance step in the transformer.
</blockquote>
<strong>feature_generation: bool, optional (default=True)</strong>
<blockquote>
Whether to apply the feature_generation step in the transformer.
</blockquote>
<strong>feature_selection: bool, optional (default=True)</strong>
<blockquote>
Whether to apply the feature_selection step in the transformer.
</blockquote>
<strong>verbose: int, optional (default=None)</strong>
<blockquote>
Verbosity level of the output. If None, it uses ATOMClassifier's verbosity.
</blockquote>
</tr>
</table>
</div>
<br />


<a name="atomclassifier-predict"></a>
<pre><em>function</em> ATOMClassifier.<strong style="color:#008AB8">predict</strong>(X, \*\*kwargs) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L711">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Transform the data and make predictions using the winning model in the pipeline.
The model has to have a `predict` method.
 <br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>X: dict, sequence, np.array or pd.DataFrame</strong>
<blockquote>
Data containing the features, with shape=(n_samples, n_features).
</blockquote>
<strong>**kwargs</strong>
<blockquote>
Same arguments as the <a href="#atomclassifier-transform">transform</a> method to
 include/exclude pre-processing steps from the transformer.
</blockquote>
</tr>
</table>
</div>
<br />


<a name="atomclassifier-predict-proba"></a>
<pre><em>function</em> ATOMClassifier.<strong style="color:#008AB8">predict_proba</strong>(X, \*\*kwargs) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L711">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Transform the data and make probability predictions using the winning model in the pipeline.
The model has to have a `predict_proba` method.
<br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>X: dict, sequence, np.array or pd.DataFrame</strong>
<blockquote>
Data containing the features, with shape=(n_samples, n_features).
</blockquote>
<strong>**kwargs</strong>
<blockquote>
Same arguments as the <a href="#atomclassifier-transform">transform</a> method to
 include/exclude pre-processing steps from the transformer.
</blockquote>
</tr>
</table>
</div>
<br />



<a name="atomclassifier-predict-log-proba"></a>
<pre><em>function</em> ATOMClassifier.<strong style="color:#008AB8">predict_log_proba</strong>(X, \*\*kwargs) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L711">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Transform the data and make logarithmic probability predictions using the winning model in the pipeline.
The model has to have a `predict_log_proba` method.
<br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>X: dict, sequence, np.array or pd.DataFrame</strong>
<blockquote>
Data containing the features, with shape=(n_samples, n_features).
</blockquote>
<strong>**kwargs</strong>
<blockquote>
Same arguments as the <a href="#atomclassifier-transform">transform</a> method to
 include/exclude pre-processing steps from the transformer.
</blockquote>
</tr>
</table>
</div>
<br />


<a name="atomclassifier-decision-function"></a>
<pre><em>function</em> ATOMClassifier.<strong style="color:#008AB8">decision_function</strong>(X, \*\*kwargs) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L711">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Transform the data and run the decision function of the winning model in the pipeline.
The model has to have a `decision_function` method.
<br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>X: dict, sequence, np.array or pd.DataFrame</strong>
<blockquote>
Data containing the features, with shape=(n_samples, n_features).
</blockquote>
<strong>**kwargs</strong>
<blockquote>
Same arguments as the <a href="#atomclassifier-transform">transform</a> method to
 include/exclude pre-processing steps from the transformer.
</blockquote>
</tr>
</table>
</div>
<br />



<a name="atomclassifier-score"></a>
<pre><em>function</em> ATOMClassifier.<strong style="color:#008AB8">score</strong>(X, y, \*\*kwargs) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L711">[source]</a></div></pre>
<div style="padding-left:3%" width="100%">
Transform the data and run the scoring method of the winning model in the pipeline.
The model has to have a `score` method.
<br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>X: dict, sequence, np.array or pd.DataFrame</strong>
<blockquote>
Data containing the features, with shape=(n_samples, n_features).
</blockquote>
<strong>y: int, str, dict, sequence, np.array or pd.Series</strong>
<blockquote>
<ul>
<li>If int: Index of the column of X which is selected as target.</li>
<li>If str: Name of the target column in X.</li>
<li>Else: Data target column with shape=(n_samples,).</li>
</ul>
</blockquote>
<strong>**kwargs</strong>
<blockquote>
Same arguments as the <a href="#atomclassifier-transform">transform</a> method to
 include/exclude pre-processing steps from the transformer.
</blockquote>
</tr>
</table>
</div>
<br />



# Plots
------
Just like the prediction methods, the plots from the training class can be called
 from the ATOMClassifier instance directly. Available plots are:

<table width="100%">
<tr>
<td><a href="../plots/plot_correlation.md">plot_correlation</a></td>
<td>Plot the correlation matrix of the dataset.</td>
</tr>

<tr>
<td><a href="../plots/plot_pca.md">plot_pca</a></td>
<td>Plot the explained variance ratio vs the number of components.</td>
</tr>

<tr>
<td><a href="../plots/plot_components.md">plot_components</a></td>
<td>Plot the explained variance ratio per component.</td>
</tr>

<tr>
<td><a href="../plots/plot_rfecv.md">plot_rfecv</a></td>
<td>Plot the scores obtained by the estimator on the RFECV.</td>
</tr>

<tr>
<td><a href="../plots/plot_bagging.md">plot_bagging</a></td>
<td>Plot a boxplot of the bagging's results.</td>
</tr>

<tr>
<td><a href="../plots/plot_successive_halving.md">plot_successive_halving</a></td>
<td>Plot the models' scores per iteration of the successive halving.</td>
</tr>

<tr>
<td><a href="../plots/plot_learning_curve.md">plot_learning_curve</a></td>
<td>Plot the model's learning curve: score vs training samples.</td>
</tr>

<tr>
<td><a href="../plots/plot_roc.md">plot_roc</a></td>
<td>Plot the Receiver Operating Characteristics curve.</td>
</tr>

<tr>
<td><a href="../plots/plot_prc.md">plot_prc</a></td>
<td>Plot the precision-recall curve.</td>
</tr>

<tr>
<td><a href="../plots/plot_permutation_importance.md">plot_permutation_importance</a></td>
<td>Plot the feature permutation importance of models.</td>
</tr>

<tr>
<td><a href="../plots/plot_feature_importance.md">plot_feature_importance</a></td>
<td>Plot the feature permutation importance of models.</td>
</tr>

<tr>
<td><a href="../plots/plot_confusion_matrix.md">plot_confusion_matrix</a></td>
<td>Plot a model's confusion matrix.</td>
</tr>

<tr>
<td><a href="../plots/plot_threshold.md">plot_threshold</a></td>
<td>Plot performance metric(s) against threshold values.</td>
</tr>

<tr>
<td><a href="../plots/plot_probabilities.md">plot_probabilities</a></td>
<td>Plot the probabilities of the different classes of belonging to the target class.</td>
</tr>

<tr>
<td><a href="../plots/plot_calibration.md">plot_calibration</a></td>
<td>Plot the calibration curve for a binary classifier.</td>
</tr>

<tr>
<td><a href="../plots/plot_gains.md">plot_gains</a></td>
<td>Plot the cumulative gains curve.</td>
</tr>

<tr>
<td><a href="../plots/plot_lift.md">plot_lift</a></td>
<td>Plot the lift curve.</td>
</tr>

<tr>
<td><a href="../plots/plot_bo.md">plot_bo</a></td>
<td>Plot the bayesian optimization scoring.</td>
</tr>

</table>
<br>



# Example
---------

    from sklearn.datasets import load_breast_cancer
    from atom import ATOMClassifier

    X, y = load_breast_cancer(return_X_y)

    # Initialize class
    atom = ATOMClassifier(X, y, logger='auto', n_jobs=2, verbose=2)

    # Apply data cleaning methods
    atom.impute(strat_num='knn', strat_cat='most_frequent', min_frac_rows=0.1)  
    atom.encode(max_onehot=10, frac_to_other=0.05)
    atom.balance(oversample=0.7)

    # Fit the models to the data
    atom.run(models=['QDA', 'CatB'],
             metric='precision',
             n_calls=25,
             n_random_starts=10,
             bo_params={'cv': 1},
             bagging=4)

    # Analyze the results
    print(f"The winning model is: {atom.winner.name}")
    print(atom.results)

    # Make some plots
	atom.plot_roc(figsize=(9, 6), filename='roc.png')  
	atom.CatB.plot_feature_importance(filename='catboost_feature_importance.png')
	
	# Run an extra model
	atom.run(models='LR',
             metric='precision',
             n_calls=25,
             n_random_starts=10,
             bo_params={'cv': 1},
             bagging=4)

    # Get the predictions for the best model on new data
    predictions = atom.predict(X_new)