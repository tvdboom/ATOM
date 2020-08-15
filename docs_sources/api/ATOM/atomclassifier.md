# ATOMClassifier
-------------------

<pre><em>class</em> atom.api.<strong style="color:#008AB8">ATOMClassifier</strong>(X, y=-1, n_rows=1, test_size=0.2, logger=None,
                              n_jobs=1, warnings=True, verbose=0, random_state=None)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/api.py#L23">[source]</a></div></pre>

ATOMClassifier is the ATOM wrapper for classification tasks. These can include binary
 and multiclass. Use this class to easily apply all data transformations and model
 management of the ATOM package on a given dataset. Note that contrary to
 scikit-learn's API, the ATOMClassifier object already contains the dataset on which
 we want to perform the analysis. Calling a method will automatically apply it on the
 dataset it contains. The class initializer always calls [StandardCleaner](../data_cleaning/standard_cleaner.md)
 with default parameters.
 The following data types can't (yet) be handled properly by ATOMClassifier and are
 therefore removed: 'datetime64', 'datetime64[ns]', 'timedelta[ns]'.


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
<li>If int: Position of the target column in X. The default value selects the last column.</li>
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

<strong>test_size: float, optional (default=0.2)</strong>
<blockquote>
Split fraction for the training and test set.
</blockquote>

<strong>n_jobs: int, optional (default=1)</strong>
<blockquote>
Number of cores to use for parallel processing.
<ul>
<li>If >0: Number of cores to use.</li>
<li>If -1: Use all available cores.</li>
<li>If <-1: Use available_cores - 1 + n_jobs.</li>
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
Note that changing this parameter will affect the PYTHONWARNINGS environment.<br>
 Note that ATOM can't manage warnings that go directly from C++ code to the
 stdout/stderr.
</blockquote>

<strong>logger: bool, str, class or None, optional (default=None)</strong>
<blockquote>
<ul>
<li>If None: Doesn't save a logging file.</li>
<li>If bool: True for logging file with default name, False for no logger.</li>
<li>If str: Name of the logging file. 'auto' for default name.</li>
<li>If class: python Logger object.</li>
</ul>
Note that warnings will not be saved to the logger in any case.
</blockquote>

<strong>random_state: int or None, optional (default=None)</strong>
<blockquote>
Seed used by the random number generator. If None, the random number
 generator is the RandomState instance used by np.random.
</blockquote>
</td>
</tr>
</table>
<br>


## Properties
-------------------


### Data properties

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
<br>


### Utility properties

<strong>models: list</strong>
<blockquote>
List of models in the pipeline.
</blockquote>

<strong>metric: str or list</strong>
<blockquote>
Metric(s) used to fit the models in the pipeline.
</blockquote>


### Plotting properties





<br>

## Attributes
-------------

<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Attributes:</strong></td>
<td width="75%" style="background:white;">

<strong>mapping: dict</strong>
<blockquote>
Dictionary of the target values mapped to their respective encoded integer.
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

<strong>collinear: pd.DataFrame</strong>
<blockquote>
Dataframe of the collinear features removed by the <a href="#atomclassifier-feature-selection">feature_selection</a> method.
 Columns include:
<ul>
<li><b>drop_feature:</b> name of the feature dropped by the method.</li>
<li><b>correlated feature:</b> Name of the correlated feature(s).</li>
<li><b>correlation_value:</b> Pearson correlation coefficient(s) of the feature pairs.</li>
</ul>
</blockquote>

<strong>errors: dict</strong>
<blockquote>
Dictionary of the encountered exceptions (if any) after calling any of the training methods.
</blockquote>

<strong>winner: model subclass</strong>
<blockquote>
Model subclass that performed best on the test set. If multi-metric,
 only the first metric is checked.
</blockquote>

<strong>results: pd.DataFrame</strong>
<blockquote>
Dataframe of the training results with the model acronyms as index. For <a href="#atomclassifier-successive-halving">successive_halving</a>
 and <a href="#atomclassifier-train-sizing">train_sizing</a>, an extra index level is added to indicate the different runs.
 Columns can include:
<ul>
<li><b>name:</b> Name of the model.</li>
<li><b>score_bo:</b> Best score achieved during the BO.</li>
<li><b>time_bo:</b> Time spent on the BO.</li>
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
<br>


## Utility methods
----------------

The ATOM class contains a variety of methods to help you handle the data and
inspect the pipeline.

<table>
<tr>
<td><a href="#atomclassifier-calibrate">calibrate</a></td>
<td>Calibrate the winning model.</td>
</tr>

<tr>
<td width="15%"><a href="#atomclassifier-clear">clear</a></td>
<td>Remove a model from the pipeline.</td>
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
<td><a href="#atomclassifier-save">save</a></td>
<td>Save the ATOMClassifier instance to a pickle file.</td>
</tr>

<tr>
<td><a href="#atomclassifier-scoring">scoring</a></td>
<td>Print the scoring of the models for a specific metric.</td>
</tr>

<tr>
<td width="15%"><a href="#atomclassifier-stats">stats</a></td>
<td>Print out a list of basic statistics on the dataset.</td>
</tr>

</table>
<br>


<a name="atomclassifier-calibrate"></a>
<pre><em>function</em> ATOMClassifier.<strong style="color:#008AB8">calibrate</strong>(\*\*kwargs)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L616">[source]</a></div></pre>
<div style="padding-left:3%">
Applies probability calibration on the winning model. The calibration is done with the
 [CalibratedClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)
 class from sklearn. The model will be trained via cross-validation on a subset
 of the training data, using the rest to fit the calibrator. The new classifier will
 replace the `model` attribute.
<br /><br />
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>**kwargs</strong>
<blockquote>
Additional keyword arguments for the [CalibratedClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)
 instance. Using cv='prefit' will use the trained model and fit the calibrator on the
 test set. Note that doing this will result in data leakage in the test set. Use this
 only if you have another, independent set for testing.
</blockquote>
</tr>
</table>
</div>
<br />


<a name="atomclassifier-clear"></a>
<pre><em>function</em> ATOMClassifier.<strong style="color:#008AB8">clear</strong>(models='all')
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L570">[source]</a></div></pre>
<div style="padding-left:3%">
Removes all traces of a model in the pipeline (except for the errors attribute).
 This includes the models and results attributes, and the model subclass.
 If all models in the pipeline are removed, the metric is reset. Use this method 
 to remove unwanted models from the pipeline or to clear memory before saving. 
<br /><br />
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>models: str, or sequence, optional (default='all')</strong>
<blockquote>
Name of the models to clear from the pipeline. If 'all', clear all models.
</blockquote>
</tr>
</table>
</div>
<br />


<a name="atomclassifier-log"></a>
<pre><em>function</em> ATOMClassifier.<strong style="color:#008AB8">log</strong>(msg, level=0)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L696">[source]</a></div></pre>
<div style="padding-left:3%">
Save information to the ATOM logger and print it to stdout.
<br /><br />
<table>
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
<div style="padding-left:3%">
Get an extensive profile analysis of the data. The report is rendered
 in HTML5 and CSS3 and saved to the `profile` attribute. Note that this method
 can be slow for n_rows > 10k.
<br /><br />
<table>
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


<a name="atomclassifier-save"></a>
<pre><em>function</em> ATOMClassifier.<strong style="color:#008AB8">save</strong>(filename=None, save_data=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L696">[source]</a></div></pre>
<div style="padding-left:3%">
Save the instance to a pickle file. Remember that the class contains the complete
 dataset as property, so the file can become large for big datasets! To avoid this,
 use `save_data=False`.

<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>filename: str or None, optional (default=None)</strong>
<blockquote>
Name to save the file with. If None or 'auto', use default name (ATOMClassifier).
</blockquote>
<strong>save_data: bool, optional (default=True)</strong>
<blockquote>
Whether to save the data as an attribute of the instance. If False, remember to
 update the data immediately after loading the pickle using the dataset's @setter.
</blockquote>
</tr>
</table>
</div>
<br>


<a name="atomclassifier-scoring"></a>
<pre><em>function</em> ATOMClassifier.<strong style="color:#008AB8">scoring</strong>(metric=None)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L616">[source]</a></div></pre>
<div style="padding-left:3%">
Print the scoring of the models for a specific metric. If a model
 shows a `XXX`, it means the metric failed for that specific model. This
 can happen if either the metric is unavailable for classification tasks or if the
 model does not have a `predict_proba` method while the metric requires it.
<br /><br />
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>metric: str or None, optional (default=None)</strong>
<blockquote>
Name of the metric to calculate. If None, returns the metric used to fit the pipeline.
If string, choose from any of sklearn's [SCORERS](https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules)
 or one of the following custom metrics:
<ul>
<li>'cm' or 'confusion_matrix' for an array of the confusion matrix.</li>
<li>'tn' for true negatives.</li>
<li>'fp' for false positives.</li>
<li>'fn' for false negatives.</li>
<li>'lift' for the lift metric.</li>
<li>'fpr' for the false positive rate.</li>
<li>'tpr' for true positive rate.</li>
<li>'sup' for the support metric.</li>
</ul>
</blockquote>
</tr>
</table>
</div>
<br />



<a name="atomclassifier-stats"></a>
<pre><em>function</em> ATOMClassifier.<strong style="color:#008AB8">stats</strong>()
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L414">[source]</a></div></pre>
<div style="padding-left:3%">
Print out a list of basic information on the dataset.
</div>
<br /><br />


## Data cleaning
----------------

ATOM provides five data cleaning methods to scale your features and handle missing values,
 categorical columns, outliers and unbalanced datasets. Calling on one of them
 will automatically apply the method on the dataset in the pipeline.

!!! tip
    Use the [report](#atomclassifier-report) method to examine the data and help you
    determine suitable parameters for the data cleaning methods.
    

<table>
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
<div style="padding-left:3%">
Scale the feature set to mean=1 and std=0. This method calls the [Cleaner](#../data_cleaning/cleaner.md)
 class under the hood and fits it only on the training set to avoid data leakage.
</div>
<br /><br />


<a name="atomclassifier-impute"></a>
<pre><em>function</em> ATOMClassifier.<strong style="color:#008AB8">impute</strong>(strat_num='drop', strat_cat='drop', min_frac_rows=0.5, min_frac_cols=0.5, missing=None) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L711">[source]</a></div></pre>
<div style="padding-left:3%">
Handle missing values according to the selected strategy. Also removes rows and
 columns with too many missing values. The imputer is fitted only on the training set
 to avoid data leakage. See [Imputer](../data_cleaning/imputer.md) for a description
 of the parameters. Note that since the Imputer can remove rows from both train and
 test set, the set's sizes may change to keep ATOM's `test_size` ratio.
</div>
<br />


<a name="atomclassifier-encode"></a>
<pre><em>function</em> ATOMClassifier.<strong style="color:#008AB8">encode</strong>(max_onehot=10, encode_type='LeaveOneOut', frac_to_other=None) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L867">[source]</a></div></pre>
<div style="padding-left:3%">
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
<div style="padding-left:3%">
Remove or replace outliers in the training set. Outliers are defined as values that
 lie further than `max_sigma` * standard_deviation away from the mean of the column.
 See [Outliers](../data_cleaning/outliers.md) for a description of the parameters.
</div>
<br />


<a name="atomclassifier-balance"></a>
<pre><em>function</em> ATOMClassifier.<strong style="color:#008AB8">balance</strong>(oversample='not majority', undersample=None, n_neighbors=5) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L1000">[source]</a></div></pre>
<div style="padding-left:3%">
Balance the number of instances per target category in the training set.
 Using oversample and undersample at the same time or not using any will
 raise an exception. Only the training set is balanced in order to maintain the
 original distribution of target categories in the test set.
 See [Balancer](../data_cleaning/balancer.md) for a description of the parameters.
</div>
<br>



## Feature engineering
----------------------

To further pre-process the data you can create new non-linear features transforming
 the existing ones or, if your dataset is too large, remove features using one
 of the provided strategies.

<table>
<tr>
<td><a href="#atomclassifier-feature-generation">feature_generation</a></td>
<td>Create new features from combinations of existing ones.</td>
</tr>

<tr>
<td><a href="#atomclassifier-feature-selection">feature_selection</a></td>
<td>Remove features according to the selected strategy.</td>
</tr>
</table>
<br>



<a name="atomclassifier-feature-generation"></a>
<pre><em>function</em> ATOMClassifier.<strong style="color:#008AB8">feature_generation</strong>(strategy='dfs', n_features=None,
                                           generations=20, population=500, operators=None)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L1131">[source]</a></div></pre>
<div style="padding-left:3%">
Use Deep feature Synthesis or a genetic algorithm to create new combinations
 of existing features to capture the non-linear relations between the original
 features.
See [FeatureGenerator](../feature_engineering/feature_generator.md) for a description of the parameters.
</div>
<br />


<a name="atomclassifier-feature-selection"></a>
<pre><em>function</em> ATOMClassifier.<strong style="color:#008AB8">feature_selection</strong>(strategy=None, solver=None, n_features=None,
                                          max_frac_repeated=1., max_correlation=1., **kwargs) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L1259">[source]</a></div></pre>
<div style="padding-left:3%">
Remove features according to the selected strategy. Ties between
features with equal scores will be broken in an unspecified way. Also
removes features with too low variance and finds pairs of collinear features
based on the Pearson correlation coefficient. For each pair above the specified
limit (in terms of absolute value), it removes one of the two.
See [FeatureSelector](../feature_engineering/feature_selector.md) for a description of the parameters.

!!! note
    <ul>
    <li>When strategy='univariate' and solver=None, [f_classif](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html)
        will be used as default solver.</li>
    <li>When strategy is one of 'SFM', 'RFE' or 'RFECV' and the solver is one of 
        ATOM's models, the algorithm will automatically select the classifier (no need to add `_class` to the solver).</li>
    <li>When strategy is one of 'SFM', 'RFE' or 'RFECV' and solver=None, ATOM will
         use the winning model (if it exists) as solver.</li>
    <li>When strategy='RFECV', ATOM will use the metric in the pipeline (if it exists)
        as the scoring parameter (only if not specified manually).</li>

</div>
<br>



## Training
----------

The training methods are where the models are fitted to the data and their
 performance is evaluated according to the selected metric. ATOMClassifier contains
 three methods to call the training classes from the ATOM package. All relevant
 attributes and methods from the training classes are attached to ATOMClassifier for
 convenience. These include the errors, winner and results attributes, the model
 subclasses, and the [prediction](#prediction-methods) and [plotting](#plots) methods.


<table>
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
<div style="padding-left:3%">
Calls a [TrainerClassifier](../training/trainerclassifier.md) instance.
 Using this class through ATOMClassifier allows subsequent runs with different models
 without losing previous information.
</div>
<br />


<a name="atomclassifier-successive-halving"></a>
<pre><em>function</em> ATOMClassifier.<strong style="color:#008AB8">successive_halving</strong>(models, metric=None, greater_is_better=True, needs_proba=False,
                                           needs_threshold=False, skip_iter=0, n_calls=0, n_random_starts=5,
                                           bo_params={}, bagging=None) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L2172">[source]</a></div></pre>
<div style="padding-left:3%">
Calls a [SuccessiveHalvingClassifier](../training/successivehalvingclassifier.md) instance.
</div>
<br />


<a name="atomclassifier-train-sizing"></a>
<pre><em>function</em> ATOMClassifier.<strong style="color:#008AB8">train_sizing</strong>(models, metric=None, greater_is_better=True, needs_proba=False,
                                     needs_threshold=False, train_sizes=np.linspace(0.2, 1.0, 5),
                                     n_calls=0, n_random_starts=5, bo_params={}, bagging=None) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L2201">[source]</a></div></pre>
<div style="padding-left:3%">
Calls a [TrainSizingClassifier](../training/trainsizingclassifier.md) instance.
</div>
<br />





## Model subclasses
-------------------

After running any of the training methods, a class for every selected model is created and
 attached to the main ATOMClassifier instance as an attribute. We call these classes model subclasses.
 They can be accessed using the models' acronyms, e.g. `atom.LGB` for the LightGBM model
 subclass. Lowercase calls are also allowed for this attribute, e.g. `atom.lgb`.
 The model subclasses contain a variety of methods and attributes to help you understand
 how every specific model performed.
 <br><br>


#### Attributes

You can see the data used to train and test every specific model using the same [data properties](#properties)
 as the ATOMClassifier has. These can differ from each other if the model needs
 scaled features and the data wasn't already scaled. Note that the data can not be updated from
 the model subclasses (i.e. the data properties have no `@setter`). A list of the
 remaining available attributes can be found hereunder:
<br>

<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Attributes:</strong></td>
<td width="75%" style="background:white;">

<strong>bo: pd.DataFrame</strong>
<blockquote>
Dataframe containing the information of every step taken by the BO. Columns include:
<ul>
<li>'params': Parameters used in the model.</li>
<li>'model': Model used for this iteration (fitted on last cross-validation).</li>
<li>'score': Score of the chosen metric. List of scores for multi-metric.</li>
<li>'time_iteration': Time spent on this iteration.</li>
<li>'time': Total ime spent since the start of the BO.</li>
</ul>
</blockquote>

<strong>best_params: dict</strong>
<blockquote>
Dictionary of the best combination of hyperparameters found by the BO.
</blockquote>

<strong>model: class</strong>
<blockquote>
Model instance with the best combination of hyperparameters fitted on the complete training set.
</blockquote>

<strong>predict_train: np.ndarray</strong>
<blockquote>
Predictions of the model on the training set.
</blockquote>

<strong> predict_test: np.ndarray</strong>
<blockquote>
Predictions of the model on the test set.
</blockquote>

<strong>predict_proba_train: np.ndarray</strong>
<blockquote>
Predict probabilities of the model on the training set. Only for models with
a `predict_proba` method.
</blockquote>

<strong>predict_proba_test: np.ndarray</strong>
<blockquote>
Predict probabilities of the model on the test set. Only for models with
a `predict_proba` method.
</blockquote>

<strong>predict_log_proba_train: np.ndarray</strong>
<blockquote>
Predict log probabilities of the model on the training set. Only for models with
a `predict_proba` method.
</blockquote>

<strong>predict_log_proba_test: np.ndarray</strong>
<blockquote>
Predict log probabilities of the model on the test set. Only for models with
a `predict_proba` method.
</blockquote>

<strong>decision_function_train: np.ndarray</strong>
<blockquote>
Decision function scores on the training set. Only for models with
a `decision_function` method.
</blockquote>

<strong>decision_function_test: np.ndarray</strong>
<blockquote>
Decision function scores on the test set. Only for models with
a `decision_function` method.
</blockquote>

<strong>time_bo: str</strong>
<blockquote>
Time it took to run the bayesian optimization algorithm.
</blockquote>

<strong>score_bo: float</strong>
<blockquote>
Best score of the model on the BO.
</blockquote>

<strong>time_fit: str</strong>
<blockquote>
Time it took to train the model on the complete training set and calculate the
 metric on the test set.
</blockquote>

<strong>score_train: float</strong>
<blockquote>
Metric score of the model on the training set.
</blockquote>

<strong>score_test: float</strong>
<blockquote>
Metric score of the model on the test set.
</blockquote>

<strong>evals: dict</strong>
<blockquote>
Dictionary of the metric calculated during training. The metric is provided by the model's
 package and is different for every model and every task. Only for models that allow
 in-training evaluation (XGB, LGB, CatB). Available keys:
<ul>
<li>'metric': Name of the used metric. </li>
<li>'train': List of scores calculated on the training set.</li>
<li>'test': List of scores calculated on the test set.</li>
</ul>
</blockquote>

<strong>score_bagging: list</strong>
<blockquote>
Array of the bagging's results.
</blockquote>

<strong>mean_bagging: float</strong>
<blockquote>
Mean of the bagging's results.
</blockquote>

<strong>std_bagging: float</strong>
<blockquote>
Standard deviation of the bagging's results.
</blockquote>

<strong>permutations: dict</strong>
<blockquote>
Dictionary of the permutation's results (if `plot_permutation_importance` was used).
</blockquote>


</tr>
</table>
<br>


#### Methods
The majority of the [plots](#plots) can be called directly from the
 subclasses. For example, to plot the ROC for the LightGBM model we could type
 `atom.lgb.plot_roc()`. A list of the remaining methods can be found hereunder:
<br>

<table>
<tr>
<td width="15%"><a href="#atomclassifier-calibrate">calibrate</a></td>
<td>Calibrate the model.</td>
</tr>

<tr>
<td width="15%"><a href="#atomclassifier-scoring">scoring</a></td>
<td>Get the scoring of a specific metric on the test set.</td>
</tr>

<tr>
<td><a href="#atomclassifier-save-model">save_model</a></td>
<td>Save the model to a pickle file.</td>
</tr>
</table>
<br>


<a name="atomclassifier-calibrate"></a>
<pre><em>function</em> BaseModel.<strong style="color:#008AB8">calibrate</strong>(\*\*kwargs)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L696">[source]</a></div></pre>
<div style="padding-left:3%">
Applies probability calibration on the winning model. The calibration is done
 with the [CalibratedClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)
 class from sklearn. The model will be trained via cross-validation on a subset of the
 training data, using the rest to fit the calibrator. The new classifier will replace
 the `model` attribute.
<br /><br />
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>**kwargs</strong>
<blockquote>
Additional keyword arguments for the CalibratedClassifierCV instance.
Using cv='prefit' will use the trained model and fit the calibrator on
the test set. Note that doing this will result in data leakage in the
test set. Use this only if you have another, independent set for testing.
</blockquote>
</tr>
</table>
</div>
<br />


<a name="atomclassifier-scoring"></a>
<pre><em>function</em> BaseModel.<strong style="color:#008AB8">scoring</strong>(metric=None)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L696">[source]</a></div></pre>
<div style="padding-left:3%">
Get the scoring of a specific metric on the test set.
<br /><br />
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>metric: str or None, optional (default=None)</strong>
<blockquote>
Name of the metric to calculate. If None, return mean_bagging if exists, else score_test.
If string, choose from any of sklearn's [SCORERS](https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules)
 or one of the following custom metrics:
<ul>
<li>'cm' or 'confusion_matrix' for an array of the confusion matrix.</li>
<li>'tn' for true negatives.</li>
<li>'fp' for false positives.</li>
<li>'fn' for false negatives.</li>
<li>'lift' for the lift metric.</li>
<li>'fpr' for the false positive rate.</li>
<li>'tpr' for true positive rate.</li>
<li>'sup' for the support metric.</li>
</ul>
</blockquote>
</tr>
</table>
</div>
<br />


<a name="atomclassifier-save-model"></a>
<pre><em>function</em> BaseModel.<strong style="color:#008AB8">save_model</strong>(filename=None)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L696">[source]</a></div></pre>
<div style="padding-left:3%">
Save the model (fitted to the complete training set) to a pickle file.
<br /><br />
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>filename: str or None, optional (default=None)</strong>
<blockquote>
Name of the file to save. If None or 'auto', use default name (<name\>_model).
</blockquote>
</tr>
</table>
</div>
<br />




## Prediction methods
---------------------

Like the majority of estimators in scikit-learn, you can use a fitted instance
 of ATOMClassifier to make predictions onto new data, e.g. `atom.predict(X)`.
 The following methods will apply all selected data pre-processing steps on the
 provided data first, and use the winning model from the pipeline (under attribute
 `winner`) to make the predictions. If you want to use a different model, you can
 call the method from the model subclass, e.g. `atom.LGB.predict(X)`.

<table>
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
<div style="padding-left:3%">
Transform new data through all the pre-processing steps. The outliers and balancer
steps are not included in the default steps since they should only be applied
on the training set.
 <br /><br />
<table>
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
Verbosity level of the output. If None, it uses ATOM's verbosity.
</blockquote>
</tr>
</table>
</div>
<br />


<a name="atomclassifier-predict"></a>
<pre><em>function</em> ATOMClassifier.<strong style="color:#008AB8">predict</strong>(X, \*\*kwargs) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L711">[source]</a></div></pre>
<div style="padding-left:3%">
Transform the data and make predictions using the winning model in the pipeline.
The model has to have a `predict` method.
 <br /><br />
<table>
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
<div style="padding-left:3%">
Transform the data and make probability predictions using the winning model in the pipeline.
The model has to have a `predict_proba` method.
<br /><br />
<table>
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
<div style="padding-left:3%">
Transform the data and make logarithmic probability predictions using the winning model in the pipeline.
The model has to have a `predict_log_proba` method.
<br /><br />
<table>
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
<div style="padding-left:3%">
Transform the data and run the decision function of the winning model in the pipeline.
The model has to have a `decision_function` method.
<br /><br />
<table>
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
<div style="padding-left:3%">
Transform the data and run the scoring method of the winning model in the pipeline.
The model has to have a `score` method.
<br /><br />
<table>
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



## Plots
--------
Plots can be called directly from the instance, e.g. `atom.plot_prc()`, or from the model
 subclasses, e.g. `atom.LDA.plot_prc()`. The plots aesthetics can be customized using
 the [plotting properties](../../../../user_guide/#aesthetics). Read more in the
 [user guide](../../../../user_guide/#plotting). Available plots are:

<table>
<tr>
<td><a href="../../plots/plot_correlation/">plot_correlation</a></td>
<td>Plot the correlation matrix of the dataset.</td>
</tr>

<tr>
<td><a href="../../plots/plot_pca/">plot_pca</a></td>
<td>Plot the explained variance ratio vs the number of components.</td>
</tr>

<tr>
<td><a href="../../plots/plot_components/">plot_components</a></td>
<td>Plot the explained variance ratio per component.</td>
</tr>

<tr>
<td><a href="../../plots/plot_rfecv/">plot_rfecv</a></td>
<td>Plot the scores obtained by the estimator on the RFECV.</td>
</tr>

<tr>
<td><a href="../../plots/plot_successive_halving/">plot_successive_halving</a></td>
<td>Plot the models' scores per iteration of the successive halving.</td>
</tr>

<tr>
<td><a href="../../plots/plot_learning_curve/">plot_learning_curve</a></td>
<td>Plot the model's learning curve: score vs training samples.</td>
</tr>

<tr>
<td><a href="../../plots/plot_bagging/">plot_bagging</a></td>
<td>Plot a boxplot of the bagging's results.</td>
</tr>

<tr>
<td><a href="../../plots/plot_bo/">plot_bo</a></td>
<td>Plot the bayesian optimization scoring.</td>
</tr>

<tr>
<td><a href="../../plots/plot_evals/">plot_evals</a></td>
<td>Plot evaluation curves for the train and test set.</td>
</tr>


<tr>
<td><a href="../../plots/plot_roc/">plot_roc</a></td>
<td>Plot the Receiver Operating Characteristics curve.</td>
</tr>

<tr>
<td><a href="../../plots/plot_prc/">plot_prc</a></td>
<td>Plot the precision-recall curve.</td>
</tr>

<tr>
<td><a href="../../plots/plot_permutation_importance/">plot_permutation_importance</a></td>
<td>Plot the feature permutation importance of models.</td>
</tr>

<tr>
<td><a href="../../plots/plot_feature_importance/">plot_feature_importance</a></td>
<td>Plot the feature permutation importance of models.</td>
</tr>

<tr>
<td><a href="../../plots/plot_confusion_matrix/">plot_confusion_matrix</a></td>
<td>Plot a model's confusion matrix.</td>
</tr>

<tr>
<td><a href="../../plots/plot_threshold/">plot_threshold</a></td>
<td>Plot performance metric(s) against threshold values.</td>
</tr>

<tr>
<td><a href="../../plots/plot_probabilities/">plot_probabilities</a></td>
<td>Plot the probabilities of the different classes of belonging to the target class.</td>
</tr>

<tr>
<td><a href="../../plots/plot_calibration/">plot_calibration</a></td>
<td>Plot the calibration curve for a binary classifier.</td>
</tr>

<tr>
<td><a href="../../plots/plot_gains/">plot_gains</a></td>
<td>Plot the cumulative gains curve.</td>
</tr>

<tr>
<td><a href="../../plots/plot_lift/">plot_lift</a></td>
<td>Plot the lift curve.</td>
</tr>

</table>
<br>



## Example
---------
```python
from sklearn.datasets import load_breast_cancer
from atom import ATOMClassifier

X, y = load_breast_cancer(return_X_y=True)

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
atom.palette = 'Blues'
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
```