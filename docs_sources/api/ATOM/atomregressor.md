# ATOMRegressor
----------------

<a name="atom"></a>
<pre><em>class</em> atom.api.<strong style="color:#008AB8">ATOMRegressor</strong>(X, y=-1, n_rows=1, test_size=0.2, logger=None,
                             n_jobs=1, warnings=True, verbose=0, random_state=None)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/api.py#L197">[source]</a></div></pre>
<div style="padding-left:3%">
ATOMRegressor is ATOM's wrapper for regression tasks. Use
 this class to easily apply all data transformations and model management provided by
 the package on a given dataset. Note that contrary to scikit-learn's API, the
 ATOMRegressor object already contains the dataset on which we want to perform the
 analysis. Calling a method will automatically apply it on the dataset it contains.
 The class initializer always calls [StandardCleaner](../data_cleaning/standard_cleaner.md)
 with default parameters. The following data types can't (yet) be handled properly
 and are therefore removed: 'datetime64', 'datetime64[ns]', 'timedelta[ns]'. 
 
You can [predict](../../../user_guide/#predicting), [plot](../../../user_guide/#plots)
 and call any [`model`](../../../user_guide/#models) from the ATOMRegressor instance.
 Read more in the [user guide](../../../user_guide/#first-steps).
<br />
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
Note that changing this parameter will affect the `PYTHONWARNINGS` environment.<br>
 Note that ATOM can't manage warnings that go directly from C++ code to the
 stdout/stderr.
</blockquote>
<strong>logger: bool, str, class or None, optional (default=None)</strong>
<blockquote>
<ul>
<li>If None: Doesn't save a logging file.</li>
<li>If bool: True for logging file with default name. False for no logger.</li>
<li>If str: Name of the logging file. 'auto' for default name.</li>
<li>If class: python `Logger` object'.</li>
</ul>
Note that warnings will not be saved to the logger in any case.
</blockquote>
<strong>random_state: int or None, optional (default=None)</strong>
<blockquote>
Seed used by the random number generator. If None, the random number
 generator is the `RandomState` instance used by `numpy.random`.
</blockquote>
</td>
</tr>
</table>
<br>



## Attributes
-------------

### Data attributes

The dataset within ATOM's pipeline can be accessed at any time through multiple
 properties, e.g. calling `atom.train` will return the training set. The data can also
 be changed through these properties, e.g. `atom.test = atom.test.drop(0)` will
 drop the first row from the test set. This will also update the other data attributes.

<a name="atom"></a>
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Attributes:</strong></td>
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
<strong>shape: tuple</strong>
<blockquote>
Dataset's shape in the form (rows x columns).
</blockquote>
<strong>columns: list</strong>
<blockquote>
List of columns in the dataset.
</blockquote>
<strong>target: str</strong>
<blockquote>
Name of the target column.
</blockquote>
</td></tr>
</table>
<br>


### Utility attributes

<a name="atom"></a>
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Attributes:</strong></td>
<td width="75%" style="background:white;">
<strong>profile: ProfileReport</strong>
<blockquote>
Profile created by pandas-profiling after calling the report method.
</blockquote>
<strong>genetic_features: pd.DataFrame</strong>
<blockquote>
Dataframe of the non-linear features created by the <a href="#ATOMRegressor-feature-generation">feature_generation</a> method.
 Columns include:
<ul>
<li><b>name:</b> Name of the feature (automatically created).</li>
<li><b>description:</b> Operators used to create this feature.</li>
<li><b>fitness:</b> Fitness score.</li>
</ul>
</blockquote>
<strong>collinear: pd.DataFrame</strong>
<blockquote>
Dataframe of the collinear features removed by the <a href="#ATOMRegressor-feature-selection">feature_selection</a> method.
 Columns include:
<ul>
<li><b>drop_feature:</b> name of the feature dropped by the method.</li>
<li><b>correlated feature:</b> Name of the correlated feature(s).</li>
<li><b>correlation_value:</b> Pearson correlation coefficient(s) of the feature pairs.</li>
</ul>
</blockquote>
<strong>models: list</strong>
<blockquote>
List of models in the pipeline.
</blockquote>
<strong>metric: str or list</strong>
<blockquote>
Metric(s) used to fit the models.
</blockquote>
<strong>errors: dict</strong>
<blockquote>
Dictionary of the encountered exceptions (if any).
</blockquote>
<strong>winner: [`model`](../../models/)</strong>
<blockquote>
Model subclass that performed best on the test set.
</blockquote>
<strong>results: pd.DataFrame</strong>
<blockquote>
Dataframe of the training results with the model acronyms as index. Columns can include:
<ul>
<li><b>name:</b> Name of the model.</li>
<li><b>metric_bo:</b> Best score achieved during the BO.</li>
<li><b>time_bo:</b> Time spent on the BO.</li>
<li><b>metric_train:</b> Metric score on the training set.</li>
<li><b>metric_test:</b>Metric score on the test set.</li>
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


### Plot attributes
 
<a name="atom"></a>
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Attributes:</strong></td>
<td width="75%" style="background:white;">
<strong>style: str</strong>
<blockquote>
Plotting style. See seaborn's <a href="https://seaborn.pydata.org/tutorial/aesthetics.html#seaborn-figure-styles">documentation</a>.
</blockquote>
<strong>palette: str</strong>
<blockquote>
Color palette. See seaborn's <a href="https://seaborn.pydata.org/tutorial/color_palettes.html">documentation</a>.
</blockquote>
<strong>title_fontsize: int</strong>
<blockquote>
Fontsize for the plot's title.
</blockquote>
<strong>label_fontsize: int</strong>
<blockquote>
Fontsize for labels and legends.
</blockquote>
<strong>tick_fontsize: int</strong>
<blockquote>
Fontsize for the ticks along the plot's axes.
</blockquote>
</td></tr>
</table>
<br><br>



## Utility methods
------------------

The ATOM class contains a variety of methods to help you handle the data and
inspect the pipeline.

<table>
<tr>
<td width="15%"><a href="#ATOMRegressor-clear">clear</a></td>
<td>Remove a model from the pipeline.</td>
</tr>

<tr>
<td width="15%"><a href="#ATOMRegressor-log">log</a></td>
<td>Save information to the logger and print to stdout.</td>
</tr>

<tr>
<td><a href="#ATOMRegressor-report">report</a></td>
<td>Get an extensive profile analysis of the data.</td>
</tr>


<tr>
<td><a href="#ATOMRegressor-save">save</a></td>
<td>Save the ATOMRegressor instance to a pickle file.</td>
</tr>

<tr>
<td><a href="#ATOMRegressor-scoring">scoring</a></td>
<td>Print the scoring of the models for a specific metric.</td>
</tr>

<tr>
<td width="15%"><a href="#ATOMRegressor-stats">stats</a></td>
<td>Print out a list of basic statistics on the dataset.</td>
</tr>

</table>
<br>


<a name="ATOMRegressor-clear"></a>
<pre><em>method</em> <strong style="color:#008AB8">clear</strong>(models='all')
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basepredictor.py#L197">[source]</a></div></pre>
<div style="padding-left:3%">
Removes all traces of a model in the pipeline (except for the `errors`
 attribute). If all models in the pipeline are removed, the metric is reset.
 Use this method to remove unwanted models from the pipeline or to clear
 memory before saving the instance. 
<br /><br />
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>models: str or sequence, optional (default='all')</strong>
<blockquote>
Model(s) to clear from the pipeline. If 'all', clear all models.
</blockquote>
</tr>
</table>
</div>
<br />


<a name="ATOMRegressor-log"></a>
<pre><em>method</em> <strong style="color:#008AB8">log</strong>(msg, level=0)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basetransformer.py#L196">[source]</a></div></pre>
<div style="padding-left:3%">
Write a message to the logger and print it to stdout.
<br /><br />
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>msg: str</strong>
<blockquote>
Message to write to the logger and print to stdout.
</blockquote>
<strong>level: int, optional (default=0)</strong>
<blockquote>
Minimum verbosity level in order to print the message.
</blockquote>
</tr>
</table>
</div>
<br />


<a name="ATOMRegressor-report"></a>
<pre><em>method</em> <strong style="color:#008AB8">report</strong>(dataset='dataset', n_rows=None, filename=None)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L250">[source]</a></div></pre>
<div style="padding-left:3%">
Get an extensive profile analysis of the data. The report is rendered
 in HTML5 and CSS3 and saved to the `profile` attribute. Note that this method
 can be slow for `n_rows` > 10k.
<br /><br />
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>df: str, optional (default='dataset')</strong>
<blockquote>
Name of the data set to get the profile from.
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


<a name="ATOMRegressor-save"></a>
<pre><em>method</em> <strong style="color:#008AB8">save</strong>(filename=None, save_data=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basetransformer.py#L220">[source]</a></div></pre>
<div style="padding-left:3%">
Save the instance to a pickle file. Remember that the class contains the complete
 dataset as property, so the file can become large for big datasets! To avoid this,
 use `save_data=False`.
<br><br>
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>filename: str or None, optional (default=None)</strong>
<blockquote>
Name to save the file with. If None or 'auto', use the name of the class.
</blockquote>
<strong>save_data: bool, optional (default=True)</strong>
<blockquote>
Whether to save the data as an attribute of the instance. If False, remember to
 update the data immediately after loading the pickle using the dataset's `@setter`.
</blockquote>
</tr>
</table>
</div>
<br>


<a name="ATOMRegressor-scoring"></a>
<pre><em>method</em> <strong style="color:#008AB8">scoring</strong>(metric=None, dataset='test')
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basepredictor#L145">[source]</a></div></pre>
<div style="padding-left:3%">
Print the scoring of the models for a specific metric. If a model
 shows a `XXX`, it means the metric failed for that specific model. This
 can happen if either the metric is unavailable for the task or if the
 model does not have a `predict_proba` method while the metric requires it.
<br /><br />
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>metric: str or None, optional (default=None)</strong>
<blockquote>
Name of the metric to calculate. Choose from any of sklearn's [SCORERS](https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules).
 If None, returns the final results for the model (ignores the `dataset` parameter).
</blockquote>
<strong>dataset: str, optional (default='test')</strong>
<blockquote>
Data set on which to calculate the metric. Options are 'train' or 'test'.
</blockquote>
</tr>
</table>
</div>
<br />


<a name="ATOMRegressor-stats"></a>
<pre><em>method</em> <strong style="color:#008AB8">stats</strong>()
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L168">[source]</a></div></pre>
<div style="padding-left:3%">
Print out a list of basic information on the dataset.
</div>
<br /><br />


## Data cleaning
----------------

ATOM provides data cleaning methods to scale your features and handle missing values,
 categorical columns and outliers. Calling on one of them will automatically apply
 the method on the dataset in the pipeline.

!!! tip
    Use the [report](#ATOMRegressor-report) method to examine the data and help you
    determine suitable parameters for the data cleaning methods.
    

<table>
<tr>
<td><a href="#ATOMRegressor-scale">scale</a></td>
<td>Scale all the features to mean=1 and std=0.</td>
</tr>

<tr>
<td><a href="#ATOMRegressor-impute">impute</a></td>
<td>Handle missing values in the dataset.</td>
</tr>

<tr>
<td><a href="#ATOMRegressor-encode">encode</a></td>
<td>Encode categorical features.</td>
</tr>

<tr>
<td><a href="#ATOMRegressor-outliers">outliers</a></td>
<td>Remove or replace outliers in the training set.</td>
</tr>
</table>
<br>


<a name="ATOMRegressor-scale"></a>
<pre><em>method</em> <strong style="color:#008AB8">scale</strong>()
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L348">[source]</a></div></pre>
<div style="padding-left:3%">
Scale the features to mean=1 and std=0.
</div>
<br /><br />


<a name="ATOMRegressor-impute"></a>
<pre><em>method</em> <strong style="color:#008AB8">impute</strong>(strat_num='drop', strat_cat='drop', min_frac_rows=0.5, min_frac_cols=0.5, missing=None) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L363">[source]</a></div></pre>
<div style="padding-left:3%">
Handle missing values according to the selected strategy. Also removes rows and
 columns with too many missing values. The imputer is fitted only on the training set
 to avoid data leakage. See [Imputer](../data_cleaning/imputer.md) for a description
 of the parameters. Note that since the Imputer can remove rows from both train and
 test set, the set's sizes may change to keep ATOM's `test_size` ratio.
</div>
<br />


<a name="ATOMRegressor-encode"></a>
<pre><em>method</em> <strong style="color:#008AB8">encode</strong>(strategy='LeaveOneOut', max_onehot=10, frac_to_other=None)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L399">[source]</a></div></pre>
<div style="padding-left:3%">
Perform encoding of categorical features. The encoding type depends on the
 number of unique values in the column:
<ul>
<li>If n_unique=2, use Label-encoding.</li>
<li>If 2 < n_unique <= max_onehot, use OneHot-encoding.</li>
<li>If n_unique > max_onehot, use `strategy`-encoding.</li>
</ul>
Also replaces classes with low occurrences with the value 'other' in
 order to prevent too high cardinality. Categorical features are defined as
 all columns whose dtype.kind not in 'ifu'. Will raise an error if it encounters
 missing values or unknown categories when transforming. The encoder is fitted only
 on the training set to avoid data leakage. See [Encoder](../data_cleaning/encoder.md)
 for a description of the parameters.
</div>
<br />


<a name="ATOMRegressor-outliers"></a>
<pre><em>method</em> <strong style="color:#008AB8">outliers</strong>(strategy='drop', max_sigma=3, include_target=False) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L433">[source]</a></div></pre>
<div style="padding-left:3%">
Remove or replace outliers in the training set. Outliers are defined as values that
 lie further than `max_sigma` * standard_deviation away from the mean of the column.
 Only outliers from the training set are removed to maintain an original sample of
 target values in the test set. Ignores categorical columns. See
 [Outliers](../data_cleaning/outliers.md) for a description of the parameters.
</div>
<br />



## Feature engineering
----------------------

To further pre-process the data you can create new non-linear features transforming
 the existing ones or, if your dataset is too large, remove features using one
 of the provided strategies.

<table>
<tr>
<td><a href="#ATOMRegressor-feature-generation">feature_generation</a></td>
<td>Create new features from combinations of existing ones.</td>
</tr>

<tr>
<td><a href="#ATOMRegressor-feature-selection">feature_selection</a></td>
<td>Remove features according to the selected strategy.</td>
</tr>
</table>
<br>



<a name="ATOMRegressor-feature-generation"></a>
<pre><em>method</em> <strong style="color:#008AB8">feature_generation</strong>(strategy='DFS', n_features=None, generations=20, population=500, operators=None)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L504">[source]</a></div></pre>
<div style="padding-left:3%">
Use Deep feature Synthesis or a genetic algorithm to create new combinations
 of existing features to capture the non-linear relations between the original
 features. See [FeatureGenerator](../feature_engineering/feature_generator.md) for
 a description of the parameters. Attributes created by the class are attached to
 the ATOM instance.
</div>
<br />


<a name="ATOMRegressor-feature-selection"></a>
<pre><em>method</em> <strong style="color:#008AB8">feature_selection</strong>(strategy=None, solver=None, n_features=None,
                         max_frac_repeated=1., max_correlation=1., **kwargs) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L540">[source]</a></div></pre>
<div style="padding-left:3%">
Remove features according to the selected strategy. Ties between
 features with equal scores will be broken in an unspecified way. Also
 removes features with too low variance and finds pairs of collinear features
 based on the Pearson correlation coefficient. For each pair above the specified
 limit (in terms of absolute value), it removes one of the two.
 See [FeatureSelector](../feature_engineering/feature_selector.md) for a description of the parameters.
 Plotting methods and attributes created by the class are attached to the instance.

!!! note
    <ul>
    <li>When strategy='univariate' and solver=None, [f_regression](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html)
        will be used as default solver.</li>
    <li>When strategy is one of 'SFM', 'RFE' or 'RFECV' and the solver is one of 
        ATOM's models, the algorithm will automatically select the classifier (no need to add `_reg` to the solver).</li>
    <li>When strategy is one of 'SFM', 'RFE' or 'RFECV' and solver=None, ATOM will
         use the winning model (if it exists) as solver.</li>
    <li>When strategy='RFECV', ATOM will use the metric in the pipeline (if it exists)
        as the scoring parameter (only if not specified manually).</li>

</div>
<br>



## Training
----------

The training methods are where the models are fitted to the data and their
 performance is evaluated according to the selected metric. ATOMRegressor contains
 three methods to call the training classes from the ATOM package. All relevant
 attributes and methods from the training classes are attached to ATOMRegressor for
 convenience. These include the errors, winner and results attributes, the `models`,
 and the [prediction](../../../user_guide/#predicting) and [plotting](#plots) methods.


<table>
<tr>
<td><a href="#ATOMRegressor-run">run</a></td>
<td>Fit the models to the data in a direct fashion.</td>
</tr>

<tr>
<td><a href="#ATOMRegressor-successive-halving">successive_halving</a></td>
<td>Fit the models to the data in a successive halving fashion.</td>
</tr>

<tr>
<td><a href="#ATOMRegressor-train-sizing">train_sizing</a></td>
<td>Fit the models to the data in a train sizing fashion.</td>
</tr>
</table>
<br>


<a name="ATOMRegressor-run"></a>
<pre><em>method</em> <strong style="color:#008AB8">run</strong>(models, metric=None, greater_is_better=True, needs_proba=False,
           needs_threshold=False, n_calls=10, n_initial_points=5, bo_params={}, bagging=None) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L664">[source]</a></div></pre>
<div style="padding-left:3%">
Runs a [TrainerRegressor](../training/trainerregressor.md) instance.
 Using this class through `atom` allows subsequent runs with different models
 without losing previous information.
</div>
<br />


<a name="ATOMRegressor-successive-halving"></a>
<pre><em>method</em> <strong style="color:#008AB8">successive_halving</strong>(models, metric=None, greater_is_better=True, needs_proba=False, needs_threshold=False,
                          skip_iter=0, n_calls=0, n_initial_points=5, bo_params={}, bagging=None) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L713">[source]</a></div></pre>
<div style="padding-left:3%">
Runs a [SuccessiveHalvingRegressor](../training/successivehalvingregressor.md) instance.
</div>
<br />


<a name="ATOMRegressor-train-sizing"></a>
<pre><em>method</em> <strong style="color:#008AB8">train_sizing</strong>(models, metric=None, greater_is_better=True, needs_proba=False, needs_threshold=False,
                    train_sizes=np.linspace(0.2, 1.0, 5), n_calls=0, n_initial_points=5, bo_params={}, bagging=None) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L754">[source]</a></div></pre>
<div style="padding-left:3%">
Runs a [TrainSizingRegressor](../training/trainsizingregressor.md) instance.
</div>
<br />



## Example
---------
```python
from sklearn.datasets import load_boston
from atom import ATOMRegressor

X, y = load_boston(return_X_y=True)

# Initialize class
atom = ATOMRegressor(X, y, logger='auto', n_jobs=2, verbose=2)

# Apply data cleaning methods
atom.outliers(strategy='min_max', max_sigma=2, include_target=True)

# Fit the models to the data
atom.run(models=['OLS', 'BR', 'CatB'],
         metric='MSE',
         n_calls=25,
         n_initial_points=10,
         bo_params={'cv': 1},
         bagging=4)

# Analyze the results
print(f"The winning model is: {atom.winner.name}")
print(atom.results)

# Make some plots
atom.palette = 'Blues'
atom.plot_errors(figsize=(9, 6), filename='errors.png')  
atom.CatB.plot_feature_importance(filename='catboost_feature_importance.png')

# Run an extra model
atom.run(models='MLP',
         metric='MSE',
         n_calls=25,
         n_initial_points=10,
         bo_params={'cv': 1},
         bagging=4)

# Get the predictions for the best model on new data
predictions = atom.predict(X_new)
```