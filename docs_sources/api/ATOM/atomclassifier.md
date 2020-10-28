# ATOMClassifier
----------------

<a name="atom"></a>
<pre><em>class</em> atom.api.<strong style="color:#008AB8">ATOMClassifier</strong>(*arrays, n_rows=1, test_size=0.2, logger=None,
                              n_jobs=1, warnings=True, verbose=0, random_state=None)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/api.py#L181">[source]</a></div></pre>
<div style="padding-left:3%">
ATOMClassifier is ATOM's wrapper for binary and multiclass classification tasks. Use
 this class to easily apply all data transformations and model management provided by
 the package on a given dataset. Note that contrary to scikit-learn's API, the
 ATOMClassifier object already contains the dataset on which we want to perform the
 analysis. Calling a method will automatically apply it on the dataset it contains.
 
You can [predict](../../../user_guide/#predicting), [plot](../../../user_guide/#plots)
 and call any [`model`](../../../user_guide/#models) from the ATOMClassifier instance.
 Read more in the [user guide](../../../user_guide/#first-steps).
<br />
<table>
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>*arrays: sequence of indexables</strong>
<blockquote>
Dataset containing the features and target. Allowed formats are:
<ul>
<li>X, y</li>
<li>train, test</li>
<li>X_train, X_test, y_train, y_test</li>
<li>(X_train, y_train), (X_test, y_test)</li>
</ul>
X, train, test: dict, list, tuple, np.array or pd.DataFrame<br>
&nbsp;&nbsp;&nbsp;&nbsp;
Feature set with shape=(n_features, n_samples). If no y is provided, the
 last column is used as target.<br><br>
y: int, str, list, tuple,  np.array or pd.Series<br>
<ul>
<li>If int: Position of the target column in X.</li>
<li>If str: Name of the target column in X.</li>
<li>Else: Data target column with shape=(n_samples,).</li>
</ul>
</blockquote>
<strong>n_rows: int or float, optional (default=1)</strong>
<blockquote>
<ul>
<li>If <=1: Fraction of the dataset to use.</li>
<li>If >1: Number of rows to use (only if input is X, y).</li>
</ul>
</blockquote>
<strong>test_size: int, float, optional (default=0.2)</strong>
<blockquote>
<ul>
<li>If <=1: Fraction of the dataset to include in the test set.</li>
<li>If >1: Number of rows to include in the test set.</li>
</ul>
This parameter is ignored if the train and test set are provided.
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
<li>If True: Default warning action (equal to "default" when string).</li>
<li>If False: Suppress all warnings (equal to "ignore" when string).</li>
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
<li>If str: Name of the logging file. "auto" for default name.</li>
<li>If class: python `Logger` object.</li>
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
</div>
<br>



## Attributes
-------------

### Data attributes

The dataset can be accessed at any time through multiple properties, e.g. calling
 `atom.train` will return the training set. The data can also be changed through
 these properties, e.g. `atom.test = atom.test.drop(0)` will drop the first row
 from the test set. This will also update the other data attributes.

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
<strong>classes: pd.DataFrame</strong>
<blockquote>
Dataframe of the number of rows per target class in the train, test and complete dataset.
</blockquote>
<strong>n_classes: int</strong>
<blockquote>
Number of unique classes in the target column.
</blockquote>
<strong>mapping: dict</strong>
<blockquote>
Dictionary of the target classes mapped to their respective encoded integer.
</blockquote>
<strong>missing: pd.Series</strong>
<blockquote>
Returns columns with number of missing values.
</blockquote>
<strong>n_missing: int</strong>
<blockquote>
Number of columns with missing values.
</blockquote>
<strong>categorical: list</strong>
<blockquote>
Returns columns with categorical features.
</blockquote>
<strong>n_categorical: int</strong>
<blockquote>
Number of columns with categorical features.
</blockquote>
<strong>scaled: bool</strong>
<blockquote>
Returns whether the feature set is scaled.
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
<strong>genetic_features: pd.DataFrame</strong>
<blockquote>
Dataframe of the non-linear features created by the <a href="#feature-generation">feature_generation</a> method.
 Columns include:
<ul>
<li><b>name:</b> Name of the feature (automatically created).</li>
<li><b>description:</b> Operators used to create this feature.</li>
<li><b>fitness:</b> Fitness score.</li>
</ul>
</blockquote>
<strong>collinear: pd.DataFrame</strong>
<blockquote>
Dataframe of the collinear features removed by the <a href="#feature-selection">feature_selection</a> method.
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
<strong>winner: [`model`](../../../user_guide/#models)</strong>
<blockquote>
Model subclass that performed best on the test set.
</blockquote>
<strong>pipeline: pd.Series</strong>
<blockquote>
Series containing all classes fitted in the pipeline. Use this attribute only to
 access the individual classes. To visualize the pipeline, use `atom`'s \_\_repr__
 or [plot_pipeline](../../plots/plot_pipeline).
</blockquote>
<strong>results: pd.DataFrame</strong>
<blockquote>
Dataframe of the training results. Columns can include:
<ul>
<li><b>metric_bo:</b> Best score achieved during the BO.</li>
<li><b>time_bo:</b> Time spent on the BO.</li>
<li><b>metric_train:</b> Metric score on the training set.</li>
<li><b>metric_test:</b> Metric score on the test set.</li>
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
<td><a href="#calibrate">calibrate</a></td>
<td>Calibrate the winning model.</td>
</tr>

<tr>
<td width="15%"><a href="#clear">clear</a></td>
<td>Remove a model from the pipeline.</td>
</tr>

<tr>
<td width="15%"><a href="#get-class-weights">get_class_weights</a></td>
<td>Return class weights for a balanced data set.</td>
</tr>

<tr>
<td width="15%"><a href="#get-sample-weights">get_sample_weights</a></td>
<td>Return sample weights for a balanced data set.</td>
</tr>

<tr>
<td width="15%"><a href="#log">log</a></td>
<td>Save information to the logger and print to stdout.</td>
</tr>

<tr>
<td><a href="#report">report</a></td>
<td>Get an extensive profile analysis of the data.</td>
</tr>

<tr>
<td><a href="#save">save</a></td>
<td>Save the instance to a pickle file.</td>
</tr>

<tr>
<td width="15%"><a href="#save-data">save_data</a></td>
<td>Save data to a csv file.</td>
</tr>

<tr>
<td><a href="#scoring">scoring</a></td>
<td>Returns the scores of the models for a specific metric.</td>
</tr>

<tr>
<td width="15%"><a href="#stats">stats</a></td>
<td>Print out a list of basic statistics on the dataset.</td>
</tr>

</table>
<br>


<a name="calibrate"></a>
<pre><em>method</em> <strong style="color:#008AB8">calibrate</strong>(\*\*kwargs)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basepredictor.py#L116">[source]</a></div></pre>
<div style="padding-left:3%">
Applies probability calibration on the winning model. The calibration is done with the
 [CalibratedClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)
 class from sklearn. The model will be trained via cross-validation on a subset
 of the training data, using the rest to fit the calibrator. The new classifier will
 replace the `estimator` attribute. After calibrating, all prediction attributes of
 the winning model will reset.
<br /><br />
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>**kwargs</strong>
<blockquote>
Additional keyword arguments for the [CalibratedClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)
 instance. Using cv="prefit" will use the trained model and fit the calibrator on the
 test set. Note that doing this will result in data leakage in the test set. Use this
 only if you have another, independent set for testing.
</blockquote>
</tr>
</table>
</div>
<br />


<a name="clear"></a>
<pre><em>method</em> <strong style="color:#008AB8">clear</strong>(models="all")
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basepredictor.py#L204">[source]</a></div></pre>
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
<strong>models: str or iterable, optional (default="all")</strong>
<blockquote>
Model(s) to clear from the pipeline. If "all", clear all models.
</blockquote>
</tr>
</table>
</div>
<br />


<a name="get-class-weights"></a>
<pre><em>method</em> <strong style="color:#008AB8">get_class_weights</strong>(dataset="train")
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basepredictor.py#L204">[source]</a></div></pre>
<div style="padding-left:3%">
Return class weights for a balanced data set. Statistically, the class weights
 re-balance the data set so that the sampled data set represents the target
 population as closely as reasonably possible. The returned weights are inversely
 proportional to class frequencies in the selected data set. 
<br /><br />
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>dataset: str, optional (default="train")</strong>
<blockquote>
Data set from which to get the weights. Choose between "train", "test" or "dataset".
</blockquote>
</tr>
</table>
</div>
<br />


<a name="get-sample-weights"></a>
<pre><em>method</em> <strong style="color:#008AB8">get_sample_weights</strong>(dataset="train")
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basepredictor.py#L204">[source]</a></div></pre>
<div style="padding-left:3%">
Return sample weights for a balanced data set. Statistically, the sampling weights
 re-balance the data set so that the sampled data set represents the target
 population as closely as reasonably possible. The returned weights are the
 reciprocal of the likelihood of being sampled (i.e. selection probability) of
 the sampling unit.
<br /><br />
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>dataset: str, optional (default="train")</strong>
<blockquote>
Data set from which to get the weights. Choose between "train", "test" or "dataset".
</blockquote>
</tr>
</table>
</div>
<br />


<a name="log"></a>
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


<a name="report"></a>
<pre><em>method</em> <strong style="color:#008AB8">report</strong>(dataset="dataset", n_rows=None, filename=None)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L264">[source]</a></div></pre>
<div style="padding-left:3%">
Get an extensive profile analysis of the data. The report is rendered
 in HTML5 and CSS3 and saved to the `profile` attribute. Note that this method
 can be slow for `n_rows` > 10k.
<br /><br />
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>dataset: str, optional (default="dataset")</strong>
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


<a name="save"></a>
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
Name to save the file with. If None or "auto", use the name of the class.
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


<a name="save-data"></a>
<pre><em>method</em> <strong style="color:#008AB8">save_data</strong>(filename=None, dataset="dataset")
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L220">[source]</a></div></pre>
<div style="padding-left:3%">
Save data to a csv file.
<br><br>
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>filename: str or None, optional (default=None)</strong>
<blockquote>
Name of the saved file. None to use default name.
</blockquote>
<strong>dataset: str, optional (default="dataset")</strong>
<blockquote>
Data set to save.
</blockquote>
</tr>
</table>
</div>
<br>


<a name="scoring"></a>
<pre><em>method</em> <strong style="color:#008AB8">scoring</strong>(metric=None, dataset="test")
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basepredictor#L152">[source]</a></div></pre>
<div style="padding-left:3%">
Returns the scores of the models for a specific metric. If a model
 returns `XXX`, it means the metric failed for that specific model. This
 can happen if either the metric is unavailable for the task or if the
 model does not have a `predict_proba` method while the metric requires it.
<br /><br />
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>metric: str or None, optional (default=None)</strong>
<blockquote>
Name of the metric to calculate. Choose from any of sklearn's [SCORERS](https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules)
 or one of the following custom metrics:
<ul>
<li>"cm" for the confusion matrix.</li>
<li>"tn" for true negatives.</li>
<li>"fp" for false positives.</li>
<li>"fn" for false negatives.</li>
<li>"tp" for true positives.</li>
<li>"lift" for the lift metric.</li>
<li>"fpr" for the false positive rate.</li>
<li>"tpr" for true positive rate.</li>
<li>"sup" for the support metric.</li>
</ul>
If None, returns the models" final results (ignores the `dataset` parameter).
</blockquote>
<strong>dataset: str, optional (default="test")</strong>
<blockquote>
Data set on which to calculate the metric. Options are "train" or "test".
</blockquote>
</tr>
</table>
</div>
<br />


<a name="stats"></a>
<pre><em>method</em> <strong style="color:#008AB8">stats</strong>()
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L182">[source]</a></div></pre>
<div style="padding-left:3%">
Print out a list of basic information on the dataset.
</div>
<br /><br />


## Data cleaning
----------------

ATOMClassifier provides data cleaning methods to scale your features and handle
 missing values, categorical columns, outliers and unbalanced datasets. Calling
 on one of them will automatically apply the method on the dataset in the pipeline.

!!! tip
    Use the [report](#report) method to examine the data and help you
    determine suitable parameters for the data cleaning methods.
    

<table>
<tr>
<td><a href="#scale">scale</a></td>
<td>Scale all the features to mean=1 and std=0.</td>
</tr>

<tr>
<td><a href="#clean">clean</a></td>
<td>Applies standard data cleaning steps on the dataset.</td>
</tr>

<tr>
<td><a href="#impute">impute</a></td>
<td>Handle missing values in the dataset.</td>
</tr>

<tr>
<td><a href="#encode">encode</a></td>
<td>Encode categorical features.</td>
</tr>

<tr>
<td><a href="#outliers">outliers</a></td>
<td>Remove or replace outliers in the training set.</td>
</tr>

<tr>
<td><a href="#balance">balance</a></td>
<td>Balance the target classes in the training set.</td>
</tr>
</table>
<br>


<a name="scale"></a>
<pre><em>method</em> <strong style="color:#008AB8">scale</strong>()
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L362">[source]</a></div></pre>
<div style="padding-left:3%">
Scale the features to mean=1 and std=0.
</div>
<br /><br />


<a name="clean"></a>
<pre><em>method</em> <strong style="color:#008AB8">clean</strong>(prohibited_types=None, strip_categorical=True, maximum_cardinality=True,
             minimum_cardinality=True, missing_target=True, map_target=None) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L377">[source]</a></div></pre>
<div style="padding-left:3%">
Applies standard data cleaning steps on the dataset. These steps can include:

* Strip categorical features from white spaces.
* Removing columns with prohibited data types.
* Removing categorical columns with maximal cardinality.
* Removing columns with minimum cardinality.
* Removing rows with missing values in the target column.
* Encode the target column.

See [Cleaner](../data_cleaning/cleaner.md) for a description of the parameters.

</div>
<br />


<a name="impute"></a>
<pre><em>method</em> <strong style="color:#008AB8">impute</strong>(strat_num="drop", strat_cat="drop", min_frac_rows=0.5, min_frac_cols=0.5, missing=None) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L377">[source]</a></div></pre>
<div style="padding-left:3%">
Handle missing values according to the selected strategy. Also removes rows and
 columns with too many missing values. The imputer is fitted only on the training set
 to avoid data leakage. See [Imputer](../data_cleaning/imputer.md) for a description
 of the parameters. Note that since the Imputer can remove rows from both train and
 test set, the set's sizes may change to keep ATOM's `test_size` ratio.
</div>
<br />


<a name="encode"></a>
<pre><em>method</em> <strong style="color:#008AB8">encode</strong>(strategy="LeaveOneOut", max_onehot=10, frac_to_other=None)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L413">[source]</a></div></pre>
<div style="padding-left:3%">
Perform encoding of categorical features. The encoding type depends on the
 number of unique values in the column:
<ul>
<li>If n_unique=2, use Label-encoding.</li>
<li>If 2 < n_unique <= max_onehot, use OneHot-encoding.</li>
<li>If n_unique > max_onehot, use `strategy`-encoding.</li>
</ul>
Also replaces classes with low occurrences with the value "other" in
 order to prevent too high cardinality. Categorical features are defined as
 all columns whose dtype.kind not in "ifu". Will raise an error if it encounters
 missing values or unknown classes when transforming. The encoder is fitted only
 on the training set to avoid data leakage. See [Encoder](../data_cleaning/encoder.md)
 for a description of the parameters.
</div>
<br />


<a name="outliers"></a>
<pre><em>method</em> <strong style="color:#008AB8">outliers</strong>(strategy="drop", max_sigma=3, include_target=False) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L447">[source]</a></div></pre>
<div style="padding-left:3%">
Remove or replace outliers in the training set. Outliers are defined as values that
 lie further than `max_sigma` * standard_deviation away from the mean of the column.
 Only outliers from the training set are removed to maintain an original sample of
 target values in the test set. Ignores categorical columns. See
 [Outliers](../data_cleaning/outliers.md) for a description of the parameters.
</div>
<br />


<a name="balance"></a>
<pre><em>method</em> <strong style="color:#008AB8">balance</strong>(strategy="ADASYN", **kwargs)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L477">[source]</a></div></pre>
<div style="padding-left:3%">
Balance the number of instances per target category in the training set.
 Only the training set is balanced in order to maintain the original distribution
 of target classes in the test set. See [Balancer](../data_cleaning/balancer.md)
 for a description of the parameters.
</div>
<br>



## Feature engineering
----------------------

To further pre-process the data you can create new non-linear features transforming
 the existing ones or, if your dataset is too large, remove features using one
 of the provided strategies.

<table>
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



<a name="feature-generation"></a>
<pre><em>method</em> <strong style="color:#008AB8">feature_generation</strong>(strategy="DFS", n_features=None, generations=20, population=500, operators=None)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L511">[source]</a></div></pre>
<div style="padding-left:3%">
Use Deep feature Synthesis or a genetic algorithm to create new combinations
 of existing features to capture the non-linear relations between the original
 features. See [FeatureGenerator](../feature_engineering/feature_generator.md) for
 a description of the parameters. Attributes created by the class are attached to
 the ATOM instance.
</div>
<br />


<a name="feature-selection"></a>
<pre><em>method</em> <strong style="color:#008AB8">feature_selection</strong>(strategy=None, solver=None, n_features=None,
                         max_frac_repeated=1., max_correlation=1., **kwargs) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L547">[source]</a></div></pre>
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
    <li>When strategy="univariate" and solver=None, [f_classif](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html)
        will be used as default solver.</li>
    <li>When strategy is one of "SFM", "RFE" or "RFECV" and the solver is one of 
        ATOM's models, the algorithm will automatically select the classifier (no need to add `_class` to the solver).</li>
    <li>When strategy is one of "SFM", "RFE" or "RFECV" and solver=None, ATOM will
         use the winning model (if it exists) as solver.</li>
    <li>When strategy="RFECV", ATOM will use the metric in the pipeline (if it exists)
        as the scoring parameter (only if not specified manually).</li>

</div>
<br>



## Training
----------

The training methods are where the models are fitted to the data and their
 performance is evaluated according to the selected metric. ATOMClassifier contains
 three methods to call the training classes from the ATOM package. All relevant
 attributes and methods from the training classes are attached to ATOMClassifier for
 convenience. These include the errors, winner and results attributes, the
 [`models`](../../../user_guide/#models), and the
 [prediction](../../../user_guide/#predicting) and
 [plotting](../../../user_guide/#plots) methods.


<table>
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
<pre><em>method</em> <strong style="color:#008AB8">run</strong>(models, metric=None, greater_is_better=True, needs_proba=False, needs_threshold=False,
           n_calls=10, n_initial_points=5, est_params={}, bo_params={}, bagging=None) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L702">[source]</a></div></pre>
<div style="padding-left:3%">
Runs a [TrainerClassifier](../training/trainerclassifier.md) instance.
</div>
<br />


<a name="successive-halving"></a>
<pre><em>method</em> <strong style="color:#008AB8">successive_halving</strong>(models, metric=None, greater_is_better=True, needs_proba=False,
                          needs_threshold=False, skip_runs=0, n_calls=0, n_initial_points=5,
                          est_params={}, bo_params={}, bagging=None) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L740">[source]</a></div></pre>
<div style="padding-left:3%">
Runs a [SuccessiveHalvingClassifier](../training/successivehalvingclassifier.md) instance.
</div>
<br />


<a name="train-sizing"></a>
<pre><em>method</em> <strong style="color:#008AB8">train_sizing</strong>(models, metric=None, greater_is_better=True, needs_proba=False,
                    needs_threshold=False, train_sizes=np.linspace(0.2, 1.0, 5), n_calls=0,
                    n_initial_points=5, est_params={}, bo_params={}, bagging=None) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L789">[source]</a></div></pre>
<div style="padding-left:3%">
Runs a [TrainSizingClassifier](../training/trainsizingclassifier.md) instance.
</div>
<br />



## Example
---------
```python
from sklearn.datasets import load_breast_cancer
from atom import ATOMClassifier

X, y = load_breast_cancer(return_X_y=True)

# Initialize class
atom = ATOMClassifier(X, y, logger="auto", n_jobs=2, verbose=2)

# Apply data cleaning methods
atom.outliers(strategy="min_max", max_sigma=2)
atom.balance(strategy="smote", sampling_strategy=0.7)

# Fit the models to the data
atom.run(
    models=["QDA", "CatB"],
    metric="precision",
    n_calls=25,
    n_initial_points=10,
    bo_params={"cv": 1},
    bagging=4
)

# Analyze the results
print(f"The winning model is: {atom.winner.name}")
print(atom.results)

# Make some plots
atom.palette = "Blues"
atom.plot_roc(figsize=(9, 6), filename="roc.png")  
atom.CatB.plot_feature_importance(filename="catboost_feature_importance.png")

# Run an extra model
atom.run(
    models="LR",
    metric="precision",
    n_calls=25,
    n_initial_points=10,
    bo_params={"cv": 1},
    bagging=4
)

# Get the predictions for the best model on new data
predictions = atom.predict(X_new)
```