# ATOMRegressor
----------------

<pre><em>class</em> atom.api.<strong style="color:#008AB8">ATOMRegressor</strong>(*arrays, y=-1, n_rows=1, test_size=0.2, logger=None,
                             n_jobs=1, warnings=True, verbose=0, random_state=None)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/api.py#L267">[source]</a></div></pre>
ATOMRegressor is ATOM's wrapper for regression tasks. Use this class to easily apply
all data transformations and model management provided by the package on a given
dataset. Note that contrary to sklearn's API, an ATOMRegressor instance already
contains the dataset on which we want to perform the analysis. Calling a method will
automatically apply it on the dataset it contains.

You can [predict](../../../user_guide/#predicting), [plot](../../../user_guide/#plots)
 and call any [model](../../../user_guide/#models) from atom. Read more in the
 [user guide](../../../user_guide/#first-steps).
<table>
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>*arrays: sequence of indexables</strong>
<blockquote>
Dataset containing features and target. Allowed formats are:
<ul>
<li>X</li>
<li>X, y</li>
<li>train, test</li>
<li>X_train, X_test, y_train, y_test</li>
<li>(X_train, y_train), (X_test, y_test)</li>
</ul>
X, train, test: dict, list, tuple, np.ndarray or pd.DataFrame<br>
&nbsp;&nbsp;&nbsp;&nbsp;
Feature set with shape=(n_features, n_samples).<br><br>
y: int, str or sequence<br>
<ul>
<li>If int: Position of the target column in X.</li>
<li>If str: Name of the target column in X.</li>
<li>Else: Target column with shape=(n_samples,).</li>
</ul>
</blockquote>
<strong>y: int, str or sequence, optional (default=-1)</strong>
<blockquote>
Target column in X. Ignored if provided through <code>arrays</code>.
<ul>
<li>If int: Position of the target column in X.</li>
<li>If str: Name of the target column in X.</li>
<li>Else: Target column with shape=(n_samples,).</li>
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
<li>If True: Default warning action (equal to "default").</li>
<li>If False: Suppress all warnings (equal to "ignore").</li>
<li>If str: One of the actions in python's warnings environment.</li>
</ul>
Note that changing this parameter will affect the <code>PYTHONWARNINGS</code> environment.
<br>
Note that ATOM can't manage warnings that go directly from C++ code to the
 stdout/stderr.
</blockquote>
<strong>logger: str, Logger or None, optional (default=None)</strong>
<blockquote>
<ul>
<li>If None: Doesn't save a logging file.</li>
<li>If str: Name of the logging file. Use "auto" for default name.</li>
<li>Else: Python <code>logging.Logger</code> instance.</li>
</ul>
The default name consists of the class' name followed by the
timestamp of the logger's creation.
</blockquote>
<strong>random_state: int or None, optional (default=None)</strong>
<blockquote>
Seed used by the random number generator. If None, the random number
generator is the <code>RandomState</code> instance used by <code>numpy.random</code>.
</blockquote>
</td>
</tr>
</table>
<br>


## Magic methods
----------------

The class contains some magic methods to help you access some of its
elements faster. Note that methods that apply on the pipeline can return
different results per branch.

* **\__repr__:** Prints an overview of atom's branches, models, metric and errors.
* **\__len__:** Returns the length of the pipeline.
* **\__iter__:** Iterate over the pipeline's transformers.
* **\__contains__:** Checks if the provided item is a column in the dataset.
* **\__getitem__:** If int, return the i-th transformer in the pipeline.
  If str, access a column in the dataset.

<br>


## Attributes
-------------

### Data attributes

The dataset can be accessed at any time through multiple attributes, e.g. calling
`trainer.train` will return the training set. The data can also be changed through
these attributes, e.g. `trainer.test = atom.test.drop(0)` will drop the first row
from the test set. Updating one of the data attributes will automatically update the
rest as well. Changing the branch will also change the response from these attributes
accordingly.

<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Attributes:</strong></td>
<td width="75%" style="background:white;">
<strong>pipeline: pd.Series</strong>
<blockquote>
Series containing all transformers fitted on the data in the current branch.
Use this attribute only to access the individual instances. To visualize the
pipeline, use the status method from the branch or the
<a href="../../plots/plot_pipeline">plot_pipeline</a> method.
</blockquote>
<strong>feature_importance: list</strong>
<blockquote>
Features ordered by most to least important. This attribute is created
after running the <a href="#feature-selection">feature_selection</a>,
<a href="../../plots/plot_permutation_importance">plot_permutation_importance</a>
or <a href="../../plots/plot_feature_importance">plot_feature_importance</a> methods.
</blockquote>
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
Dataset's shape: (n_rows x n_columns) or
(n_rows, (shape_sample), n_cols) for deep learning datasets.
</blockquote>
<strong>columns: list</strong>
<blockquote>
Names of the columns in the dataset.
</blockquote>
<strong>n_columns: int</strong>
<blockquote>
Number of columns in the dataset.
</blockquote>
<strong>features: list</strong>
<blockquote>
Names of the features in the dataset.
</blockquote>
<strong>n_features: int</strong>
<blockquote>
Number of features in the dataset.
</blockquote>
<strong>target: str</strong>
<blockquote>
Name of the target column.
</blockquote>
<strong>scaled: bool</strong>
<blockquote>
Whether the feature set is scaled. It is considered scaled when
it has mean=0 and std=1, or when atom has a scaler in the pipeline.
</blockquote>
<strong>duplicates: int</strong>
<blockquote>
Number of duplicate rows in the dataset.
</blockquote>
<strong>nans: pd.Series</strong>
<blockquote>
Columns with the number of missing values in them.
</blockquote>
<strong>n_nans: int</strong>
<blockquote>
Number of samples containing missing values.
</blockquote>
<strong>numerical: list</strong>
<blockquote>
Names of the numerical features in the dataset.
</blockquote>
<strong>n_numerical: int</strong>
<blockquote>
Number of numerical features in the dataset.
</blockquote>
<strong>categorical: list</strong>
<blockquote>
Names of the categorical features in the dataset.
</blockquote>
<strong>n_categorical: int</strong>
<blockquote>
Number of categorical features in the dataset.
</blockquote>
<strong>outliers: pd.Series</strong>
<blockquote>
Columns in training set with amount of outlier values.
</blockquote>
<strong>n_outliers: int</strong>
<blockquote>
Number of samples in the training set containing outliers.
</blockquote>
</td></tr>
</table>
<br>


### Utility attributes

<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Attributes:</strong></td>
<td width="75%" style="background:white;">
<strong>missing: list</strong>
<blockquote>
List of values that are considered "missing" (used by the <a href="#clean">clean</a>
and <a href="#impute">impute</a> methods). Default values are: "", "?", "None", "NA",
"nan", "NaN" and "inf". Note that <code>None</code>, <code>NaN</code>, <code>+inf</code>
and <code>-inf</code> are always considered missing since they are incompatible with
sklearn estimators.
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
<strong>winner: <a href="../../../user_guide/#models">model</a></strong>
<blockquote>
Model subclass that performed best on the test set.
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

The ATOMRegressor class contains a variety of methods to help you handle the data and
inspect the pipeline.

<table>
<tr>
<td><a href="#add">add</a></td>
<td>Add a transformer to the current branch.</td>
</tr>

<tr>
<td><a href="#apply">apply</a></td>
<td>Apply a function to the dataset.</td>
</tr>

<tr>
<td><a href="#automl">automl</a></td>
<td>Use AutoML to search for an optimized pipeline.</td>
</tr>

<tr>
<td width="15%"><a href="#canvas">canvas</a></td>
<td>Create a figure with multiple plots.</td>
</tr>

<tr>
<td width="15%"><a href="#delete">delete</a></td>
<td>Remove a model from the pipeline.</td>
</tr>

<tr>
<td width="15%"><a href="#distribution">distribution</a></td>
<td>Get statistics on a column's distribution.</td>
</tr>

<tr>
<td><a href="#drop">drop</a></td>
<td>Drop columns from the dataset.</td>
</tr>

<tr>
<td><a href="#export-pipeline">export_pipeline</a></td>
<td>Export atom's pipeline to a sklearn's Pipeline object.</td>
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
<td><a href="#reset-aesthetics">reset_aesthetics</a></td>
<td>Reset the plot aesthetics to their default values.</td>
</tr>

<tr>
<td><a href="#reset-predictions">reset_predictions</a></td>
<td>Clear the prediction attributes from all models.</td>
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
<td><a href="#stacking">stacking</a></td>
<td>Add a Stacking instance to the models in the pipeline.</td>
</tr>

<tr>
<td width="15%"><a href="#stats">stats</a></td>
<td>Print out a list of basic statistics on the dataset.</td>
</tr>

<tr>
<td width="15%"><a href="#status">status</a></td>
<td>Get an overview of atom's branches, models and errors.</td>
</tr>

<tr>
<td><a href="#voting">voting</a></td>
<td>Add a Voting instance to the models in the pipeline.</td>
</tr>
</table>
<br>


<a name="add"></a>
<pre><em>method</em> <strong style="color:#008AB8">add</strong>(transformer, columns=None, train_only=False)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L597">[source]</a></div></pre>
Add a transformer to the current branch. If the transformer is
not fitted, it is fitted on the complete training set. Afterwards,
the data set is transformed and the transformer is added to atom's
pipeline. If the transformer is a sklearn Pipeline, every transformer
is merged independently with atom.

!!!note
    If the transformer doesn't return a dataframe,  the column naming happens as
    follows. If the transformer returns the same number of columns, the names are
    kept equal. If the number of columns change, old columns will keep their name
    (as long as the column is unchanged) and new columns will receive the name
    `Feature n`, where n stands for the n-th feature. This means that a transformer
    should only transform, add or drop columns, not combinations of these.

!!!note
    If the transformer has a `n_jobs` and/or `random_state` parameter and it
    is left to its default value, it adopts atom's value.

<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>transformer: estimator</strong>
<blockquote>
Transformer to add to the pipeline. Should implement a <code>transform</code> method.
</blockquote>
<strong>columns: int, str, slice, sequence or None, optional (default=None)</strong>
<blockquote>
Names or indices of the columns in the dataset to transform.
If None, transform all columns.
</blockquote>
<strong>train_only: bool, optional (default=False)</strong>
<blockquote>
Whether to apply the transformer only on the train set or
on the complete dataset.
</blockquote>
</tr>
</table>
<br />


<a name="apply"></a>
<pre><em>method</em> <strong style="color:#008AB8">apply</strong>(func, column)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L556">[source]</a></div></pre>
Transform one column in the dataset using a function (can
be a lambda). If the provided column is present in the dataset,
that same column is transformed. If it's not a column in the
dataset, a new column with that name is created. The input of
function is the complete dataset as pd.DataFrame.

!!! note
    This approach is preferred over changing the dataset directly
    through the property's `@setter` since the transformation
    is saved to atom's pipeline.

<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>func: callable</strong>
<blockquote>
Function to apply to the dataset.
</blockquote>
<strong>column: int or str</strong>
<blockquote>
Name or index of the column in the dataset to create or transform.
</blockquote>
</tr>
</table>
<br />


<a name="automl"></a>
<pre><em>method</em> <strong style="color:#008AB8">automl</strong>(**kwargs)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L940">[source]</a></div></pre>
Uses the [TPOT](http://epistasislab.github.io/tpot/) package to perform
an automated search of transformers and a final estimator that maximizes
a metric on the dataset. The resulting transformations and estimator are
merged with atom's pipeline. The tpot instance can be accessed through the
`tpot` attribute. Read more in the [user guide](../../../user_guide/#automl).
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>**kwargs</strong>
<blockquote>
Keyword arguments for tpot's regressor.
</blockquote>
</tr>
</table>
<br />


<a name="canvas"></a>
<pre><em>method</em> <strong style="color:#008AB8">canvas</strong>(nrows=1, ncols=2, title=None, figsize=None, filename=None, display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L422">[source]</a></div></pre>
This `@contextmanager` allows you to draw many plots in one figure. The default
option is to add two plots side by side. See the [user guide](../../../user_guide/#canvas)
for an example use case.
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>nrows: int, optional (default=1)</strong>
<blockquote>
Number of plots in length.
</blockquote>
<strong>ncols: int, optional (default=2)</strong>
<blockquote>
Number of plots in width.
</blockquote>
<strong>title: str or None, optional (default=None)</strong>
<blockquote>
Plot's title. If None, no title is displayed.
</blockquote>
<strong>figsize: tuple or None, optional (default=None)</strong>
<blockquote>
Figure's size, format as (x, y). If None, adapts size to the number of plots
 in the canvas.
</blockquote>
<strong>filename: str or None, optional (default=None)</strong>
<blockquote>
Name of the file. If None, the figure is not saved.
</blockquote>
<strong>display: bool, optional (default=True)</strong>
<blockquote>
Whether to render the plot.
</blockquote>
</tr>
</table>
<br />


<a name="delete"></a>
<pre><em>method</em> <strong style="color:#008AB8">delete</strong>(models=None)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basepredictor.py#L411">[source]</a></div></pre>
Removes a model from the pipeline. If all models in the pipeline are removed,
the metric is reset. Use this method to remove unwanted models or to free
some memory before saving the instance.
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>models: str or sequence, optional (default=None)</strong>
<blockquote>
Name of the models to clear from the pipeline. If None, clear all models.
</blockquote>
</tr>
</table>
<br />


<a name="distribution"></a>
<pre><em>method</em> <strong style="color:#008AB8">distribution</strong>(column=0)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L310">[source]</a></div></pre>
Compute the [KS-statistic](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)
for various distributions against a column in the dataset. Missing values are ignored.

!!!tip
    Use the [plot_distribution](../../plots/plot_distribution) method to plot
    the column's distribution.

<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>column: int or str, optional (default=0)</strong>
<blockquote>
Index or name of the column to get the statistics from. Only
numerical columns are accepted.
</blockquote>
</tr>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Returns:</strong></td>
<td width="75%" style="background:white;">
<strong>stats: pd.DataFrame</strong>
<blockquote>
Dataframe with the statistic results.
</blockquote>
</tr>
</table>
<br />


<a name="drop"></a>
<pre><em>method</em> <strong style="color:#008AB8">drop</strong>(columns)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L526">[source]</a></div></pre>
Drop columns from the dataset.

!!! note
    This approach is preferred over dropping columns from the
    dataset directly through the property's `@setter` since
    the transformation is saved to atom's pipeline.

<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>columns: int, str, slice or sequence</strong>
<blockquote>
Names or indices of the columns to drop.
</blockquote>
</tr>
</table>
<br />


<a name="export-pipeline"></a>
<pre><em>method</em> <strong style="color:#008AB8">export_pipeline</strong>(model=None)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L482">[source]</a></div></pre>
Export atom's pipeline to a sklearn's Pipeline. Optionally, you can add a model
as final estimator. If the model needs feature scaling and there is no scaler in
the pipeline, a [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
is added. The returned pipeline is already fitted.
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>model: str or None, optional (default=None)</strong>
<blockquote>
Name of the model to add as a final estimator to the
pipeline. If None, no model is added.
</blockquote>
</tr>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Returns:</strong></td>
<td width="75%" style="background:white;">
<strong>pipeline: <a href="https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html">Pipeline</a></strong>
<blockquote>
Pipeline in the current branch as a sklearn object.
</blockquote>
</tr>
</table>
<br />


<a name="log"></a>
<pre><em>method</em> <strong style="color:#008AB8">log</strong>(msg, level=0)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basetransformer.py#L318">[source]</a></div></pre>
Write a message to the logger and print it to stdout.
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
Minimum verbosity level to print the message.
</blockquote>
</tr>
</table>
<br />


<a name="report"></a>
<pre><em>method</em> <strong style="color:#008AB8">report</strong>(dataset="dataset", n_rows=None, filename=None)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L352">[source]</a></div></pre>
Create an extensive profile analysis report of the data. The report is rendered
in HTML5 and CSS3. Note that this method can be slow for `n_rows` > 10k.
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>dataset: str, optional (default="dataset")</strong>
<blockquote>
Data set to get the report from.
</blockquote>
<strong>n_rows: int or None, optional (default=None)</strong>
<blockquote>
Number of (randomly picked) rows to process. None for all rows.
</blockquote>
<strong>filename: str or None, optional (default=None)</strong>
<blockquote>
Name to save the file with (as .html). None to not save anything.
</blockquote>
</tr>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Returns:</strong></td>
<td width="75%" style="background:white;">
<strong>report: <a href="https://pandas-profiling.github.io/pandas-profiling/docs/master/rtd/pages/api/_autosummary/pandas_profiling.profile_report.ProfileReport.html#pandas_profiling.profile_report.ProfileReport">ProfileReport</a></strong>
<blockquote>
Created profile object.
</blockquote>
</tr>
</table>
<br />


<a name="reset-aesthetics"></a>
<pre><em>method</em> <strong style="color:#008AB8">reset_aesthetics</strong>()
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L200">[source]</a></div></pre>
Reset the [plot aesthetics](../../../user_guide/#aesthetics) to their default values.
<br /><br /><br />


<a name="reset-predictions"></a>
<pre><em>method</em> <strong style="color:#008AB8">reset_predictions</strong>()
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basepredictor.py#L122">[source]</a></div></pre>
Clear the [prediction attributes](../../../user_guide/#predicting) from all models.
Use this method to free some memory before saving the trainer.
<br /><br /><br />


<a name="save"></a>
<pre><em>method</em> <strong style="color:#008AB8">save</strong>(filename=None, save_data=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basetransformer.py#L339">[source]</a></div></pre>
Save the instance to a pickle file. Remember that the class contains the complete
dataset as attribute, so the file can become large for big datasets! To avoid this,
use `save_data=False`.
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>filename: str or None, optional (default=None)</strong>
<blockquote>
Name to save the file with. None or "auto" to save with
 the __name__ of the class.
</blockquote>
<strong>save_data: bool, optional (default=True)</strong>
<blockquote>
Whether to save the data as an attribute of the instance. If False, remember to
 add the data to <a href="../../ATOM/atomloader">ATOMLoader</a> when loading the file.
</blockquote>
</tr>
</table>
<br>


<a name="save-data"></a>
<pre><em>method</em> <strong style="color:#008AB8">save_data</strong>(filename=None, dataset="dataset")
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L457">[source]</a></div></pre>
Save the data in the current branch to a csv file.
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>filename: str or None, optional (default=None)</strong>
<blockquote>
Name to save the file with. None or "auto" for default name.
</blockquote>
<strong>dataset: str, optional (default="dataset")</strong>
<blockquote>
Data set to save.
</blockquote>
</tr>
</table>
<br>


<a name="scoring"></a>
<pre><em>method</em> <strong style="color:#008AB8">scoring</strong>(metric=None, dataset="test", **kwargs)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basepredictor.py#L363">[source]</a></div></pre>
Print all the models' scoring for a specific metric.
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>metric: str or None, optional (default=None)</strong>
<blockquote>
Name of the metric to calculate. Choose from any of sklearn's regression <a href="https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules">SCORERS</a>.
If None, returns the models' final results (ignores the <code>dataset</code> parameter).
</blockquote>
<strong>dataset: str, optional (default="test")</strong>
<blockquote>
Additional keyword arguments for the metric function.
</blockquote>
</table>
<br />


<a name="stacking"></a>
<pre><em>method</em> <strong style="color:#008AB8">stacking</strong>(models=None, estimator=None, stack_method="auto", passthrough=False)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basepredictor#L275">[source]</a></div></pre>
Add a Stacking instance to the models in the pipeline.
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>models: sequence or None, optional (default=None)</strong>
<blockquote>
Models that feed the stacking. If None, it selects all models
depending on the current branch.
</blockquote>
<strong>estimator: str, callable or None, optional (default=None)</strong>
<blockquote>
The final estimator, which is used to combine the base estimators. If str,
choose from ATOM's <a href="../../../user_guide/#predefined-models">predefined models</a>.
If None, <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html">Ridge</a> is selected.
</blockquote>
<strong>stack_method: str, optional (default="auto")</strong>
<blockquote>
Methods called for each base estimator. If "auto", it will try to 
invoke <code>predict_proba</code>, <code>decision_function</code>
or <code>predict</code> in that order.
</blockquote>
<strong>passthrough: bool, optional (default=False)</strong>
<blockquote>
When False, only the predictions of estimators are used
as training data for the final estimator. When True, the
estimator is trained on the predictions as well as the
original training data. The passed dataset is scaled
if any of the models require scaled features and they are
not already.
</blockquote>
</tr>
</table>
<br />


<a name="stats"></a>
<pre><em>method</em> <strong style="color:#008AB8">stats</strong>()
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L256">[source]</a></div></pre>
Print basic information about the dataset.
<br /><br /><br />


<a name="status"></a>
<pre><em>method</em> <strong style="color:#008AB8">status</strong>()
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L251">[source]</a></div></pre>
Get an overview of the branches, models and errors in the current instance.
This method prints the same information as atom's \__repr__ but will also
save it to the logger.
<br /><br /><br />


<a name="voting"></a>
<pre><em>method</em> <strong style="color:#008AB8">voting</strong>(models=None, weights=None)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basepredictor#L242">[source]</a></div></pre>
Add a Voting instance to the models in the pipeline.
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>models: sequence or None, optional (default=None)</strong>
<blockquote>
Models that feed the voting. If None, it selects all models
depending on the current branch.
</blockquote>
<strong>weights: sequence or None, optional (default=None)</strong>
<blockquote>
Sequence of weights (int or float) to weight the
 occurrences of predicted class labels (hard voting)
 or class probabilities before averaging (soft voting).
 Uses uniform weights if None.
</blockquote>
</tr>
</table>
<br /><br />




## Data cleaning
----------------

ATOMRegressor provides data cleaning methods to scale your features and handle
missing values, categorical columns and outliers. Calling on one of them will
automatically apply the method on the dataset in the pipeline.

!!! tip
    Use the [report](#report) method to examine the data and help you
    determine suitable parameters for the data cleaning methods.
    

<table>
<tr>
<td><a href="#scale">scale</a></td>
<td>Scale the dataset.</td>
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
<td><a href="#prune">prune</a></td>
<td>Prune outliers from the training set.</td>
</tr>
</table>
<br>


<a name="scale"></a>
<pre><em>method</em> <strong style="color:#008AB8">scale</strong>(strategy="standard")
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L640">[source]</a></div></pre>
Applies one of sklearn's scalers. Non-numerical columns are ignored (instead
of raising an exception). See the [Scaler](../data_cleaning/scaler.md) class.
<br /><br /><br />


<a name="clean"></a>
<pre><em>method</em> <strong style="color:#008AB8">clean</strong>(prohibited_types=None, strip_categorical=True, maximum_cardinality=True,
             minimum_cardinality=True, drop_duplicates=False, missing_target=True, encode_target=None) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L655">[source]</a></div></pre>
Applies standard data cleaning steps on the dataset. Use the parameters
to choose which transformations to perform. The available steps are:

* Drop columns with prohibited data types.
* Drop categorical columns with maximal cardinality.
* Drop columns with minimum cardinality.
* Strip categorical features from white spaces.
* Drop duplicate rows.
* Drop rows with missing values in the target column.
* Encode the target column.

See [Cleaner](../data_cleaning/cleaner.md) for a description of the parameters.
<br /><br /><br />


<a name="impute"></a>
<pre><em>method</em> <strong style="color:#008AB8">impute</strong>(strat_num="drop", strat_cat="drop", min_frac_rows=None, min_frac_cols=None, missing=None) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L704">[source]</a></div></pre>
Impute or remove missing values according to the selected strategy. Also removes
rows and columns with too many missing values. The imputer is fitted only on the
training set to avoid data leakage. Use the `missing` attribute to customize what
are considered "missing values". See [Imputer](../data_cleaning/imputer.md) for a
description of the parameters. Note that since the Imputer can remove rows from both
the train and test set, the size of the sets may change after the tranformation.
<br /><br /><br />


<a name="encode"></a>
<pre><em>method</em> <strong style="color:#008AB8">encode</strong>(strategy="LeaveOneOut", max_onehot=10, frac_to_other=None)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L738">[source]</a></div></pre>
Perform encoding of categorical features. The encoding type depends on the
 number of unique values in the column:
<ul>
<li>If n_unique=2, use Label-encoding.</li>
<li>If 2 < n_unique <= max_onehot, use OneHot-encoding.</li>
<li>If n_unique > max_onehot, use `strategy`-encoding.</li>
</ul>
Also replaces classes with low occurrences with the value `other` in
order to prevent too high cardinality. Categorical features are defined as
all columns whose dtype.kind not in `ifu`. Will raise an error if it encounters
missing values or unknown classes when transforming. The encoder is fitted only
on the training set to avoid data leakage. See [Encoder](../data_cleaning/encoder.md)
for a description of the parameters.

!!!note
    This method only encodes the categorical features. It does not encode
    the target column! Use the [clean](#clean) method for that.

<br /><br /><br />


<a name="prune"></a>
<pre><em>method</em> <strong style="color:#008AB8">prune</strong>(strategy="z-score", method="drop", max_sigma=3, include_target=False, **kwargs)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L774">[source]</a></div></pre>
Prune outliers from the training set. The definition of outlier depends
on the selected strategy and can greatly differ from one each other. 
Ignores categorical columns. Only outliers from the training set are pruned
in order to maintain the original distribution of samples in the test
set. Ignores categorical columns. See [Pruner](../data_cleaning/pruner.md)
for a description of the parameters.
<br /><br /><br />



## Feature engineering
----------------------

To further pre-process the data, you can create new non-linear features transforming
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
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L840">[source]</a></div></pre>
Use Deep feature Synthesis or a genetic algorithm to create new combinations
of existing features to capture the non-linear relations between the original
features. See [FeatureGenerator](../feature_engineering/feature_generator.md) for
a description of the parameters. Attributes created by the class are attached to
atom.
<br /><br /><br />


<a name="feature-selection"></a>
<pre><em>method</em> <strong style="color:#008AB8">feature_selection</strong>(strategy=None, solver=None, n_features=None,
                         max_frac_repeated=1., max_correlation=1., **kwargs) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L879">[source]</a></div></pre>
Remove features according to the selected strategy. Ties between features with
equal scores are broken in an unspecified way. Also removes features with
too low variance and finds pairs of collinear features based on the Pearson
correlation coefficient. For each pair above the specified limit (in terms of
absolute value), it removes one of the two. See [FeatureSelector](../feature_engineering/feature_selector.md)
for a description of the parameters. Plotting methods and attributes created
by the class are attached to atom.

!!! note
    <ul>
    <li>When strategy="univariate" and solver=None, [f_classif](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html)
        is used as default solver.</li>
    <li>When strategy is one of SFM, RFE, RFECV or SFS and the solver is one of 
        ATOM's [predefined models](../../../user_guide/#predefined-models), the
        algorithm will automatically select the classifier (no need to add `_class`
        to the solver).</li>
    <li>When strategy is one of SFM, RFE, RFECV or SFS and solver=None, ATOM will
         use the winning model (if it exists) as solver.</li>
    <li>When strategy is RFECV or SFS, ATOM will use the metric in the pipeline
        (if it exists) as the scoring parameter (only if not specified).</li>

<br /><br />



## Training
----------

The training methods are where the models are fitted to the data and their
performance is evaluated according to the selected metric. There are three
methods to call the three different training approaches in ATOM. All relevant
attributes and methods from the training classes are attached to atom for
convenience. These include the errors, winner and results attributes, as well
as the [models](../../../user_guide/#models), and the
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
           n_calls=10, n_initial_points=5, est_params=None, bo_params=None, bagging=0) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L1095">[source]</a></div></pre>
Runs a [DirectRegressor](../training/directregressor.md) instance.
<br /><br /><br />


<a name="successive-halving"></a>
<pre><em>method</em> <strong style="color:#008AB8">successive_halving</strong>(models, metric=None, greater_is_better=True, needs_proba=False,
                          needs_threshold=False, skip_runs=0, n_calls=0, n_initial_points=5,
                          est_params=None, bo_params=None, bagging=0) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L1134">[source]</a></div></pre>
Runs a [SuccessiveHalvingRegressor](../training/successivehalvingregressor.md) instance.
<br /><br /><br />


<a name="train-sizing"></a>
<pre><em>method</em> <strong style="color:#008AB8">train_sizing</strong>(models, metric=None, greater_is_better=True, needs_proba=False,
                    needs_threshold=False, train_sizes=np.linspace(0.2, 1.0, 5), n_calls=0,
                    n_initial_points=5, est_params=None, bo_params=None, bagging=0) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L1180">[source]</a></div></pre>
Runs a [TrainSizingRegressor](../training/trainsizingregressor.md) instance.
<br /><br /><br />



## Example
---------
```python
from sklearn.datasets import load_boston
from atom import ATOMRegressor

X, y = load_boston(return_X_y=True)

# Initialize class
atom = ATOMRegressor(X, y, logger="auto", n_jobs=2, verbose=2)

# Apply data cleaning methods
atom.prune(strategy="z-score", method="min_max", max_sigma=2, include_target=True)

# Fit the models to the data
atom.run(
    models=["OLS", "BR", "CatB"],
    metric="MSE",
    n_calls=25,
    n_initial_points=10,
    bo_params={"cv": 1},
    bagging=4,
)

# Analyze the results
print(f"The winning model is: {atom.winner.name}")
print(atom.results)

# Make some plots
atom.plot_errors(figsize=(9, 6), filename="errors.png")  
atom.CatB.plot_feature_importance(filename="catboost_feature_importance.png")

# Run an extra model
atom.run(
    models="MLP",
    metric="MSE",
    n_calls=25,
    n_initial_points=10,
    bo_params={"cv": 1},
    bagging=4,
)

# Get the predictions for the best model on new data
predictions = atom.predict(X_new)
```