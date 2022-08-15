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

The dataset can be accessed at any time through multiple attributes,
e.g. calling `atom.train` will return the training set. Updating one
of the data attributes will automatically update the rest as well.
Changing the branch will also change the response from these attributes
accordingly.

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

:: table:
    - attributes:
        from_docstring: False
        include:
            - models
            - metric
            - errors
            - n_nans
            - winners
            - winner
            - outliers
            - n_outliers
            - classes
            - n_classes

<br>

### Plot attributes
 
:: table:
    - attributes:
        from_docstring: False
        include:
            - style
            - palette
            - title_fontsize
            - label_fontsize
            - tick_fontsize


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


<a name="canvas"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">canvas</strong>(nrows=1,
ncols=2, title=None, figsize=None, filename=None, display=True)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L427">[source]</a>
</span>
</div>
This `@contextmanager` allows you to draw many plots in one figure.
The default option is to add two plots side by side. See the
[user guide](../../../user_guide/plots/#canvas) for an example.
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>nrows: int, default=1</strong><br>
Number of plots in length.
</p>
<p>
<strong>ncols: int, default=2</strong><br>
Number of plots in width.
</p>
<p>
<strong>title: str or None, default=None</strong><br>
Plot's title. If None, no title is displayed.
</p>
<p>
<strong>figsize: tuple or None, default=None</strong><br>
Figure's size, format as (x, y). If None, it adapts the size to the
number of plots in the canvas.
</p>
<p>
<strong>filename: str or None, default=None</strong><br>
Name of the file. Use "auto" for automatic naming.
If None, the figure is not saved.
</p>
<p>
<strong>display: bool, default=True</strong><br>
Whether to render the plot.
</p>
</td>
</tr>
</table>
<br />


<a name="clear"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">clear</strong>()
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/baserunner.py#L536">[source]</a>
</span>
</div>
Reset all model attributes to their initial state, deleting potentially
large data arrays. Use this method to free some memory before saving
the class. The cleared attributes per model are:

* [Prediction attributes](../../../user_guide/predicting).
* [Metrics scores](../../../user_guide/training/#metric).
* [Shap values](../../../user_guide/plots/#shap).
* [Dashboard instance](../../../user_guide/data_management/#dashboard).

<br /><br /><br />


<a name="delete"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">delete</strong>(models=None)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/baserunner.py#L551">[source]</a>
</span>
</div>
Delete models. If all models are removed, the metric
is reset. Use this method to drop unwanted models from the pipeline
or to free some memory before saving. Deleted models are not removed
from any active mlflow experiment.
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<strong>models: str or sequence, default=None</strong><br>
Models to delete. If None, delete them all.
</td>
</tr>
</table>
<br />


<a name="distribution"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">distribution</strong>(distributions=None, columns=0)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L334">[source]</a>
</span>
</div>
Compute the [Kolmogorov-Smirnov test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)
for various distributions against columns in the dataset. Only for
numerical columns. Missing values are ignored.

!!! tip
    Use the [plot_distribution](../../plots/plot_distribution) method to plot
    a column's distribution.

<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>distributions: str, sequence or None, default=None</strong><br>
Names of the distributions in <code>scipy.stats</code> to get the
statistics on. If None, a selection of the most common ones is used.
</p>
<p>
<strong>columns: int, str, slice, sequence or None, default=None</strong><br>
Names, indices or dtypes of the columns in the dataset to
perform the test on. If None, select all numerical columns.
</p>
</td>
</tr>
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Returns:</strong></td>
<td width="80%" class="td_params">
<strong>pd.DataFrame</strong><br>
Statistic results with multiindex levels:
<ul style="line-height:1.2em;margin-top:5px">
<li><b>dist:</b> Name of the distribution.</li>
<li><b>stat:</b> Statistic results:
    <ul style="line-height:1.2em;margin-top:5px">
    <li><b>score:</b> KS-test score.</li>
    <li><b>p_value:</b> Corresponding p-value.</li>
    </ul>
</li>
</ul>
</td>
</tr>
</table>
<br />


<a name="drop"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">drop</strong>(columns)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L977">[source]</a>
</span>
</div>
Drop columns from the dataset.

!!! note
    This approach is preferred over dropping columns from the
    dataset directly through the property's `@setter` since
    the transformation is saved to atom's pipeline.

<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<strong>columns: int, str, slice or sequence</strong><br>
Names or indices of the columns to drop.
</td>
</tr>
</table>
<br />


<a name="evaluate"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">evaluate</strong>(metric=None,
dataset="test", threshold=0.5, sample_weight=None)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/baserunner.py#L578">[source]</a>
</span>
</div>
Get all the models' scores for the provided metrics.
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>metric: str, func, scorer, sequence or None, default=None</strong><br>
Metrics to calculate. If None, a selection of the most common
metrics per task are used.
</p>
<p>
<strong>dataset: str, default="test"</strong><br>
Data set on which to calculate the metric. Choose from: "train",
"test" or "holdout".
</p>
<strong>threshold: float, default=0.5</strong><br>
Threshold between 0 and 1 to convert predicted probabilities
to class labels. Only used when:
<ul style="line-height:1.2em;margin-top:5px">
<li>The task is binary classification.</li>
<li>The model has a <code>predict_proba</code> method.</li>
<li>The metric evaluates predicted target values.</li>
</ul>
<p>
<strong>sample_weight: sequence or None, default=None</strong><br>
Sample weights corresponding to y in <code>dataset</code>.
</p>
</td>
</tr>
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Returns:</strong></td>
<td width="80%" class="td_params">
<strong>pd.DataFrame</strong><br>
Scores of the models.
</td>
</tr>
</table>
<br />


<a name="export-pipeline"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">export_pipeline</strong>(model=None,
memory=None, verbose=None)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L409">[source]</a>
</span>
</div>
Export atom's pipeline to a sklearn-like Pipeline object. Optionally, you
can add a model as final estimator. The returned pipeline is already fitted
on the training set.

!!! info
    ATOM's Pipeline class behaves the same as a sklearn <a href="https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html">Pipeline</a>,
    and additionally:
    <ul style="line-height:1.2em;margin-top:5px">
    <li>Accepts transformers that change the target column.</li>
    <li>Accepts transformers that drop rows.</li>
    <li>Accepts transformers that only are fitted on a subset of the
        provided dataset.</li>
    <li>Always outputs pandas objects.</li>
    <li>Uses transformers that are only applied on the training set (see the
        <a href="#balance">balance</a> or <a href="#prune">prune</a> methods)
        to fit the pipeline, not to make predictions on new data.</li>
    </ul>

<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>model: str or None, default=None</strong><br>
Name of the model to add as a final estimator to the pipeline. If the
model used <a href="../../../user_guide/training/#automated-feature-scaling">automated feature scaling</a>,
the <code>scaler</code> is added to the pipeline. If None, only the
transformers are added.
</p>
<strong>memory: bool, str, Memory or None, default=None</strong><br>
Used to cache the fitted transformers of the pipeline.
<ul style="line-height:1.2em;margin-top:5px">
<li>If None or False: No caching is performed.</li>
<li>If True: A default temp directory is used.</li>
<li>If str: Path to the caching directory.</li>
<li>If Memory: Object with the <a href="https://joblib.readthedocs.io/en/latest/generated/joblib.Memory.html">joblib.Memory</a> interface.</li>
</ul>
<p>
<strong>verbose: int or None, default=None</strong><br>
Verbosity level of the transformers in the pipeline. If None, it leaves
them to their original verbosity. Note that this is not the pipeline's
own verbose parameter. To change that, use the <code>set_params</code>
method.
</p>
</td>
</tr>
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Returns:</strong></td>
<td width="80%" class="td_params">
<strong><a href="https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html">Pipeline</a></strong><br>
Current branch as a sklearn-like Pipeline object.
</td>
</tr>
</table>
<br />


<a name="get-class-weight"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">get_class_weights</strong>(dataset="train")
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/baserunner.py#L624">[source]</a>
</span>
</div>
Return class weights for a balanced data set. Statistically, the class
weights re-balance the data set so that the sampled data set represents
the target population as closely as possible. The returned weights are
inversely proportional to the class frequencies in the selected data set. 
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<strong>dataset: str, default="train"</strong><br>
Data set from which to get the weights. Choose from: "train", "test" or "dataset".
</tr>
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Returns:</strong></td>
<td width="80%" class="td_params">
<strong>dict</strong><br>
Classes with the corresponding weights.
</td>
</tr>
</table>
<br />


<a name="inverse-transform"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">inverse_transform</strong>(X=None, y=None, verbose=None)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L762">[source]</a>
</span>
</div>
Inversely transform new data through the pipeline. Transformers that
are only applied on the training set are skipped. The rest should all
implement a `inverse_transform` method. If only `X` or only `y` is
provided, it ignores transformers that require the other parameter.
This can be of use to, for example, inversely transform only the target
column.
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>X: dataframe-like or None, default=None</strong><br>
Transformed feature set with shape=(n_samples, n_features).
If None, X is ignored in the transformers.
</p>
<strong>y: int, str, dict, sequence or None, default=None</strong><br>
<ul style="line-height:1.2em;margin-top:5px">
<li>If None: y is ignored in the transformers.</li>
<li>If int: Position of the target column in X.</li>
<li>If str: Name of the target column in X.</li>
<li>Else: Array with shape=(n_samples,) to use as target.</li>
</ul>
<p>
<strong>verbose: int or None, default=None</strong><br>
Verbosity level of the output. If None, it uses the transformer's
own verbosity.
</p>
</td>
</tr>
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Returns:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>pd.DataFrame</strong><br>
Original feature set. Only returned if provided.
</p>
<p>
<strong>pd.Series</strong><br>
Original target column. Only returned if provided.
</p>
</td>
</tr>
</table>
<br /><br />


<a name="log"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">log</strong>(msg, level=0)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/basetransformer.py#L590">[source]</a>
</span>
</div>
Write a message to the logger and print it to stdout.
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>msg: str</strong><br>
Message to write to the logger and print to stdout.
</p>
<p>
<strong>level: int, default=0</strong><br>
Minimum verbosity level to print the message.
</p>
</td>
</tr>
</table>
<br />


<a name="merge"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">merge</strong>(other, suffix="2")
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/baserunner.py#L659">[source]</a>
</span>
</div>
Merge another instance of the same class into this one. Branches,
models, metrics and attributes of the other instance are merged into
this one. If there are branches and/or models with the same name,
they are merged adding the `suffix` parameter to their name. The
errors and missing attributes are extended with those of the other
instance. It's only possible to merge two instances if they are
initialized with the same dataset and trained with the same metric.
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>other: ATOMClassifier</strong><br>
Instance with which to merge. Should be of the same class as self.
</p>
<p>
<strong>suffix: str, default="2"</strong><br>
Conflicting branches and models are merged adding <code>suffix</code>
to the end of their names.
</p>
</td>
</tr>
</table>
<br />


<a name="report"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">report</strong>(dataset="dataset",
n_rows=None, filename=None, **kwargs)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L483">[source]</a>
</span>
</div>
Create an extensive profile analysis report of the data. The report
is rendered in HTML5 and CSS3. Note that this method can be slow for
`n_rows` > 10k.
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>dataset: str, default="dataset"</strong><br>
Data set to get the report from.
</p>
<p>
<strong>n_rows: int or None, default=None</strong><br>
Number of (randomly picked) rows to process. None to use all rows.
</p>
<p>
<strong>filename: str or None, default=None</strong><br>
Name to save the file with (as .html). None to not save anything.
</p>
<p>
<strong>**kwargs</strong><br>
Additional keyword arguments for the ProfileReport instance.
</p>
</td>
</tr>
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Returns:</strong></td>
<td width="80%" class="td_params">
<strong><a href="https://pandas-profiling.github.io/pandas-profiling/docs/master/rtd/pages/api/_autosummary/pandas_profiling.profile_report.ProfileReport.html#pandas_profiling.profile_report.ProfileReport">ProfileReport</a></strong><br>
Created profile object.
</td>
</tr>
</table>
<br />


<a name="reset"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">reset</strong>()
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L533">[source]</a>
</span>
</div>
Reset the instance to it's initial state, i.e. it deletes all branches
and models. The dataset is also reset to its form after initialization.
<br /><br /><br />


<a name="reset-aesthetics"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">reset_aesthetics</strong>()
<span style="float:right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L213">[source]</a>
</span>
</div>
Reset the [plot aesthetics](../../../user_guide/plots/#aesthetics) to their default values.
<br /><br /><br />


<a name="save"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">save</strong>(filename="auto", save_data=True)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/basetransformer.py#L611">[source]</a>
</span>
</div>
Save the instance to a pickle file. Remember that the class contains
the complete dataset as attribute, so the file can become large for
big datasets! To avoid this, use `save_data=False`.
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>filename: str, default="auto"</strong><br>
Name of the file. Use "auto" for automatic naming.
</p>
<p>
<strong>save_data: bool, default=True</strong><br>
Whether to save the data as an attribute of the instance. If False,
remember to add the data to <a href="../../ATOM/atomloader">ATOMLoader</a>
when loading the file.
</p>
</td>
</tr>
</table>
<br>


<a name="save-data"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">save_data</strong>(filename="auto", dataset="dataset")
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L550">[source]</a>
</span>
</div>
Save the data in the current branch to a csv file.
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>filename: str, default="auto"</strong><br>
Name of the file. Use "auto" for automatic naming.
</p>
<p>
<strong>dataset: str, default="dataset"</strong><br>
Data set to save.
</p>
</td>
</tr>
</table>
<br>


<a name="shrink"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">shrink</strong>(obj2cat=True,
int2uint=False, dense2sparse=False, columns=None)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L571">[source]</a>
</span>
</div>
Converts the columns to the smallest possible matching dtype. Examples
are: `float64` -> `float32`, `int64` -> `int8`, etc... Sparse arrays also
transform their non-fill value. Use this method for memory optimization.
Note that applying transformers to the data may alter the types again.
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>obj2cat: bool, default=True</strong><br>
Whether to convert <code>object</code> to <code>category</code>. Only if the
number of categories would be less than 30% of the length
of the column.
</p>
<p>
<strong>int2uint: bool, default=False</strong><br>
Whether to convert <code>int</code> to <code>uint</code> (unsigned integer).
Only if the values in the column are strictly positive.
</p>
<p>
<strong>dense2sparse: bool, default=False</strong><br>
Whether to convert all features to sparse format. The value that is
compressed is the most frequent value in the column.
</p>
<p>
<strong>columns: int, str, slice, sequence or None, default=None</strong><br>
Names, indices or dtypes of the columns in the dataset to shrink.
If None, transform all columns.
</p>
</td>
</tr>
</table>
<br />


<a name="stacking"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">stacking</strong>(name="Stack",
models=None, **kwargs)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/baserunner.py#L728">[source]</a>
</span>
</div>
Add a [Stacking](../../../user_guide/models/#stacking) model to the pipeline.
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>name: str, default="Stack"</strong><br>
Name of the model. The name is always presided with the
model's acronym: <code>Stack</code>.
</p>
<p>
<strong>models: sequence or None, default=None</strong><br>
Models that feed the stacking estimator. If None, it selects
all non-ensemble models trained on the current branch.
</p>
<p>
<strong>**kwargs</strong><br>
Additional keyword arguments for sklearn's <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html">StackingClassifier</a>
instance. The <a href="../../../user_guide/models/#predefined-models">predefined model's</a>
acronyms can be used for the <code>final_estimator</code> parameter.
</td>
</tr>
</table>
<br />


<a name="stats"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">stats</strong>()
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L680">[source]</a>
</span>
</div>
Print basic information about the dataset. The count and balance of
classes is shown, followed by the ratio (between parentheses) of the
class with respect to the rest of the classes in the same data set,
i.e. the class with the fewer samples is followed by `(1.0)`. This
information can be used to quickly assess if the data set is unbalanced.
<br /><br /><br />


<a name="status"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">status</strong>()
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L757">[source]</a>
</span>
</div>
Get an overview of the branches, models and errors in the instance.
This method prints the same information as atom's \__repr__ and also
saves it to the logger.
<br /><br /><br />


<a name="transform"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">transform</strong>(X=None, y=None, verbose=None)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L762">[source]</a>
</span>
</div>
Transform new data through the pipeline. Transformers that are only
applied on the training set are skipped. If only `X` or only `y` is
provided, it ignores transformers that require the other parameter.
This can be of use to, for example, transform only the target column.
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>X: dataframe-like or None, default=None</strong><br>
Feature set with shape=(n_samples, n_features). If None, X is ignored
in the transformers.
</p>
<strong>y: int, str, dict, sequence or None, default=None</strong><br>
<ul style="line-height:1.2em;margin-top:5px">
<li>If None: y is ignored in the transformers.</li>
<li>If int: Position of the target column in X.</li>
<li>If str: Name of the target column in X.</li>
<li>Else: Array with shape=(n_samples,) to use as target.</li>
</ul>
<p>
<strong>verbose: int or None, default=None</strong><br>
Verbosity level of the output. If None, it uses the transformer's
own verbosity.
</p>
</td>
</tr>
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Returns:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>pd.DataFrame</strong><br>
Transformed feature set. Only returned if provided.
</p>
<p>
<strong>pd.Series</strong><br>
Transformed target column. Only returned if provided.
</p>
</td>
</tr>
</table>
<br /><br />


<a name="voting"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">voting</strong>(name="Vote",
models=None, **kwargs)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/baserunner.py#L798">[source]</a>
</span>
</div>
Add a [Voting](../../../user_guide/models/#voting) model to the pipeline.
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>name: str, default="Vote"</strong><br>
Name of the model. The name is always presided with the
model's acronym: <code>Vote</code>.
</p>
<p>
<strong>models: sequence or None, default=None</strong><br>
Models that feed the voting estimator. If None, it selects
all non-ensemble models trained on the current branch.
</p>
<p>
<strong>**kwargs</strong><br>
Additional keyword arguments for sklearn's <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html">VotingClassifier</a>
instance.
</td>
</tr>
</table>
<br /><br />



## Data cleaning

The class provides data cleaning methods to scale or transform the
features and handle missing values, categorical columns, outliers and
unbalanced datasets. Calling on one of them will automatically apply the
method on the dataset in the pipeline.

!!! tip
    Use the [report](#report) method to examine the data and help you
    determine suitable parameters for the data cleaning methods.
    

<table style="font-size:16px;margin-top:5px">
<tr>
<td><a href="#balance">balance</a></td>
<td>Balance the target classes in the training set.</td>
</tr>

<tr>
<td><a href="#clean">clean</a></td>
<td>Applies standard data cleaning steps on the dataset.</td>
</tr>

<tr>
<td><a href="#discretize">discretize</a></td>
<td>Bin continuous data into intervals.</td>
</tr>

<tr>
<td><a href="#encode">encode</a></td>
<td>Encode categorical features.</td>
</tr>

<tr>
<td><a href="#impute">impute</a></td>
<td>Handle missing values in the dataset.</td>
</tr>

<tr>
<td><a href="#normalize">normalize</a></td>
<td>Transform the data to follow a Normal/Gaussian distribution.</td>
</tr>

<tr>
<td><a href="#prune">prune</a></td>
<td>Prune outliers from the training set.</td>
</tr>

<tr>
<td><a href="#scale">scale</a></td>
<td>Scale the dataset.</td>
</tr>
</table>
<br>


<a name="balance"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">balance</strong>(strategy="ADASYN", **kwargs)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L1235">[source]</a>
</span>
</div>
Balance the number of samples per class in the target column. When
oversampling, the newly created samples have an increasing integer
index for numerical indices, and an index of the form [estimator]_N
for non-numerical indices, where N stands for the N-th sample in the
data set. The estimator created by the class is attached to atom.
See [Balancer](../data_cleaning/balancer.md) for a description of the
parameters.

!!! note
    This transformation is only applied to the training set in order to
    maintain the original distribution of target classes in the test set.

<br /><br /><br />


<a name="clean"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">clean</strong>(drop_types=None,
strip_categorical=True, drop_max_cardinality=True, drop_min_cardinality=True,
drop_duplicates=False, drop_missing_target=True, encode_target=True)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L1045">[source]</a>
</span>
</div>
Applies standard data cleaning steps on the dataset. Use the parameters
to choose which transformations to perform. The available steps are:

* Drop columns with specific data types.
* Strip categorical features from white spaces.
* Drop categorical columns with maximal cardinality.
* Drop columns with minimum cardinality.
* Drop duplicate rows.
* Drop rows with missing values in the target column.
* Encode the target column.

See the [Cleaner](../data_cleaning/cleaner.md) class for a description of the parameters.
<br /><br /><br />


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
