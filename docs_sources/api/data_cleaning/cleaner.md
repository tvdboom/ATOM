# Cleaner
---------

<div style="font-size:20px">
<em>class</em> atom.data_cleaning.<strong style="color:#008AB8">Cleaner</strong>(drop_types=None,
strip_categorical=True, drop_max_cardinality=True, drop_min_cardinality=True,
drop_duplicates=False, drop_missing_target=True, encode_target=True, gpu=False,
verbose=0, logger=None)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/data_cleaning.py#L427">[source]</a>
</span>
</div>

Performs standard data cleaning steps on a dataset. Use the parameters
to choose which transformations to perform. The available steps are:

* Drop columns with specific data types.
* Strip categorical features from white spaces.
* Drop categorical columns with maximal cardinality.
* Drop columns with minimum cardinality.
* Drop duplicate rows.
* Drop rows with missing values in the target column.
* Encode the target column.

This class can be accessed from atom through the [clean](../../ATOM/atomclassifier/#clean)
method. Read more in the [user guide](../../../user_guide/data_cleaning/#standard-data-cleaning).

<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>drop_types: str, sequence or None, optional (default=None)</strong><br>
Columns with these types are dropped from the dataset.
</p>
<p>
<strong>strip_categorical: bool, optional (default=True)</strong><br>
Whether to strip the spaces from the categorical columns.
</p>
<p>
<strong>drop_max_cardinality: bool, optional (default=True)</strong><br>
Whether to drop categorical columns with maximum cardinality,
i.e. the number of unique values is equal to the number of
samples. Usually the case for names, IDs, etc...
</p>
<p>
<strong>drop_min_cardinality: bool, optional (default=True)</strong><br>
Whether to drop columns with minimum cardinality, i.e. all values in the
column are the same.
</p>
<p>
<strong>drop_duplicates: bool, optional (default=False)</strong><br>
Whether to drop duplicate rows. Only the first occurrence of
every duplicated row is kept.
</p>
<p>
<strong>drop_missing_target: bool, optional (default=True)</strong><br>
Whether to drop rows with missing values in the target column.
This parameter is ignored if <code>y</code> is not provided.
</p>
<p>
<strong>encode_target: bool, optional (default=True)</strong><br>
Whether to Label-encode the target column. This parameter is ignored
if <code>y</code> is not provided.
</p>
<strong>gpu: bool or str, optional (default=False)</strong><br>
Train estimator on GPU (instead of CPU). Only for encode_target=True.
<ul style="line-height:1.2em;margin-top:5px">
<li>If False: Always use CPU implementation.</li>
<li>If True: Use GPU implementation if possible.</li>
<li>If "force": Force GPU implementation.</li>
</ul>
<strong>verbose: int, optional (default=0)</strong><br>
Verbosity level of the class. Possible values are:
<ul style="line-height:1.2em;margin-top:5px">
<li>0 to not print anything.</li>
<li>1 to print basic information.</li>
<li>2 to print detailed information.</li>
</ul>
<strong>logger: str, Logger or None, optional (default=None)</strong><br>
<ul style="line-height:1.2em;margin-top:5px">
<li>If None: Doesn't save a logging file.</li>
<li>If str: Name of the log file. Use "auto" for automatic naming.</li>
<li>Else: Python <code>logging.Logger</code> instance.</li>
</ul>
</td>
</tr>
</table>
<br>



## Attributes

<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Attributes:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>missing: list</strong><br>
Values that are considered "missing". Default values are: "", "?",
"None", "NA", "nan", "NaN" and "inf". Note that <code>None</code>,
<code>NaN</code>, <code>+inf</code> and <code>-inf</code> are always
considered missing since they are incompatible with sklearn estimators.
</p>
<p>
<strong>mapping: dict</strong><br>
Target values mapped to their respective encoded integer. Only
available if encode_target=True.
</p>
</td>
</tr>
</table>
<br>



## Methods

<table style="font-size:16px">
<tr>
<td><a href="#fit-transform">fit_transform</a></td>
<td>Same as transform.</td>
</tr>

<tr>
<td><a href="#get-params">get_params</a></td>
<td>Get parameters for this estimator.</td>
</tr>

<tr>
<td><a href="#log">log</a></td>
<td>Write information to the logger and print to stdout.</td>
</tr>

<tr>
<td><a href="#save">save</a></td>
<td>Save the instance to a pickle file.</td>
</tr>


<tr>
<td><a href="#set-params">set_params</a></td>
<td>Set the parameters of this estimator.</td>
</tr>

<tr>
<td><a href="#transform">transform</a></td>
<td>Transform the data.</td>
</tr>
</table>
<br>


<a name="fit-transform"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">fit_transform</strong>(X, y=None)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/data_cleaning.py#L77">[source]</a>
</span>
</div>
Apply the data cleaning steps to the data.
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>X: dataframe-like</strong><br>
Feature set with shape=(n_samples, n_features).
</p>
<strong>y: int, str, sequence or None, optional (default=None)</strong><br>
<ul style="line-height:1.2em;margin-top:5px">
<li>If None: y is ignored.</li>
<li>If int: Index of the target column in X.</li>
<li>If str: Name of the target column in X.</li>
<li>Else: Target column with shape=(n_samples,).</li>
</ul>
</tr>
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Returns:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>pd.DataFrame</strong><br>
Transformed feature set.
</p>
<p>
<strong>pd.Series</strong><br>
Transformed target column. Only returned if provided.
</p>
</td>
</tr>
</table>
<br />

<a name="get-params"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">get_params</strong>(deep=True)
<span style="float:right">
<a href="https://github.com/scikit-learn/scikit-learn/blob/0fb307bf3/sklearn/base.py#L189">[source]</a>
</span>
</div>
Get parameters for this estimator.
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>deep: bool, optional (default=True)</strong><br>
If True, will return the parameters for this estimator and contained
subobjects that are estimators.
</p>
</td>
</tr>
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Returns:</strong></td>
<td width="80%" class="td_params">
<strong>dict</strong><br>
Parameter names mapped to their values.
</td>
</tr>
</table>
<br />


<a name="log"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">log</strong>(msg, level=0)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/basetransformer.py#L525">[source]</a>
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
<strong>level: int, optional (default=0)</strong><br>
Minimum verbosity level to print the message.
</p>
</td>
</tr>
</table>
<br />


<a name="save"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">save</strong>(filename="auto")
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/basetransformer.py#L546">[source]</a>
</span>
</div>
Save the instance to a pickle file.
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<strong>filename: str, optional (default="auto")</strong><br>
Name of the file. Use "auto" for automatic naming.
</td>
</tr>
</table>
<br>


<a name="set-params"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">set_params</strong>(**params)
<span style="float:right">
<a href="https://github.com/scikit-learn/scikit-learn/blob/0fb307bf3/sklearn/base.py#L221">[source]</a>
</span>
</div>
Set the parameters of this estimator.
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<strong>**params: dict</strong><br>
Estimator parameters.
</tr>
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Returns:</strong></td>
<td width="80%" class="td_params">
<strong>Cleaner</strong><br>
Estimator instance.
</td>
</tr>
</table>
<br />


<a name="transform"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">transform</strong>(X, y=None)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/data_cleaning.py#L521">[source]</a>
</span>
</div>
Apply the data cleaning steps to the data.
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>X: dataframe-like</strong><br>
Feature set with shape=(n_samples, n_features).
</p>
<strong>y: int, str, sequence or None, optional (default=None)</strong><br>
<ul style="line-height:1.2em;margin-top:5px">
<li>If None: y is ignored.</li>
<li>If int: Index of the target column in X.</li>
<li>If str: Name of the target column in X.</li>
<li>Else: Target column with shape=(n_samples,).</li>
</ul>
</tr>
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Returns:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>pd.DataFrame</strong><br>
Transformed feature set.
</p>
<p>
<strong>pd.Series</strong><br>
Transformed target column. Only returned if provided.
</p>
</td>
</tr>
</table>
<br />



## Example

=== "atom"
    ```python
    from atom import ATOMClassifier
    
    atom = ATOMClassifier(X, y)
    atom.clean(maximum_cardinality=False)
    ```

=== "stand-alone"
    ```python
    from atom.data_cleaning import Cleaner
    
    cleaner = Cleaner(maximum_cardinality=False)
    X, y = cleaner.transform(X, y)
    ```