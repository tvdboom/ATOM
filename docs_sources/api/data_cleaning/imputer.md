# Imputer
---------

<div style="font-size:20px">
<em>class</em> atom.data_cleaning.<strong style="color:#008AB8">Imputer</strong>(strat_num="drop",
strat_cat="drop", max_nan_rows=None, max_nan_cols=None, gpu=False,
verbose=0, logger=None)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/data_cleaning.py#L661">[source]</a>
</span>
</div>

Impute or remove missing values according to the selected strategy.
Also removes rows and columns with too many missing values. Use
the `missing` attribute to customize what are considered "missing
values". This class can be accessed from atom through the
[impute](../../ATOM/atomclassifier/#impute) method. Read more in the
[user guide](../../../user_guide/data_cleaning/#imputing-missing-values).

<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<strong>strat_num: str, int or float, optional (default="drop")</strong><br>
Imputing strategy for numerical columns. Choose from:
<ul style="line-height:1.2em;margin-top:5px">
<li>"drop": Drop rows containing missing values.</li>
<li>"mean": Impute with mean of column.</li>
<li>"median": Impute with median of column.</li>
<li>"knn": Impute using a K-Nearest Neighbors approach.</li>
<li>"most_frequent": Impute with most frequent value.</li>
<li>int or float: Impute with provided numerical value.</li>
</ul>
<strong>strat_cat: str, optional (default="drop")</strong><br>
Imputing strategy for categorical columns. Choose from:
<ul style="line-height:1.2em;margin-top:5px">
<li>"drop": Drop rows containing missing values.</li>
<li>"most_frequent": Impute with most frequent value.</li>
<li>str: Impute with provided string.</li>
</ul>
<p>
<strong>max_nan_rows: int, float or None, optional (default=None)</strong><br>
Maximum number or fraction of missing values in a row
(if more, the row is removed). If None, ignore this step.
</p>
<p>
<strong>max_nan_cols: int, float, optional (default=None)</strong><br>
Maximum number or fraction of missing values in a column
(if more, the column is removed). If None, ignore this step.
</p>
<strong>gpu: bool or str, optional (default=False)</strong><br>
Train strategies on GPU (instead of CPU).
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

!!! tip
    Use atom's [nans](../../ATOM/atomclassifier/#data-attributes) attribute
    for an overview of the number of missing values per column.

<br>



## Attributes

<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Attributes:</strong></td>
<td width="80%" class="td_params">
<strong>missing: list</strong><br>
Values that are considered "missing". Default values are: "", "?",
"None", "NA", "nan", "NaN" and "inf". Note that <code>None</code>,
<code>NaN</code>, <code>+inf</code> and <code>-inf</code> are always
considered missing since they are incompatible with sklearn estimators.
</td>
</tr>
</table>
<br>



## Methods

<table style="font-size:16px">
<tr>
<td><a href="#fit">fit</a></td>
<td>Fit to data.</td>
</tr>

<tr>
<td><a href="#fit-transform">fit_transform</a></td>
<td>Fit to data, then transform it.</td>
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


<a name="fit"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">fit</strong>(X, y=None)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/data_cleaning.py#L747">[source]</a>
</span>
</div>
Fit to data.
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>X: dataframe-like</strong><br>
Feature set with shape=(n_samples, n_features).
</p>
<p>
<strong>y: int, str, sequence or None, optional (default=None)</strong><br>
Does nothing. Implemented for continuity of the API.
</p>
</td>
</tr>
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Returns:</strong></td>
<td width="80%" class="td_params">
<strong>Imputer</strong><br>
Fitted instance of self.
</tr>
</table>
<br />


<a name="fit-transform"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">fit_transform</strong>(X, y=None)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/data_cleaning.py#L101">[source]</a>
</span>
</div>
Fit to data, then impute the missing values. Note that leaving y=None
can lead to inconsistencies in data length between X and y if rows are
dropped during the transformation.
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
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/basetransformer.py#L582">[source]</a>
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
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/basetransformer.py#L603">[source]</a>
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
<strong>Imputer</strong><br>
Estimator instance.
</td>
</tr>
</table>
<br />


<a name="transform"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">fit_transform</strong>(X, y=None)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/data_cleaning.py#L844">[source]</a>
</span>
</div>
Impute the missing values. Note that leaving y=None can lead to
inconsistencies in data length between X and y if rows are dropped
during the transformation.
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
<li>Else: Target column with shape=(n_samples,)</li>
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
    atom.impute(strat_num="knn", strat_cat="drop", max_nan_cols=0.8)
    ```

=== "stand-alone"
    ```python
    from atom.data_cleaning import Imputer
    
    imputer = Imputer(strat_num="knn", strat_cat="drop", max_nan_cols=0.8)
    imputer.fit(X_train, y_train)
    X = imputer.transform(X)
    ```