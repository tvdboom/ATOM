# Cleaner
---------

<pre><em>class</em> atom.data_cleaning.<strong style="color:#008AB8">Cleaner</strong>(prohibited_types=None, maximum_cardinality=True, minimum_cardinality=True,
                                 strip_categorical=True, drop_duplicates=False, missing_target=True,
                                 encode_target=True, verbose=0, logger=None)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/data_cleaning.py#L182">[source]</a></div></pre>
Performs standard data cleaning steps on a dataset. Use the parameters to choose
which transformations to perform. The available steps are:

  * Remove columns with prohibited data types.
  * Remove categorical columns with maximal cardinality.
  * Remove columns with minimum cardinality.
  * Strip categorical features from white spaces.
  * Drop duplicate rows.
  * Drop rows with missing values in the target column.
  * Encode the target column.

This class can be accessed from atom through the [clean](../../ATOM/atomclassifier/#clean)
 method. Read more in the [user guide](../../../user_guide/#standard-data-cleaning).
<table>
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>prohibited_types: str, sequence or None, optional (default=None)</strong>
<blockquote>
Columns with these types will be removed from the dataset.
</blockquote>
<strong>maximum_cardinality: bool, optional (default=True)</strong>
<blockquote>
Whether to remove categorical columns with maximum cardinality,
i.e. the number of unique values is equal to the number of
instances. Usually the case for names, IDs, etc...
</blockquote>
<strong>minimum_cardinality: bool, optional (default=True)</strong>
<blockquote>
Whether to remove columns with minimum cardinality, i.e. all values in the
 column are the same.
</blockquote>
<strong>strip_categorical: bool, optional (default=True)</strong>
<blockquote>
Whether to strip the spaces from the categorical columns.
</blockquote>
<strong>drop_duplicates: bool, optional (default=False)</strong>
<blockquote>
Whether to drop duplicate rows. Only the first occurrence of
every duplicated row is kept.
</blockquote>
<strong>missing_target: bool, optional (default=True)</strong>
<blockquote>
Whether to drop rows with missing values in the target column.
Is ignored if y is not provided.
</blockquote>
<strong>encode_target: bool, optional (default=True)</strong>
<blockquote>
Whether to Label-encode the target column. Is ignored if y is not provided.
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
<strong>logger: str, Logger or None, optional (default=None)</strong>
<blockquote>
<ul>
<li>If None: Doesn't save a logging file.</li>
<li>If str: Name of the logging file. Use "auto" for default name.</li>
<li>Else: Python <code>logging.Logger</code> instance.</li>
</ul>
The default name consists of the class' name followed by
 the timestamp of the logger's creation.
</blockquote>
</td>
</tr>
</table>
<br>



## Attributes
-------------

<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Attributes:</strong></td>
<td width="75%" style="background:white;">
<strong>missing: list</strong>
<blockquote>
List of values that are considered "missing". Default values are: "",
"?", "None", "NA", "nan", "NaN" and "inf". Note that <code>None</code>,
<code>NaN</code>, <code>+inf</code> and <code>-inf</code> are always
considered missing since they are incompatible with sklearn estimators.
</blockquote>
<strong>mapping: dict</strong>
<blockquote>
Dictionary of the target values mapped to their respective encoded integer.
 Only available if encode_target=True.
</blockquote>
</td>
</tr>
</table>
<br>



## Methods
---------

<table width="100%">
<tr>
<td><a href="#fit-transform">fit_transform</a></td>
<td>Same as transform.</td>
</tr>

<tr>
<td><a href="#get-params">get_params</a></td>
<td>Get parameters for this estimator.</td>
</tr>

<tr>
<td width="15%"><a href="#log">log</a></td>
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
<pre><em>method</em> <strong style="color:#008AB8">fit_transform</strong>(X, y=None) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/data_cleaning.py#L39">[source]</a></div></pre>
Fit to data, then transform it.
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>X: dict, list, tuple, np.ndarray or pd.DataFrame</strong>
<blockquote>
Feature set with shape=(n_samples, n_features).
</blockquote>
<strong>y: int, str, sequence or None, optional (default=None)</strong>
<blockquote>
<ul>
<li>If None: y is ignored.</li>
<li>If int: Index of the target column in X.</li>
<li>If str: Name of the target column in X.</li>
<li>Else: Target column with shape=(n_samples,).</li>
</ul>
</blockquote>
</tr>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Returns:</strong></td>
<td width="75%" style="background:white;">
<strong>X: pd.DataFrame</strong>
<blockquote>
Transformed feature set.
</blockquote>
<strong>y: pd.Series</strong>
<blockquote>
Transformed target column. Only returned if provided.
</blockquote>
</tr>
</table>
<br />

<a name="get-params"></a>
<pre><em>method</em> <strong style="color:#008AB8">get_params</strong>(deep=True) 
<div align="right"><a href="https://github.com/scikit-learn/scikit-learn/blob/0fb307bf3/sklearn/base.py#L189">[source]</a></div></pre>
Get parameters for this estimator.
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>deep: bool, default=True</strong>
<blockquote>
If True, will return the parameters for this estimator and contained subobjects that are estimators.
</blockquote>
</tr>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Returns:</strong></td>
<td width="75%" style="background:white;">
<strong>params: dict</strong>
<blockquote>
Dictionary of the parameter names mapped to their values.
</blockquote>
</tr>
</table>
<br />


<a name="log"></a>
<pre><em>method</em> <strong style="color:#008AB8">log</strong>(msg, level=0)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basetransformer.py#L315">[source]</a></div></pre>
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


<a name="save"></a>
<pre><em>method</em> <strong style="color:#008AB8">save</strong>(filename=None)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basetransformer.py#L336">[source]</a></div></pre>
Save the instance to a pickle file.
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>filename: str or None, optional (default=None)</strong>
<blockquote>
Name to save the file with. None or "auto" to save with the __name__ of the class.
</blockquote>
</tr>
</table>
<br>


<a name="set-params"></a>
<pre><em>method</em> <strong style="color:#008AB8">set_params</strong>(**params) 
<div align="right"><a href="https://github.com/scikit-learn/scikit-learn/blob/0fb307bf3/sklearn/base.py#L221">[source]</a></div></pre>
<div style="padding-left:3%">
Set the parameters of this estimator.
<br><br>
</div>
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>**params: dict</strong>
<blockquote>
Estimator parameters.
</blockquote>
</tr>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Returns:</strong></td>
<td width="75%" style="background:white;">
<strong>self: Cleaner</strong>
<blockquote>
Estimator instance.
</blockquote>
</tr>
</table>
<br />


<a name="transform"></a>
<pre><em>method</em> <strong style="color:#008AB8">transform</strong>(X, y=None) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/data_cleaning.py#L278">[source]</a></div></pre>
Apply the data cleaning steps on the data.
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>X: dict, list, tuple, np.ndarray or pd.DataFrame</strong>
<blockquote>
Feature set with shape=(n_samples, n_features).
</blockquote>
<strong>y: int, str, sequence or None, optional (default=None)</strong>
<blockquote>
<ul>
<li>If None: y is ignored.</li>
<li>If int: Index of the target column in X.</li>
<li>If str: Name of the target column in X.</li>
<li>Else: Target column with shape=(n_samples,).</li>
</ul>
</blockquote>
</tr>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Returns:</strong></td>
<td width="75%" style="background:white;">
<strong>X: pd.DataFrame</strong>
<blockquote>
Transformed feature set.
</blockquote>
<strong>y: pd.Series</strong>
<blockquote>
Transformed target column. Only returned if provided.
</blockquote>
</tr>
</table>
<br />


## Example
----------

```python
from atom import ATOMClassifier

atom = ATOMClassifier(X, y)
atom.clean(maximum_cardinality=False)
```
or
```python
from atom.data_cleaning import Cleaner

cleaner = Cleaner(maximum_cardinality=False)
X, y = cleaner.transform(X, y)
```