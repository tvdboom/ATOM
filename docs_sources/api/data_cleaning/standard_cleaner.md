# StandardCleaner
-----------------

<a name="atom"></a>
<pre><em>class</em> atom.data_cleaning.<strong style="color:#008AB8">StandardCleaner</strong>(prohibited_types=[], strip_categorical=True, maximum_cardinality=True,
                                         minimum_cardinality=True, missing_target=True, map_target=None,
                                         verbose=0, logger=None)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/data_cleaning.py#L141">[source]</a></div></pre>
<div style="padding-left:3%">
Performs standard data cleaning steps on a dataset. Use the parameters to choose
 which transformations to perform. The available steps are:

  * Remove columns with prohibited data types.
  * Strip categorical features from white spaces.
  * Remove categorical columns with maximal cardinality.
  * Remove columns with minimum cardinality.
  * Remove rows with missing values in the target column.
  * Label-encode the target column.

This class is automatically called when initializing `atom`. Read more in the
 [user guide](../../../user_guide/#standard-data-cleaning).
<table>
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>prohibited_types: str or sequence, optional (default=[])</strong>
<blockquote>
Columns with any of these types will be removed from the dataset.
</blockquote>
<strong>strip_categorical: bool, optional (default=True)</strong>
<blockquote>
Whether to strip the spaces from values in the categorical columns.
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
<strong>missing_target: bool, optional (default=True)</strong>
<blockquote>
Whether to remove rows with missing values in the target column.
 Ignored if y is not provided.
</blockquote>
<strong>map_target: bool or None, optional (default=None)</strong>
<blockquote>
Whether to map the target column to numerical values. Should only
 be used for classification tasks. If None, infer task from the
 provided target column and set to True if it is classification.
 Ignored if y is not provided or if it already consists of ordered integers.
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
<strong>logger: bool, str, class or None, optional (default=None)</strong>
<blockquote>
<ul>
<li>If None: Doesn't save a logging file.</li>
<li>If bool: True for logging file with default name. False for no logger.</li>
<li>If str: Name of the logging file. 'auto' to create an automatic name.</li>
<li>If class: python `Logger` object.</li>
</ul>
</blockquote>
</td>
</tr>
</table>
</div>
<br>


## Attributes
-------------

<a name="atom"></a>
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Attributes:</strong></td>
<td width="75%" style="background:white;">
<strong>mapping: dict</strong>
<blockquote>
Dictionary of the target values mapped to their respective encoded integer.
 Only available if `map_target` was performed.
</blockquote>
</td>
</tr>
</table>
<br>


## Methods
---------

<table width="100%">
<tr>
<td><a href="#standardcleaner-fit-transform">fit_transform</a></td>
<td>Same as transform.</td>
</tr>

<tr>
<td><a href="#standardcleaner-get-params">get_params</a></td>
<td>Get parameters for this estimator.</td>
</tr>

<tr>
<td width="15%"><a href="#standardcleaner-log">log</a></td>
<td>Write information to the logger and print to stdout.</td>
</tr>

<tr>
<td><a href="#standardcleaner-save">save</a></td>
<td>Save the instance to a pickle file.</td>
</tr>


<tr>
<td><a href="#standardcleaner-set-params">set_params</a></td>
<td>Set the parameters of this estimator.</td>
</tr>

<tr>
<td><a href="#standardcleaner-transform">transform</a></td>
<td>Transform the data.</td>
</tr>
</table>
<br>



<a name="standardcleaner-fit-transform"></a>
<pre><em>method</em> <strong style="color:#008AB8">fit_transform</strong>(X, y=None) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/data_cleaning.py#L40">[source]</a></div></pre>
<div style="padding-left:3%">
Apply the data cleaning steps on the data.
<br><br>
</div>
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>X: dict, sequence, np.array or pd.DataFrame</strong>
<blockquote>
Data containing the features, with shape=(n_samples, n_features).
</blockquote>
<strong>y: int, str, sequence, np.array, pd.Series or None, optional (default=None)</strong>
<blockquote>
<ul>
<li>If None: y is ignored in the transformation.</li>
<li>If int: Index of the target column in X.</li>
<li>If string: Name of the target column in X.</li>
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

<a name="standardcleaner-get-params"></a>
<pre><em>method</em> <strong style="color:#008AB8">get_params</strong>(deep=True) 
<div align="right"><a href="https://github.com/scikit-learn/scikit-learn/blob/0fb307bf3/sklearn/base.py#L189">[source]</a></div></pre>
<div style="padding-left:3%">
Get parameters for this estimator.
<br><br>
</div>
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


<a name="standardcleaner-log"></a>
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


<a name="standardcleaner-save"></a>
<pre><em>method</em> <strong style="color:#008AB8">save</strong>(filename=None)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basetransformer.py#L220">[source]</a></div></pre>
<div style="padding-left:3%">
Save the instance to a pickle file.
<br><br>
<table>
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
<br>


<a name="standardcleaner-set-params"></a>
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
<strong>\*\*params: dict</strong>
<blockquote>
Estimator parameters.
</blockquote>
</tr>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Returns:</strong></td>
<td width="75%" style="background:white;">
<strong>self: StandardCleaner</strong>
<blockquote>
Estimator instance.
</blockquote>
</tr>
</table>
<br />


<a name="standardcleaner-transform"></a>
<pre><em>method</em> <strong style="color:#008AB8">transform</strong>(X, y=None) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/data_cleaning.py#L213">[source]</a></div></pre>
<div style="padding-left:3%">
Apply the data cleaning steps on the data.
<br><br>
</div>
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>X: dict, sequence, np.array or pd.DataFrame</strong>
<blockquote>
Data containing the features, with shape=(n_samples, n_features).
</blockquote>
<strong>y: int, str, sequence, np.array or pd.Series, optional (default=None)</strong>
<blockquote>
<ul>
<li>If None: y is ignored in the transformation.</li>
<li>If int: Index of the target column in X.</li>
<li>If string: Name of the target column in X.</li>
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

# ATOM's initializer calls StandardCleaner automatically
atom = ATOMClassifier(X, y)
```
or
```python
from atom.data_cleaning import StandardCleaner

cleaner = StandardCleaner(prohibited_types=['str'], missing_target=True)
X, y = cleaner.transform(X, y)
```