# Outliers
----------

<a name="atom"></a>
<pre><em>class</em> atom.data_cleaning.<strong style="color:#008AB8">Outliers</strong>(strategy="drop", max_sigma=3, include_target=False, verbose=0, logger=None)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/data_cleaning.py#L754">[source]</a></div></pre>
<div style="padding-left:3%">
Remove or replace outliers in the data. Outliers are defined as values that lie
 further than `max_sigma` * standard_deviation away from the mean of the column.
 Ignores categorical columns. This class can be accessed from `atom` through the
 [outliers](../../ATOM/atomclassifier/#outliers) method. Read more
 in the [user guide](../../../user_guide/#handling-outliers).
<br /><br />
<table>
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>strategy: int, float or str, optional (default="drop")</strong>
<blockquote>
Strategy to apply on the outliers. Choose from:
<ul>
<li>"drop": Drop any row with outliers.</li>
<li>"min_max": Replace the outlier with the min or max of the column.</li>
<li>Any numerical value with which to replace the outliers.</li>
</ul>
</blockquote>
<strong>max_sigma: int or float, optional (default=3)</strong>
<blockquote>
Maximum allowed standard deviations from the mean of the column.
 If more, it is considered an outlier.
</blockquote>
<strong>include_target: bool, optional (default=False)</strong>
<blockquote>
Whether to include the target column in the transformation. This can be useful for
 regression tasks.
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
<li>If None: Doesn"t save a logging file.</li>
<li>If bool: True for logging file with default name. False for no logger.</li>
<li>If str: Name of the logging file. "auto" to create an automatic name.</li>
<li>If class: python `Logger` object.</li>
</ul>
</blockquote>
</td>
</tr>
</table>
</div>
<br>


## Methods
---------

<table>
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
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/data_cleaning.py#L40">[source]</a></div></pre>
<div style="padding-left:3%">
Apply the outlier strategy on the data.
<br><br>
</div>
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>X: dict, list, tuple, np.array or pd.DataFrame</strong>
<blockquote>
Data containing the features, with shape=(n_samples, n_features).
</blockquote>
<strong>y: int, str, sequence, np.array, pd.Series or None, optional (default=None)</strong>
<blockquote>
<ul>
<li>If None: y is ignored in the transformation.</li>
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
<strong>X: pd.Series</strong>
<blockquote>
Transformed target column. Only returned if provided.
</blockquote>
</tr>
</table>
<br />

<a name="get-params"></a>
<pre><em>method</em> <strong style="color:#008AB8">get_params</strong>(deep=True) 
<div align="right"><a href="https://github.com/scikit-learn/scikit-learn/blob/0fb307bf3/sklearn/base.py#L189">[source]</a></div></pre>
<div style="padding-left:3%">
Get parameters for this estimator.
<br><br>
</div>
<table>
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


<a name="save"></a>
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


<a name="set-params"></a>
<pre><em>method</em> <strong style="color:#008AB8">set_params</strong>(**params) 
<div align="right"><a href="https://github.com/scikit-learn/scikit-learn/blob/0fb307bf3/sklearn/base.py#L221">[source]</a></div></pre>
<div style="padding-left:3%">
Set the parameters of this estimator.
<br><br>
</div>
<table>
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
<strong>self: Outliers</strong>
<blockquote>
Estimator instance.
</blockquote>
</tr>
</table>
<br />


<a name="transform"></a>
<pre><em>method</em> <strong style="color:#008AB8">transform</strong>(X, y=None) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/data_cleaning.py#L802">[source]</a></div></pre>
<div style="padding-left:3%">
Apply the outlier strategy on the data.
<br><br>
</div>
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>X: dict, list, tuple, np.array or pd.DataFrame</strong>
<blockquote>
Data containing the features, with shape=(n_samples, n_features).
</blockquote>
<strong>y: int, str, sequence, np.array, pd.Series or None, optional (default=None)</strong>
<blockquote>
<ul>
<li>If None: y is ignored in the transformation.</li>
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
<strong>X: pd.Series</strong>
<blockquote>
Transformed target column. Only returned if provided.
</blockquote>
</tr>
</table>
<br />


## Example
----------

```python
from atom import ATOMRegressor

atom = ATOMRegressor(X, y)
atom.outliers(strategy="min_max", max_sigma=2, include_target=True)
```
or
```python
from atom.data_cleaning import Outliers

outliers = Outliers(strategy="min_max", max_sigma=2, include_target=True)
X_train, y_train = outliers.transform(X_train, y_train)
```