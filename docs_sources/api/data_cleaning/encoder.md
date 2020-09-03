# Encoder
--------

<a name="atom"></a>
<pre><em>class</em> atom.data_cleaning.<strong style="color:#008AB8">Encoder</strong>(max_onehot=10, encode_type='Target',
                                 frac_to_other=None, verbose=0, logger=None, **kwargs)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/data_cleaning.py#L543">[source]</a></div></pre>
<div style="padding-left:3%">
Perform encoding of categorical features. The encoding type depends on the number
 of unique values in the column:

- If n_unique=2, use label-encoding.
- If 2 < n_unique <= max_onehot, use one-hot-encoding.
- If n_unique > max_onehot, use `encode_type`.

Also replaces classes with low occurrences with the value `other` in
 order to prevent too high cardinality. Categorical features are defined as
 all columns whose dtype.kind not in `ifu`. Will raise an error if it encounters
 missing values or unknown categories while transforming.
<br /><br />
<table>
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>max_onehot: int or None, optional (default=10)</strong>
<blockquote>
Maximum number of unique values in a feature to perform one-hot-encoding.
 If None, it will never do one-hot-encoding.
</blockquote>
<strong>encode_type: str, optional (default='Target')</strong>
<blockquote>
Type of encoding to use for high cardinality features. Choose from one of the
 encoders available in the [category-encoders](http://contrib.scikit-learn.org/category_encoders/)
 package (except for OneHotEncoder and HashingEncoder).
</blockquote>
<strong>frac_to_other: float, optional (default=None)</strong>
<blockquote>
Categories with less rows than n_rows * `fraction_to_other` are replaced with the
 string `other`. If None, skip this step.
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
<strong>**kwargs</strong>
<blockquote>
Additional keyword arguments passed to the encoder selected by the `encode_type` parameter.
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
<td><a href="#encoder-fit">fit</a></td>
<td>Fit the class.</td>
</tr>

<tr>
<td><a href="#encoder-fit-transform">fit_transform</a></td>
<td>Fit the class and return the transformed data.</td>
</tr>

<tr>
<td><a href="#encoder-get-params">get_params</a></td>
<td>Get parameters for this estimator.</td>
</tr>

<tr>
<td width="15%"><a href="#encoder-log">log</a></td>
<td>Write information to the logger and print to stdout.</td>
</tr>

<tr>
<td><a href="#encoder-save">save</a></td>
<td>Save the instance to a pickle file.</td>
</tr>


<tr>
<td><a href="#encoder-set-params">set_params</a></td>
<td>Set the parameters of this estimator.</td>
</tr>

<tr>
<td><a href="#encoder-transform">transform</a></td>
<td>Transform the data.</td>
</tr>
</table>
<br>


<a name="encoder-fit"></a>
<pre><em>method</em> <strong style="color:#008AB8">fit</strong>(X, y) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/data_cleaning.py#L626">[source]</a></div></pre>
<div style="padding-left:3%">
Fit the class.
<br><br>
</div>
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>X: dict, sequence, np.array or pd.DataFrame</strong>
<blockquote>
Data containing the features, with shape=(n_samples, n_features).
</blockquote>
<strong>y: int, str, sequence, np.array or pd.Series</strong>
<blockquote>
<ul>
<li>If int: Index of the target column in X.</li>
<li>If string: Name of the target column in X.</li>
<li>Else: Target column with shape=(n_samples,).</li>
</ul>
</blockquote>
</tr>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Returns:</strong></td>
<td width="75%" style="background:white;">
<strong>self: Encoder</strong>
<blockquote>
Fitted instance of self.
</blockquote>
</tr>
</table>
<br />


<a name="encoder-fit-transform"></a>
<pre><em>method</em> <strong style="color:#008AB8">fit_transform</strong>(X, y) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/data_cleaning.py#L42">[source]</a></div></pre>
<div style="padding-left:3%">
Fit the Encoder and return the encoded data.
<br><br>
</div>
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>X: dict, sequence, np.array or pd.DataFrame</strong>
<blockquote>
Data containing the features, with shape=(n_samples, n_features).
</blockquote>
<strong>y: int, str, sequence, np.array, pd.Series</strong>
<blockquote>
<ul>
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
</tr>
</table>
<br />

<a name="encoder-get-params"></a>
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


<a name="encoder-log"></a>
<pre><em>method</em> <strong style="color:#008AB8">log</strong>(msg, level=0)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basetransformer.py#L194">[source]</a></div></pre>
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


<a name="encoder-save"></a>
<pre><em>method</em> <strong style="color:#008AB8">save</strong>(filename=None)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basetransformer.py#L218">[source]</a></div></pre>
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


<a name="encoder-set-params"></a>
<pre><em>method</em> <strong style="color:#008AB8">set_params</strong>(**params) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L2155">[source]</a></div></pre>
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
<strong>self: Encoder</strong>
<blockquote>
Estimator instance.
</blockquote>
</tr>
</table>
<br />


<a name="encoder-transform"></a>
<pre><em>method</em> <strong style="color:#008AB8">transform</strong>(X, y=None) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/data_cleaning.py#L692">[source]</a></div></pre>
<div style="padding-left:3%">
Encode the data.
<br><br>
</div>
<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>X: dict, sequence, np.array or pd.DataFrame</strong>
<blockquote>
Data containing the features, with shape=(n_samples, n_features).
</blockquote>
<strong>y: int, str, sequence, np.array, pd.Series or None, optional (default=None)</strong>
<blockquote>
Does nothing. Implemented for continuity of the API.
</blockquote>
</tr>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Returns:</strong></td>
<td width="75%" style="background:white;">
<strong>X: pd.DataFrame</strong>
<blockquote>
Transformed feature set.
</blockquote>
</tr>
</table>
<br />


## Example
---------
```python
from atom import ATOMClassifier

atom = ATOMClassifier(X, y)
atom.encode(max_onehot=5, encode_type='LeaveOneOut')
```
or
```python
from atom.data_cleaning import Encoder

encoder = Encoder(max_onehot=5, encode_type='LeaveOneOut')
encoder.fit(X_train, y_train)
X = encoder.transform(X)
```