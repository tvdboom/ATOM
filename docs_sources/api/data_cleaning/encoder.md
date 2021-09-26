# Encoder
---------

<div style="font-size:20px">
<em>class</em> atom.data_cleaning.<strong style="color:#008AB8">Encoder</strong>(strategy="LeaveOneOut",
max_onehot=10, ordinal=None, frac_to_other=None, verbose=0, logger=None, **kwargs)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/data_cleaning.py#L897">[source]</a>
</span>
</div>

Perform encoding of categorical features. The encoding type depends on
the number of classes in the column:

* If n_classes=2 or ordinal feature, use Ordinal-encoding.
* If 2 < n_classes <= `max_onehot`, use OneHot-encoding.
* If n_classes > `max_onehot`, use `strategy`-encoding.

Missing values are propagated to the output column. Unknown
classes encountered during transforming are imputed according
to the selected strategy. Classes with low occurrences can be
replaced with the value `other` in order to prevent too high
cardinality. It can be accessed from atom through the
[encode](../../ATOM/atomclassifier/#encode) method. Read more
in the [user guide](../../../user_guide/data_cleaning/#encoding-categorical-features).

<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<strong>strategy: str or estimator, optional (default="LeaveOneOut")</strong><br>
Type of encoding to use for high cardinality features. Choose
from any of the estimators in the <a href="https://contrib.scikit-learn.org/category_encoders/">category-encoders</a>
package or provide a custom one.
<p>
<strong>max_onehot: int or None, optional (default=10)</strong><br>
Maximum number of unique values in a feature to perform one-hot
encoding. If None, <code>strategy</code>-encoding is always used
for columns with more than two classes.
</p>
<p>
<strong>ordinal: dict or None, optional (default=None)</strong><br>
Order of ordinal features, where the dict key is the feature's
name and the value is the class order, e.g. {"salary": ["low",
"medium", "high"]}.
</p>
<p>
<strong>frac_to_other: int, float or None, optional (default=None)</strong><br>
Classes with less occurrences than n_rows * <code>frac_to_other</code>
are replaced with the string <code>other</code>. This transformation
is done before the encoding of the column. If None, skip this step.
</p>
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
<strong>**kwargs</strong><br>
Additional keyword arguments passed to the <code>strategy</code> estimator.
</td>
</tr>
</table>

!!! tip
    Use atom's [categorical](../../ATOM/atomclassifier/#data-attributes) attribute
    for a list of the categorical columns in the dataset.

!!!warning
    Two category-encoders estimators are unavailable:

    * OneHotEncoder: Use the `max_onehot` parameter.
    * HashingEncoder: Incompatibility of APIs.

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
<div style="font-size:18px"><em>method</em> <strong style="color:#008AB8">fit</strong>(X, y=None)
<span style="float:right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/data_cleaning.py#L976">[source]</a></span></div>
Fit to data. Note that leaving y=None can lead to errors if the
`strategy` encoder requires target values.
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>X: dict, list, tuple, np.ndarray or pd.DataFrame</strong><br>
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
<strong>self: Encoder</strong><br>
Fitted instance of self.
</tr>
</table>
<br />


<a name="fit-transform"></a>
<div style="font-size:18px"><em>method</em> <strong style="color:#008AB8">fit_transform</strong>(X, y=None)
<span style="float:right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/data_cleaning.py#L74">[source]</a></span></div>
Fit to data, then transform it. Note that leaving y=None can lead
to errors if the `strategy` encoder requires target values.
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>X: dict, list, tuple, np.ndarray or pd.DataFrame</strong><br>
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
<strong>X: pd.DataFrame</strong><br>
Transformed feature set.
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
If True, will return the parameters for this estimator and contained subobjects that are estimators.
</p>
</td>
</tr>
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Returns:</strong></td>
<td width="80%" class="td_params">
<strong>params: dict</strong><br>
Dictionary of the parameter names mapped to their values.
</td>
</tr>
</table>
<br />


<a name="log"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">log</strong>(msg, level=0)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/basetransformer.py#L349">[source]</a>
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
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/basetransformer.py#L370">[source]</a>
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
<strong>self: Encoder</strong><br>
Estimator instance.
</td>
</tr>
</table>
<br />


<a name="transform"></a>
<div style="font-size:18px"><em>method</em> <strong style="color:#008AB8">transform</strong>(X, y=None) 
<span style="float:right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/data_cleaning.py#L1075">[source]</a></span></div>
Encode the data.
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>X: dict, list, tuple, np.ndarray or pd.DataFrame</strong><br>
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
<strong>X: pd.DataFrame</strong><br>
Transformed feature set.
</tr>
</table>
<br />



## Example

```python
from atom import ATOMClassifier

atom = ATOMClassifier(X, y)
atom.encode(strategy="CatBoost", max_onehot=5)
```
or
```python
from atom.data_cleaning import Encoder

encoder = Encoder(strategy="CatBoost", max_onehot=5)
encoder.fit(X_train, y_train)
X = encoder.transform(X)
```