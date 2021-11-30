# Gauss
-------

<div style="font-size:20px">
<em>class</em> atom.data_cleaning.<strong style="color:#008AB8">Gauss</strong>(strategy="yeo-johnson",
verbose=0, logger=None, **kwargs)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/data_cleaning.py#L273">[source]</a>
</span>
</div>

Transform the data to follow a Gaussian distribution. This
transformation is useful for modeling issues related to
heteroscedasticity (non-constant variance), or other situations
where normality is desired. Missing values are disregarded in
fit and maintained in transform. Categorical columns are ignored.
This class can be accessed from atom through the [gauss](../../ATOM/atomclassifier/#gauss)
method. Read more in the [user guide](../../../user_guide/data_cleaning/#making-gaussian-like-features).

<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<strong>strategy: str, optional (default="yeo-johnson")</strong><br>
The transforming strategy. Choose from:
<ul style="line-height:1.2em;margin-top:5px">
<li><a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html">yeo-johnson</a></li>
<li><a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html">box-cox</a> (only works with strictly positive values)</li>
<li><a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html">quantile</a> (non-linear transformation)</li>
</ul>
<strong>verbose: int, optional (default=0)</strong><br>
Verbosity level of the class. Possible values are:
<ul style="line-height:1.2em;margin-top:5px">
<li>0 to not print anything.</li>
<li>1 to print basic information.</li>
</ul>
<strong>logger: str, Logger or None, optional (default=None)</strong><br>
<ul style="line-height:1.2em;margin-top:5px">
<li>If None: Doesn't save a logging file.</li>
<li>If str: Name of the log file. Use "auto" for automatic naming.</li>
<li>Else: Python <code>logging.Logger</code> instance.</li>
</ul>
<strong>random_state: int or None, optional (default=None)</strong><br>
Seed used by the quantile strategy. If None, the random number generator
is the <code>RandomState</code> used by <code>numpy.random</code>.
<p>
<strong>**kwargs</strong><br>
Additional keyword arguments for the <code>strategy</code> estimator.
</p>
</td>
</tr>
</table>

!!! info
    The yeo-johnson and box-cox strategies apply zero-mean, unit-variance
    normalization after transforming. Use the `kwargs` parameter to change
    this behaviour.

!!! tip
    Use atom's [plot_distribution](../../plots/plot_distribution) method to
    visualize the transformation.

!!! warning
    Note that the quantile strategy performs a non-linear transformation.
    This may distort linear correlations between variables measured at the 
    same scale but renders variables measured at different scales more
    directly comparable.

<br>


## Attributes

<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Attributes:</strong></td>
<td width="80%" class="td_params">
<strong>estimator: sklearn transformer</strong><br>
Estimator's instance with which the data is transformed.
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
<div style="font-size:18px"><em>method</em> <strong style="color:#008AB8">fit</strong>(X, y=None)
<span style="float:right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/data_cleaning.py#L338">[source]</a></span></div>
Fit to data.
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>X: dict, list, tuple, np.array, sps.matrix or pd.DataFrame</strong><br>
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
<strong>self: Gauss</strong><br>
Fitted instance of self.
</tr>
</table>
<br />


<a name="fit-transform"></a>
<div style="font-size:18px"><em>method</em> <strong style="color:#008AB8">fit_transform</strong>(X, y=None)
<span style="float:right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/data_cleaning.py#L74">[source]</a></span></div>
Fit to data, then transform it.
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>X: dict, list, tuple, np.array, sps.matrix or pd.DataFrame</strong><br>
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
Scaled feature set.
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
<strong>self: Gauss</strong><br>
Estimator instance.
</td>
</tr>
</table>
<br />


<a name="transform"></a>
<div style="font-size:18px"><em>method</em> <strong style="color:#008AB8">transform</strong>(X, y=None) 
<span style="float:right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/data_cleaning.py#L386">[source]</a></span></div>
Apply the transformations to the data.
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width=80%" style="background:white;">
<p>
<strong>X: dict, list, tuple, np.array, sps.matrix or pd.DataFrame</strong><br>
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
from atom import ATOMRegressor

atom = ATOMRegressor(X, y)
atom.gauss()
```
or
```python
from atom.data_cleaning import Gauss

gauss = Gauss()
gauss.fit(X_train)
X = gauss.transform(X)
```