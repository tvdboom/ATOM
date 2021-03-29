# Pruner
--------

<pre><em>class</em> atom.data_cleaning.<strong style="color:#008AB8">Pruner</strong>(strategy="z-score", method="drop", max_sigma=3,
                                include_target=False, verbose=0, logger=None, **kwargs)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/data_cleaning.py#L920">[source]</a></div></pre>
Replace or remove outliers. The definition of outlier depends
on the selected strategy and can greatly differ from one
another. Ignores categorical columns. This class can be accessed
from atom through the [prune](../../ATOM/atomclassifier/#prune)
method. Read more in the [user guide](../../../user_guide/#handling-outliers).
<table>
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>strategy: str, optional (default="z-score")</strong>
<blockquote>
Strategy with which to select the outliers. Choose from:
<ul>
<li>"z-score": Uses the z-score of each data value.</li>
<li>"iForest": Uses an <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html">Isolation Forest</a>.</li>
<li>"EE": Uses an <a href="https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EllipticEnvelope.html">Elliptic Envelope</a>.</li>
<li>"LOF": Uses a <a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html">Local Outlier Factor</a>.</li>
<li>"SVM": Uses a <a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html">One-class SVM</a>.</li>
<li>"DBSCAN": Uses <a href="https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html">DBSCAN</a> clustering.</li>
<li>"OPTICS": Uses <a href="https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html">OPTICS</a> clustering.</li>
</ul>
</blockquote>
<strong>method: int, float or str, optional (default="drop")</strong>
<blockquote>
Method to apply on the outliers. Only the z-score strategy
accepts another method than "drop". Choose from:
<ul>
<li>"drop": Drop any sample with outlier values.</li>
<li>"min_max": Replace the outlier with the min or max of the column.</li>
<li>Any numerical value with which to replace the outliers.</li>
</ul>
</blockquote>
<strong>max_sigma: int or float, optional (default=3)</strong>
<blockquote>
Maximum allowed standard deviations from the mean of the column.
If more, it is considered an outlier. Only if strategy="z-score".
</blockquote>
<strong>include_target: bool, optional (default=False)</strong>
<blockquote>
Whether to include the target column in the transformation. This can be
useful for regression tasks.
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
The default name consists of the class' name followed by the
timestamp of the logger's creation.
</blockquote>
<strong>**kwargs</strong>
<blockquote>
Additional keyword arguments passed to the <code>strategy</code> estimator.
</blockquote>
</td>
</tr>
</table>
<br>

!!!tip
    Use atom's [outliers](../../ATOM/atomclassifier/#data-attributes) attribute
    for an overview of the number of outlier values per column.

<br>


## Attributes
-------------

<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Attributes:</strong></td>
<td width="75%" style="background:white;">
<strong>&lt;strategy>: sklearn estimator</strong>
<blockquote>
Estimator instance (lowercase strategy) used to prune the data, e.g.
<code>pruner.iforest</code> for the isolation forest strategy.
</blockquote>
</td>
</tr>
</table>
<br>


## Methods
---------

<table>
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


<a name="get-params"></a>
<pre><em>method</em> <strong style="color:#008AB8">get_params</strong>(deep=True) 
<div align="right"><a href="https://github.com/scikit-learn/scikit-learn/blob/0fb307bf3/sklearn/base.py#L189">[source]</a></div></pre>
Get parameters for this estimator.
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


<a name="save"></a>
<pre><em>method</em> <strong style="color:#008AB8">save</strong>(filename=None)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basetransformer.py#L339">[source]</a></div></pre>
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
Set the parameters of this estimator.
<table>
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
<strong>self: Outliers</strong>
<blockquote>
Estimator instance.
</blockquote>
</tr>
</table>
<br />


<a name="transform"></a>
<pre><em>method</em> <strong style="color:#008AB8">transform</strong>(X, y=None) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/data_cleaning.py#L999">[source]</a></div></pre>
Apply the outlier strategy on the data.
<table>
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
atom.prune(strategy="z-score", max_sigma=2, include_target=True)
```
or
```python
from atom.data_cleaning import Pruner

pruner = Pruner(strategy="z-score", max_sigma=2, include_target=True)
X_train, y_train = pruner.transform(X_train, y_train)
```