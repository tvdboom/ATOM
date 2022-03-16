# Pruner
--------

<div style="font-size:20px">
<em>class</em> atom.data_cleaning.<strong style="color:#008AB8">Pruner</strong>(strategy="z-score",
method="drop", max_sigma=3, include_target=False, verbose=0, logger=None, **kwargs)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/data_cleaning.py#L1412">[source]</a>
</span>
</div>

Replace or remove outliers. The definition of outlier depends
on the selected strategy and can greatly differ from one
another. Ignores categorical columns. This class can be accessed
from atom through the [prune](../../ATOM/atomclassifier/#prune)
method. Read more in the [user guide](../../../user_guide/data_cleaning/#handling-outliers).

<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<strong>strategy: str or sequence, optional (default="z-score")</strong><br>
Strategy with which to select the outliers. If sequence of
strategies, only samples marked as outliers by all chosen
strategies are dropped. Choose from:
<ul style="line-height:1.2em;margin-top:5px">
<li>"z-score": Z-score of each data value.</li>
<li>"<a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html">iforest</a>": Isolation Forest.</li>
<li>"<a href="https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EllipticEnvelope.html">ee</a>": Elliptic Envelope.</li>
<li>"<a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html">lof</a>": Local Outlier Factor.</li>
<li>"<a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html">svm</a>": One-class SVM.</li>
<li>"<a href="https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html">dbscan</a>": Density-Based Spatial Clustering.</li>
<li>"<a href="https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html">optics</a>": DBSCAN-like clustering approach.</li>
</ul>
<strong>method: int, float or str, optional (default="drop")</strong><br>
Method to apply on the outliers. Only the z-score strategy
accepts another method than "drop". Choose from:
<ul style="line-height:1.2em;margin-top:5px">
<li>"drop": Drop any sample with outlier values.</li>
<li>"min_max": Replace the outlier with the min or max of the column.</li>
<li>Any numerical value with which to replace the outliers.</li>
</ul>
<p>
<strong>max_sigma: int or float, optional (default=3)</strong><br>
Maximum allowed standard deviations from the mean of the column.
If more, it is considered an outlier. Only if strategy="z-score".
</p>
<p>
<strong>include_target: bool, optional (default=False)</strong><br>
Whether to include the target column in the search for
outliers. This can be useful for regression tasks. Only
if strategy="z-score".
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
Additional keyword arguments for the <code>strategy</code>
estimator. If sequence of strategies, the params should be provided
in a dict with the strategy's name as key.
</td>
</tr>
</table>

!!! tip
    Use atom's [outliers](../../ATOM/atomclassifier/#data-attributes) attribute
    for an overview of the number of outlier values per column.

<br>



## Attributes

<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Attributes:</strong></td>
<td width="80%" class="td_params">
<strong>&lt;strategy>: sklearn estimator</strong><br>
Object used to prune the data, e.g.<code>pruner.iforest</code> for the
isolation forest strategy.
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
Apply the outlier strategy to the data.
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
<strong>Pruner</strong><br>
Estimator instance.
</td>
</tr>
</table>
<br />


<a name="transform"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">transform</strong>(X, y=None)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/data_cleaning.py#L1495">[source]</a>
</span>
</div>
Apply the outlier strategy to the data.
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>X: dataframe-like</strong><br>
Feature set with shape=(n_samples, n_features).
</p>
<strong>y: int, str, sequence or None, optional (default=None)</strong>
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
<strong>X: pd.Series</strong><br>
Transformed target column. Only returned if provided.
</p>
</td>
</tr>
</table>
<br />



## Example

=== "atom"
    ```python
    from atom import ATOMRegressor
    
    atom = ATOMRegressor(X, y)
    atom.prune(strategy="z-score", max_sigma=2, include_target=True)
    ```

=== "stand-alone"
    ```python
    from atom.data_cleaning import Pruner
    
    pruner = Pruner(strategy="z-score", max_sigma=2, include_target=True)
    X_train, y_train = pruner.transform(X_train, y_train)
    ```