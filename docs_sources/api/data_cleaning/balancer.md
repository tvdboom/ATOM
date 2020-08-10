# Balancer
----------

<pre><em>class</em> atom.data_cleaning.<strong style="color:#008AB8">Balancer</strong>(oversample='not majority', undersample=None, n_neighbors=5,
                                  n_jobs=1, verbose=0, logger=None, random_state=None)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L114">[source]</a></div></pre>

Balance the number of rows per target category. Using oversample and
 undersample at the same time or not using any will raise an exception.
 Use only for classification tasks.

<table>
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>oversample: float, string or None, optional (default='not majority')</strong>
<blockquote>
Oversampling strategy using <a href="https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.ADASYN.html">ADASYN</a>.
 Choose from:
<ul>
<li>None: Don't oversample.</li>
<li>float: Fraction minority/majority (only for binary classification).</li>
<li>'minority': Resample only the minority category.</li>
<li>'not minority': Resample all but the minority category.</li>
<li>'not majority': Resample all but the majority category.</li>
<li>'all': Resample all categories.</li>
</ul>
</blockquote>

<strong>undersample: float, string or None, optional (default=None)</strong>
<blockquote>
Undersampling strategy using <a href="https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.under_sampling.NearMiss.html">NearMiss</a>.
 Choose from:
<ul>
<li>None: Don't oversample.</li>
<li>float: Fraction minority/majority (only for binary classification).</li>
<li>'majority': Resample only the majority category.</li>
<li>'not minority': Resample all but the minority category.</li>
<li>'not majority': Resample all but the majority category.</li>
<li>'all': Resample all categories.</li>
</ul>
</blockquote>

<strong>n_neighbors: int, optional (default=5)</strong>
<blockquote>
Number of nearest neighbors used for the ADASYN and NearMiss algorithms.
</blockquote>

<strong>n_jobs: int, optional (default=1)</strong>
<blockquote>
Number of cores to use for parallel processing.
<ul>
<li>If >0: Number of cores to use.</li>
<li>If -1: Use all available cores.</li>
<li>If <-1: Use available_cores - 1 + n_jobs.</li>
</ul>
Beware that using multiple processes on the same machine may cause
memory issues for large datasets.
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
<li>If bool: True for logging file with default name, False for no logger.</li>
<li>If str: Name of the logging file. 'auto' to create an automatic name.</li>
<li>If class: python Logger object.</li>
</ul>
</blockquote>

<strong>random_state: int or None, optional (default=None)</strong>
<blockquote>
Seed used by the random number generator. If None, the random number
 generator is the RandomState instance used by np.random.
</blockquote>

</td>
</tr>
</table>
<br>


## Methods
---------

<table>

<tr>
<td><a href="#Balancer-fit-transform">fit_transform</a></td>
<td>Same as transform.</td>
</tr>

<tr>
<td><a href="#Balancer-get-params">get_params</a></td>
<td>Get parameters for this estimator.</td>
</tr>

<tr>
<td><a href="#Balancer-save">save</a></td>
<td>Save the instance to a pickle file.</td>
</tr>

<tr>
<td><a href="#Balancer-set-params">set_params</a></td>
<td>Set the parameters of this estimator.</td>
</tr>

<tr>
<td><a href="#Balancer-transform">transform</a></td>
<td>Transform the data.</td>
</tr>
</table>
<br>


<a name="Balancer-fit-transform"></a>
<pre><em>function</em> Balancer.<strong style="color:#008AB8">fit_transform</strong>(X, y) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L2155">[source]</a></div></pre>
<div style="padding-left:3%">
Oversample or undersample the data.
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
<li>If int: Position of the target column in X.</li>
<li>If string: Name of the target column in X</li>
<li>Else: Data target column with shape=(n_samples,)</li>
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
Transformed target column.
</blockquote>
</tr>
</table>
<br />

<a name="Balancer-get-params"></a>
<pre><em>function</em> Balancer.<strong style="color:#008AB8">get_params</strong>(deep=True) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L2155">[source]</a></div></pre>
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


<a name="Balancer-save"></a>
<pre><em>function</em> Balancer.<strong style="color:#008AB8">save</strong>(filename=None)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L696">[source]</a></div></pre>
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


<a name="Balancer-set-params"></a>
<pre><em>function</em> Balancer.<strong style="color:#008AB8">set_params</strong>(**params) 
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
<strong>self: Balancer</strong>
<blockquote>
Estimator instance.
</blockquote>
</tr>
</table>
<br />


<a name="Balancer-transform"></a>
<pre><em>function</em> Balancer.<strong style="color:#008AB8">transform</strong>(X, y) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L2155">[source]</a></div></pre>
<div style="padding-left:3%">
Oversample or undersample the data.
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
<li>If int: Position of the target column in X.</li>
<li>If string: Name of the target column in X</li>
<li>Else: Data target column with shape=(n_samples,)</li>
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
Transformed target column.
</blockquote>
</tr>
</table>
<br />


## Example
----------

```python
from atom.data_cleaning import Balancer

Balancer = Balancer(undersample='not majority', n_neigbors=10, verbose=2, random_state=1)
X_balanced, y_balanced = Balancer.transform(X, y)
```