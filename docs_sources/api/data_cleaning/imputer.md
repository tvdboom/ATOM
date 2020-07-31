# Imputer
--------

<pre><em>class</em> atom.data_cleaning.<strong style="color:#008AB8">Imputer</strong>(strat_num='drop', strat_cat='drop', min_frac_rows=0.5,
                                 min_frac_cols=0.5, missing=None, verbose=0, logger=None)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L114">[source]</a></div></pre>

Impute or remove missing values according to the selected strategy.
 Also removes rows and columns with too many missing values.

<table>
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>strat_num: str, int or float, optional (default='drop')</strong>
<blockquote>
Imputing strategy for numerical columns. Choose from:
<ul>
<li>'drop': Drop rows containing missing values.</li>
<li>'mean': Impute with mean of column.</li>
<li>'median': Impute with median of column.</li>
<li>'knn': Impute using a K-Nearest Neighbors approach.</li>
<li>'most_frequent': Impute with most frequent value.</li>
<li>int or float: Impute with provided numerical value.</li>
</ul>
</blockquote>

<strong>strat_cat: str, optional (default='drop')</strong>
<blockquote>
Imputing strategy for categorical columns. Choose from:
<ul>
<li>'drop': Drop rows containing missing values.</li>
<li>'most_frequent': Impute with most frequent value.</li>
<li>string: Impute with provided string.</li>
</ul>
</blockquote>

<strong>min_frac_rows: float, optional (default=0.5)</strong>
<blockquote>
Minimum fraction of non-missing values in a row. If less, the row is removed.
</blockquote>

<strong>min_frac_cols: float, optional (default=0.5)</strong>
<blockquote>
Minimum fraction of non-missing values in a column. If less, the column is removed.
</blockquote>

<strong>missing: int, float or list, optional (default=None)</strong>
<blockquote>
List of values to treat as 'missing'. None to use the default values:
 [None, np.NaN, np.inf, -np.inf, '', '?', 'NA', 'nan', 'inf']. Note that np.NaN,
 None, np.inf and -np.inf will always be imputed because of the
 incompatibility with the models. 
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
</td>
</tr>
</table>
<br>


## Methods
---------

<table width="100%">
<tr>
<td><a href="#imputer-fit">fit</a></td>
<td>Fit the class.</td>
</tr>

<tr>
<td><a href="#imputer-fit-transform">fit_transform</a></td>
<td>Fit the class and return the transformed data.</td>
</tr>

<tr>
<td><a href="#imputer-get-params">get_params</a></td>
<td>Get parameters for this estimator.</td>
</tr>

<tr>
<td><a href="#imputer-save">save</a></td>
<td>Save the instance to a pickle file.</td>
</tr>


<tr>
<td><a href="#imputer-set-params">set_params</a></td>
<td>Set the parameters of this estimator.</td>
</tr>

<tr>
<td><a href="#imputer-transform">transform</a></td>
<td>Transform the data.</td>
</tr>
</table>
<br>


<a name="imputer-fit"></a>
<pre><em>function</em> Imputer.<strong style="color:#008AB8">fit</strong>(X, y=None) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L2155">[source]</a></div></pre>
<div style="padding-left:3%">
Fit the class.
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
Does nothing. Implemented for continuity of the API.
</blockquote>
</tr>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Returns:</strong></td>
<td width="75%" style="background:white;">
<strong>self: Imputer</strong>
<blockquote>
Fitted instance of self.
</blockquote>
</tr>
</table>
<br />


<a name="imputer-fit-transform"></a>
<pre><em>function</em> Imputer.<strong style="color:#008AB8">fit_transform</strong>(X, y=None) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L2155">[source]</a></div></pre>
<div style="padding-left:3%">
Fit the Imputer and return the imputed data.
 
!!!note
    Leaving `y=None` can lead to inconsistencies in data length between X and y
    if rows are dropped in the transformation. Use this option only for
    prediction methods.

</div>
<table width="100%">
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
<li>If None: y is not used in the transformation.</li>
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
<strong>y: pd.Series</strong>
<blockquote>
Transformed target column. Only returned if provided.
</blockquote>
</tr>
</table>
<br />

<a name="imputer-get-params"></a>
<pre><em>function</em> Imputer.<strong style="color:#008AB8">get_params</strong>(deep=True) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L2155">[source]</a></div></pre>
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


<a name="imputer-save"></a>
<pre><em>function</em> Imputer.<strong style="color:#008AB8">save</strong>(filename=None)
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


<a name="imputer-set-params"></a>
<pre><em>function</em> Imputer.<strong style="color:#008AB8">set_params</strong>(**params) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L2155">[source]</a></div></pre>
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
<strong>self: imputer</strong>
<blockquote>
Estimator instance.
</blockquote>
</tr>
</table>
<br />


<a name="imputer-transform"></a>
<pre><em>function</em> Imputer.<strong style="color:#008AB8">transform</strong>(X, y=None) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L2155">[source]</a></div></pre>
<div style="padding-left:3%">
Impute the data.

!!!note
    Leaving `y=None` can lead to inconsistencies in data length between X and y
    if rows are dropped in the transformation. Use this option only for
    prediction methods.

</div>
<table width="100%">
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
<li>If None: y is not used in the transformation.</li>
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
<strong>y: pd.Series</strong>
<blockquote>
Transformed target column. Only returned if provided.
</blockquote>
</tr>
</table>
<br />


## Example
---------
```python
from atom.data_cleaning import Imputer

imputer = Imputer(strat_num='knn', strat_cat='drop', min_frac_cols=0.8)
X_imputed, y = imputer.fit_transform(X, y)
```