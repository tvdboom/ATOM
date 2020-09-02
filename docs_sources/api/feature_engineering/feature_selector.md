# FeatureSelector
------------------

<pre><em>class</em> atom.feature_selection.<strong style="color:#008AB8">FeatureSelector</strong>(strategy=None, solver=None, n_features=None, max_frac_repeated=1.,
                                             max_correlation=1., n_jobs=1, verbose=0, logger=None,
                                             random_state=None, **kwargs)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L114">[source]</a></div></pre>

Remove features according to the selected strategy. Ties between
 features with equal scores will be broken in an unspecified way.
 Also removes features with too low variance and finds pairs of
 collinear features based on the Pearson correlation coefficient. For
 each pair above the specified limit (in terms of absolute value), it
 removes one of the two. See the [user guide](../../../user_guide/#selecting-useful-features)
 for more information.

<table>
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>strategy: string or None, optional (default=None)</strong>
<blockquote>
Feature selection strategy to use. Choose from:
<ul>
<li>None: Do not perform any feature selection algorithm.</li>
<li>'univariate': Select best features according to a univariate F-test.</li>
<li>'PCA': Perform principal component analysis.</li>
<li>'SFM': Select best features according to a model.</li>
<li>'RFE': Perform recursive feature elimination.</li>
<li>'RFECV': Perform RFE with cross-validated selection.</li>
</ul>
</blockquote>

<strong>solver: string, callable or None, optional (default=None)</strong>
<blockquote>
Solver or model to use for the feature selection strategy. See the
sklearn documentation for an extended description of the choices.
Select None for the default option per strategy (not applicable
for SFM, RFE and RFECV).
<ul>
<li>for 'univariate', choose from:
    <ul>
    <li>'f_classif'</li>
    <li>'f_regression'</li>
    <li>'mutual_info_classif'</li>
    <li>'mutual_info_regression'</li>
    <li>'chi2'</li>
    <li>Any function taking two arrays (X, y), and returning
        arrays (scores, p-values). See the sklearn <a href="https://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection">documentation</a>.</li>
    </ul></li>
<li>for 'PCA', choose from:
    <ul>
    <li>'auto' (default)</li>
    <li>'full'</li>
    <li>'arpack'</li>
    <li>'randomized'</li>
    </ul></li>
<li>for 'SFM', 'RFE' and 'RFECV:<br>
Choose a supervised learning model. The estimator must have either feature_importances_
 or coef_ attribute after fitting. You can use a model from the ATOM package (add _class
 or _reg after the model name to specify a classification or regression task respectively,
 e.g. solver='LGB_reg'). No default option. Note that the RFE and RFECV strategies don't
 work when the solver is a CatBoost model due to incompatibility of the APIs.</li>
</ul>
</blockquote>

<strong>n_features: int, float or None, optional (default=None)</strong>
<blockquote>
Number of features to select. Choose from:
<ul>
<li>if None: Select all features.</li>
<li>if < 1: Fraction of the total features to select.</li>
<li>if >= 1: Number of features to select.</li>
</ul>
If strategy='SFM' and the threshold parameter is not specified, the threshold will be
 set to -np.inf in order to make this parameter the number of features to select.<br>
If strategy='RFECV', it's the minimum number of features to select.
</blockquote>

<strong>max_frac_repeated: float or None, optional (default=1.)</strong>
<blockquote>
Remove features with the same value in at least this fraction of
 the total rows. The default is to keep all features with non-zero
 variance, i.e. remove the features that have the same value in all
 samples. None to skip this step.
</blockquote>

<strong>max_correlation: float or None, optional (default=1.)</strong>
<blockquote>
Minimum value of the Pearson correlation coefficient to identify
 correlated features. A value of 1 removes on of 2 equal columns. A dataframe
 of the removed features and their correlation values can be accessed
 through the collinear attribute. None to skip this step.
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

<strong>**kwargs</strong>
<blockquote>
Any extra keyword argument for the PCA, SFM, RFE or RFECV estimators.
 See the corresponding sklearn documentation for the available options.
</blockquote>

</td>
</tr>
</table>
<br>


## Properties

The plot aesthetics can be customized using the properties described hereunder, e.g.
 `FeatureSelector.tick_fontsize = 12`. Note that the properties are bound to the class,
 not the instance.

<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Plotting properties:</strong></td>
<td width="75%" style="background:white;">
<strong>style: str</strong>
<blockquote>
Seaborn plotting style. See the <a href="https://seaborn.pydata.org/tutorial/aesthetics.html#seaborn-figure-styles">documentation</a>.
</blockquote>

<strong>palette: str</strong>
<blockquote>
Seaborn color palette. See the <a href="https://seaborn.pydata.org/tutorial/color_palettes.html">documentation</a>.
</blockquote>

<strong>title_fontsize: int</strong>
<blockquote>
Fontsize for plot titles.
</blockquote>

<strong>label_fontsize: int</strong>
<blockquote>
Fontsize for labels and legends.
</blockquote>

<strong>tick_fontsize: int</strong>
<blockquote>
Fontsize for ticks.
</blockquote>

</td></tr>
</table>
<br>


## Attributes
-------------

<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Attributes:</strong></td>
<td width="75%" style="background:white;">

<strong>collinear: pd.DataFrame</strong>
<blockquote>
Dataframe of the removed collinear features.
 Columns include:
<ul>
<li><b>drop_feature:</b> name of the feature dropped by the method.</li>
<li><b>correlated feature:</b> Name of the correlated feature(s).</li>
<li><b>correlation_value:</b> Pearson correlation coefficient(s) of the feature pairs.</li>
</ul>
</blockquote>

<strong>univariate: class</strong>
<blockquote>
<a href="https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html">SelectKBest</a>
 instance used to fit the estimator. Only if strategy='univariate'.
</blockquote>

<strong>scaler: class</strong>
<blockquote>
<a href="../../data_cleaning/scaler/">Scaler</a>
 instance used to scale the data. Only if strategy='PCA' and data was not scaled.
</blockquote>

<strong>pca: class</strong>
<blockquote>
<a href="https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html">PCA</a>
 instance used to fit the estimator. Only if strategy='PCA'.
</blockquote>

<strong>sfm: class</strong>
<blockquote>
<a href="https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html">SelectFromModel</a>
 instance used to fit the estimator. Only if strategy='SFM'.
</blockquote>

<strong>rfe: class</strong>
<blockquote>
<a href="https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html#sklearn.feature_selection.RFE">RFE</a>
 instance used to fit the estimator. Only if strategy='RFE'.
</blockquote>

<strong>rfecv: class</strong>
<blockquote>
<a href="https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html#sklearn.feature_selection.RFECV">RFECV</a>
 instance used to fit the estimator. Only if strategy='RFECV'.
</blockquote>

</td>
</tr>
</table>
<br>



## Methods
----------

<table width="100%">
<tr>
<td><a href="#FeatureSelector-fit">fit</a></td>
<td>Fit the class.</td>
</tr>

<tr>
<td><a href="#FeatureSelector-fit-transform">fit_transform</a></td>
<td>Fit the class and return the transformed data.</td>
</tr>

<tr>
<td><a href="#FeatureSelector-get-params">get_params</a></td>
<td>Get parameters for this estimator.</td>
</tr>

<tr>
<td><a href="#FeatureSelector-save">save</a></td>
<td>Save the instance to a pickle file.</td>
</tr>

<tr>
<td><a href="#FeatureSelector-set-params">set_params</a></td>
<td>Set the parameters of this estimator.</td>
</tr>

<tr>
<td><a href="#FeatureSelector-transform">transform</a></td>
<td>Transform the data.</td>
</tr>

<tr>
<td><a href="#FeatureSelector-plot-pca">plot_pca</a></td>
<td>Plot the explained variance ratio vs the number of components.</td>
</tr>

<tr>
<td><a href="#FeatureSelector-plot-components">plot_components</a></td>
<td>Plot the explained variance ratio per component.</td>
</tr>

<tr>
<td><a href="#FeatureSelector-plot-rfecv">plot_rfecv</a></td>
<td>Plot the scores obtained by the estimator on the RFECV.</td>
</tr>

</table>
<br>


<a name="FeatureSelector-fit"></a>
<pre><em>function</em> FeatureSelector.<strong style="color:#008AB8">fit</strong>(X, y=None) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L2155">[source]</a></div></pre>
<div style="padding-left:3%">
Fit the class. Note that the univariate, sfm (when model is not fitted), rfe and
 rfecv strategies all need a target column. Leaving it None will raise an exception.
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
<li>If None: y is not used in this estimator.</li>
<li>If int: Position of the target column in X.</li>
<li>If string: Name of the target column in X</li>
<li>Else: Data target column with shape=(n_samples,)</li>
</ul>
</blockquote>
</tr>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Returns:</strong></td>
<td width="75%" style="background:white;">
<strong>self: FeatureSelector</strong>
<blockquote>
Fitted instance of self.
</blockquote>
</tr>
</table>
<br />


<a name="FeatureSelector-fit-transform"></a>
<pre><em>function</em> FeatureSelector.<strong style="color:#008AB8">fit_transform</strong>(X, y) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L2155">[source]</a></div></pre>
<div style="padding-left:3%">
Fit the FeatureSelector and return the transformed feature set. Note that the
 univariate, sfm (when model is not fitted), rfe and rfecv strategies need a target
 column. Leaving it None will raise an exception.
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
<li>If None: y is not used in this estimator.</li>
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
</tr>
</table>
<br />

<a name="FeatureSelector-get-params"></a>
<pre><em>function</em> FeatureSelector.<strong style="color:#008AB8">get_params</strong>(deep=True) 
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

<a name="FeatureSelector-save"></a>
<pre><em>function</em> FeatureSelector.<strong style="color:#008AB8">save</strong>(filename=None)
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

<a name="FeatureSelector-set-params"></a>
<pre><em>function</em> FeatureSelector.<strong style="color:#008AB8">set_params</strong>(**params) 
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
<strong>self: FeatureSelector</strong>
<blockquote>
Estimator instance.
</blockquote>
</tr>
</table>
<br />


<a name="FeatureSelector-transform"></a>
<pre><em>function</em> FeatureSelector.<strong style="color:#008AB8">transform</strong>(X, y=None) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L2155">[source]</a></div></pre>
<div style="padding-left:3%">
Transform the feature set.
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
<strong>X: pd.DataFrame</strong>
<blockquote>
Transformed feature set.
</blockquote>
</tr>
</table>
<br />



<a name="FeatureSelector-plot-pca"></a>
<pre><em>function</em> atom.ATOM.<strong style="color:#008AB8">plot_pca</strong>(title=None, figsize=(10, 6), filename=None, display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L86">[source]</a></div></pre>
<div style="padding-left:3%">
Plot the explained variance ratio vs the number of components.
See <a href="../../plots/plot_pca">plot_pca</a> for a description of the parameters.
</div>
<br />


<a name="FeatureSelector-plot-components"></a>
<pre><em>function</em> atom.ATOM.<strong style="color:#008AB8">plot_components</strong>(show=None, title=None, figsize=None, filename=None, display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L146">[source]</a></div></pre>
<div style="padding-left:3%">
Plot the explained variance ratio per components.
See <a href="../../plots/plot_components">plot_components</a> for a description of the parameters.
</div>
<br />


<a name="FeatureSelector-plot-rfecv"></a>
<pre><em>function</em> atom.ATOM.<strong style="color:#008AB8">plot_rfecv</strong>(title=None, figsize=(10, 6), filename=None, display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L213">[source]</a></div></pre>
<div style="padding-left:3%">
Plot the scores obtained by the estimator fitted on every subset of
 the data. See <a href="../../plots/plot_rfecv">plot_rfecv</a> for a description of the parameters.
</div>
<br />



## Example
---------
```python
from atom.feature_engineering import FeatureSelector

feature_selector = FeatureSelector(stratgey='pca', n_features=12, whiten=True, max_correlation=0.96)
X_transformed = FeatureSelector.fit_transform(X, y)

feature_selector.plot_pca(filename='pca', figsize=(8, 5))
```