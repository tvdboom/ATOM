# FeatureGenerator
------------------

<pre><em>class</em> atom.feature_selection.<strong style="color:#008AB8">FeatureGenerator</strong>(strategy='DFS', n_features=None, generations=20, population=500,
                                              operators=None, n_jobs=1, verbose=0, logger=None, random_state=None)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L114">[source]</a></div></pre>

Use Deep feature Synthesis or a genetic algorithm to create new combinations
 of existing features to capture the non-linear relations between the original
 features. See the [user guide](../../../user_guide/#generating-new-features) for more information.
<table>
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>strategy: str, optional (default='DFS')</strong>
<blockquote>
Strategy to crate new features. Choose from:
<ul>
<li>'DFS' to use Deep Feature Synthesis.</li>
<li>'GFG' or 'genetic' to use Genetic Feature Generation.</li>
</ul>
</blockquote>
<strong>n_features: int or None, optional (default=None)</strong>
<blockquote>
Number of newly generated features to add to the dataset (if strategy='genetic', no
 more than 1% of the population). If None, select all created.
</blockquote>
<strong>generations: int, optional (default=20)</strong>
<blockquote>
Number of generations to evolve. Only if strategy='genetic'.
</blockquote>
<strong>population: int, optional (default=500)</strong>
<blockquote>
Number of programs in each generation. Only if strategy='genetic'.
</blockquote>
<strong>operators: str, sequence or None, optional (default=None)</strong>
<blockquote>
Name of the operators to be used on the features (for both strategies). None to use all.
 Valid options are: 'add', 'sub', 'mul', 'div', 'sqrt', 'log', 'sin', 'cos', 'tan'.

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


## Attributes
-------------

<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Attributes:</strong></td>
<td width="75%" style="background:white;">

<strong>symbolic_transformer: class</strong>
<blockquote>
Class used to calculate the genetic features, from <a href="https://gplearn.readthedocs.io/en/stable/reference.html#symbolic-transformer">SymbolicTransformer</a>.
 Only if strategy='genetic'.
</blockquote>

<strong>genetic_features: pd.DataFrame</strong>
<blockquote>
Dataframe of the newly created non-linear features. Only if strategy='genetic'. 
 Columns include:
<ul>
<li><b>name:</b> Name of the feature (automatically created).</li>
<li><b>description:</b> Operators used to create this feature.</li>
<li><b>fitness:</b> Fitness score.</li>
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
<td><a href="#FeatureGenerator-fit">fit</a></td>
<td>Fit the class.</td>
</tr>

<tr>
<td><a href="#FeatureGenerator-fit-transform">fit_transform</a></td>
<td>Fit the class and return the transformed data.</td>
</tr>

<tr>
<td><a href="#FeatureGenerator-get-params">get_params</a></td>
<td>Get parameters for this estimator.</td>
</tr>

<tr>
<td><a href="#FeatureGenerator-save">save</a></td>
<td>Save the instance to a pickle file.</td>
</tr>

<tr>
<td><a href="#FeatureGenerator-set-params">set_params</a></td>
<td>Set the parameters of this estimator.</td>
</tr>

<tr>
<td><a href="#FeatureGenerator-transform">transform</a></td>
<td>Transform the data.</td>
</tr>
</table>
<br>


<a name="FeatureGenerator-fit"></a>
<pre><em>function</em> FeatureGenerator.<strong style="color:#008AB8">fit</strong>(X, y) 
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
<strong>self: FeatureGenerator</strong>
<blockquote>
Fitted instance of self.
</blockquote>
</tr>
</table>
<br />


<a name="FeatureGenerator-fit-transform"></a>
<pre><em>function</em> FeatureGenerator.<strong style="color:#008AB8">fit_transform</strong>(X, y) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L2155">[source]</a></div></pre>
<div style="padding-left:3%">
Fit the FeatureGenerator and return the transformed data.
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
<strong>y: int, str, sequence, np.array, pd.Series</strong>
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
Feature set with the newly generated features.
</blockquote>
</tr>
</table>
<br />

<a name="FeatureGenerator-get-params"></a>
<pre><em>function</em> FeatureGenerator.<strong style="color:#008AB8">get_params</strong>(deep=True) 
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


<a name="FeatureGenerator-save"></a>
<pre><em>function</em> FeatureGenerator.<strong style="color:#008AB8">save</strong>(filename=None)
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


<a name="FeatureGenerator-set-params"></a>
<pre><em>function</em> FeatureGenerator.<strong style="color:#008AB8">set_params</strong>(**params) 
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
<strong>self: FeatureGenerator</strong>
<blockquote>
Estimator instance.
</blockquote>
</tr>
</table>
<br />


<a name="FeatureGenerator-transform"></a>
<pre><em>function</em> FeatureGenerator.<strong style="color:#008AB8">transform</strong>(X, y=None) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L2155">[source]</a></div></pre>
<div style="padding-left:3%">
Create new features in the dataset.
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
Feature set with the newly generated features.
</blockquote>
</tr>
</table>
<br />


## Example
---------
```python
from atom.feature_engineering import FeatureGenerator

feature_generator = FeatureGenerator(n_features=3, generations=30, n_jobs=3, verbose=2)
X_transformed = FeatureGenerator.fit_transform(X, y)

print(feature_generator.genetic_features)
```