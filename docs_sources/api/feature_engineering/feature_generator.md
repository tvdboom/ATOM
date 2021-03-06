# FeatureGenerator
------------------

<a name="atom"></a>
<pre><em>class</em> atom.feature_engineering.<strong style="color:#008AB8">FeatureGenerator</strong>(strategy="DFS", n_features=None, generations=20, population=500,
                                                operators=None, n_jobs=1, verbose=0, logger=None, random_state=None)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/feature_engineering.py#L49">[source]</a></div></pre>
Use Deep feature Synthesis or a genetic algorithm to create new combinations
of existing features to capture the non-linear relations between the original
features. This class can be accessed from atom through the
[feature_generation](../../ATOM/atomclassifier/#feature-generation)
method. Read more in the [user guide](../../../user_guide/#generating-new-features).
<table>
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>strategy: str, optional (default="DFS")</strong>
<blockquote>
Strategy to crate new features. Choose from:
<ul>
<li>"DFS" to use Deep Feature Synthesis.</li>
<li>"GFG" or "genetic" to use Genetic Feature Generation.</li>
</ul>
</blockquote>
<strong>n_features: int or None, optional (default=None)</strong>
<blockquote>
Number of newly generated features to add to the dataset (no
more than 1% of the population for the genetic strategy). If
None, select all created features.
</blockquote>
<strong>generations: int, optional (default=20)</strong>
<blockquote>
Number of generations to evolve. Only for the genetic strategy.
</blockquote>
<strong>population: int, optional (default=500)</strong>
<blockquote>
Number of programs in each generation. Only for the genetic strategy.
</blockquote>
<strong>operators: str, list, tuple or None, optional (default=None)</strong>
<blockquote>
Name of the operators to be used on the features. None to use all.
Choose from: "add", "sub", "mul", "div", "sqrt", "log", "sin", "cos", "tan".
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
<strong>logger: str, class or None, optional (default=None)</strong>
<blockquote>
<ul>
<li>If None: Doesn't save a logging file.</li>
<li>If str: Name of the logging file. Use "auto" for default name.</li>
<li>If class: python <code>Logger</code> object.</li>
</ul>
The default name consists of the class' name followed by the
timestamp of the logger's creation.
</blockquote>
<strong>random_state: int or None, optional (default=None)</strong>
<blockquote>
Seed used by the random number generator. If None, the random number
generator is the <code>RandomState</code> instance used by <code>numpy.random</code>.
</blockquote>
</td>
</tr>
</table>
<br>


!!! tip
    DFS can create many new features and not all of them will be useful. Use
    [FeatureSelector](/API/feature_engineering/feature_selector) to reduce
    the number of features!

!!! warning
    Using the div, log or sqrt operators can return new features with `inf` or
    `NaN` values. Check the warnings that may pop up or use atom's
    [missing](/API/ATOM/atomclassifier/#data-attributes) property.

!!! warning
    When using DFS with `n_jobs>1`, make sure to protect your code with `if __name__
    == "__main__"`. Featuretools uses [dask](https://dask.org/), which uses python
    multiprocessing for parallelization. The spawn method on multiprocessing starts
    a new python process, which requires it to import the \__main__ module before it
    can do its task.

<br>

## Attributes
-------------

<table>
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Attributes:</strong></td>
<td width="75%" style="background:white;">
<strong>symbolic_transformer: <a href="https://gplearn.readthedocs.io/en/stable/reference.html#symbolic-transformer">SymbolicTransformer</a></strong>
<blockquote>
Instance used to calculate the genetic features. Only for the genetic strategy.
</blockquote>
<strong>genetic_features: pd.DataFrame</strong>
<blockquote>
Dataframe of the newly created non-linear features. Only for the genetic strategy.
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


<a name="fit"></a>
<pre><em>method</em> <strong style="color:#008AB8">fit</strong>(X, y) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/feature_engineering.py#L149">[source]</a></div></pre>
Fit to data.
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>X: dict, list, tuple, np.ndarray or pd.DataFrame</strong>
<blockquote>
Feature set with shape=(n_samples, n_features).
</blockquote>
<strong>y: int, str or sequence</strong>
<blockquote>
<ul>
<li>If int: Index of the target column in X.</li>
<li>If str: Name of the target column in X.</li>
<li>Else: Target column with shape=(n_samples,).</li>
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


<a name="fit-transform"></a>
<pre><em>method</em> <strong style="color:#008AB8">fit_transform</strong>(X, y) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/data_cleaning.py#L39">[source]</a></div></pre>
Fit to data, then transform it.
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>X: dict, list, tuple, np.ndarray or pd.DataFrame</strong>
<blockquote>
Feature set with shape=(n_samples, n_features).
</blockquote>
<strong>y: int, str or sequence</strong>
<blockquote>
<ul>
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
Feature set with the newly generated features.
</blockquote>
</tr>
</table>
<br />

<a name="get-params"></a>
<pre><em>method</em> <strong style="color:#008AB8">get_params</strong>(deep=True) 
<div align="right"><a href="https://github.com/scikit-learn/scikit-learn/blob/0fb307bf3/sklearn/base.py#L189">[source]</a></div></pre>
Get parameters for this estimator.
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


<a name="log"></a>
<pre><em>method</em> <strong style="color:#008AB8">log</strong>(msg, level=0)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basetransformer.py#L315">[source]</a></div></pre>
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
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/basetransformer.py#L336">[source]</a></div></pre>
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
<table width="100%">
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
<strong>self: FeatureGenerator</strong>
<blockquote>
Estimator instance.
</blockquote>
</tr>
</table>
<br />


<a name="transform"></a>
<pre><em>method</em> <strong style="color:#008AB8">transform</strong>(X, y=None) 
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/feature_engineering.py#L309">[source]</a></div></pre>
Generate new features.
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>X: dict, list, tuple, np.ndarray or pd.DataFrame</strong>
<blockquote>
Feature set with shape=(n_samples, n_features).
</blockquote>
<strong>y: int, str, sequence or None, optional (default=None)</strong>
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
from atom import ATOMClassifier

atom = ATOMClassifier(X, y)
atom.feature_generation(strategy="genetic", n_features=3, generations=30, population=400)
```
or
```python
from atom.feature_engineering import FeatureGenerator

feature_generator = FeatureGenerator(strategy="genetic", n_features=3, generations=30, population=400)
feature_generator.fit(X_train, y_train)
X = feature_generator.transform(X)
```