
<pre><strong><em>class</em> atom.<strong style="color:#008AB8">ATOM</strong>(X,
                y=None,
                n_rows=1,
                test_size=0.3,
                log=None,
                n_jobs=1,
                warnings=True,
                verbose=0,
                random_state=None)</strong><br>
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/atom.py#L114">[source]</a></div></pre>

Main class of the package. The `ATOM` class is a parent class of the `ATOMClassifier`
 and `ATOMRegressor` classes. These will inherit all methods and attributes described
 in this page. Note that contrary to scikit-learn's API, the ATOM object already contains
 the dataset on which we want to perform the analysis. Calling a method will automatically
 apply it on the dataset it contains.

!!! warning
    Don't call the `ATOM` class directly! Use `ATOMClassifier` or `ATOMRegressor`
     depending on the task at hand. Click [here](../getting_started/#usage) for an example.

The class initializer will label-encode the target column if its labels are
not ordered integers. It will also apply some standard data
cleaning steps unto the dataset. These steps include:

  * Transforming the input data into a pd.DataFrame (if it wasn't one already)
   that can be accessed through the class' data attributes.
  * Strip categorical features from white spaces.
  * Removing columns with prohibited data types ('datetime64',
   'datetime64[ns]', 'timedelta[ns]'). ATOM can't (yet) handle these types.
  * Removing categorical columns with maximal cardinality (the number of
   unique values is equal to the number of instances. Usually the case for
    names, IDs, etc...).
  * Removing columns with minimum cardinality (all values are the same).
  * Removing rows with missing values in the target column.


<table>
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>X: dict, sequence, np.array or pd.DataFrame</strong>
<blockquote>
Dataset containing the features, with shape=(n_samples, n_features).
</blockquote>

<strong>y: string, sequence, np.array or pd.Series, optional (default=None)</strong>
<blockquote>
<ul>
<li>If None: the last column of X is selected as target column</li>
<li>If string: name of the target column in X</li>
<li>Else: data target column with shape=(n_samples,)</li>
</blockquote>

<strong>n_rows: int or float, optional (default=1)</strong>
<blockquote>
<ul>
<li>if <=1: fraction of the data to use.</li>
<li>if >1: number of rows to use.</li>
</ul>
</blockquote>

<strong>test_size: float, optional (default=0.3)</strong>
<blockquote>
Split fraction of the train and test set.
</blockquote>

<strong>log: string or None, optional (default=None)</strong>
<blockquote>
Name of the logging file. 'auto' for default name with date and time. None to not save any log.
</blockquote>

<strong>n_jobs: int, optional (default=1)</strong>
<blockquote>
Number of cores to use for parallel processing.
<ul>
<li>If -1, use all available cores</li>
<li>If <-1, use available_cores - 1 + n_jobs</li>
Beware that using multiple processes on the same machine may cause
memory issues for large datasets.
</blockquote>

<strong>verbose: int, optional (default=0)</strong>
<blockquote>
Verbosity level of the class. Possible values are:
<ul>
<li>0 to not print anything</li>
<li>1 to print minimum information</li>
<li>2 to print average information</li>
<li>3 to print maximum information</li>
</ul>
</blockquote>

<strong>warnings: bool, optional (default=True)</strong>
<blockquote>
If False, suppresses all warnings. Note that this will change the
PYTHONWARNINGS environment.
</blockquote>

<strong>random_state: int or None, optional (default=None)</strong>
<blockquote>
Seed used by the random number generator. If None, the random number
 generator is the RandomState instance used by np.random.
</blockquote>

<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Data attributes:</strong></td>
<td width="75%" style="background:white;">
<strong>dataset: pd.DataFrame</strong>
<blockquote>
Complete dataset in the pipeline.
</blockquote>

<strong>train: pd.DataFrame</strong>
<blockquote>
Training set.
</blockquote>

<strong>test: pd.DataFrame</strong>
<blockquote>
Test set.
</blockquote>

<strong>X: pd.DataFrame</strong>
<blockquote>
Feature set.
</blockquote>

<strong>y: pd.Series</strong>
<blockquote>
Target column.
</blockquote>

<strong>X_train: pd.DataFrame</strong>
<blockquote>
Training features.
</blockquote>

<strong>y_train: pd.Series</strong>
<blockquote>
Training target.
</blockquote>

<strong>X_test: pd.DataFrame</strong>
<blockquote>
Test features.
</blockquote>

<strong>y_test: pd.Series</strong>
<blockquote>
Test target.
</blockquote>
</td></tr>

<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Attributes:</strong></td>
<td width="75%" style="background:white;">

<strong>mapping: dict</strong>
<blockquote>
Dictionary of the target values mapped to their respective encoded integer. Only for classification tasks.
</blockquote>

<strong>errors: dict</strong>
<blockquote>
Dictionary of the encountered exceptions (if any) while fitting the models.
</blockquote>

<strong>winner: callable</strong>
<blockquote>
Model subclass that performed best on the test set.
</blockquote>

<strong>scores: pd.DataFrame</strong>
<blockquote>
Dataframe (or list of dataframes if the pipeline was called using successive_halving
 or train_sizing) of the results. Columns can include:
<ul>
<li><b>model:</b> model's name (acronym).</li>
<li><b>total_time:</b> time spent on this model.</li>
<li><b>score_train:</b> metric score on the training set.</li>
<li><b>score_test:</b> metric score on the test set.</li>
<li><b>fit_time:</b> time spent fitting and predicting.</li>
<li><b>bagging_mean:</b> mean score of the bagging's results.</li>
<li><b>bagging_std:</b> standard deviation score of the bagging's results.</li>
<li><b>bagging_time:</b> time spent on the bagging algorithm.</li>
</ul>
</blockquote>
</td>
</tr>
</table>
<br>
