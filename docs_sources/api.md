# ATOM

<pre><strong><em>class</em> <strong style="color:#008AB8">ATOM</strong>(X, y=None, percentage=100, test_size=0.3, log=None,
           n_jobs=1, warnings=False, verbose=0, random_state=None, verbose=0)</strong><br>
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/base.py#L111">[source]</a></div></pre>

Main class of the package. The `ATOM` class is a parent class of the `ATOMClassifier`
 and `ATOMRegressor` classes. These will inherit all methods and attributes described
 in this page. Note that contrary to scikit-learn's API, the ATOM object already contains
 the dataset on which we want to perform the analysis. Calling a method will automatically
 apply it on the dataset it contains.

!!! warning
    Don't call the `ATOM` class directly! Use `ATOMClassifier` or `ATOMRegressor`
     depending on the task at hand. Click [here](/getting_started/#usage) for an example.

The class initializer will automatically proceed to apply some standard data
cleaning steps unto the data. These steps include:

  * Transforming the input data into a pd.DataFrame (if it wasn't one already)
   that can be accessed through the class' data attributes.
  * Removing columns with prohibited data types ('datetime64',
   'datetime64[ns]', 'timedelta[ns]').
  * Removing categorical columns with maximal cardinality (the number of
   unique values is equal to the number of instances. Usually the case for IDs,
    names, etc...).
  * Remove columns with minimum cardinality.
  * Removing duplicate rows.
  * Remove rows with missing values in the target column.


<table>
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>X: dict, iterable, np.array or pd.DataFrame</strong>
<blockquote>
Dataset containing the features, with shape=(n_samples, n_features).
</blockquote>

<strong>y: string, list, np.array or pd.Series, optional (default=None)</strong>
<blockquote>
<ul>
<li>If None: the last column of X is selected as target column</li>
<li>If string: name of the target column in X</li>
<li>Else: data target column with shape=(n_samples,)</li>
</blockquote>

<strong>percentage: int or float, optional (default=100)</strong>
<blockquote>
Percentage of the provided dataset to use.
</blockquote>

<strong>test_size: float, optional (default=0.3)</strong>
<blockquote>
Split fraction of the train and test set.
</blockquote>

<strong>log: string or None, optional (default=None)</strong>
<blockquote>
Name of the log file. 'auto' for default name with date and time. None to not save any log.
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

<strong> warnings: bool, optional (default=False)</strong>
<blockquote>
If False, it will supress all warnings.
</blockquote>

<strong>verbose: int, optional (default=0)</strong>
<blockquote>
Verbosity level of the class. Possible values are:
<ul>
<li>0 to not print anything</li>
<li>1 to print minimum information</li>
<li>2 to print average information</li>
<li>3 to print maximum information</li>
</blockquote>

<strong>random_state: int or None, optional (default=None)</strong>
<blockquote>
Seed used by the random number generator. If None, the random number
 generator is the RandomState instance used by np.random.
</blockquote>

<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Data attributes:</strong></td>
<td width="80%" style="background:white;">
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
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Attributes:</strong></td>
<td width="80%" style="background:white;">

<strong>mapping: dict</strong>
<blockquote>
Dictionary of the target values mapped to their respective encoded integer. Only for classification tasks.
</blockquote>

<strong>genetic_features: pd.DataFrame</strong>
<blockquote>
Contains the description of the generated features by the feature_insertion method and their scores.
</blockquote>

<strong>collinear: pd.DataFrame</strong>
<blockquote>
Dataframe of the collinear features and their correlation values (if feature_selection was used).
</blockquote>

<strong>errors: dict</strong>
<blockquote>
Dictionary of the encountered exceptions (if any) while fitting the models.
</blockquote>

<strong>winner: class</strong>
<blockquote>
Model subclass that performed best on the test set.
</blockquote>

<strong>scores: pd.DataFrame</strong>
<blockquote>
Dataframe (or list of dataframes if successive_halving=True) of the results. Columns can include:
<ul>
<li>model: model's name (acronym)</li>
<li>total_time: time spent on this model</li>
<li>score_train: metric score on the training set</li>
<li>score_test: metric score on the test set</li>
<li>fit_time: time spent fitting and predicting</li>
<li>bagging_mean: mean score of the bagging's results</li>
<li>bagging_std: standard deviation score of the bagging's results</li>
<li>bagging_time: time spent on the bagging algorithm</li>
</blockquote>


</td>
<tr>
</table>

## Example

```Python
from atom import ATOMClassifier
from sklearn.datasets import load_wine

X, y = load_wine(return_X_y=True)

atom = ATOMClassifier(X, y, log='auto', n_jobs=2, verbose=2)

atom.pipeline(models=['LR', 'LDA', 'RF', 'lSVM'],
	          metric='f1_macro',
	          max_iter=10,
	          max_time=1000,
	          init_points=3,
	          cv=4,
	          bagging=10)  
```

<br>

## Utilities

The ATOM class contains a variety of methods to help you handle the data and
inspect the pipeline.

<table width="100%">
<tr>
<td width="20%"><a href="#atom-stats">stats</a></td>
<td>Print out a list of basic statistics on the dataset.</td>
</tr>

<tr>
<td><a href="#atom-scale">scale</a></td>
<td>Scale all the features to mean=1 and std=0.</td>
</tr>

<tr>
<td><a href="#atom-update">update</a></td>
<td>Update all data attributes.</td>
</tr>

<tr>
<td><a href="#atom-report">report</a></td>
<td>Get an extensive profile analysis of the data.</td>
</tr>

<tr>
<td><a href="#atom-results">results</a></td>
<td>Print final results for a specific metric.</td>
</tr>

<tr>
<td><a href="#atom-save">save</a></td>
<td>Save the ATOM class to a pickle file.</td>
</tr>
</table>
<br>


<a name="atom-stats"></a>
<pre><em>function</em> ATOM.<strong style="color:#008AB8">stats</strong>()
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/base.py#L420">[source]</a></div></pre>
<div style="padding-left:5%" width="100%">
Print out a list of basic statistics on the dataset.
</div>
<br /><br />


<a name="atom-scale"></a>
<pre><em>function</em> ATOM.<strong style="color:#008AB8">scale</strong>()
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/base.py#L481">[source]</a></div></pre>
<div style="padding-left:5%" width="100%">
Scale all the features to mean=1 and std=0.
</div>
<br /><br />


<a name="atom-update"></a>
<pre><em>function</em> ATOM.<strong style="color:#008AB8">update</strong>(df='dataset')
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/base.py#L510">[source]</a></div></pre>
<div style="padding-left:5%" width="100%">
If you change any of the class' data attributes in between the pipeline, you
 should call this method to change all other data attributes to their correct
  values. Independent attributes are updated in unison, that is, setting
   df='X_train' will also update X_test, y_train and y_test, or df='train'
    will also update the test set, etc... This means that you can change both
     X_train and y_train and update them with one call of the method.
<br /><br />
<table width="100%">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>df: string, optional(default='dataset')</strong>
<blockquote>
Data attribute that has been changed.
</blockquote>
</td></tr>
</table>
</div>
<br />


<a name="atom-report"></a>
<pre><em>function</em> ATOM.<strong style="color:#008AB8">report</strong>(df='dataset', rows=None, filename=None)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/base.py#L568">[source]</a></div></pre>
<div style="padding-left:5%" width="100%">
Get an extensive profile analysis of the data. The report is rendered
in HTML5 and CSS3. Note that this method can be slow for very large datasets.
Dependency: [pandas-profiling](https://pandas-profiling.github.io/pandas-profiling/docs/).
<br /><br />
<table width="100%">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>df: string, optional(default='dataset')</strong>
<blockquote>
Name of the data class attribute to get the report from.
</blockquote>
<strong>rows: int or None, optional(default=None)</strong>
<blockquote>
Number of rows to process (randomly picked). None for all rows.
</blockquote>
<strong>filename: str or None, optional (default=None)</strong>
<blockquote>
Name of the file when saved (as .html). None to not save anything.
</blockquote>
</tr>
</table>
</div>
<br />


<a name="atom-results"></a>
<pre><em>function</em> ATOM.<strong style="color:#008AB8">results</strong>(metric=None)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/base.py#L616">[source]</a></div></pre>
<div style="padding-left:5%" width="100%">
Print the final results for a specific metric. This method can only
 be called after running the [`pipeline`](#atom-pipeline) method.
<br /><br />
<table width="100%">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>metric: string or None, optional (default=None)</strong>
<blockquote>
String of one of sklearn's predefined metrics. If None, the metric
used to fit the pipeline is selected.
</blockquote>
</tr>
</table>
</div>
<br />


<a name="atom-save"></a>
<pre><em>function</em> ATOM.<strong style="color:#008AB8">save</strong>(filename=None)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/base.py#L664">[source]</a></div></pre>
<div style="padding-left:5%" width="100%">
Save the class to a pickle file.
<br /><br />
<table width="100%">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>filename: str or None, optional (default=None)</strong>
<blockquote>
Name to save the file with. None to save with default name.
</blockquote>
</tr>
</table>
</div>
<br />



## Data cleaning

Before throwing your data in a model, it is crucial to apply some standard data
cleaning steps. ATOM provides four data cleaning methods to handle missing values,
 categorical columns, outliers and unbalanced datasets.

<table width="100%">
<tr>
<td><a href="#atom-impute">impute</a></td>
<td>Handle missing values in the dataset.</td>
</tr>

<tr>
<td><a href="#atom-encode">encode</a></td>
<td>Encode categorical columns.</td>
</tr>

<tr>
<td><a href="#atom-outliers">outliers</a></td>
<td>Remove outliers from the training set.</td>
</tr>

<tr>
<td><a href="#atom-balance">balance</a></td>
<td>Balance the number of instances per target class.</td>
</tr>
</table>
<br>



## Feature selection

To further pre-process the data you can create new non-linear features using a
 genetic algorithm or, if your dataset is too large, remove features using one
 of the provided strategies.

<table width="100%">
<tr>
<tr>
<td><a href="#atom-feature-insertion">feature_insertion</a></td>
<td>Use a genetic algorithm to create new combinations of existing features.</td>
</tr>

<tr>
<td><a href="#atom-feature-selection">feature_selection</a></td>
<td>Remove features according to the selected strategy.</td>
</tr>
</table>
<br>



## Pipeline

The pipeline method is where the models are fitted to the data and their
 performance is evaluated according to the selected metric. For every model, the
 pipeline applies the following steps:

1. The optimal hyperparameters are selectred using a Bayesian Optimization (BO)
 algorithm with gaussian process as kernel. The resulting score of each step of
 the BO is either computed by cross-validation on the complete training set or
 by randomly splitting the training set every iteration into a (sub) training
 set and a validation set. This process can create some data leakage but
 ensures a maximal use of the provided data. The test set, however, does not
 contain any leakage and will be used to determine the final score of every model.
 Note that, if the dataset is relatively small, the best score on the BO can
 consistently be lower than the final score on the test set (despite the
 leakage) due to the considerable fewer instances on which it is trained.
<div></div>
 
2. Once the best hyperparameters are found, the model is trained again, now
 using the complete training set. After this, predictions are made on the test set.
<div></div>

3. You can choose to evaluate the robustness of each model's applying a bagging
 algorithm, i.e. the model will be trained multiple times on a bootstrapped
 training set, returning a distribution of its performance on the test set.

After running the pipeline, the models are attached as subclasses

If you want to compare similar models, you can choose to use a successive halving
 approach when running the pipeline. This technique fits N models to 1/N of the
 data. The best half are selected to go to the next iteration where the process
 is repeated. This continues until only one model remains, which is fitted on
 the complete dataset. Beware that a model's performance can depend greatly on
 the amount of data on which it is trained. For this reason we recommend only to
 use this technique with similar models, e.g. only using tree-based models. 

## Plots

The ATOM class provides a variety of plot methods to analyze the results of the
 pipeline. The plots can be called either directly from the ATOM class, e.g.
  `atom.plot_PRC(['LDA', 'LGB'])`, or from a model subclass,
 e.g. `atom.LDA.plot_PRC()`. 
 
The former can be used to compare multiple models. Filling only one model for
 the models parameter is equal to calling the method from the model subclass.
The latter only plots the results for the specified model. This possibility is
 not available for the [`plot_correlation`](#atom-plot-correlation) and
  [`plot_PCA`](#atom-plot-pca) methods since they are
 unrelated to the models fitted in the pipeline.

The plots aesthetics can be customized using various
 [classmethods](#atom-plot-customization).

<table width="100%">
<tr>
<td><a href="#atom-plot-correlation">plot_correlation</a></td>
<td>Correlation matrix plot of the data.</td>
</tr>

<tr>
<td><a href="#atom-plot-PCA">plot_PCA</a></td>
<td>Plot the explained variance ratio of the components.</td>
</tr>

<tr>
<td><a href="#atom-plot-bagging">plot_bagging</a></td>
<td>Plot a boxplot of the bagging's results.</td>
</tr>

<tr>
<td><a href="#atom-plot-successive-halving">plot_successive_halving</a></td>
<td>Plot the models' scores per iteration of the successive halving.</td>
</tr>

<tr>
<td><a href="#atom-plot-ROC">plot_ROC</a></td>
<td>Plot the Receiver Operating Characteristics curve.</td>
</tr>

<tr>
<td><a href="#atom-plot-PRC">plot_PRC</a></td>
<td>Plot the precision-recall curve.</td>
</tr>

<tr>
<td><a href="#atom-plot-permutation-importance">plot_permutation_importance</a></td>
<td>Plot the feature permutation importance of models.</td>
</tr>

<tr>
<td><a href="#atom-plot-feature-importance">plot_feature_importance</a></td>
<td>Plot the feature permutation importance of models.</td>
</tr>

<tr>
<td><a href="#atom-plot-permutation-importance">plot_permutation_importance</a></td>
<td>Plot tree-based model's normalized feature importances.</td>
</tr>

<tr>
<td><a href="#atom-plot-confusion-matrix">plot_confusion_matrix</a></td>
<td>Plot a model's confusion matrix.</td>
</tr>

<tr>
<td><a href="#atom-plot-threshold">plot_threshold</a></td>
<td>Plot performance metric(s) against threshold values.</td>
</tr>

<tr>
<td><a href="#atom-plot-probabilities">plot_probabilities</a></td>
<td>Plot the probabilities of the different classes of belonging to the target class.</td>
</tr>

</table>
<br>

<a name="atom-plot-correlation"></a>
<pre><em>function</em> ATOM.<strong style="color:#008AB8">plot_correlation</strong>(title=None, figsize=(10, 10),
                               filename=None, display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#37">[source]</a></div></pre>
<div style="padding-left:5%" width="100%">
Correlation matrix plot of the dataset. Ignores non-numeric columns.
<br /><br />
<table width="100%">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>title: string or None, optional (default=None)</strong>
<blockquote>
Plot's title. If None, the default option is used.
</blockquote>
<strong>figsize: tuple, optional (default=(10, 10))</strong>
<blockquote>
Figure's size, format as (x, y).
</blockquote>
<strong>filename: string or None, optional (default=None)</strong>
<blockquote>
Name of the file (to save). If None, the figure is not saved.
</blockquote>
<strong>display: bool, optional (default=True)</strong>
<blockquote>
Wether to render the plot.
</blockquote>
</tr>
</table>
</div>
<br />


<a name="atom-plot-PCA"></a>
<pre><em>function</em> ATOM.<strong style="color:#008AB8">plot_PCA</strong>(show=None, title=None,
                       figsize=None, filename=None, display=True)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#85">[source]</a></div></pre>
<div style="padding-left:5%" width="100%">
Plot the explained variance ratio of the components. Only if a Principal Component Analysis
 was applied on the dataset through the [`feature_selection`](#atom-feature-selection) method.
<br /><br />
<table width="100%">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>show: int or None, optional (default=None)</strong>
<blockquote>
Number of components to show. If None, all are plotted.
</blockquote>
<strong>title: string or None, optional (default=None)</strong>
<blockquote>
Plot's title. If None, the default option is used.
</blockquote>
<strong>figsize: tuple, optional (default=None)</strong>
<blockquote>
Figure's size, format as (x, y). If None, adapts size to `show` parameter.
</blockquote>
<strong>filename: string or None, optional (default=None)</strong>
<blockquote>
Name of the file (to save). If None, the figure is not saved.
</blockquote>
<strong>display: bool, optional (default=True)</strong>
<blockquote>
Wether to render the plot.
</blockquote>
</tr>
</table>
</div>
<br />


<a name="atom-plot-customization"></a>
### Plot customization

The plotting aesthetics can be customized with the use of the `@classmethods` described hereunder, e.g. `ATOMClassifier.set_style('white')`.


<table width="100%">
<tr>
<td width="20%"><a href="#atom-set-style">set_style</a></td>
<td>Change the seaborn plotting style.</td>
</tr>

<tr>
<td width="20%"><a href="#atom-set-palette">set_palette</a></td>
<td>Change the seaborn color palette.</td>
</tr>

<tr>
<td width="20%"><a href="#atom-set-title-fontsize">set_title_fontsize</a></td>
<td>Change the fontsize of the plot's title.</td>
</tr>

<tr>
<td width="20%"><a href="#atom-set-label-fontsize">set_label_fontsize</a></td>
<td>Change the fontsize of the plot's labels and legends.</td>
</tr>

<tr>
<td width="20%"><a href="#atom-set-tick-fontsize">set_tick_fontsize</a></td>
<td>Change the fontsize of the plot's ticks.</td>
</tr>
</table>
<br>

<a name="atom-set-style"></a>
<pre><em>classmethod</em> ATOM.<strong style="color:#008AB8">set_style</strong>(style='darkgrid')
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/base.py#L2188">[source]</a></div></pre>
<div style="padding-left:5%" width="100%">
Change the plotting style. See the seaborn [documentation](https://seaborn.pydata.org/tutorial/aesthetics.html#seaborn-figure-styles).
<br /><br />
<table width="100%">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>style: string, optional (default='darkgrid')</strong>
<blockquote>
Style to change to. Available options are: 'darkgrid', 'whitegrid', 'dark', 'white', and 'ticks'.
</blockquote>
</tr>
</table>
</div>
<br />

<a name="atom-set-palette"></a>
<pre><em>classmethod</em> ATOM.<strong style="color:#008AB8">set_palette</strong>(palette='GnBu_d')
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/base.py#L2205">[source]</a></div></pre>
<div style="padding-left:5%" width="100%">
Change the plotting palette. See the seaborn [documentation](https://seaborn.pydata.org/tutorial/color_palettes.html)
 for the available options.
<br /><br />
<table width="100%">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>palette: string, optional (default='GnBu_d')</strong>
<blockquote>
Palette to change to.
</blockquote>
</tr>
</table>
</div>
<br />

<a name="atom-set-title-fontsize"></a>
<pre><em>classmethod</em> ATOM.<strong style="color:#008AB8">set_title_fontsize</strong>(fontsize=20)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/base.py#L2222">[source]</a></div></pre>
<div style="padding-left:5%" width="100%">
Change the fontsize of the plot's title.
<br /><br />
<table width="100%">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>fontsize: int, optional (default=20)</strong>
<blockquote>
Fontsize to change to.
</blockquote>
</tr>
</table>
</div>
<br />

<a name="atom-set-label-fontsize"></a>
<pre><em>classmethod</em> ATOM.<strong style="color:#008AB8">set_label_fontsize</strong>(fontsize=16)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/base.py#L2237">[source]</a></div></pre>
<div style="padding-left:5%" width="100%">
Change the fontsize of the plot's labels and legends.
<br /><br />
<table width="100%">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>fontsize: int, optional (default=16)</strong>
<blockquote>
Fontsize to change to.
</blockquote>
</tr>
</table>
</div>
<br />

<a name="atom-set-tick-fontsize"></a>
<pre><em>classmethod</em> ATOM.<strong style="color:#008AB8">set_tick_fontsize</strong>(fontsize=12)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/base.py#L2252">[source]</a></div></pre>
<div style="padding-left:5%" width="100%">
Change the fontsize of the plot's ticks.
<br /><br />
<table width="100%">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>fontsize: int, optional (default=12)</strong>
<blockquote>
Fontsize to change to.
</blockquote>
</tr>
</table>
</div>
<br />
