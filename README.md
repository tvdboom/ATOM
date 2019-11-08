<p align="center">
	<img src="https://github.com/tvdboom/ATOM/blob/master/images/logo.png?raw=true" alt="ATOM" title="ATOM" width="500" height="140"/>
</p>

# Automated Tool for Optimized Modelling
Author: tvdboom  
Email: m.524687@gmail.com

[![Python 3.6|3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![License: MIT](https://img.shields.io/github/license/tvdboom/ATOM)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/atom-ml)](https://pypi.org/project/atom-ml/)
  
Description  
------------------------  
Automated Tool for Optimized Modelling (ATOM) is a python package designed for fast exploration of ML solutions. With just a few lines of code, you can perform basic data cleaning steps, feature selection and compare the performance of multiple machine learning models on a given dataset. ATOM should be able to provide quick insights on which algorithms perform best for the task at hand and provide an indication of the feasibility of the ML solution.

| NOTE: A data scientist with knowledge of the data will quickly outperform ATOM if he applies usecase-specific feature engineering or data cleaning methods. Use ATOM only for a fast exploration of the problem! |
| --- |

Possible steps taken by the ATOM pipeline:
1. Data Cleaning
	* Handle missing values
	* Encode categorical features
	* Remove outliers
	* Balance the dataset
2. Perform feature selection
	* Remove features with too high collinearity
	* Remove features with too low variance
	* Select best features according to a chosen strategy
3. Fit all selected models (either direct or via successive halving)
	* Select hyperparameters using a Bayesian Optimization approach
	* Perform bagging to assess the robustness of the model
4. Analyze the results using the provided plotting functions!

<br/><br/>

<p align="center">
	<img src="https://github.com/tvdboom/ATOM/blob/master/images/diagram.png?raw=true" alt="diagram" title="diagram" width="700" height="250" />
</p>


Installation
------------------------  
Intall ATOM easily using `pip`
	
	pip install atom-ml

Usage  
------------------------  
Call the `ATOMClassifier` or `ATOMRegressor` class and provide the data you want to use:  

    from atom import ATOMClassifier  
    atom = ATOMClassifier(X, Y, log='atom_log', n_jobs=2, verbose=1)

ATOM has multiple data cleaning methods to help you prepare the data for modelling:

    atom.impute(strat_num='mean', strat_cat='most_frequent',  max_frac=0.1)  
    atom.encode(max_onehot=10)  
    atom.outliers(max_sigma=4)  
    atom.balance(oversample=0.8, neighbors=15)  
    atom.feature_selection(strategy='univariate', max_features=0.9)

Fit the data to different models:

    atom.fit(models=['logreg', 'LDA', 'XGB', 'lSVM'],
	         metric='Accuracy',
	         successive_halving=True,
	         max_iter=10,
	         max_time=1000,
	         init_points=3,
	         cv=4,
	         bagging=5)  

Make plots and analyze results: 

	atom.boxplot(filename='boxplot.png')  
	atom.lSVM.plot_probabilities()  
	atom.lda.plot_confusion_matrix()  
  

API
----------------------------- 

**ATOMClassifier(X, Y=None, target=None, percentage=100, test_size=0.3, log=None, n_jobs=1, warnings=False, verbose=0, random_state=None)**  
ATOM class for classification tasks. When initializing the class, ATOM will automatically proceed to apply some standard data cleaning steps unto the data. These steps include transforming the input data into a pd.DataFrame (if it wasn't one already) that can be accessed through the class' attributes, removing columns with prohibited data types, removing categorical columns with maximal cardinality (the number of unique values is equal to the number of instances, usually the case for IDs, names, etc...), and removing duplicate rows and rows with missing values in the target column.

* **X: np.array or pd.DataFrame**  
Data features with shape = [n_samples, n_features]. If Y and target are None, the last column of X is selected as target column. 
* **Y: np.array or pd.Series, optional (default=None)**  
Data target column with shape = [n_samples].
* **target: string, optional (default=None)**  
Name of the target column in X (X needs to be a pd.DataFrame). If Y is provided, target will be ignored.
* **percentage: int, optional (default=100)**  
Percentage of data to use.
* **test_size: float, optional (default=0.3)**  
Split ratio of the train and test set.
* **log: string, optional (default=None)**  
Name of the log file, None to not save any log.
* **n_jobs: int, optional (default=1)**  
Number of cores to use for parallel processing.
	+ If -1, use all available cores
	+ If <-1, use available_cores - 1 + n_jobs  
* **warnings: bool, optional (default=False)**  
Wether to show warnings when running the pipeline.
* **verbose: int, optional (default=0)**  
Verbosity level of the class. Possible values are:  
	+ 0 to not print anything  
	+ 1 to print minimum information
	+ 2 to print medium information
	+ 3 to print maximum information
* **random_state: int, optional (default=None)**  
Seed used by the random number generator. If None, the random number generator is the RandomState instance used by `np.random`.


**ATOMRegressor(X, Y=None, target=None, percentage=100, test_size=0.3, log=None, n_jobs=1, warnings=False, verbose=0, random_state=None)**  
ATOM class for regression tasks. See `ATOMClassifier` for an explanation of the class' parameters.


Class methods
----------------------------- 
ATOM contains multiple methods for standard data cleaning and feature selection processes. Calling on one of them will automatically apply the method on the dataset in the class and update the class' attributes accordingly.

| TIP: Use Pandas Profiling to examine the data first and help you determine suitable parameters for the methods |
| --- |

* **impute(strat_num='remove', strat_cat='remove', max_frac=0.5, missing=[np.nan, None, '', '?', 'NA', 'nan', 'NaN', np.inf, -np.inf])**  
Handle missing values according to the selected strategy. Also removes columns with too many missing values.
	+ strat_num: int, float or string, optional (default='remove')  
	Imputing strategy for numerical columns. Possible values are:
		- 'remove': remove row if any missing value
		- 'mean': impute with mean of column
		- 'median': impute with median of column
		- 'most_frequent': impute with most frequent value
		- int or float: impute with provided numerical value
	+ strat_cat: string, optional (default='remove')  
	Imputing strategy for categorical columns. Possible values are:
		- 'remove': remove row if any missing value
		- 'most_frequent': impute with most frequent value
		- string: impute with provided string
	+ max_frac: float, optional (default=0.5)  
	Maximum allowed fraction of rows with any missing values. If more, the column is removed.
	+ missing: value or list of values, optional (default=[np.nan, None, '', '?', 'NA', 'nan', 'NaN', np.inf, -np.inf])  
	List of values to consider as missing. None, np.nan, '', np.inf and -np.inf are always added to the list since they are incompatible with sklearn models.
* **encode(max_onehot=10)**  
Perform encoding of categorical features. The encoding type depends on the number of unique values in the column: label-encoding for n_unique=2, one-hot-encoding for 2 < n_unique <= max_onehot and target-encoding for n_unique > max_onehot.
	+ max_onehot: int, optional (default=10)  
	Maximum number of unique values in a feature to perform one-hot-encoding.
* **outliers(max_sigma=3, include_target=False)**  
Remove outliers from the training set.
	+ max_sigma: int or float, optional (default=3)  
	Remove rows containing any value with a maximum standard deviation (on the respective column) above max_sigma.
	+ include_target: bool, optional (default=False)  
	Wether to include the target column when searching for outliers.
* **balance(oversample=None, undersample=None, neighbors=5)**  
Balance the number of instances per target class. Only for classification tasks.
	+ oversample: float or string, optional (default=None)  
	Oversampling strategy using [SMOTE](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html). Choose from:
		- None: do not perform oversampling
		- float: fraction minority/majority (only for binary classification)
		- 'minority': resample only the minority class
		- 'not minority': resample all but minority class
		- 'not majority': resample all but majority class
		- 'all': resample all classes
	+ neighbors: int, optional (default=5)  
	Number of nearest neighbors used for SMOTE.
	+ undersample: float or string, optional (default=None)  
	Undersampling strategy using [RandomUnderSampler](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.under_sampling.RandomUnderSampler.html#imblearn.under_sampling.RandomUnderSampler). Choose from:
		- None: do not perform undersampling
		- float: fraction majority/minority (only for binary classification)
		- 'minority': resample only the minority class
		- 'not minority': resample all but minority class
		- 'not majority': resample all but majority class
		- 'all': resample all classes
* **feature_selection(strategy='univariate', solver=None, max_features=None, threshold=-np.inf, frac_variance=1., max_correlation=0.98)**  
Select best features according to the selected strategy. Ties between features with equal scores will be broken in an unspecified way. Also removes features with too low variance and too high collinearity.
	+ strategy: string, optional (default='univariate')  
	Feature selection strategy to use. Choose from:
		- None: do not perform any feature selection algorithm (it does still look for multicollinearity and variance)
		- 'univariate': perform a univariate statistical test
		- 'PCA': perform a principal component analysis
		- 'SFM': select best features from an existing model
	+ solver: string or model class (default=depend on strategy)  
	Solver or model to use for the feature selection strategy. See the scikit-learn documentation for an extended descrition of the choices. Select None for the default option per strategy (not applicable for SFM).
		- for 'univariate', choose from:
			* 'f_classif' (default for classification tasks)
			* 'f_regression' (default for regression tasks)
			* 'mutual_info_classif'
			* 'mutual_info_regression'
			* 'chi2'
		- for 'PCA', choose one from:
			* 'auto' (default)
			* 'full'
			* 'arpack'
			* 'randomized'
		- for 'SFM', choose a model class (not yet fitted). No default.
	+ max_features: int or float, optional (default=None)  
	Number of features to select.
		- None: select all features
		- if >= 1: number of features to select
		- if < 1: fraction of features to select
	+ threshold: string or float, optional (default=-np.inf)  
	Threshold value to attain when selecting the best features (works only when strategy='SFM'). Features whose importance is greater or equal are kept while the others are discarded.
		- if 'mean': set the mean of feature_importances as threshold
		- if 'median': set the median of feature_importances as threshold
	+ frac_variance: float, optional (default=1)  
	Remove features with the same value in at least this fraction of the total.
	+ max_correlation: float, optional (default=0.98)  
	Minimum value of the Pearson correlation cofficient to identify correlated features.
* **fit(models=None, metric=None, successive_halving=False, skip_steps=0, max_iter=15, max_time=np.inf, eps=1e-08, batch_size=1, init_points=5, plot_bo=False, cv=3, bagging=None)**  
Fit class to the selected models. The optimal hyperparameters per model are selectred using a Bayesian Optimization algorithm with gaussian process as kernel. The resulting score of each step of the BO is either computed by cross-validation on the complete training set or by creating a validation set from the training set. This process will create some minimal leakage but ensures a maximal use of the provided data. The test set, however, does not contain any leakage and will be used to determine the final score of every model. After this process, you can choose to test the robustness of the model selecting bootstrapped samples of the training set on which to fit and test (again on the test set) the model, providing a distribution of the models' performance.
	+ models: string or list of strings, optional (default=None)  
 	List of models to fit on the data. If None, all models are chosen. Possible values are (case insensitive):    
		- 'GNB' for Gaussian Naïve Bayes (no hyperparameter tuning)
		- 'MNB' for Multinomial Naïve Bayes  
		- 'BNB' for Bernoulli Naïve Bayes  
		- 'GP' for Gaussian Process (no hyperparameter tuning)
		- 'LinReg' for Linear Regression (OLS, ridge, lasso and elasticnet)  
		- 'LogReg' for Logistic Regression  
		- 'LDA' for Linear Discriminant Analysis  
		- 'QDA' for Quadratic Discriminant Analysis  
		- 'KNN' for K-Nearest Neighbors  
		- 'Tree' for a single Decision Tree  
		- 'Bag' for Bagging (with decision tree as base estimator)
		- 'ET' for Extra-Trees 
		- 'RF' for Random Forest
		- 'AdaB' for AdaBoost  
		- 'GBM' for Gradient Boosting Machine  
		- 'XGB' for XGBoost (if package is available)  
		- 'LGB' for LightGBM (if package is available)
		- 'CatB' for CatBoost (if package is available)
		- 'lSVM' for Linear Support Vector Machine  
		- 'kSVM' for Non-linear Support Vector Machine  
		- 'PA' for Passive Aggressive  
		- 'SGD' for Stochastic Gradient Descent  
		- 'MLP' for Multilayer Perceptron  
	+ metric: string, optional (default=None)  
	Metric on which the pipeline fits the models. If None, the default option is selected depending on the task's type. Possible values are (case insensitive):  
		+ For binary and multiclass classification or regression:  
			- 'max_error'  
			- 'R2'  
			- 'MAE' for Mean Absolute Error  
			- 'MSE' for Mean Squared Error (default)
			- 'MSLE' for Mean Squared Log Error  
		+ Only binary classification:  
			- 'Precision'  
			- 'Recall'  
			- 'Accuracy'
			- 'F1' (default)
			- 'Jaccard'  
			- 'AUC' for Area Under Curve  
	+ successive_halving: bool, optional (default=False)  
	Fit the pipeline using a successive halving approach, that is, fitting the model on 1/N of the data, where N stands for the number of models still in the pipeline. After this, the best half of the models are selected for the next iteration. This process is repeated until only one model is left. Since models perform quite differently depending on the size of the training set, we recommend to use this feature when fitting similar models (e.g: only using tree-based models).
	+ skip_iter: int, optional (default=0)  
	Skip n last iterations of the successive halving.
	+ max_iter: int, optional (default=15)  
	Maximum number of iterations of the BO.
	+ max_time: int, optional (default=np.inf)  
	Maximum time allowed for the BO (in seconds).
	+ eps: float, optional (default=1e-08)  
	Minimum hyperparameter distance between two consecutive steps in the BO.
	+ batch_size: int, optional (default=1)  
	Size of the batch in which the objective is evaluated.
	+ init_points: int, optional (default=5)  
	Initial number of random tests of the BO. If 1, the model is fitted on the default hyperparameters of the package.
	+ plot_bo: bool, optional (default=False)  
	Wether to plot the BO's progress as it runs. Creates a canvas with two plots: the first plot shows the score of every trial and the second shows the distance between the last consecutive steps. Don't forget to call `%matplotlib` at the start of the cell if you are using jupyter notebook!
	+ cv: bool, optional (default=3)  
		- if 1, randomly split the set to a train and validation set and fit and score the BO's selected model on them
		- if >1, perform a k-fold cross validation on the training set and score the BO as the output
	+ bagging: int, optional (default=None)  
	Number of bootstrapped samples used for bagging. If None, no bagging is performed.


Class methods (utilities)
----------------------------- 
* **stats()**  
Print out a list of basic statistics on the dataset.
* **boxplot(iteration=-1, figsize=None, filename=None)**  
Make a boxplot of the bagging's results after fitting the class.
	+ iteration, int, optional (default=-1)  
	Iteration of the successive_halving to plot. If -1, use the last iteration.
	+ figsize, 2d-tuple, optional (default=None)  
	Figure size: format as (x, y). If None, adjust to number of models.
	+ filename: string, optional (default=None)  
	Name of the file when saved. None to not save anything.
* **plot_correlation(X, figsize=(10, 6), filename=None)**  
Make a correlation maxtrix plot of the dataset. Ignores non-numeric columns.
	+ X: array or pd.Dataframe, optional if class is fitted  
	+ figsize, 2d-tuple, optional (default=(10, 6))  
	Figure size: format as (x, y).
	+ filename: string, optional (default=None)  
	Name of the file when saved. None to not save anything.
* **plot_successive_halving(figsize=(10, 6), filename=None)**  
Make a plot of the models' scores per iteration of the successive halving.
	+ figsize, 2d-tuple, optional (default=(10, 6))  
	Figure size: format as (x, y).
	+ filename: string, optional (default=None)  
	Name of the file when saved. None to not save anything.


Class attributes  
-----------------------------  
* **dataset**: Dataframe of the complete dataset.
* **X, Y**: Data features and target.
* **train, test**: Train and test set.
* **X_train, Y_train**: Training set features and target.
* **X_test, Y_test**: Test set features and target.
* **target_mapping**: Dictionary of the target values mapped to their encoded integer (only for classification tasks).
* **collinear**: Dataframe of the collinear features and their correlation values (only if feature_selection was used).
* **univariate**: Univariate feature selection class (if used), from scikit-learn [SelectKBest](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html).
* **PCA**: Principal component analysis class (if used), from scikit-learn [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).
* **SFM**: Select from model class (if used), from scikit-learn [SelectFromModel](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html).
* **errors**: Dictionary of the encountered exceptions (if any) while fitting the models.
* **results**: Dataframe (or array of dataframes if successive_halving=True) of the results.


### After fitting, the models become subclasses of the main class. They can be called upon for  handy plot functions and attributes. If successive_halving=True, the model subclass corresponds to the last fitted model.


Subclass methods (utilities)  
-----------------------------  
* **plot_threshold(metric=None, steps=100, figsize=(10, 6), filename=None)**  
Plot performance metrics against multiple threshold values. Only for binary classification tasks.  
	+ metric: string or list of strings, optional (default=None)  
	Metric(s) to plot. If None, the selected metric will be the one chosen to fit the model.
    + steps: int, optional (default=100)  
    Number of thresholds to try between 0 and 1.
   	+ figsize, 2d-tuple, optional (default=(10, 6))  
	Figure size: format as (x, y).
	+ filename: string, optional (default=None)  
	Name of the file when saved. None to not save anything.
* **plot_probabilities(target_class=1, figsize=(10, 6), filename=None)**  
Plots the probability of every class in the target variable against the class selected by target_class. Only for classification tasks.
	+ target_class: int, optional (default=1)
	Target class to plot the probabilities against. A value of 0 corresponds to the first class, 1 to the second class, etc...
	+ figsize, 2d-tuple, optional (default=(10, 6))  
	Figure size: format as (x, y).
	+ filename: string, optional (default=None)  
	Name of the file when saved. None to not save anything.
* **plot_feature_importance(figsize=(10, 6), filename=None)**  
Plots the feature importance scores. Only works with tree based algorithms (Tree, Bag, ET, RF, AdaBoost, GBM, XGB and LGBM).
	+ figsize, 2d-tuple, optional (default=(10, 6))  
	Figure size: format as (x, y).
	+ filename: string, optional (default=None)  
	Name of the file when saved. None to not save anything.
* **plot_ROC(figsize=(10, 6), filename=None)**  
Plots the ROC curve. Only for binary classification tasks.  
 	+ figsize, 2d-tuple, optional (default=(10, 6))  
	Figure size: format as (x, y).
	+ filename: string, optional (default=None)  
	Name of the file when saved. None to not save anything.
* **plot_confusion_matrix(normalize=True, figsize=(10, 6), filename=None)**  
Plot the confusion matrix for the model. Only for binary classification tasks.  
	+ normalize: bool, optional (default=True)
	Wether to normalize the confusion matrix.
   	+ figsize, 2d-tuple, optional (default=(10, 6))  
	Figure size: format as (x, y).
	+ filename: string, optional (default=None)  
	Name of the file when saved. None to not save anything.
* **plot_tree(num_trees=0, max_depth=None, rotate=False, figsize=(10, 6), filename=None)**  
Plot a single decision tree of the model. Only for tree-based algorithms. Dependency: [graphviz](https://graphviz.gitlab.io/download/).
	+ num_trees: int, otional (default=0 --> first tree)  
	Number of the tree to plot (if ensemble).
	+ max_depth: int, optional (default=None)  
	Maximum depth of the plotted tree. None for no limit.
	+ rotate: bool, optional (default=False)  
	When True, orientate the tree left-right instead of top-bottom.
   	+ figsize, 2d-tuple, optional (default=(10, 6))  
	Figure size: format as (x, y).
	+ filename: string, optional (default=None)  
	Name of the file when saved. None to not save anything.  
* **save(filename=None)**  
Save the best found model as a pickle file.
	 + filename: string, optional (default=None)  
	Name of the file when saved. If None, it will be saved as ATOM_[model_type].


Sublass methods (metrics)  
-----------------------------
Call any of the metrics as a method. It will return the metric (evaluated on the cross-validation) for the best model found by the BO.
+ **atom.knn.AUC()**: Returns the AUC score for the best trained KNN  
+ **atom.adaboost.MSE()**: Returns the MSE score for the best trained AdaBoost 
+ **atom.xgb.Accuracy()**: Returns the accuracy score for the best trained XGBoost model


Subclass attributes
-----------------------------  
* **atom.MLP.best_params**: Get parameters of the model with highest score.
* **atom.SVM.best_model**: Get the model with highest score (not fitted).  
* **atom.SVM.best_model_fit**: Get the model with highest score fitted on the training set.  
* **atom.lgbm.score**: Metric score of the BO's selected model on the test set.
* **atom.Tree.predict**: Get the predictions on the test set.  
* **atom.rf.predict_proba**: Get the predicted probabilities on the test set.  
* **atom.MNB.error**: If the model encountered an exception, this shows it.  
* **atom.PA.bagging_scores**: Array of the bagging's results.
* **atom.<span>KNN.BO</span>**: Dictionary containing the information of every step taken by the BO.
	+ 'params': Parameters used for the model
	+ 'score': Score of the chosen metric


Dependencies
-----------------------------
* **[NumPy](https://numpy.org/)** (>=1.17.2)
* **[Pandas](https://pandas.pydata.org/)** (>=0.25.1)
* **[scikit-learn](https://scikit-learn.org/stable/)** (>=0.21.3)
* **[tqdm](https://tqdm.github.io/)** (>=4.35.0)
* **[GpyOpt](https://sheffieldml.github.io/GPyOpt/)** (>=1.2.5)
* **[Matplotlib](https://matplotlib.org/)** (>=3.1.0)
* **[Seaborn](https://seaborn.pydata.org/)** (>=0.9.0)
* **[XGBoost](https://xgboost.readthedocs.io/en/latest/)**, optional (>=0.90)
* **[LightGBM](https://lightgbm.readthedocs.io/en/latest/)**, optional (>=2.3.0)
* **[imbalanced-learn](https://imbalanced-learn.readthedocs.io/en/stable/api.html)**, optional (>=0.5.0)
