## Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom  
Email: m.524687@gmail.com
  
Description  
------------------------  
ATOM is a python package for exploration of ML problems. With just a few lines of code, you can compare the performance of multiple machine learning models on a given dataset, providing a quick insight on which algorithms performs best for the task at hand. Furthermore, ATOM contains a variety of plotting functions to help you analyze the models' performances. All ML algorithms are  implemented using the [scikit-learn](https://scikit-learn.org/stable/) python package except for the Extreme Gradient Booster, which uses [XGBoost]([https://xgboost.readthedocs.io/en/latest/).  
The pipeline, first applies the imputing of missing values, the encoding of categorical features and the selection of best features. After that, it starts selecting the optimal hyperparameters per model using a Bayesian Optimization (BO) approach implemented with the [GPyOpt](https://sheffieldml.github.io/GPyOpt/) library. The data is fitted to the  selected metric. Hereafter, the pipleine performs a K-fold cross validation on the complete data set. This is needed to avoid having a bias towards the hyperparameters selected by the BO and provides a better statistical overview of the final results. The class contains the models as subclasses, on which you can call extra methods and attributes. 

  
Usage  
------------------------  
Call the pipeline class:  

    atom = ATOM(models=['LinReg', 'KNN', 'RF', 'GBM', MLP'],
                metric="MAE",
                impute='median',
                features=0.8,
                ratio=0.25,
                max_iter=10,
                n_splits=5,
                verbose=1)
Run the pipeline:  

    atom.fit(X, Y)  
Make plots and analyse results: 

	atom.boxplot('boxplot.png')  
	atom.RF.plot_probabilities()  
  
  Alternatively, the preprocessing methods can be called independently of the fit method to further tune specific parameters.
  
  	# Create an optimized Random Forest for feature_selection
	aml = ATOM('RF', cv=False)
	aml.fit(X, Y, percentage=10)
	aml.rf.plot_feature_importance()  # Visualize the feature ranking

	# Call new ATOM class for ML task exploration
	atom = ATOM(models=['LogReg', 'AdaBoost', 'XGBoost'],
		    metric="f1",
		    max_iter=20,
		    init_points=1,
		    n_splits=3,
		    log='ATOM_log',
		    n_jobs=2,
		    verbose=3)

	X = atom.imputer(X, strategy='mean', max_frac_missing=0.8)
	X = atom.encoder(X, max_number_onehot=20)
	X = atom.feature_selection(X, Y, k=10, model=aml.RF.best_model)
	atom.fit(X, Y)


Class parameters
----------------------------- 
* **models: string or list of strings, optional (default=all)**  
List of models on which to apply the pipeline. Possible values are (case insensitive):    
    + 'GNB' for Gaussian Naïve Bayes  
    + 'MNB' for Multinomial Naïve Bayes  
    + 'BNB' for Bernoulli Naïve Bayes  
    + 'GP' for Gaussian Process  
	+ 'LinReg' for linear regression (with elasticnet regularization)  
	+ 'LogReg' for Logistic Regression  
	+ 'LDA' for Linear Discriminant Analysis  
	+ 'QDA' for Quadratic Discriminant Analysis  
	+ 'KNN' for K-Nearest Neighbors  
	+ 'Tree' for a single Decision Tree  
	+ 'ET' for Extra-Trees  
	+ 'RF' for Random Forest  
	+ 'AdaBoost' for Adaptive Boosting  
	+ 'GBM' for Gradient Boosting Machine  
	+ 'XGBoost' for Extreme Gradient Boosting (if package is available)  
	+ 'lSVM' for Linear Support Vector Machine  
	+ 'kSVM' for Kernel Support Vector Machine  
	+ 'PA' for Passive Aggressive  
	+ 'SGD' for Stochastic Gradient Descent  
	+ 'MLP' for Multilayer Perceptron  
* **metric: string, optional (default='F1' or 'MSE')**  
Metric on which the pipeline fits the models. Possible values are (case insensitive):  
	+ For binary and multiclass classification or regression:  
		- 'max_error'  
		- 'R2'  
		- 'MAE' for Mean Absolute Error  
		- 'MSE' for Mean Squared Error
		- 'MSLE' for Mean Squared Log Error  
	+ Only binary classification:  
		- 'Precision'  
		- 'Recall'  
		- 'Accuracy'
		- 'F1'
		- 'Jaccard'  
		- 'AUC' for Area Under Curve  
		- 'LogLoss' for binary cross-entropy  
* **impute: string, optional (default=None)**  
Strategy for the imputing of missing values. Possible strategies are:
	+ None to not perform any imputation  
	+ 'mean' to impute with the mean of feature  
	+ 'median' to impute with the median of feature  
	+ 'most_frequent' to impute with the most frequent value (only option for categorical features)  
* **features: int or float, optional (default=None)**  
Select best features according to a univariate F-test.
	+ if >= 1: number of features to select
	+ if < 1: fraction of features to select
* **ratio: float, optional (default=0.3)**  
Split ratio of the train and test set used for the BO.
* **max_iter: int, optional (default=15)**  
Maximum number of iterations of the BO.
* **max_time: int, optional (default=inf)**  
Maximum time allowed for the BO (in seconds).
* **eps: float, optional (default=1e-08)**  
Minimum distance in hyperparameters between two consecutive steps in the BO.
* **batch_size: int, optional (default=1)**  
Size of the batch in which the objective is evaluated 
* **init_points: int, optional (default=5)**  
Initial number of random tests of the BO. If 1, the model is fitted on the default hyperparameters of the package.
* **plot_bo: bool, optional (default=False)**  
Wether to plot the BO's progress as it runs.
* **cv: bool, optional (default=True)**  
Wether to perform a K-fold cross validation n every model after the BO.
* **n_splits: int, optional (default=4)**  
Number of splits for the K-fold cross validation. Only if cv=True.
* **log: string, optional (default=None)**  
Name of the log file, None to not save any log.
* **n_jobs: int, optional (default=1)**  
Number of cores to use for parallel processing.
	+ If -1, use all available cores.
	+ If <-1, use available_cores - 1 + n_jobs  
* **verbose: int, optional (default=0)**  
Verbosity level of the pipeline. Possible values:  
	+ 0 to not print anything  
	+ 1 to print minimum information
	+ 2 to print medium information
	+ 3 to print maximum information
  
Class methods
-----------------------------  
* **fit(X, Y, percentage=100)**  
Run the ATOM pipeline.
	+ X: array or pd.Dataframe, shape = [n_samples, n_features]
	+ Y: array or pd.Series, shape = [n_samples]
	+ percentage: int, optional (default=100)  
	Percentage of data to use in the pipeline.
* **imputer(X, strategy='median' , max_frac_missing=0.5, missing=[NaN, None, '', '?', 'NA', inf, -inf])**  
Impute missing values. Non-numeric features are always imputed with the most_frequent strategy. Also removes columns with more than max_frac fraction of missing values.
	+ X: array or pd.Dataframe, optional if class is fitted
	+ strategy: string, optional (default='median')  
	See ATOM's class parameters for impute strategies.
	+ max_frac_missing: float, optional (default=0.5)  
	Maximum fraction of instances with missing values before removing the feature.
	+ missing: string or list of strings, optional (default=[NaN, None, '', '?', 'NA', inf, -inf])  
	List of values to impute. None and NaN are always imputed.
* **encoder(X, max_number_onehot=10)**  
Performs one-hot-encoding on categorical features if the number of unique values is smaller or equal to max_number_onehot, else Label-encoding. Also removes columns with only one unique category.
	+ X: array or pd.Dataframe, optional if class is fitted
	+ max_number_onehot: int, optional (default=10)  
	Maximum number of unique values in a feature to perform one-hot-encoding.
* **feature_selection(X, Y, strategy='univariate', max_features=0.9, threshold=-np.inf, frac_variance=1, max_correlation=0.98)**  
Select best features according to a univariate F-test or with a recursive feature selector (RFS). Ties between features with equal scores will be broken in an unspecified way. Also removes features with too low variance and too high collinearity.
	+ X: array or pd.Dataframe, optional if class is fitted  
	+ Y: array or pd.Series, optional if class is fitted
	+ strategy: string or model class, optional (default='univariate')
	Strategy for the feature selector. Choose from:
		- 'univariate' for the univariate F-test
		- model class (not fitted) for the RFS
	+ max_features: int or float, optional (default=0.9)  
	Number or fraction of features to select.
		- if >= 1: number of features to select
		- if < 1: fraction of features to select
	+ threshold: string or float, optional (default=-np.inf)  
	The threshold value to use. Features whose importance is greater or equal are kept while the others are discarded. Only for RFS.
		- if 'mean': set the mean of feature_importances as threshold
		- if 'median': set the median of feature_importances as threshold
	+ frac_variance: float, optional (default=1)  
	Remove features with constant instances in at least this fraction of the total.
	+ max_correlation: float, optional (default=0.98)  
	Minimum value of the Pearson correlation cofficient to identify correlated features.
* **boxplot(figsize, filename=None)**  
Make a boxplot of the results of the cross validation. Only after the class is fitted.
	+ figsize, 2d-tuple, otional (default=dependent on # of models)
	+ filename: string, optional (default=None)  
	Name of the file when saved. None to not save anything.
* **plot_correlation(X, figsize=(10, 6), filename=None)**  
Make a correlation maxtrix plot of the dataset. Ignores non-numeric columns.
	+ X: array or pd.Dataframe, optional if class is fitted  
	+ figsize, 2d-tuple, otional (default=dependent on # of models)
	+ filename: string, optional (default=None)  
	Name of the file when saved. None to not save anything.

Class attributes  
-----------------------------  
* **dataset**: dataframe of the features and target after pre-processing (not yet scaled)
* **X, Y**: data features and target
* **X_train, Y_train**: training set features and target
* **X_test, Y_test**: validation set features and target
* **errors**: contains a list of the encountered exceptions (if any) while fitting the models.
* **collinear**: dataframe containing the collinear features (if any) and their correlation value. Only if feature_selection was ran.

  
### The models chosen become subclasses of the ATOM class after calling the fit method. They can be called upon for  handy plot functions and attributes (case unsensitive).
  
Subclass methods (plots)  
-----------------------------  
* **plot_probabilities(target_class=1, figsize=(10, 6), filename=None)**  
Plots the probability of every class in the target variable against the class selected by target_class. Only for classification tasks.
	+ target_class:  int, optional (default=1 -->2nd class)
	Target class to plot the probabilities against.
	+ figsize, 2d-tuple, otional (default=dependent on # of models)
	+ filename: string, optional (default=None)  
	Name of the file when saved. None to not save anything.
* **plot_feature_importance(figsize=(10, 6), filename=None)**  
Plots the feature importance scores. Only works with tree based algorithms (Tree, ET, RF, AdaBoost, GBM and XGBoost).
	+ figsize, 2d-tuple, otional (default=dependent on # of models)
	+ filename: string, optional (default=None)  
	Name of the file when saved. None to not save anything.
* **plot_ROC(figsize=(10, 6), filename=None)**  
Plots the ROC curve. Only for binary classification tasks.  
 	+ figsize, 2d-tuple, otional (default=dependent on # of models)
	+ filename: string, optional (default=None)  
	Name of the file when saved. None to not save anything.
* **plot_confusion_matrix(normalize=True, figsize=(10, 6), filename=None)**  
Plot the confusion matrix for the model. Only for binary classification.  
	+ normalize: bool, otional (default=True)
   	+ figsize, 2d-tuple, otional (default=dependent on # of models)
	+ filename: string, optional (default=None)  
	Name of the file when saved. None to not save anything.
* **plot_decision_tree(num_trees=0, max_depth=None, rotate=False, figsize=(10, 6), filename=None)**  
Plot a single decision tree of the model. Only for tree-based algorithms.
	+ num_trees: int, otional (default=0 --> first tree)  
	Number of the tree to plot (if ensemble).
	+ max_depth: int, optional (default=None)  
	Maximum depth of the plotted tree. None for no limit.
	+ rotate: bool, optional (default=False)  
	When True, orientate the tree left-right instead of top-bottom.
   	+ figsize, 2d-tuple, otional (default=dependent on # of models)
	+ filename: string, optional (default=None)  
	Name of the file when saved. None to not save anything.  
  
Sublass methods (metrics)  
-----------------------------
Call any of the metrics as a method. It will return the metric (evaluated on the test set) for the best model found by the BO.
+ **atom.knn.AUC()**: Returns the AUC score for the best trained KNN  
+ **atom.adaboost.MSE()**: Returns the MSE score for the best trained AdaBoost  
  
Subclass attributes
-----------------------------  
* **atom.MLP.best_params**: Get parameters of the MLP with highest score.
* **atom.SVM.best_model**: Get the SVM model with highest score (not fitted).  
* **atom.SVM.model_fit**: Get the SVM model with highest score (fitted).  
* **atom.Tree.prediction**: Get the predictions on the test set.  
* **atom.<span>KNN.BO</span>**: Dictionary containing the information of every step taken by the BO.
	+ 'params': Parameters used for the model
	+ 'score': Score of the chosen metric
* **atom.GBM.error**: If the model encountered an exception, this shows it.
