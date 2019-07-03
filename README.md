Title: AutoML pipeline
Author: Marco van den Boom
Date: 21-Jun-2019

Description
------------------------
Compare multiple ML models on the same data. All models are implemented
using the SKlearn python package. Note that the data needs to be adapted to
the models you want to use in terms of categorical/missing data.
The algorithm first starts selecting the optimal hyperparameters per model
using a Bayesian Optimization (BO) approach implemented with the GPyOpt
library. The data is fitted to the provided metric. The parameters and
domains of the algorithms are pre-set. For this, the data is split in a train
and test set with sizes of 70% and 30% respectively.
Hereafter it performs a K-fold cross validation on the complete data set
provided. This is needed to avoid having a bias towards the hyperparamters
plotted in a boxplot.
The function returns a dictionary of the model as classes, on which you
can call any metric or extra plots.

Usage
------------------------
Load module in with:
from automl import AutoML

Call the pipeline function (it returns a dictionary of the models used):
models = AutoML(X, Y,
                models=['LinReg', 'KNN', 'RF', 'GBM', MLP'],
                metric="MAE",
                ratio=0.25,
                max_iter=5,
                batch_size=1,
                cross_val=True,
                n_splits=5,
                verbose=1)

Call the plotting functions for the specific models:
models['SVM'].plot_proba(target_class=1)
models['GBM'].plot_feature_importance(save_plot='feature_importance.png')


Parameters
------------------------
X      --> array or dataframe of target features
Y      --> array or dataframe of target classes
models --> list of models to use. Possible values are:
               LinReg for linear regression (with elasticnet regularization)
               LogReg for Logistic Regression
               KNN for K_Nearest Neighbors
               Tree for a single Decision Tree
               ET for Extra-Trees
               RF for Random Forest
               AdaBoost for Adaptive Boosting
               GBM for Gradient Boosting Machine
               XGBoost for Extreme Gradient Boosting (if library is available)
               SVM for Support Vector Machine
               MLP for Multilayer Perceptron
metric --> metric on which the BO performs its fit. Possible values are:
               For binary and multiclass classification or regression:
                   max_error
                   r2
                   MAE for Mean Absolute Error
                   MSE for Mean Squared Error
                   MSLE for Mean Squared Log Error
               Only binary classification:
                   Precision
                   Recall
                   Accuracy
                   F1
                   Jaccard
                   AUC for Area Under Curve
                   LogLoss for binary cross-entropy
ratio      --> train/test split ratio
max_iter   --> Maximum number of iterations of the BO algorithm
batch_size --> Size of the batches processed in the BO before fitting
cross_val  --> Boolean wether to perform K-fold cross validation
n_splits   --> Number of splits for the K-fold cross validation
n_jobs     --> Number of cores for parallel processing
percentage --> percentage of the data to use
save_plot  --> Directory to save plot to. If None, plot is not saved
verbose    --> verbosity level of the pipeline. Only works if n_jobs=1.
               Possible values:
                  0 to print only the final stats
                  1 to print stats per algorithm as it gets fit by the BO
                  2 to print every step of the BO


Class methods (plots)
--------------------------
plot_proba(target_class, save_plot):
    Plots the probability of every class in the target variable against the
    class selected by target_class (default=2nd class). Works for multi-class.

plot_feature_importance(save_plot):
    Plots the feature importance scores. Only works with tree based
    algorithms (Tree, ET, RF, AdaBoost, GBM and XGBoost).

plot_ROC(save_plot):
    Plots the ROC curve. Works only for binary classification.

plot_confusion_matrix(normalize, save_plot):
    Plot the confusion matrix for the model. Works only for binary
    classification.

plot_decision_tree(num_trees, rotate, save_plot):
    Plot a single decision tree of a tree-based model. Only works with
    tree-based algorithms.

Class methods (metrics)
--------------------------
Call any of the possible metrics as a method. It will return the metric
(evaluated on the test set) for the best model found by the BO.
e.g. models['KNN'].AUC()        # Get AUC score for the best trained KNN
     models['AdaBoost'].MSE()   # Get MSE score for the best trained AdaBoost

Class attributes
--------------------------
The dictionary returned by the AutoML pipeline can be used to call for the
plot functions described above as well as for other handy features.
e.g. models['MLP'].best_params  # Get parameters of the MLP with highest score
     models['SVM'].best_model   # Get model of the SVM with highest score
     models['Tree'].prediction  # Get the predictions on the test set
