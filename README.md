<p align="center">
	<img src="https://github.com/tvdboom/ATOM/blob/master/images/logo.png?raw=true" alt="ATOM" title="ATOM" width="500" height="140"/>
</p>

# Automated Tool for Optimized Modelling
Author: tvdboom  
Email: m.524687@gmail.com

[![Project Status: Active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Build Status](https://travis-ci.com/tvdboom/ATOM.svg?branch=master)](https://travis-ci.com/tvdboom/ATOM)
[![codecov](https://codecov.io/gh/tvdboom/ATOM/branch/master/graph/badge.svg)](https://codecov.io/gh/tvdboom/ATOM)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/tvdboom/ATOM.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/tvdboom/ATOM/context:python)
[![Python 3.6|3.7|3.8](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue)](https://www.python.org/downloads/release/python-380/)
[![License: MIT](https://img.shields.io/github/license/tvdboom/ATOM)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/atom-ml)](https://pypi.org/project/atom-ml/)


Description  
------------------------  
Automated Tool for Optimized Modelling (ATOM) is a python package designed for fast exploration and experimentation of supervised machine learning tasks. With just a few lines of code, you can perform basic data cleaning steps, feature selection and compare the performance of multiple models on a given dataset. ATOM should be able to provide quick insights on which algorithms perform best for the task at hand and provide an indication of the feasibility of the ML solution. This package supports binary classification, multiclass classification, and regression tasks.

| NOTE: A data scientist with knowledge of the data will quickly outperform ATOM if he applies usecase-specific feature engineering or data cleaning methods. Use ATOM only for a fast exploration of the problem! |
| --- |

Possible steps taken by the ATOM pipeline:
1. Data Cleaning
	* Handle missing values
	* Encode categorical features
	* Balance the dataset
	* Remove outliers
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
Intall ATOM easily using `pip`.
	
	pip install atom-ml


Usage
------------------------  
Call the `ATOMClassifier` or `ATOMRegressor` class and provide the data you want to use:  

    from atom import ATOMClassifier  
    
    atom = ATOMClassifier(X, y, log='auto', n_jobs=2, verbose=2)

ATOM has multiple data cleaning methods to help you prepare the data for modelling:

    atom.impute(strat_num='knn', strat_cat='most_frequent',  max_frac_rows=0.1)  
    atom.encode(max_onehot=10, frac_to_other=0.05)  
    atom.outliers(max_sigma=4)  
    atom.balance(oversample=0.8, n_neighbors=15)  
    atom.feature_selection(strategy='univariate', solver='chi2', max_features=0.9)

Run the pipeline with different models:

    atom.pipeline(models=['LR', 'LDA', 'XGB', 'lSVM'],
	          metric='f1',
	          max_iter=10,
	          max_time=1000,
	          init_points=3,
	          cv=4,
	          bagging=10)  

Make plots and analyze results: 

	atom.plot_bagging(filename='bagging_results.png')  
	atom.lSVM.plot_probabilities()  
	atom.lda.plot_confusion_matrix()


Documentation
------------------------  
For further information about ATOM, please see the project [documentation](tvdboom.github.io/ATOM).
