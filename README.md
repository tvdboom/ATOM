<p align="center">
	<img src="https://github.com/tvdboom/ATOM/blob/master/images/logo.png?raw=true" alt="ATOM" title="ATOM" width="600" height="180"/>
</p>


Automated Tool for Optimized Modelling
-----------------

Author: tvdboom  
Email: m.524687@gmail.com

[![Project Status: Active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Build Status](https://travis-ci.com/tvdboom/ATOM.svg?branch=master)](https://travis-ci.com/tvdboom/ATOM)
[![codecov](https://codecov.io/gh/tvdboom/ATOM/branch/master/graph/badge.svg)](https://codecov.io/gh/tvdboom/ATOM)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/tvdboom/ATOM.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/tvdboom/ATOM/context:python)
[![Python 3.6|3.7|3.8](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue)](https://www.python.org/downloads/release/python-380/)
[![License: MIT](https://img.shields.io/github/license/tvdboom/ATOM)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/atom-ml)](https://pypi.org/project/atom-ml/)

<br><br>


Description  
-----------------

There is no magic formula in data science that can tell us which type of machine
 learning algorithm will perform best for a specific use-case. Best practices tell
 us to start with a simple model (e.g. linear regression) and build up to more
 complicated models (e.g. logistic regression -> random forest -> multilayer perceptron)
 if you are not satisfied with the results. Unfortunately, different models require
 different data cleaning steps, tuning a new set of hyperparameters, etc. Refactoring
 the code for all these steps can be very time consuming. This result in many data
 scientists just using the model best known to them and fine-tuning this particular
 model without ever trying other ones. This can result in poor performance (because
 the model is just not the right one for the task) or in poor time management (because you
 could have achieved a similar performance with a simpler/faster model).  
 
ATOM is here to help us solve these issues. With just a few lines of code, you can
 perform basic data cleaning steps, select relevant features and compare the
 performance of multiple models on a given dataset. ATOM should be able to provide
 quick insights on which algorithms perform best for the task at hand and provide an
 indication of the feasibility of the ML solution.

It is important to realize that ATOM is not here to replace all the work a data
 scientist has to do before getting his model into production. ATOM doesn't spit out
 production-ready models just by tuning some parameters in its API. After helping you
 to determine the right model, you will most probably need to fine-tune it using
 use-case specific features and data cleaning steps in order to achieve maximum
 performance.

So, this sounds a bit like AutoML, how is ATOM different than 
 [auto-sklearn](https://automl.github.io/auto-sklearn/master/) or
 [TPOT](http://epistasislab.github.io/tpot/)? Well, ATOM does AutoML in the sense
 that it helps you find the best model for a specific task, but contrary to the
 aforementioned packages, it does not actively search for the best model. It just
 runs all of them and let you pick the one that you think suites the best.
 AutoML packages are often black boxes to which you provide data, and magically,
 a good model comes out. Although it works great, they often produce complicated
 pipelines with low explainability, hard to sell to the business. In this, ATOM excels.
 Every step of the pipeline is accounted for, and using the provided plotting methods,
 its easy to demonstrate why a model is better/worse than the other. 

<br>
Possible steps taken by ATOM's pipeline:

1. Data Cleaning
	* Handle missing values
	* Encode categorical features
	* Balance the dataset
	* Remove outliers
2. Perform feature selection
	* Remove features with too high collinearity
	* Remove features with too low variance
	* Select best features according to a chosen strategy
3. Train and validate models
	* Select hyperparameters using a Bayesian Optimization approach
	* Train and test the models on the provided data
	* Perform bagging to assess the robustness of the models
4. Analyze the results

<br/><br/>

<p align="center">
	<img src="https://github.com/tvdboom/ATOM/blob/master/images/diagram.png?raw=true" alt="diagram" title="diagram" width="900" height="300" />
</p>

<br><br>


Installation
-----------------

| NOTE: Since atom was already taken, download the package under the name `atom-ml`! |
| --- |


Install ATOM's newest release easily via `pip`:

```Python
	$ pip install -U atom-ml
```

or via `conda`:

```Python
	$ conda install -c conda-forge atom-ml
```

<br><br>


Usage  
-----------------

Call the `ATOMClassifier` or `ATOMRegressor` class and provide the data you want to use:  

    from sklearn.datasets import load_breast_cancer
    from atom import ATOMClassifier
    
    X, y = load_breast_cancer(return_X_y)
    atom = ATOMClassifier(X, y, logger='auto', n_jobs=2, verbose=2)

ATOM has multiple data cleaning methods to help you prepare the data for modelling:

    atom.impute(strat_num='knn', strat_cat='most_frequent',  min_frac_rows=0.1)  
    atom.encode(max_onehot=10, frac_to_other=0.05)  
    atom.feature_selection(strategy='PCA', n_features=12)

Run the pipeline with the models you want to compare:

    atom.run(models=['LR', 'LDA', 'XGB', 'lSVM'],
             metric='f1',
             n_calls=25,
             n_random_starts=10,
             bagging=4)

Make plots to analyze the results: 

	atom.plot_bagging(figsize=(9, 6), filename='bagging_results.png')  
	atom.LDA.plot_confusion_matrix(normalize=True, filename='cm.png')

<br><br>


Documentation
-----------------
  
For further information about ATOM, please see the project's [documentation](https://tvdboom.github.io/ATOM).
