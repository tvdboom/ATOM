<p align="center">
	<img src="https://github.com/tvdboom/ATOM/blob/master/images/logo.png?raw=true" alt="ATOM" title="ATOM" width="500" height="140"/>
</p>

Automated Tool for Optimized Modelling (ATOM) is a python package designed for fast exploration and experimentation of supervised machine learning tasks. With just a few lines of code, you can perform basic data cleaning steps, feature selection and compare the performance of multiple models on a given dataset. ATOM should be able to provide quick insights on which algorithms perform best for the task at hand and provide an indication of the feasibility of the ML solution. This package supports binary classification, multiclass classification, and regression tasks.

!!!note
    A data scientist with domain knowledge can outperform ATOM if he applies usecase-specific feature engineering or data cleaning steps! 

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
