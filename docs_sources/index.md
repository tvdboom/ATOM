<center>
    <img src="./img/logo.png" height="150" width="600"/>
</center>
<br><br>

# Automated Tool for Optimized Modelling
-----------------

ATOM is a python package designed for fast exploration and experimentation of
 supervised machine learning tasks. With just a few lines of code, you can
 perform basic data cleaning steps, select relevant features and compare the
 performance of multiple models on a given dataset. ATOM should be able to provide
 quick insights on which algorithms perform best for the task at hand and provide an
 indication of the feasibility of the ML solution. This package supports binary
 classification, multiclass classification, and regression tasks.

!!!note
    A data scientist with domain knowledge can outperform ATOM if he applies
    usecase-specific feature engineering or data cleaning steps! 

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

<center>
    <img src="./img/diagram.png" height="300" width="1000"/>
</center>