<center>
    <img src="./img/logo.png" height="150" width="600"/>
</center>
<br><br>

# Automated Tool for Optimized Modelling
----------------------------------------

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

!!!note
    A data scientist with domain knowledge can outperform ATOM if he applies
    usecase-specific feature engineering or data cleaning steps! 

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

<center>
    <img src="./img/diagram.png" height="300" width="1000"/>
</center>