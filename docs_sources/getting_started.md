Installation
------------------------  
Intall ATOM easily using `pip`.

```Python
	pip install atom-ml
```

!!! note
    Since atom was already taken, the name of the package in pypi is `atom-ml`!
   

Usage  
------------------------  
Call the `ATOMClassifier` or `ATOMRegressor` class and provide the data you want to use:  

    from sklearn.datasets import load_breast_cancer
    from atom import ATOMClassifier
    
    X, y = load_breast_cancer(return_X_y)
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
	atom.lSVM.plot_probabilities(figsize=(9, 6))  
	atom.lda.plot_confusion_matrix(normalize=True)