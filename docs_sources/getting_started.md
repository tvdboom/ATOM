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

    atom.impute(strat_num='knn', strat_cat='most_frequent',  min_frac_rows=0.1)  
    atom.encode(max_onehot=10, frac_to_other=0.05)  
    atom.feature_selection(strategy='PCA', n_features=12)

Run the pipeline with different models:

    atom.pipeline(models=['LR', 'LDA', 'XGB', 'lSVM'],
    	          metric='f1',
    	          n_calls=25,
    	          n_random_starts=10,
    	          bagging=4)

Make plots and analyze results: 

	atom.plot_bagging(figsize=(9, 6), filename='bagging_results.png')  
	atom.LDA.plot_confusion_matrix(normalize=True, filename='cm.png')