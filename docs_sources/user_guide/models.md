# Models
--------

## Predefined models

ATOM provides 32 estimators for classification and regression tasks
that can be used to fit the data in the pipeline. After fitting, a
class containing the estimator is attached to the trainer as an
attribute. We refer to these "subclasses" as models. Apart from the
estimator, the models contain a variety of attributes and methods to
help you understand how the underlying estimator performed. They can
be accessed using their acronyms, e.g. `atom.LGB` to access the
LightGBM's model. The available models and their corresponding
acronyms are: 

* "Dummy" for [Dummy Classification/Regression](../../API/models/dummy)
* "GP" for [Gaussian Process](../../API/models/gp)
* "GNB" for [Gaussian Naive Bayes](../../API/models/gnb)
* "MNB" for [Multinomial Naive Bayes](../../API/models/mnb)
* "BNB" for [Bernoulli Naive Bayes](../../API/models/bnb)
* "CatNB" for [Categorical Naive Bayes](../../API/models/catnb)
* "CNB" for [Complement Naive Bayes](../../API/models/cnb)
* "OLS" for [Ordinary Least Squares](../../API/models/ols)
* "Ridge" for [Ridge Classification/Regression](../../API/models/ridge)
* "Lasso" for [Lasso Regression](../../API/models/lasso)
* "EN" for [Elastic Net](../../API/models/en)
* "BR" for [Bayesian Ridge](../../API/models/br)
* "ARD" for [Automated Relevance Determination](../../API/models/ard)
* "LR" for [Logistic Regression](../../API/models/lr)
* "LDA" for [Linear Discriminant Analysis](../../API/models/lda)
* "QDA" for [Quadratic Discriminant Analysis](../../API/models/qda)
* "KNN" for [K-Nearest Neighbors](../../API/models/knn)
* "RNN" for [Radius Nearest Neighbors](../../API/models/rnn)
* "Tree" for [Decision Tree](../../API/models/tree)
* "Bag" for [Bagging](../../API/models/bag)
* "ET" for [Extra-Trees](../../API/models/et)
* "RF" for [Random Forest](../../API/models/rf)
* "AdaB" for [AdaBoost](../../API/models/adab)
* "GBM" for [Gradient Boosting Machine](../../API/models/gbm)
* "XGB" for [XGBoost](../../API/models/xgb)
* "LGB" for [LightGBM](../../API/models/lgb)
* "CatB" for [CatBoost](../../API/models/catb)
* "lSVM" for [Linear-SVM](../../API/models/lsvm)
* "kSVM" for [Kernel-SVM](../../API/models/ksvm)
* "PA" for [Passive Aggressive](../../API/models/pa)
* "SGD" for [Stochastic Gradient Descent](../../API/models/sgd)
* "MLP" for [Multi-layer Perceptron](../../API/models/mlp)

!!! tip
    The acronyms are case-insensitive, e.g. `atom.lgb` also calls
    the LightGBM's model.

!!! warning
    The models can not be initialized directly by the user! Only use
    them through the trainers.


<br>

## Custom models

It is also possible to create your own models in ATOM's pipeline. For
example, imagine we want to use sklearn's [Lars](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lars.html)
estimator (note that is not included in ATOM's [predefined models](#predefined-models)).
There are two ways to achieve this:

* Using [ATOMModel](../../API/ATOM/atommodel) (recommended). With this
approach you can pass the required model characteristics to the pipeline.

```python
from sklearn.linear_model import Lars
from atom import ATOMRegressor, ATOMModel

model = ATOMModel(models=Lars, fullname="Lars Regression", needs_scaling=True)

atom = ATOMRegressor(X, y)
atom.run(model)
```

* Using the estimator's class or an instance of the class. This approach will
also call [ATOMModel](../../API/ATOM/atommodel) under the hood, but it will
leave its parameters to their default values.

```python
from sklearn.linear_model import Lars
from atom import ATOMRegressor, ATOMModel

atom = ATOMRegressor(X, y)
atom.run(Lars)
```

Additional things to take into account:

* Custom models can be accessed through their acronym like any other model, e.g.
  `atom.lars` in the example above.
* Custom models are not restricted to sklearn estimators, but they should
  follow [sklearn's API](https://scikit-learn.org/stable/developers/develop.html),
  i.e. have a fit and predict method.
* [Parameter customization](#parameter-customization) (for the initializer)
  is only possible for custom models which provide an estimator that has a
  `set_params()` method, i.e. it's a child class of [BaseEstimator](https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html).
* [Hyperparameter optimization](#hyperparameter-tuning) for custom
  models is ignored unless appropriate dimensions are provided through
  `bo_params`.
* If the estimator has a `n_jobs` and/or `random_state` parameter that is
  left to its default value, it will automatically adopt the values from
  the trainer it's called from.

<br>


## Deep learning

Deep learning models can be used through ATOM's [custom models](#custom-models)
as long as they follow [sklearn's API](https://scikit-learn.org/stable/developers/develop.html).
For example, models implemented with the Keras package should use the sklearn wrappers
[KerasClassifier](https://www.tensorflow.org/api_docs/python/tf/keras/wrappers/scikit_learn/KerasClassifier)
or [KerasRegressor](https://www.tensorflow.org/api_docs/python/tf/keras/wrappers/scikit_learn/KerasRegressor).

Many deep learning use cases, for example in computer vision, use datasets
with more than 2 dimensions, e.g. image data can have shape (n_samples,
length, width, rgb). These data structures are not intended to store in
a two-dimensional pandas dataframe, but, since ATOM requires a dataframe
for its internal API, datasets with more than two dimensions are stored
in a single column called "Multidimensional feature", where every row
contains one (multidimensional) sample. Note that the [data cleaning](../data_cleaning),
[feature engineering](../feature_engineering) and some of the [plotting](../plots)
methods are unavailable when this is the case.

See in this [example](../../examples/deep_learning) how to use ATOM to train
and validate a Convolutional Neural Network implemented with Keras.

