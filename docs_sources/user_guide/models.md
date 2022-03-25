# Models
--------

## Predefined models

ATOM provides many models for classification and regression tasks
that can be used to fit the data in the pipeline. After fitting, a
class containing the underlying estimator is attached to the trainer
as an attribute. We refer to these "subclasses" as models. Apart from
the estimator, the models contain a variety of attributes and methods
to help you understand how the underlying estimator performed. They
can be accessed using their acronyms, e.g. `atom.LGB` to access the
LightGBM's model. The available models and their corresponding
acronyms are:

* "Dummy" for [Dummy Estimator](../../API/models/dummy)
* "GP" for [Gaussian Process](../../API/models/gp)
* "GNB" for [Gaussian Naive Bayes](../../API/models/gnb)
* "MNB" for [Multinomial Naive Bayes](../../API/models/mnb)
* "BNB" for [Bernoulli Naive Bayes](../../API/models/bnb)
* "CatNB" for [Categorical Naive Bayes](../../API/models/catnb)
* "CNB" for [Complement Naive Bayes](../../API/models/cnb)
* "OLS" for [Ordinary Least Squares](../../API/models/ols)
* "Ridge" for [Ridge Estimator](../../API/models/ridge)
* "Lasso" for [Lasso Regression](../../API/models/lasso)
* "EN" for [ElasticNet Regression](../../API/models/en)
* "Lars" for [Least Angle Regression](../../API/models/lars)
* "BR" for [Bayesian Ridge](../../API/models/br)
* "ARD" for [Automated Relevance Determination](../../API/models/ard)
* "Huber" for [Huber Regression](../../API/models/huber)
* "Perc" for [Perceptron](../../API/models/perc)
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
* "hGBM" for [HistGBM](../../API/models/hgbm)
* "XGB" for [XGBoost](../../API/models/xgb)
* "LGB" for [LightGBM](../../API/models/lgb)
* "CatB" for [CatBoost](../../API/models/catb)
* "lSVM" for [Linear SVM](../../API/models/lsvm)
* "kSVM" for [Kernel SVM](../../API/models/ksvm)
* "PA" for [Passive Aggressive](../../API/models/pa)
* "SGD" for [Stochastic Gradient Descent](../../API/models/sgd)
* "MLP" for [Multi-layer Perceptron](../../API/models/mlp)

!!! tip
    The acronyms are case-insensitive, e.g. `atom.lgb` also calls
    the LightGBM's model.

!!! warning
    The models can not be initialized directly by the user! Use them
    only through the trainers.


<br>

## Custom models

It is also possible to create your own models in ATOM's pipeline. For
example, imagine we want to use sklearn's [RANSACRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html)
estimator (note that is not included in ATOM's [predefined models](#predefined-models)).
There are two ways to achieve this:

* Using [ATOMModel](../../API/ATOM/atommodel) (recommended). With this
approach you can pass the required model characteristics to the pipeline.

```python
from atom import ATOMRegressor, ATOMModel
from sklearn.linear_model import RANSACRegressor

ransac = ATOMModel(
    models=RANSACRegressor,
    acronym="RANSAC",
    fullname="Random Sample Consensus",
    needs_scaling=True,
)

atom = ATOMRegressor(X, y)
atom.run(ransac)
```

* Using the estimator's class or an instance of the class. This approach will
also call [ATOMModel](../../API/ATOM/atommodel) under the hood, but it will
leave its parameters to their default values.

```python
from atom import ATOMRegressor
from sklearn.linear_model import RANSACRegressor

atom = ATOMRegressor(X, y)
atom.run(RANSACRegressor)
```

Additional things to take into account:

* Custom models can be accessed through their acronym like any other model, e.g.
  `atom.ransac` in the example above.
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
For example, models implemented with the Keras package should use the scikeras
wrappers [KerasClassifier](https://www.adriangb.com/scikeras/refs/heads/master/generated/scikeras.wrappers.KerasClassifier.html#scikeras.wrappers.KerasClassifier)
or [KerasRegressor](https://www.adriangb.com/scikeras/refs/heads/master/generated/scikeras.wrappers.KerasRegressor.html#scikeras.wrappers.KerasRegressor).

Many deep learning use cases, for example in computer vision, use datasets
with more than 2 dimensions, e.g. image data can have shape (n_samples,
length, width, rgb). These data structures are not intended to store in
a two-dimensional pandas dataframe, but, since ATOM requires a dataframe
for its internal API, datasets with more than two dimensions are stored
in a single column called "multidim feature", where every row
contains one (multidimensional) sample. Note that the [data cleaning](../data_cleaning),
[feature engineering](../feature_engineering) and some [plotting](../plots)
methods are unavailable when this is the case.

See in this [example](../../examples/deep_learning) how to use ATOM to train
and validate a Convolutional Neural Network implemented with Keras.

!!! warning
    Keras' models can only use [custom hyperparameter tuning](../training/#hyperparameter-tuning)
    when `n_jobs=1` or `bo_params={"cv": 1}`. Using n_jobs > 1 and
    cv > 1 raises a PicklingError due to incompatibilities of the APIs.

<br>


## Ensembles

Ensemble models use multiple estimators to obtain better predictive
performance than could be obtained from any of the constituent learning
algorithms alone. ATOM implements two ensemble techniques: voting and
stacking. Click [here](../../examples/ensembles) to see an example that uses
ensemble models.

If the ensemble's underlying estimator is a model that used [automated feature scaling](../training/#automated-feature-scaling),
it's added as a Pipeline containing the `scaler` and estimator. If an
[mlflow experiment](../logging/#tracking) is active, the ensembles start
their own run, just like the [predefined  models](#predefined-models) do.

!!! warning
    Combining models trained on different branches into one ensemble is
    not allowed and will raise an exception.


### Voting

The idea behind voting is to combine the predictions of conceptually
different models to make new predictions. Such a technique can be
useful for a set of equally well performing models in order to balance
out their individual weaknesses. Read more in sklearn's [documentation](https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier).

A voting model is created from a trainer through the [voting](../../API/ATOM/atomclassifier/#voting)
method. The voting model is added automatically to the list of
models in the pipeline, under the `Vote` acronym. The underlying
estimator is a custom adaptation of [VotingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html)
or [VotingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html)
depending on the task. The differences between ATOM's and sklearn's
implementation are:

- ATOM's implementation doesn't fit estimators if they're already fitted.
- ATOM's instance is considered fitted at initialization when all underlying
  estimators are.
- ATOM's VotingClassifier doesn't implement a LabelEncoder to encode the
  target column.

The two estimators are customized in this way to save time and computational
resources, since the classes are always initialized with fitted estimators.
As a consequence of this, the VotingClassifier can not use sklearn's build-in
LabelEncoder for the target column since it can't be fitted when initializing
the class. For the vast majority of use cases, the changes will have no effect.
If you want to export the estimator and retrain it on different data, just make
sure to [clone](https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html)
the underlying estimators first.


<br>

### Stacking

Stacking is a method for combining estimators to reduce their biases.
More precisely, the predictions of each individual estimator are
stacked together and used as input to a final estimator to compute the
prediction. Read more in sklearn's [documentation](https://scikit-learn.org/stable/modules/ensemble.html#stacked-generalization).

A stacking model is created from a trainer through the [stacking](../../API/ATOM/atomclassifier/#stacking)
method. The stacking model is added automatically to the list of
models in the pipeline, under the `Stack` acronym. The underlying
estimator is a custom adaptation of [StackingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html)
or [StackingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html)
depending on the task. The only difference between ATOM's and sklearn's
implementation is that ATOM's implementation doesn't fit estimators if
they're already fitted. The two estimators are customized in this way to
save time and computational resources, since the classes are always
initialized with fitted estimators. For the vast majority of use cases,
the changes will have no effect. If you want to export the estimator and
retrain it on different data, just make sure to [clone](https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html)
the underlying estimators first.
