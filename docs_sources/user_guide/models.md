# Models
--------

## Predefined models

ATOM provides many models for classification and regression tasks
that can be used to fit the data in the pipeline. After fitting, a
class containing the underlying estimator is attached to atom as an
attribute. We refer to these "subclasses" as models. Apart from the
estimator, the models contain a variety of attributes and methods that
can help you understand how the underlying estimator performed. They
can be accessed using their acronyms, e.g., `atom.LGB` to access the
LightGBM model. The available models and their corresponding
acronyms are:

:: atom.models:MODELS
    :: toc

!!! warning
    The model classes cannot be initialized directly by the user! Use
    them only through atom.

!!! tip
    The acronyms are case-insensitive, e.g., `atom.lgb` also calls
    the LightGBM model.

<br>

## Custom models

It is also possible to create your own models in ATOM's pipeline. For
example, imagine we want to use sklearn's [RANSACRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html)
estimator (note that is not included in ATOM's [predefined models][]).
There are two ways to achieve this:

* Using [ATOMModel][] (recommended). With this approach you can pass
  the required model characteristics to the pipeline.

```pycon
from atom import ATOMRegressor, ATOMModel
from sklearn.datasets import load_diabetes
from sklearn.linear_model import RANSACRegressor

ransac = ATOMModel(RANSACRegressor, name="RANSAC", needs_scaling=True)

X, y = load_diabetes(return_X_y=True, as_frame=True)

atom = ATOMRegressor(X, y)
atom.run(ransac)
```

* Using the estimator's class or an instance of the class. This approach
  will also call [ATOMModel][] under the hood, but it will leave its
  parameters to their default values.

```pycon
from atom import ATOMRegressor
from sklearn.datasets import load_diabetes
from sklearn.linear_model import RANSACRegressor

X, y = load_diabetes(return_X_y=True, as_frame=True)

atom = ATOMRegressor(X, y)
atom.run(RANSACRegressor)
```

Additional things to take into account:

* Custom models can be accessed through their acronym like any other model, e.g.
  `atom.ransac` in the example above.
* Custom models are not restricted to sklearn estimators, but they should
  follow [sklearn's API][api], i.e., have a fit and predict method.
* [Parameter customization][] (for the initializer) is only possible for
  custom models which provide an estimator that has a `set_params()` method,
  i.e., it's a child class of [BaseEstimator](https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html).
* [Hyperparameter tuning][] for custom models is ignored unless appropriate
  dimensions are provided through `ht_params`.

<br>


## Deep learning

Deep learning models can be used through ATOM's [custom models][]
as long as they follow [sklearn's API][api]. For example, models
implemented with the Keras package should use the [scikeras](https://www.adriangb.com/scikeras/stable/)
wrappers [KerasClassifier](https://www.adriangb.com/scikeras/refs/heads/master/generated/scikeras.wrappers.KerasClassifier.html#scikeras.wrappers.KerasClassifier)
or [KerasRegressor](https://www.adriangb.com/scikeras/refs/heads/master/generated/scikeras.wrappers.KerasRegressor.html#scikeras.wrappers.KerasRegressor).

Many deep learning use cases, for example in computer vision, use datasets
with more than 2 dimensions, e.g., image data can have shape (n_samples,
length, width, rgb). Luckily, scikeras has a workaround to be able to work
with such datasets. Learn with this [example][example-deep-learning] how to
use ATOM to train and validate a Convolutional Neural Network on an image
dataset.

!!! warning
    Models implemented with [keras](https://keras.io/) can only use
    [custom hyperparameter tuning][hyperparameter-tuning] when `#!python n_jobs=1`
    or `#!python ht_params={"cv": 1}`. Using n_jobs > 1 and cv > 1 raises a
    PicklingError due to incompatibilities of the APIs.

<br>


## Ensembles

Ensemble models use multiple estimators to obtain better predictive
performance than could be obtained from any of the constituent learning
algorithms alone. ATOM implements two ensemble techniques: voting and
stacking. Click [here][example-ensembles] to see an example that uses
ensemble models.

If the ensemble's underlying estimator is a model that used [automated feature scaling][],
it's added as a Pipeline containing the `scaler` and estimator. If a
[mlflow experiment][tracking] is active, the ensembles start their own
run, just like the [predefined models][] do.

!!! warning
    Combining models trained on different branches into one ensemble is
    not allowed and will raise an exception.


### Voting

The idea behind voting is to combine the predictions of conceptually
different models to make new predictions. Such a technique can be
useful for a set of equally well performing models in order to balance
out their individual weaknesses. Read more in sklearn's [documentation](https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier).

A voting model is created from a trainer through the [voting][atomclassifier-voting]
method. The voting model is added automatically to the list of
models in the trainer, under the `Vote` acronym. The underlying
estimator is a custom adaptation of [VotingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html)
or [VotingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html)
depending on the task. The differences between ATOM's and sklearn's
implementation are:

* ATOM's implementation doesn't fit estimators if they're already fitted.
* ATOM's instance is considered fitted at initialization when all underlying
  estimators are.
* ATOM's VotingClassifier doesn't implement a LabelEncoder to encode the
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

A stacking model is created from a trainer through the [stacking][atomclassifier-stacking]
method. The stacking model is added automatically to the list of
models in the trainer, under the `Stack` acronym. The underlying
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
