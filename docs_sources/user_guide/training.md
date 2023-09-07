# Training
----------

The training phase is where the models are fitted on the training data.
After this, you can use the [plots][] and [prediction methods][] to
evaluate the results. The training applies the following steps for all
models:

1. Use [hyperparameter tuning][] to select the optimal hyperparameters for 
   the model (optional).
2. The model is fitted on the training set using the best combination
   of hyperparameters found. After that, the model is evaluated on the tes set.
3. Calculate various scores on the test set using a [bootstrap](#bootstrapping)
   algorithm (optional).

There are three approaches to run the training.

* Direct training:
    - [DirectClassifier][]
    - [DirectForecaster][]
    - [DirectRegressor][]
* Training via [successive halving][]:
    - [SuccessiveHalvingClassifier][]
    - [SuccessiveHalvingForecaster][]
    - [SuccessiveHalvingRegressor][]
* Training via [train sizing][]:
    - [TrainSizingClassifier][]
    - [TrainSizingForecaster][]
    - [TrainSizingRegressor][]

The direct fashion repeats the aforementioned steps only once, while the
other two approaches repeats them more than once. Just like the [data cleaning][]
and [feature engineering][] classes, it's discouraged to use these classes
directly. Instead, every approach can be called directly from atom through
the [run][atomclassifier-run], [successive_halving][atomclassifier-successive_halving]
and [train_sizing][atomclassifier-train_sizing] methods respectively.

Models are called through their [acronyms][models], e.g., `#!python atom.run(models="RF")`
will train a [RandomForest][]. If you want to run the same model multiple
times, add a tag after the acronym to differentiate them. the tag must be 
separated from the accronym by an underscore.

```python
atom.run(
    models=["RF_1", "RF_2"],
    est_params={
        "RF_1": {"n_estimators": 100},
        "RF_2": {"n_estimators": 200},
    }
)
```

For example, this pipeline fits two Random Forest models, one with 100
and the other with 200 decision trees. The models can be accessed through
`atom.rf_1` and `atom.rf_2`. Use tagged models to test how the same model
performs when fitted with different parameters or on different data sets.
See the [Imbalanced datasets][example-imbalanced-datasets] example.

Additional things to take into account:

* If an exception is encountered while fitting an estimator, the
  pipeline will automatically jump to the next model. The exceptions are
  stored in the `errors` attribute. Note that when a model is skipped,
  there is no model subclass for that estimator.
* When showing the final results, a `!` indicates the highest score
  and a `~` indicates that the model is possibly overfitting (training
  set has a score at least 20% higher than the test set).

<br>

## Metric

ATOM uses sklearn's [scorers](https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules)
for model evaluation. A scorer consists of a metric function and
some parameters that define the scorer's properties , such as if
a higher or lower score is better (score or loss function) or if
the function needs probability estimates or rounded predictions
(see the [make_scorer](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html)
function). The [`metric`][directclassifier-metric] parameter accepts
three ways of defining the scorer:

* Using the name of one of the [predefined scorers][].
* Using a function with signature `#!python function(y_true, y_pred) -> score`.
  In this case, ATOM uses [make_scorer](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html)
  with default parameters.
* Using a scorer object.

Note that all scorers follow the convention that higher return values
are better than lower return values. Thus, metrics which measure the
distance between the model and the data (i.e. loss functions), like
`max_error` or `mean_squared_error`, will return the negated value of
the metric.

<br>

#### Predefined scorers

ATOM accepts all sklearn's [scorers](https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules)
as well as some custom acronyms and custom scorers. Since some of
sklearn's scorers have quite long names and ATOM is all about
<s>lazy</s>fast experimentation, the package provides acronyms
for some of the most commonly used ones. These acronyms are
case-insensitive and can be used in the [`metric`][directclassifier-metric]
parameter instead of the scorer's full name, e.g., `#!python atom.run("LR", metric="BA")`
uses `balanced_accuracy`. The available acronyms are:

* "AP" for "average_precision"
* "BA" for "balanced_accuracy"
* "AUC" for "roc_auc"
* "LogLoss" for "neg_log_loss"
* "EV" for "explained_variance"
* "ME" for "max_error"
* "MAE" for "neg_mean_absolute_error"
* "MSE" for "neg_mean_squared_error"
* "RMSE" for "neg_root_mean_squared_error"
* "MSLE" for "neg_mean_squared_log_error"
* "MEDAE" for "neg_median_absolute_error"
* "MAPE" for "neg_mean_absolute_percentage_error"
* "POISSON" for "neg_mean_poisson_deviance"
* "GAMMA" for "neg_mean_gamma_deviance"


ATOM also provides some extra common metrics for binary classification
tasks. 

* "TN" for True Negatives
* "FP" for False Positives
* "FN" for False Negatives
* "TP" for True Positives
* "FPR" for False Positive rate (fall-out)
* "TPR" for True Positive Rate (sensitivity, recall)
* "TNR" for True Negative Rate (specificity)
* "FNR" for False Negative Rate (miss rate)
* "MCC" for Matthews Correlation Coefficient (also for multiclass classification)

<br>

#### Multi-metric runs

Sometimes it is useful to measure the performance of the models in more
than one way. ATOM lets you run the pipeline with multiple metrics at
the same time. To do so, provide the `metric` parameter with a list of
desired metrics, e.g., `#!python atom.run("LDA", metric=["r2", "mse"])`.

When fitting multi-metric runs, the resulting scores will return a list
of metrics. For example, if you provided three metrics to the pipeline,
`#!python atom.knn.score_train` could return [0.8734, 0.6672, 0.9001].
Only the first metric of a multi-metric run (this metric is called the
**main** metric) is used to select the [winning][atomclassifier-winner]
model.

!!! info
    * The [`winning`][atomclassifier-winner] model is retrieved comparing only
      the main metric.
    * Some plots let you choose which of the metrics in a multi-metric run
      to show using the `metric` parameter, e.g., [plot_results][].

<br>

## Automated feature scaling

Models that require feature scaling will automatically do so before
training, unless the data is [sparse][sparse-datasets] or already scaled.
The data is considered scaled if it has one of the following prerequisites:

* The mean value over the mean of all columns lies between -0.05 and 0.05
  and the mean of the standard deviation over all columns lies between 0.85
  and 1.15. Categorical and binary columns (only 0s and 1s) are excluded
  from the calculation.
* There is a transformer in the pipeline whose \__name__ contains the
  word `scaler`.

The scaling is applied using a [Scaler][] with default parameters. It
can be accessed from the model through the `scaler` attribute. The
scaled dataset can be examined through the model's [data attributes][].
Use the [available_models][atomclassifier-available_models] method to
see which models require feature scaling. See [here][example-automated-feature-scaling]
an example.

<br>

## In-training validation

Some [predefined models][] allow in-training validation. This means
that the estimator is evaluated (using only the **main metric**) on the
train and test set after every round of the training (a round can be an
iteration for linear models or an added tree for boosted tree models).
The validation scores are stored in the `evals` attribute, a dictionary
of the train and test performances per round (also when pruning isn't
applied). Click [here][example-in-training-validation] for an example
using in-training validation.

The predefined models that support in-training validation are:

* [CatBoost][]
* [LightGBM][]
* [MultiLayerPerceptron][]
* [PassiveAggressive][]
* [Perceptron][]
* [StochasticGradientDescent][]
* [XGBoost][]

To apply in-training validation to a [custom model][custom-models], use the
[`has_validation`][atommodel-has_validation] parameter when creating the
custom model.

!!! warning
    * In-training validation is **not** calculated during [hyperparameter tuning][].
    * CatBoost selects the weights achieved by the best evaluation on the
    test set after training. This means that, by default, there is some
    minor data leakage in the test set. Use the `#!python use_best_model=False`
    parameter to avoid this behavior or use a [holdout set][data-sets] to
    evaluate the final estimator.

!!! tip
    Use the [plot_evals][] method to visualize the in-training validation
    on the train and test sets.

<br>

## Parameter customization

By default, every estimator uses the default parameters they get from
their respective packages. To select different ones, use the [`est_params`][directclassifier-est_params].
parameter of the [run][atomclassifier-run] method. There are two ways
to add custom parameters to the models: adding them directly to the
dictionary as key-value pairs or through dictionaries.

Adding the parameters directly to `est_params` (or using a dict with
the key 'all') shares them across all models in the trainer. In
this example, both the [XGBoost][] and the [LightGBM][] model use
200 boosted trees. Make sure all the models do have the specified
parameters or an exception will be raised!

```python
atom.run(models=["XGB", "LGB"], est_params={"n_estimators": 200})
```

To specify parameters per model, use the model name as key and a dict
of the parameters as value. In this example, the [XGBoost][] model uses
`n_estimators=200` and the [MultiLayerPerceptron][] uses one hidden
layer with 75 neurons.

```python
atom.run(
    models=["XGB", "MLP"],
    est_params={
        "XGB": {"n_estimators": 200},
        "MLP": {"hidden_layer_sizes": (75,)},
    }
)
```

Some estimators allow you to pass extra parameters to the fit method
(besides X and y). This can be done adding `_fit` at the end of the
parameter. For example, to change [XGBoost][]'s verbosity, we can run:

```python
atom.run(models="XGB", est_params={"verbose_fit": True})
```

!!! note
    If a parameter is specified through `est_params`, it's ignored
    by the study, even if it's added manually to `#!python ht_params["distributions"]`.

!!! info
    The estimator's `n_jobs` and `random_state` parameters adopt atom's
    values (when available), unless specified through `est_params`.

<br>

## Hyperparameter tuning

In order to achieve maximum performance, it's important to tune an
estimator's hyperparameters before training it. ATOM provides
[hyperparameter tuning](https://en.wikipedia.org/wiki/Hyperparameter_optimization)
through the [optuna](https://optuna.org/) package. Just like optuna,
we use the terms `study` and `trial` as follows:

* Study: optimization based on an objective function.
* Trial: a single execution of the objective function.

Each trial is either computed by cross-validation on the complete training
set or by randomly splitting the training set every iteration into a
(sub)training and validation set. This process can create some minimum
data leakage towards specific parameters (since the estimator is evaluated
on data that is used to train the next estimator), but it ensures maximal
use of the provided data. However, the leakage is not present in the
independent test set, thus the final score of every model is unbiased.
Note that, if the dataset is relatively small, the tuning's best score can
consistently be lower than the final score on the test set due to the
considerable lower fraction of instances on which it is trained. After
finishing the study, the parameters that resulted in the best score are
used to fit the final model on the complete training set.

!!! info
    * Unless specified differently by the user, the used [samplers](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html)
      are [TPESampler](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html)
      for single-metric runs and [NSGAIISampler](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.NSGAIISampler.html)
      for [multi-metric runs][].
    * For [multi-metric runs][], the selected [best trial][adaboost-best_trial]
      is the trial that performed best on the main metric. Use the property's
      `@setter` to change it to any other trial. See the [hyperparameter tuning][example-hyperparameter-tuning]
      example.

There are many possibilities to tune the study to your liking. The main
parameter is [`n_trials`][directclassifier-n_trials], which determine the
number of trials that are performed.

Extra things to take into account:

* The train/validation splits are different per trial but equal for all models.
* Re-evaluating the objective function at the same point (with the same
  hyperparameters) automatically skips the calculation and returns the
  same score as the equivalent trial.

!!! tip
    The hyperparameter tuning output can become quite wide for models
    with many hyperparameters. If you are working in a Jupyter Notebook,
    you can change the output's width running the following code in a cell:
    ```python
    from IPython.display import display, HTML
    display(HTML("<style>.container { width:100% !important; }</style>"))
    ```

Other settings can be changed through the [`ht_params`][directclassifier-ht_params]
parameter, a dictionary where every key-value combination can be used
to further customize the optimization.

By default, which hyperparameters are tuned and their corresponding
distributions are predefined by ATOM. Use the 'distributions' key to
customize these. Just like with `est_params`, it's possible to share the
same parameters across models or use a dictionary with the model name as
key to specify the parameters for every individual model. Use the key
'all' to tune some hyperparameters for all models when you also want to
tune other parameters only for specific ones. The following example tunes
the `n_estimators` parameter for both models but the `max_depth` parameter
only for the [RandomForest][].

```python
atom.run(
    models=["ET", "RF"],
    n_trials=30,
    ht_params={"distributions": {"all": "n_estimators", "RF": "max_depth"}},
)
```

Like the [`columns`][atomclassifier-add] parameter in atom's methods, you
can exclude parameters from the optimization adding `!` before its name.
It's possible to exclude multiple parameters, but not to combine inclusion
and exclusion for the same model. For example, to optimize a [RandomForest][]
using all its predefined parameters except `n_estimators`, run:

```python
atom.run(
    models="ET",
    n_trials=15,
    ht_params={"distributions": "!n_estimators"},
)
```

If just the parameter name is provided, the predefined distribution is
used. It's also possible to provide custom distributions spaces, but make
sure they are compliant with [optuna's API](https://optuna.readthedocs.io/en/stable/reference/distributions.html).
See every model's individual documentation in ATOM's API section for an
overview of their hyperparameters and distributions.

```python
from optuna.distributions import (
    IntDistribution, FloatDistribution, CategoricalDistribution
)

atom.run(
    models=["ET", "RF"],
    n_trials=30,
    ht_params={
        "dimensions": {
            "all": {"n_estimators": IntDistribution(10, 100, step=10)},
            "RF": {
                "max_depth": IntDistribution(1, 10),
                "max_features": CategoricalDistribution(["sqrt", "log2"]),
           },
        },
    }
)
```

Parameters for optuna's [study][] and the study's [optimize][] method can
be added as kwargs to `ht_params`. For example, to use a different sampler
or add a custom callback.

```python
from optuna.samplers import RandomSampler

atom.run(
    models="LR",
    n_trials=30,
    ht_params={
        "sampler": RandomSampler(seed=atom.random_state),
        "callbacks": custom_callback(),
    },
)
```

!!! note
    * If you use the default sampler, it’s recommended to consider setting
      larger `n_trials` to make full use of the characteristics of TPESampler
      because TPESampler uses some (by default, 10) trials for its startup.
    * When specifying distributions manually, make sure to import the
      distribution types from optuna: `#!python from optuna.distributions import ...`.

!!! warning
    Keras' models can only use hyperparameter tuning when `#!python n_jobs=1`
    or `#!python ht_params={"cv": 1}`. Using n_jobs > 1 and cv > 1 raises
    a PicklingError due to incompatibilities of the APIs. Read [here][deep-learning]
    more about deep learning models.

!!! tip
    ATOM has several plots that can help you examine a model's study and
    trials. Have a look at them [here][hyperparameter-tuning-plots].

<br>

## Pruning

During hyperparameter tuning, pruning stops unpromising trials at the
early stages of the training (a.k.a., automated early-stopping). This
can save the pipeline much time that would otherwise be wasted on an
estimator that is unlikely to yield the best results. A pruned trial
can't be selected as [`best_trial`][adaboost-best_trial]. Click
[here][example-pruning] to see an example that uses pruning.

The study uses [MedianPruner](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.MedianPruner.html)
as default pruner. You can use any other of optuna's [pruners](https://optuna.readthedocs.io/en/stable/reference/pruners.html)
through the [`ht_params`][directclassifier-ht_params] parameter.

```python
from optuna.pruners import HyperbandPruner

atom.run("SGD", n_trials=30, ht_params={"pruner": HyperbandPruner()})
```

!!! warning
    * Pruning is disabled for [multi-metric runs][].
    * Pruning is only available for models that support [in-training validation][].

<br>

## Bootstrapping

After fitting the estimator, you can assess the robustness of the model
using the [bootstrap](https://en.wikipedia.org/wiki/Bootstrapping_(statistics))
technique. This technique creates several new data sets selecting random 
samples from the training set (with replacement) and evaluates them on 
the test set. This way you can get a distribution of the performance of
the model. The sets are the same for every model. The number of sets can
be chosen through the [`n_bootstrap`][directclassifier-n_bootstrap] parameter.

!!! tip
    Use the [plot_results][] method to plot the boostrap scores in a boxplot.


<br>

## Successive halving

Successive halving is a bandit-based algorithm that fits N models to
1/N of the data. The best half are selected to go to the next iteration
where the process is repeated. This continues until only one model
remains, which is fitted on the complete dataset. Beware that a model's
performance can depend greatly on the amount of data on which it is
trained. For this reason, we recommend only to use this technique with
similar models, e.g., only using tree-based models.

Run successive halving from atom via the [successive_halving][atomclassifier-successive_halving]
method. Consecutive runs of the same model are saved with the model's
acronym followed by the number of models in the run. For example, a
[RandomForest][] in a run with 4 models would become model `RF4`.

See [here][example-successive-halving] a successive halving example.

!!! tip
    Use the [plot_successive_halving][] method to see every model's
    performance per iteration of the successive halving.

<br>

## Train sizing

When training models, there is usually a trade-off between model
performance and computation time, that is regulated by the number of
samples in the training set. Train sizing can be used to create
insights in this trade-off, and help determine the optimal size of
the training set. The models are fitted multiple times, ever-increasing
the number of samples in the training set.

Run train sizing from atom via the [train_sizing][atomclassifier-train_sizing]
method. The number of iterations and the number of samples per training
can be specified with the [`train_sizes`][trainsizingclassifier-train_sizes]
parameter. Consecutive runs of the same model are saved with the model's
acronym followed by the fraction of rows in the training set (the `.` is
removed from the fraction!). For example, a [RandomForest][] in a run with
80% of the training samples would become model `RF08`.

See [here][example-train-sizing] a train sizing example.

!!! tip
    Use the [plot_learning_curve][] method to see the model's performance
    per size of the training set.
