# Training
----------

The training phase is where the models are fitted and evaluated. After
this, the models are attached to the trainer, and you can use the
[plotting](../plots) and [predicting](../predicting) methods.  The
pipeline applies the following steps iteratively for all models:

1. The optimal hyperparameters for the model are selected using a [bayesian
   optimization](#hyperparameter-tuning) algorithm (optional).
2. The model is fitted on the training set using the best combination
   of hyperparameters found. After that, the model is evaluated on the tes set.
3. Calculate various scores on the test set using a [bootstrap](#bootstrapping)
   algorithm (optional).

There are three approaches to run the training.

* Direct training:
    - [DirectClassifier](../../API/training/directclassifier)
    - [DirectRegressor](../../API/training/directregressor)
* Training via [successive halving](#successive-halving):
    - [SuccessiveHalvingClassifier](../../API/training/successivehalvingclassifier)
    - [SuccessiveHavingRegressor](../../API/training/successivehalvingregressor)
* Training via [train sizing](#train-sizing):
    - [TrainSizingClassifier](../../API/training/trainsizingclassifier)
    - [TrainSizingRegressor](../../API/training/trainsizingregressor)

The direct fashion repeats the aforementioned steps only once, while the
other two approaches repeats them more than once. Every approach can be
directly called from atom through the [run](../../API/ATOM/atomclassifier/#run),
[successive_halving](../../API/ATOM/atomclassifier/#successive-halving)
and [train_sizing](../../API/ATOM/atomclassifier/#train-sizing) methods
respectively.

Models are called through their [acronyms](../models), e.g. `atom.run(models="RF")`
will train a [Random Forest](../../API/models/rf). If you want to run
the same model multiple times, add a tag after the acronym to
differentiate them.

```python
atom.run(
    models=["RF1", "RF2"],
    est_params={
        "RF1": {"n_estimators": 100},
        "RF2": {"n_estimators": 200},
    }
) 
```

For example, this pipeline will fit two Random Forest models, one
with 100 and the other with 200 decision trees. The models can be
accessed through `atom.rf1` and `atom.rf2`. Use tagged models to
test how the same model performs when fitted with different
parameters or on different data sets. See the
[Imbalanced datasets](../../examples/imbalanced_datasets) example.


Additional things to take into account:

* If an exception is encountered while fitting an estimator, the
  pipeline will automatically jump to the next model. The exceptions are
  stored in the `errors` attribute. Note that when a model is skipped,
  there is no model subclass for that estimator.
* When showing the final results, a `!` indicates the highest score
  and a `~` indicates that the model is possibly overfitting (training
  set has a score at least 20% higher than the test set).
* The winning model (the one with the highest `mean_bootstrap` or
  `metric_test`) can be accessed through the `winner` attribute.

<br>

## Metric

ATOM uses sklearn's [scorers](https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules)
for model evaluation. A scorer consists of a metric function and
some parameters that define the scorer's properties , such as if
a higher or lower score is better (score or loss function) or if
the function needs probability estimates or rounded predictions
(see the [make_scorer](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html)
function). ATOM lets you define the scorer for the pipeline in three ways:

* The `metric` parameter is the name of a [predefined scorer](#predefined-scorers).
* The `metric` parameter is a function with signature metric(y, y_pred).
  In this case, use the `greater_is_better`, `needs_proba` and `needs_threshold`
  parameters to specify the scorer's properties.
* The `metric` parameter is a scorer object.

Note that all scorers follow the convention that higher return values
are better than lower return values. Thus, metrics which measure the
distance between the model and the data (i.e. loss functions), like
`max_error` or `mean_squared_error`, will return the negated value of
the metric.


<a name="predefined-scorers"></a>
**Predefined scorers**<br>
ATOM accepts all of sklearn's [SCORERS](https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules)
as well as some custom acronyms and custom scorers. Since some of
sklearn's scorers have quite long names and ATOM is all about
<s>lazy</s>fast experimentation, the package provides acronyms
for some of the most commonly used ones. These acronyms are
case-insensitive and can be used in the `metric` parameter instead
of the scorer's full name, e.g. `atom.run("LR", metric="BA")` will
use `balanced_accuracy`. The available acronyms are:

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
* "Lift" for Lift
* "MCC" for Matthews Correlation Coefficient (also for multiclass classification)


**Multi-metric runs**<br>
Sometimes it is useful to measure the performance of the models in more
than one way. ATOM lets you run the pipeline with multiple metrics at
the same time. To do so, provide the `metric` parameter with a list of
desired metrics, e.g. `atom.run("LDA", metric=["r2", "mse"])`. If you
provide metric functions, don't forget to also provide a sequence of
values to the `greater_is_better`, `needs_proba` and `needs_threshold`
parameters, where the n-th value in corresponds to the n-th function.
If you leave them as a single value, that value will apply to every
provided metric.

When fitting multi-metric runs, the resulting scores will return a list
of metrics. For example, if you provided three metrics to the pipeline,
`atom.knn.metric_bo` could return [0.8734, 0.6672, 0.9001]. Only the
first metric of a multi-metric run is used to evaluate every step of
the bayesian optimization and to select the winning model.

!!! info
    Some plots let you choose which of the metrics to show using the
    `metric` parameter.

<br>

## Automated feature scaling

Models that require feature scaling will automatically do so before
training, if the data is not already scaled. The data is considered
scaled if it has one of the following prerequisites:

* The mean value over the mean of all columns is <0.05 and the mean of
  the standard deviation over all columns lies between 0.93 and 1.07.
* There is a transformer in the pipeline whose \__name__ contains the
  word `scaler`.

The scaling is applied using a [Scaler](../../API/data_cleaning/scaler)
with default parameters. It can be accessed from the model through the
`scaler` attribute. The scaled dataset can be examined through the
model's [data attributes](../../API/models/gnb/#data-attributes). Use
the [available_models](../../API/ATOM/atomclassifier/#available-models)
method to see which models require feature scaling. 

<br>

## Parameter customization

By default, the parameters every estimator uses are the same default
parameters they get from their respective packages. To select different
ones, use `est_params`. There are two ways to add custom parameters to
the models: adding them directly to the dictionary as key-value pairs
or through various dictionaries with the model names as keys.

Adding the parameters directly to `est_params` will share them across
all models in the pipeline. In this example, both the XGBoost and the
LightGBM model will use n_estimators=200. Make sure all the models do 
have the specified parameters or an exception will be raised!

```python
atom.run(models=["XGB", "LGB"], est_params={"n_estimators": 200})
```

To specify parameters per model, use the model name as key and a dict
of the parameters as value. In this example, the XGBoost model will
use n_estimators=200 and the Multi-layer Perceptron will use one hidden
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
parameter. For example, to change XGBoost's verbosity, we can run:

```python
atom.run(models="XGB", est_params={"verbose_fit": True})
```

!!! note
    If a parameter is specified through `est_params`, it is
    ignored by the bayesian optimization! 


<br>

## Hyperparameter tuning

In order to achieve maximum performance, it's important to tune an
estimator's hyperparameters before training it. ATOM provides
[hyperparameter tuning](https://en.wikipedia.org/wiki/Hyperparameter_optimization)
using a [bayesian optimization](https://en.wikipedia.org/wiki/Bayesian_optimization#:~:text=Bayesian%20optimization%20is%20a%20sequential,expensive%2Dto%2Devaluate%20functions.)
(BO) approach implemented with [scikit-optimize](https://scikit-optimize.github.io/stable/).
The BO is optimized on the first metric provided with the `metric`
parameter. Each step is either computed by cross-validation on the
complete training set or by randomly splitting the training set every
iteration into a (sub) training and validation set. This process can
create some minimum data leakage towards specific parameters (since
the estimator is evaluated on data that is used to train the next
estimator), but it ensures maximal use of the provided data. However,
the leakage is not present in the independent test set, thus the final
score of every model is unbiased. Note that, if the dataset is relatively
small, the BO's best score can consistently be lower than the final score
on the test set due to the considerable fewer instances on which it is
trained.

There are many possibilities to tune the BO to your liking. Use
`n_calls` and `n_initial_points` to determine the number of iterations
that are performed randomly at the start (exploration) and the number
of iterations spent optimizing (exploitation). If `n_calls` is equal to
`n_initial_points`, every iteration of the BO will select its
hyperparameters randomly. This means the algorithm is technically
performing a [random search](https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf).

Extra things to take into account:

* The `n_calls` parameter includes the iterations in `n_initial_points`,
  i.e. calling `atom.run(models="LR", n_calls=20, n_intial_points=10)`
  will run 20 iterations of which the first 10 are random.
* If `n_initial_points=1`, the first call is equal to the
  estimator's default parameters.
* The train/validation splits are different per call but equal for all models.
* Re-evaluating the objective function at the same point automatically
  skips the calculation and returns the same score as the equivalent call.

!!! tip
    The hyperparameter tuning output can become quite wide for models
    with many hyperparameters. If you are working in a Jupyter Notebook,
    you can change the output's width running the following code in a cell:
    ```python
    from IPython.core.display import display, HTML
    display(HTML("<style>.container { width:100% !important; }</style>"))
    ```

Other settings can be changed through the `bo_params` parameter, a
dictionary where every key-value combination can be used to further
customize the BO.

By default, the hyperparameters and corresponding dimensions per model
are predefined by ATOM. Use the `dimensions` key to use custom ones.
Just like with `est_params`, you can share the same dimensions across
models or use a dictionary with the model names as keys to specify the
dimensions for every individual model. Note that the provided search
space dimensions must be compliant with skopt's API.

```python
atom.run(
    models="LR",
    n_calls=30,
    bo_params={"dimensions": [Integer(100, 1000, name="max_iter")]},
)
```

!!! warning
    Keras' models can only use hyperparameter tuning when `n_jobs=1` or
    `bo_params={"cv": 1}`. Using n_jobs > 1 and cv > 1 raises a PicklingError
    due to incompatibilities of the APIs. Read [here](../models/#deep-learning)
    more about deep learning models.

The majority of skopt's callbacks to stop the optimizer early can be
accessed through `bo_params`. Other callbacks can be included through
the `callbacks` key.

```python
atom.run(
    models="LR",
    n_calls=30,
    bo_params={"callbacks": custom_callback()},
)
```

It's also possible to include additional parameters for skopt's optimizer
as key-value pairs.

```python
atom.run("LR", n_calls=10, bo_params={"acq_func": "EI"})
```


<br>

## Bootstrapping

After fitting the estimator, you can assess the robustness of the model
using the [bootstrap](https://en.wikipedia.org/wiki/Bootstrapping_(statistics))
technique. This technique creates several new data sets selecting random 
samples from the training set (with replacement) and evaluates them on 
the test set. This way we get a distribution of the performance of the
model. The sets are the same for every model. The number of sets can be
chosen through the `n_bootstrap` parameter.

!!! tip
    Use the [plot_results](../../API/plots/plot_results) method to plot
    the boostrap scores in a boxplot.


<br>

## Early stopping

[XGBoost](../../API/models/xgb), [LightGBM](../../API/models/lgb) and [CatBoost](../../API/models/catb)
allow in-training evaluation. This means that the estimator is evaluated
after every round of the training, and that the training is stopped
early if it didn't improve in the last `early_stopping` rounds. This
can save the pipeline much time that would otherwise be wasted on an
estimator that is unlikely to improve further. Note that this technique
is applied both during the BO and at the final fit on the complete
training set.

There are two ways to apply early stopping on these models:

* Through the `early_stopping` key in `bo_params`. This approach applies
  early stopping to all models in the pipeline and allows the input of a
  fraction of the total number of rounds.
* Filling the `early_stopping_rounds` parameter directly in `est_params`.
  Don't forget to add `_fit` to the parameter to call it from the fit method.

After fitting, the model gets the `evals` attribute, a dictionary of the
train and test performances per round (also if early stopping wasn't
applied). Click [here](../../examples/early_stopping) for an example using
early stopping.

!!! tip
    Use the [plot_evals](../../API/plots/plot_evals) method to plot the
    in-training evaluation on the train and test set.


<br>

## Successive halving

Successive halving is a bandit-based algorithm that fits N models to
1/N of the data. The best half are selected to go to the next iteration
where the process is repeated. This continues until only one model
remains, which is fitted on the complete dataset. Beware that a model's
performance can depend greatly on the amount of data on which it is
trained. For this reason, we recommend only to use this technique with
similar models, e.g. only using tree-based models.

Use successive halving through the [SuccessiveHalvingClassifier](../../API/training/successivehalvingclassifier)/[SuccessiveHalvingRegressor](../../API/training/successivehalvingregressor)
classes or from atom via the [successive_halving](../../API/ATOM/atomclassifier/#successive-halving)
method. Consecutive runs of the same model are saved with the model's acronym
followed by the number of models in the run. For example, a
[Random Forest](../../API/models/rf) in a run with 4 models would become model
`RF4`.

Click [here](../../examples/successive_halving) for a successive halving example.

!!! tip
    Use the [plot_successive_halving](../../API/plots/plot_successive_halving)
    method to see every model's performance per iteration of the
    successive halving.

<br>

## Train sizing

When training models, there is usually a trade-off between model
performance and computation time, that is regulated by the number of
samples in the training set. Train sizing can be used to create
insights in this trade-off, and help determine the optimal size of
the training set. The models are fitted multiple times, ever-increasing
the number of samples in the training set.

Use train sizing through the [TrainSizingClassifier](../../API/training/trainsizingclassifier)/[TrainSizingRegressor](../../API/training/trainsizingregressor)
classes or from atom via the [train_sizing](../../API/ATOM/atomclassifier/#train-sizing)
method. The number of iterations and the number of samples per training
can be specified with the `train_sizes` parameter. Consecutive runs of the
same model are saved with the model's acronym followed by the fraction of
rows in the training set (the `.` is removed from the fraction!). For example,
a [Random Forest](../../API/models/rf) in a run with 80% of the training samples
would become model `RF08`.

Click [here](../../examples/train_sizing) for a train sizing example.

!!! tip
    Use the [plot_learning_curve](../../API/plots/plot_learning_curve)
    method to see the model's performance per size of the training set.
