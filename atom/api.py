# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing the API classes.

"""

from copy import deepcopy
from logging import Logger
from typing import Any, Optional, Union

import dill as pickle
from sklearn.base import clone
from typeguard import typechecked

from atom.atom import ATOM
from atom.basetransformer import BaseTransformer
from atom.utils import (
    INT, SCALAR, SEQUENCE_TYPES, Y_TYPES, Predictor, custom_transform,
)


# Functions ======================================================== >>

@typechecked
def ATOMLoader(
    filename: str,
    data: Optional[SEQUENCE_TYPES] = None,
    *,
    transform_data: bool = True,
    verbose: Optional[INT] = None,
) -> Any:
    """Load a class instance from a pickle file.

    If the file is an atom instance that was saved using
    `save_data=False`, it's possible to load new data into it
    and apply all data transformations.

    Parameters
    ----------
    filename: str
        Name of the pickle file to load.

    data: sequence of indexables or None, default=None
        Original dataset. Only use this parameter if the file is an
        atom instance that was saved using `save_data=False`. Allowed
        formats are:

        - X
        - X, y
        - train, test
        - train, test, holdout
        - X_train, X_test, y_train, y_test
        - X_train, X_test, X_holdout, y_train, y_test, y_holdout
        - (X_train, y_train), (X_test, y_test)
        - (X_train, y_train), (X_test, y_test), (X_holdout, y_holdout)

        **X, train, test: dataframe-like**<br>
        Feature set with shape=(n_samples, n_features).

        **y: int, str or sequence**<br>
        Target column corresponding to X.

        - If int: Position of the target column in X.
        - If str: Name of the target column in X.
        - Else: Array with shape=(n_samples,) to use as target.

    transform_data: bool, default=True
        If False, the `data` is left as provided. If True, it is
        transformed through all the steps in the instance's pipeline.
        This parameter is ignored if the loaded pickle is not an atom
        instance.

    verbose: int or None, default=None
        Verbosity level of the transformations applied on the new
        data. If None, use the verbosity from the loaded instance.
        This parameter is ignored if `transform_data=False`.

    Returns
    -------
    class instance
        Unpickled instance.

    Examples
    --------

    ```pycon
    >>> from atom import ATOMClassifier, ATOMLoader
    >>> from sklearn.datasets import load_breast_cancer

    >>> atom = ATOMClassifier(X, y)
    >>> atom.scale()
    >>> atom.run(["LR", "RF", "SGD"], metric="AP")
    >>> atom.save("atom", save_data=False)  # Save atom to a pickle file

    # Load the class and add the data to the new instance
    >>> atom_2 = ATOMLoader("atom", data=(X, y), verbose=2)

    Transforming data for branch master:
    Scaling features...
    ATOMClassifier successfully loaded.

    >>> print(atom_2.results)

          score_train   score_test time_fit    time
    LR       0.998179     0.998570   0.016s  0.016s
    RF       1.000000     0.995568   0.141s  0.141s
    SGD      0.998773     0.994313   0.016s  0.016s

    ```

    """
    with open(filename, "rb") as f:
        cls = pickle.load(f)

    # Reassign the transformer attributes (warnings random_state, etc...)
    if issubclass(cls.__class__, BaseTransformer):
        BaseTransformer.__init__(
            cls, **{x: getattr(cls, x) for x in BaseTransformer.attrs if hasattr(cls, x)}
        )

    if data is not None:
        if not hasattr(cls, "_branches"):
            raise TypeError(
                "Data is provided but the class is not an "
                f"atom instance, got {cls.__class__.__name__}."
            )
        elif any(branch._data is not None for branch in cls._branches.values()):
            raise ValueError(
                f"The loaded {cls.__class__.__name__} instance already contains data!"
            )

        # Prepare the provided data
        data, idx, cls.holdout = cls._get_data(data, use_n_rows=transform_data)

        # Apply transformations per branch
        step = {}  # Current step in the pipeline per branch
        for b1, v1 in cls._branches.items():
            branch = cls._branches[b1]

            # Provide the input data if not already filled from another branch
            if branch._data is None:
                branch._data, branch._idx = data, idx

            if transform_data:
                if len(cls._branches) > 2 and not v1.pipeline.empty:
                    cls.log(f"Transforming data for branch {b1}:", 1)

                for i, est1 in enumerate(v1.pipeline):
                    # Skip if the transformation was already applied
                    if step.get(b1, -1) < i:
                        custom_transform(est1, branch, verbose=verbose)

                        for b2, v2 in cls._branches.items():
                            if b1 != b2 and v2.pipeline.get(i) is est1:
                                # Update the data and step for the other branch
                                cls._branches[b2]._data = deepcopy(branch._data)
                                cls._branches[b2]._idx = deepcopy(branch._idx)
                                step[b2] = i

    cls.log(f"{cls.__class__.__name__} successfully loaded.", 1)

    return cls


@typechecked
def ATOMModel(
    estimator: Predictor,
    *,
    acronym: Optional[str] = None,
    needs_scaling: bool = False,
    has_validation: Optional[str] = None,
) -> Predictor:
    """Convert an estimator to a model that can be ingested by atom.

    This function adds the relevant attributes to the estimator so
    that they can be used by atom. Note that only estimators that follow
    [sklearn's API][api] are compatible.

    Read more about using custom models in the [user guide][custom-models].

    Parameters
    ----------
    estimator: Predictor
        Custom estimator. Should implement a `fit` and `predict` method.

    acronym: str or None, default=None
        Model's acronym. Used to call the model from atom. If None, the
        capital letters in the estimator's \__name__ are used (only if
        two or more, else it uses the entire \__name__).

    needs_scaling: bool, default=False
        Whether the model needs scaled features.

    has_validation: str or None, default=None
        Whether the model allows [in-training validation][]. If str,
        name of the estimator's parameter that states the number of
        iterations. If None, no support for in-training validation.

    Returns
    -------
    estimator
        Clone of the provided estimator with custom attributes.

    Examples
    --------

    ```pycon
    >>> from atom import ATOMRegressor, ATOMModel
    >>> from sklearn.linear_model import RANSACRegressor

    >>> ransac =  ATOMModel(
    ...      estimator=RANSACRegressor(),
    ...      acronym="RANSAC",
    ...      needs_scaling=False,
    ...  )

    >>> atom = ATOMRegressor(X, y)
    >>> atom.run(ransac, verbose=2)

    Training ========================= >>
    Models: RANSAC
    Metric: r2

    Results for RANSACRegressor:
    Fit ---------------------------------------------
    Train evaluation --> r2: -2.1784
    Test evaluation --> r2: -9.4592
    Time elapsed: 0.072s
    -------------------------------------------------
    Total time: 0.072s

    Final results ==================== >>
    Total time: 0.072s
    -------------------------------------
    RANSACRegressor --> r2: -9.4592 ~

    ```

    """
    estimator = clone(estimator)

    if acronym:
        estimator.acronym = acronym
    estimator.needs_scaling = needs_scaling
    estimator.has_validation = has_validation

    return estimator


# Classes ========================================================== >>

class ATOMClassifier(BaseTransformer, ATOM):
    """Main class for binary and multiclass classification tasks.

    Apply all data transformations and model management provided by
    the package on a given dataset. Note that, contrary to sklearn's
    API, the instance contains the dataset on which to perform the
    analysis. Calling a method will automatically apply it on the
    dataset it contains.

    All [data cleaning][], [feature engineering][], [model training]
    [training], [predicting][], and [plotting][plots] functionality
    can be accessed from an instacne of this class.

    Parameters
    ----------
    *arrays: sequence of indexables
        Dataset containing features and target. Allowed formats are:

        - X
        - X, y
        - train, test
        - train, test, holdout
        - X_train, X_test, y_train, y_test
        - X_train, X_test, X_holdout, y_train, y_test, y_holdout
        - (X_train, y_train), (X_test, y_test)
        - (X_train, y_train), (X_test, y_test), (X_holdout, y_holdout)

        **X, train, test: dataframe-like**<br>
        Feature set with shape=(n_samples, n_features).

        **y: int, str or sequence**<br>
        Target column corresponding to X.

        - If int: Position of the target column in X.
        - If str: Name of the target column in X.
        - Else: Array with shape=(n_samples,) to use as target.

    y: int, str or sequence, default=-1
        Target column corresponding to X.

        - If int: Position of the target column in X.
        - If str: Name of the target column in X.
        - Else: Array with shape=(n_samples,) to use as target.

        This parameter is ignored if the target column is provided
        through `arrays`.

    index: bool, int, str or sequence, default=False
        Handle the index in the resulting dataframe.

        - If False: Reset to [RangeIndex][].
        - If True: Use the provided index.
        - If int: Position of the column to use as index.
        - If str: Name of the column to use as index.
        - If sequence: Array with shape=(n_samples,) to use as index.

    test_size: int or float, default=0.2
        - If <=1: Fraction of the dataset to include in the test set.
        - If >1: Number of rows to include in the test set.

        This parameter is ignored if the test set is provided
        through `arrays`.

    holdout_size: int, float or None, default=None
        - If None: No holdout data set is kept apart.
        - If <=1: Fraction of the dataset to include in the holdout set.
        - If >1: Number of rows to include in the holdout set.

        This parameter is ignored if the holdout set is provided
        through `arrays`.

    shuffle: bool, default=True
        Whether to shuffle the dataset before splitting the train and
        test set. Be aware that not shuffling the dataset can cause
        an unequal distribution of target classes over the sets.

    stratify: bool, int, str or sequence, default=True
        Handle stratification of the target classes over the data sets.

        - If False: The data is split randomly.
        - If True: The data is stratified over the target column.
        - Else: Name or position of the columns to stratify by. The
          columns can't contain `NaN` values.

        This parameter is ignored if `shuffle=False` or if the test
        set is provided through `arrays`.

    n_rows: int or float, default=1
        Random subsample of the dataset to use. The default value selects
        all rows.

        - If <=1: Fraction of the dataset to select.
        - If >1: Exact number of rows to select. Only if `arrays` is X
                 or X, y.

    n_jobs: int, default=1
        Number of cores to use for parallel processing.

        - If >0: Number of cores to use.
        - If -1: Use all available cores.
        - If <-1: Use number of cores - 1 + `n_jobs`.

    device: str, default="cpu"
        Device on which to train the estimators. Use any string
        that follows the [SYCL_DEVICE_FILTER][] filter selector,
        e.g. `device="gpu"` to use the GPU. Read more in the
        [user guide][accelerating-pipelines].

    engine: str, default="sklearn"
        Execution engine to use for the estimators. Refer to the
        [user guide][accelerating-pipelines] for an explanation
        regarding every choice. Choose from:

        - "sklearn" (only if device="cpu")
        - "sklearnex"
        - "cuml" (only if device="gpu")

    verbose: int, default=0
        Verbosity level of the class. Choose from:

        - 0 to not print anything.
        - 1 to print basic information.
        - 2 to print detailed information.

    warnings: bool or str, default=False
        - If True: Default warning action (equal to "default").
        - If False: Suppress all warnings (equal to "ignore").
        - If str: One of python's [warnings filters][warnings].

        Changing this parameter affects the `PYTHONWARNINGS` environment.
        ATOM can't manage warnings that go from C/C++ code to stdout.

    logger: str, Logger or None, default=None
        - If None: Doesn't save a logging file.
        - If str: Name of the log file. Use "auto" for automatic name.
        - Else: Python `logging.Logger` instance.

    experiment: str or None, default=None
        Name of the [mlflow experiment][experiment] to use for tracking.
        If None, no mlflow tracking is performed.

    random_state: int or None, default=None
        Seed used by the random number generator. If None, the random
        number generator is the `RandomState` used by `np.random`.

    See Also
    --------
    atom.api:ATOMRegressor

    Examples
    --------

    ```pycon
    >>> from atom import ATOMClassifier
    >>> from sklearn.datasets import load_breast_cancer

    >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    >>> # Initialize atom
    >>> atom = ATOMClassifier(X, y, logger="auto", n_jobs=2, verbose=2)

    << ================== ATOM ================== >>
    Algorithm task: binary classification.
    Parallel processing with 2 cores.

    Dataset stats ==================== >>
    Shape: (569, 31)
    Memory: 138.96 kB
    Scaled: False
    Outlier values: 160 (1.1%)
    -------------------------------------
    Train set size: 456
    Test set size: 113
    -------------------------------------
    |   |     dataset |       train |        test |
    | - | ----------- | ----------- | ----------- |
    | 0 |   212 (1.0) |   170 (1.0) |    42 (1.0) |
    | 1 |   357 (1.7) |   286 (1.7) |    71 (1.7) |

    >>> # Apply data cleaning and feature engineering methods
    >>> atom.balance(strategy="smote")

    Oversampling with SMOTE...
     --> Adding 116 samples to class 0.

    >>> atom.feature_selection(strategy="rfecv", solver="xgb", n_features=22)

    Fitting FeatureSelector...
    Performing feature selection ...
     --> RFECV selected 26 features from the dataset.
       --> Dropping feature mean perimeter (rank 4).
       --> Dropping feature mean symmetry (rank 3).
       --> Dropping feature perimeter error (rank 2).
       --> Dropping feature worst compactness (rank 5).

    >>> # Train models
    >>> atom.run(
    ...    models=["LR", "RF", "XGB"],
    ...    metric="precision",
    ...    n_bootstrap=4,
    ... )

    Training ========================= >>
    Models: LR, RF, XGB
    Metric: precision

    Results for Logistic Regression:
    Fit ---------------------------------------------
    Train evaluation --> precision: 0.9895
    Test evaluation --> precision: 0.9467
    Time elapsed: 0.028s
    -------------------------------------------------
    Total time: 0.028s

    Results for Random Forest:
    Fit ---------------------------------------------
    Train evaluation --> precision: 1.0
    Test evaluation --> precision: 0.9221
    Time elapsed: 0.181s
    -------------------------------------------------
    Total time: 0.181s

    Results for XGBoost:
    Fit ---------------------------------------------
    Train evaluation --> precision: 1.0
    Test evaluation --> precision: 0.9091
    Time elapsed: 0.124s
    -------------------------------------------------
    Total time: 0.124s

    Final results ==================== >>
    Total time: 0.333s
    -------------------------------------
    Logistic Regression --> precision: 0.9467 !
    Random Forest       --> precision: 0.9221
    XGBoost             --> precision: 0.9091

    >>> # Analyze the results
    >>> atom.evaluate()

         accuracy  average_precision  ...    recall   roc_auc
    LR   0.970588           0.995739  ...  0.981308  0.993324
    RF   0.958824           0.982602  ...  0.962617  0.983459
    XGB  0.964706           0.996047  ...  0.971963  0.993473

    [3 rows x 9 columns]

    ```

    """

    @typechecked
    def __init__(
        self,
        *arrays,
        y: Y_TYPES = -1,
        index: Union[bool, INT, str, SEQUENCE_TYPES] = False,
        shuffle: bool = True,
        stratify: Union[bool, INT, str, SEQUENCE_TYPES] = True,
        n_rows: SCALAR = 1,
        test_size: SCALAR = 0.2,
        holdout_size: Optional[SCALAR] = None,
        n_jobs: INT = 1,
        device: str = "cpu",
        engine: str = "sklearn",
        verbose: INT = 0,
        warnings: Union[bool, str] = False,
        logger: Optional[Union[str, Logger]] = None,
        experiment: Optional[str] = None,
        random_state: Optional[INT] = None,
    ):
        super().__init__(
            n_jobs=n_jobs,
            device=device,
            engine=engine,
            verbose=verbose,
            warnings=warnings,
            logger=logger,
            experiment=experiment,
            random_state=random_state,
        )

        self.goal = "class"
        ATOM.__init__(
            self,
            arrays=arrays,
            y=y,
            index=index,
            test_size=test_size,
            holdout_size=holdout_size,
            shuffle=shuffle,
            stratify=stratify,
            n_rows=n_rows,
        )


class ATOMRegressor(BaseTransformer, ATOM):
    """Main class for regression tasks.

    Apply all data transformations and model management provided by
    the package on a given dataset. Note that, contrary to sklearn's
    API, the instance contains the dataset on which to perform the
    analysis. Calling a method will automatically apply it on the
    dataset it contains.

    All [data cleaning][], [feature engineering][], [model training]
    [training], [predicting][], and [plotting][plots] functionality
    can be accessed from an instacne of this class.

    Parameters
    ----------
    *arrays: sequence of indexables
        Dataset containing features and target. Allowed formats are:

        - X
        - X, y
        - train, test
        - train, test, holdout
        - X_train, X_test, y_train, y_test
        - X_train, X_test, X_holdout, y_train, y_test, y_holdout
        - (X_train, y_train), (X_test, y_test)
        - (X_train, y_train), (X_test, y_test), (X_holdout, y_holdout)

        **X, train, test: dataframe-like**<br>
        Feature set with shape=(n_samples, n_features).

        **y: int, str or sequence**<br>
        Target column corresponding to X.

        - If int: Position of the target column in X.
        - If str: Name of the target column in X.
        - Else: Array with shape=(n_samples,) to use as target.

    y: int, str or sequence, default=-1
        Target column corresponding to X.

        - If int: Position of the target column in X.
        - If str: Name of the target column in X.
        - Else: Array with shape=(n_samples,) to use as target.

        This parameter is ignored if the target column is provided
        through `arrays`.

    index: bool, int, str or sequence, default=False
        Handle the index in the resulting dataframe.

        - If False: Reset to [RangeIndex][].
        - If True: Use the provided index.
        - If int: Position of the column to use as index.
        - If str: Name of the column to use as index.
        - If sequence: Array with shape=(n_samples,) to use as index.

    test_size: int or float, default=0.2
        - If <=1: Fraction of the dataset to include in the test set.
        - If >1: Number of rows to include in the test set.

        This parameter is ignored if the test set is provided
        through `arrays`.

    holdout_size: int, float or None, default=None
        - If None: No holdout data set is kept apart.
        - If <=1: Fraction of the dataset to include in the holdout set.
        - If >1: Number of rows to include in the holdout set.

        This parameter is ignored if the holdout set is provided
        through `arrays`.

    shuffle: bool, default=True
        Whether to shuffle the dataset before splitting the train and
        test set. Be aware that not shuffling the dataset can cause
        an unequal distribution of target classes over the sets.

    n_rows: int or float, default=1
        Random subsample of the dataset to use. The default value selects
        all rows.

        - If <=1: Fraction of the dataset to select.
        - If >1: Exact number of rows to select. Only if `arrays` is X
                 or X, y.

    n_jobs: int, default=1
        Number of cores to use for parallel processing.

        - If >0: Number of cores to use.
        - If -1: Use all available cores.
        - If <-1: Use number of cores - 1 + `n_jobs`.

    device: str, default="cpu"
        Device on which to train the estimators. Use any string
        that follows the [SYCL_DEVICE_FILTER][] filter selector,
        e.g. `device="gpu"` to use the GPU. Read more in the
        [user guide][accelerating-pipelines].

    engine: str, default="sklearn"
        Execution engine to use for the estimators. Refer to the
        [user guide][accelerating-pipelines] for an explanation
        regarding every choice. Choose from:

        - "sklearn" (only if device="cpu")
        - "sklearnex"
        - "cuml" (only if device="gpu")

    verbose: int, default=0
        Verbosity level of the class. Choose from:

        - 0 to not print anything.
        - 1 to print basic information.
        - 2 to print detailed information.

    warnings: bool or str, default=False
        - If True: Default warning action (equal to "default").
        - If False: Suppress all warnings (equal to "ignore").
        - If str: One of python's [warnings filters][warnings].

        Changing this parameter affects the `PYTHONWARNINGS` environment.
        ATOM can't manage warnings that go from C/C++ code to stdout.

    logger: str, Logger or None, default=None
        - If None: Doesn't save a logging file.
        - If str: Name of the log file. Use "auto" for automatic name.
        - Else: Python `logging.Logger` instance.

    experiment: str or None, default=None
        Name of the [mlflow experiment][experiment] to use for tracking.
        If None, no mlflow tracking is performed.

    random_state: int or None, default=None
        Seed used by the random number generator. If None, the random
        number generator is the `RandomState` used by `np.random`.

    See Also
    --------
    atom.api:ATOMClassifier

    Examples
    --------

    ```pycon
    >>> from atom import ATOMRegressor
    >>> from sklearn.datasets import load_diabetes

    >>> X, y = load_diabetes(return_X_y=True, as_frame=True)

    >>> # Initialize atom
    >>> atom = ATOMRegressor(X, y, logger="auto", n_jobs=2, verbose=2)

    << ================== ATOM ================== >>
    Algorithm task: regression.
    Parallel processing with 2 cores.

    Dataset stats ==================== >>
    Shape: (442, 11)
    Memory: 39.02 kB
    Scaled: False
    Outlier values: 10 (0.3%)
    -------------------------------------
    Train set size: 310
    Test set size: 132

    >>> # Apply data cleaning and feature engineering methods
    >>> atom.scale()

    Fitting Scaler...
    Scaling features...

    >>> atom.feature_selection(strategy="rfecv", solver="xgb", n_features=12)

    Fitting FeatureSelector...
    Performing feature selection ...
     --> rfecv selected 6 features from the dataset.
       --> Dropping feature age (rank 5).
       --> Dropping feature sex (rank 4).
       --> Dropping feature s1 (rank 2).
       --> Dropping feature s3 (rank 3).

    >>> # Train models
    >>> atom.run(
    ...    models=["LR", "RF", "XGB"],
    ...    metric="precision",
    ...    n_bootstrap=4,
    ... )

    Training ========================= >>
    Models: OLS, BR, RF
    Metric: r2

    Results for Ordinary Least Squares:
    Fit ---------------------------------------------
    Train evaluation --> r2: 0.5223
    Test evaluation --> r2: 0.4012
    Time elapsed: 0.010s
    -------------------------------------------------
    Total time: 0.010s

    Results for Bayesian Ridge:
    Fit ---------------------------------------------
    Train evaluation --> r2: 0.522
    Test evaluation --> r2: 0.4037
    Time elapsed: 0.010s
    -------------------------------------------------
    Total time: 0.010s

    Results for Random Forest:
    Fit ---------------------------------------------
    Train evaluation --> r2: 0.9271
    Test evaluation --> r2: 0.259
    Time elapsed: 0.175s
    -------------------------------------------------
    Total time: 0.175s


    Final results ==================== >>
    Total time: 0.195s
    -------------------------------------
    Ordinary Least Squares --> r2: 0.4012 ~
    Bayesian Ridge         --> r2: 0.4037 ~ !
    Random Forest          --> r2: 0.259 ~

    >>> # Analyze the results
    >>> atom.evaluate()

         neg_mean_absolute_error  ...  neg_root_mean_squared_error
    OLS               -43.756992  ...                   -54.984345
    BR                -43.734975  ...                   -54.869543
    RF                -48.327879  ...                   -61.167760

    [3 rows x 7 columns]

    ```

    """

    @typechecked
    def __init__(
        self,
        *arrays,
        y: Y_TYPES = -1,
        index: Union[bool, INT, str, SEQUENCE_TYPES] = False,
        shuffle: bool = True,
        n_rows: SCALAR = 1,
        test_size: SCALAR = 0.2,
        holdout_size: Optional[SCALAR] = None,
        n_jobs: INT = 1,
        device: str = "cpu",
        engine: str = "sklearn",
        verbose: INT = 0,
        warnings: Union[bool, str] = False,
        logger: Optional[Union[str, Logger]] = None,
        experiment: Optional[str] = None,
        random_state: Optional[INT] = None,
    ):
        super().__init__(
            n_jobs=n_jobs,
            device=device,
            engine=engine,
            verbose=verbose,
            warnings=warnings,
            logger=logger,
            experiment=experiment,
            random_state=random_state,
        )

        self.goal = "reg"
        ATOM.__init__(
            self,
            arrays=arrays,
            y=y,
            index=index,
            test_size=test_size,
            holdout_size=holdout_size,
            shuffle=shuffle,
            stratify=False,
            n_rows=n_rows,
        )
