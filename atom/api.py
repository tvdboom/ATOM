"""Automated Tool for Optimized Modeling (ATOM).

Author: Mavs
Description: Module containing the API classes.

"""

from __future__ import annotations

from logging import Logger
from pathlib import Path
from typing import TypeVar

from beartype import beartype
from joblib.memory import Memory
from sklearn.base import clone

from atom.atom import ATOM
from atom.utils.types import (
    Backend, Bool, ColumnSelector, Engine, IndexSelector, IntLargerEqualZero,
    NJobs, Predictor, Scalar, Seasonality, SPDict, Verbose, Warnings,
    YSelector,
)
from atom.utils.utils import Goal


T_Predictor = TypeVar("T_Predictor", bound=Predictor)


@beartype
def ATOMModel(
    estimator: T_Predictor,
    name: str | None = None,
    *,
    acronym: str | None = None,
    needs_scaling: Bool = False,
    native_multilabel: Bool = False,
    native_multioutput: Bool = False,
    validation: str | None = None,
) -> T_Predictor:
    """Convert an estimator to a model that can be ingested by atom.

    This function adds the relevant tags to the estimator so that they
    can be used by `atom`. Note that only estimators that follow
    [sklearn's API][api] are compatible.

    Read more about custom models in the [user guide][custom-models].

    Parameters
    ----------
    estimator: Predictor
        Custom estimator. Should implement a `fit` and `predict` method.

    name: str or None, default=None
        Name for the model. This is the value used to call the
        model from atom. The value should start with the model's
        `acronym` when specified. If None, the capital letters of the
        estimator's name are used (only if two or more, else it uses
        the entire name).

    acronym: str or None, default=None
        Model's acronym. If None, it uses the model's `name`. Specify
        this parameter when you want to train multiple custom models
        that share the same estimator.

    needs_scaling: bool, default=False
        Whether the model should use [automated feature scaling][].

    native_multilabel: bool, default=False
        Whether the model has native support for [multilabel][] tasks.
        If False and the task is multilabel, a multilabel meta-estimator
        is wrapper around the estimator.

    native_multioutput: bool, default=False
        Whether the model has native support for [multioutput tasks][].
        If False and the task is multioutput, a multioutput
        meta-estimator is wrapped around the estimator.

    validation: str or None, default=None
        Whether the model allows [in-training validation][].

        - If None: No support for in-training validation.
        - If str: Name of the estimator's parameter that states the
          number of iterations, e.g., `n_estimators` for
          [RandomForestClassifier][].

    Returns
    -------
    Predictor
        Estimator with provided information. Provide this instance to
        the `models` parameter of the [run][atomclassifier-run] method.

    Examples
    --------
    ```pycon
    from atom import ATOMRegressor, ATOMModel
    from sklearn.datasets import load_diabetes
    from sklearn.linear_model import RANSACRegressor

    ransac = ATOMModel(
        estimator=RANSACRegressor(),
        name="RANSAC",
        needs_scaling=False,
    )

    X, y = load_diabetes(return_X_y=True, as_frame=True)

    atom = ATOMRegressor(X, y, verbose=2)
    atom.run(ransac)
    ```

    """
    if callable(estimator):
        estimator_c = estimator()
    else:
        estimator_c = clone(estimator)

    if name:
        estimator_c.name = name
    if acronym:
        estimator_c.acronym = acronym
    estimator_c.needs_scaling = needs_scaling
    estimator_c.native_multilabel = native_multilabel
    estimator_c.native_multioutput = native_multioutput
    estimator_c.validation = validation

    return estimator_c


@beartype
class ATOMClassifier(ATOM):
    """Main class for classification tasks.

    Apply all data transformations and model management provided by
    the package on a given dataset. Note that, contrary to sklearn's
    API, the instance contains the dataset on which to perform the
    analysis. Calling a method will automatically apply it on the
    dataset it contains.

    All [data cleaning][], [feature engineering][], [model training]
    [training] and [plotting][plots] functionality can be accessed
    from an instance of this class.

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

        **y: int, str, dict, sequence or dataframe**<br>
        Target column(s) corresponding to `X`.

        - If int: Position of the target column in `X`.
        - If str: Name of the target column in `X`.
        - If dict: Name of the target column and sequence of values.
        - If sequence: Target column with shape=(n_samples,) or
          sequence of column names or positions for multioutput tasks.
        - If dataframe: Target columns for multioutput tasks.

    y: int, str, dict, sequence or dataframe, default=-1
        Target column(s) corresponding to `X`.

        - If int: Position of the target column in `X`.
        - If str: Name of the target column in `X`.
        - If dict: Name of the target column and sequence of values.
        - If sequence: Target column with shape=(n_samples,) or
          sequence of column names or positions for multioutput tasks.
        - If dataframe: Target columns for multioutput tasks.

        This parameter is ignored if the target column is provided
        through `arrays`.

    index: bool, int, str or sequence, default=False
        Handle the index in the resulting dataframe.

        - If False: Reset to [RangeIndex][].
        - If True: Use the provided index.
        - If int: Position of the column to use as index.
        - If str: Name of the column to use as index.
        - If sequence: Array with shape=(n_samples,) to use as index.

    ignore: int, str, sequence or None, default=None
        Features in X to ignore during data transformations and model
        training. The features are still used in the remaining methods.

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

        For [multioutput tasks][], stratification applies to the joint
        target columns.

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
        Device on which to run the estimators. Use any string that
        follows the [SYCL_DEVICE_FILTER][] filter selector, e.g.
        `#!python device="gpu"` to use the GPU. Read more in the
        [user guide][gpu-acceleration].

    engine: str, dict or None, default=None
        Execution engine to use for [data][data-acceleration] and
        [estimators][estimator-acceleration]. The value should be
        one of the possible values to change one of the two engines,
        or a dictionary with keys `data` and `estimator`, with their
        corresponding choice as values to change both engines. If
        None, the default values are used. Choose from:

        - "data":

            - "pandas" (default)
            - "pyarrow"
            - "modin"

        - "estimator":

            - "sklearn" (default)
            - "sklearnex"
            - "cuml"

    backend: str, default="loky"
        Parallelization backend. Read more in the
        [user guide][parallel-execution]. Choose from:

        - "loky": Single-node, process-based parallelism.
        - "multiprocessing": Legacy single-node, process-based
          parallelism. Less robust than `loky`.
        - "threading": Single-node, thread-based parallelism.
        - "ray": Multi-node, process-based parallelism.
        - "dask": Multi-node, process-based parallelism.

    memory: bool, str, Path or Memory, default=False
        Enables caching for memory optimization. Read more in the
        [user guide][memory-considerations].

        - If False: No caching is performed.
        - If True: A default temp directory is used.
        - If str: Path to the caching directory.
        - If Path: A [pathlib.Path][] to the caching directory.
        - If Memory: Object with the [joblib.Memory][] interface.

    verbose: int, default=0
        Verbosity level of the class. Choose from:

        - 0 to not print anything.
        - 1 to print basic information.
        - 2 to print detailed information.

    warnings: bool or str, default=False
        - If True: Default warning action (equal to "once").
        - If False: Suppress all warnings (equal to "ignore").
        - If str: One of python's [warnings filters][warnings].

        Changing this parameter affects the `PYTHONWarnings` environment.
        ATOM can't manage warnings that go from C/C++ code to stdout.

    logger: str, Logger or None, default=None
        - If None: Logging isn't used.
        - If str: Name of the log file. Use "auto" for automatic name.
        - If Path: A [pathlib.Path][] to the log file.
        - Else: Python `logging.Logger` instance.

    experiment: str or None, default=None
        Name of the [mlflow experiment][experiment] to use for tracking.
        If None, no mlflow tracking is performed.

    random_state: int or None, default=None
        Seed used by the random number generator. If None, the random
        number generator is the `RandomState` used by `np.random`.

    See Also
    --------
    atom.api:ATOMForecaster
    atom.api:ATOMRegressor

    Examples
    --------
    ```pycon
    from atom import ATOMClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    # Initialize atom
    atom = ATOMClassifier(X, y, verbose=2)

    # Apply data cleaning and feature engineering methods
    atom.balance(strategy="smote")
    atom.feature_selection(strategy="rfe", solver="lr", n_features=22)

    # Train models
    atom.run(models=["LR", "RF", "XGB"])

    # Analyze the results
    print(atom.results)

    print(atom.evaluate())
    ```

    """

    _goal = Goal.classification

    def __init__(
        self,
        *arrays,
        y: YSelector = -1,
        index: IndexSelector = False,
        ignore: ColumnSelector | None = None,
        shuffle: Bool = True,
        stratify: IndexSelector = True,
        n_rows: Scalar = 1,
        test_size: Scalar = 0.2,
        holdout_size: Scalar | None = None,
        n_jobs: NJobs = 1,
        device: str = "cpu",
        engine: Engine = None,
        backend: Backend = "loky",
        memory: Bool | str | Path | Memory = False,
        verbose: Verbose = 0,
        warnings: Bool | Warnings = False,
        logger: str | Path | Logger | None = None,
        experiment: str | None = None,
        random_state: IntLargerEqualZero | None = None,
    ):
        super().__init__(
            arrays=arrays,
            y=y,
            index=index,
            ignore=ignore,
            test_size=test_size,
            holdout_size=holdout_size,
            shuffle=shuffle,
            stratify=stratify,
            n_rows=n_rows,
            n_jobs=n_jobs,
            device=device,
            engine=engine,
            backend=backend,
            memory=memory,
            verbose=verbose,
            warnings=warnings,
            logger=logger,
            experiment=experiment,
            random_state=random_state,
        )


@beartype
class ATOMForecaster(ATOM):
    """Main class for forecasting tasks.

    Apply all data transformations and model management provided by
    the package on a given dataset. Note that, contrary to sklearn's
    API, the instance contains the dataset on which to perform the
    analysis. Calling a method will automatically apply it on the
    dataset it contains.

    All [data cleaning][], [feature engineering][], [model training]
    [training] and [plotting][plots] functionality can be accessed
    from an instance of this class.

    Parameters
    ----------
    *arrays: sequence of indexables
        Dataset containing exogenous features and time series. Allowed
        formats are:

        - X
        - y
        - X, y
        - train, test
        - train, test, holdout
        - X_train, X_test, y_train, y_test
        - X_train, X_test, X_holdout, y_train, y_test, y_holdout
        - (X_train, y_train), (X_test, y_test)
        - (X_train, y_train), (X_test, y_test), (X_holdout, y_holdout)

        **X, train, test: dataframe-like**<br>
        Exogenous feature set corresponding to y, with shape=(n_samples,
        n_features).

        **y: int, str, dict, sequence or dataframe**<br>
        Time series.

        - If int: Position of the target column in `X`.
        - If str: Name of the target column in `X`.
        - If dict: Name of the target column and sequence of values.
        - If sequence: Target column with shape=(n_samples,) or
          sequence of column names or positions for multioutput tasks.
        - If dataframe: Target columns for multioutput tasks.

    y: int, str, dict, sequence or dataframe, default=-1
        Time series.

        - If None: `y` is ignored.
        - If int: Position of the target column in `X`.
        - If str: Name of the target column in `X`.
        - If dict: Name of the target column and sequence of values.
        - If sequence: Target column with shape=(n_samples,) or
          sequence of column names or positions for multioutput tasks.
        - If dataframe: Target columns for multioutput tasks.

        This parameter is ignored if the time series is provided
        through `arrays`.

    ignore: int, str, sequence or None, default=None
        Exogenous features in X to ignore during data transformations
        and model training. The features are still used in the remaining
        methods.

    sp: int, str, sequence, dict or None, default=None
        [Seasonality][] of the time series.

        - If None: No seasonality.
        - If int: Seasonal period, e.g., 7 for weekly data, and 12 for
          monthly data. The value must be >=2.
        - If str:

            - Seasonal period provided as [PeriodAlias][], e.g., "M" for
              12 or "H" for 24.
            - "index": The frequency of the data index is mapped to a
              seasonal period.
            - "infer": Automatically infer the seasonal period from the
              data (calls [get_seasonal_period][self-get_seasonal_period]
              under the hood, using default parameters).

        - If sequence: Multiple seasonal periods provided as int or str.
        - If dict: Dictionary with keys:

            - "sp": Seasonal period provided as one of the aforementioned
              options.
            - "seasonal_model" (optional): "additive" or "multiplicative".
            - "trend_model" (optional): "additive" or "multiplicative".

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

    n_rows: int or float, default=1
        Subsample of the dataset to use. The cut is made from the head
        of the dataset (older entries are dropped when sorted by date
        ascending). The default value selects all rows.

        - If <=1: Fraction of the dataset to select.
        - If >1: Exact number of rows to select. Only if `arrays` is X
                 or X, y.

    n_jobs: int, default=1
        Number of cores to use for parallel processing.

        - If >0: Number of cores to use.
        - If -1: Use all available cores.
        - If <-1: Use number of cores - 1 + `n_jobs`.

    device: str, default="cpu"
        Device on which to run the estimators. Use any string that
        follows the [SYCL_DEVICE_FILTER][] filter selector, e.g.
        `#!python device="gpu"` to use the GPU. Read more in the
        [user guide][gpu-acceleration].

    engine: str, dict or None, default=None
        Execution engine to use for [data][data-acceleration] and
        [estimators][estimator-acceleration]. The value should be
        one of the possible values to change one of the two engines,
        or a dictionary with keys `data` and `estimator`, with their
        corresponding choice as values to change both engines. If
        None, the default values are used. Choose from:

        - "data":

            - "pandas" (default)
            - "pyarrow"
            - "modin"

        - "estimator":

            - "sklearn" (default)
            - "sklearnex"
            - "cuml"

    backend: str, default="loky"
        Parallelization backend. Read more in the
        [user guide][parallel-execution]. Choose from:

        - "loky": Single-node, process-based parallelism.
        - "multiprocessing": Legacy single-node, process-based
          parallelism. Less robust than `loky`.
        - "threading": Single-node, thread-based parallelism.
        - "ray": Multi-node, process-based parallelism.
        - "dask": Multi-node, process-based parallelism.

    memory: bool, str, Path or Memory, default=False
        Enables caching for memory optimization. Read more in the
        [user guide][memory-considerations].

        - If False: No caching is performed.
        - If True: A default temp directory is used.
        - If str: Path to the caching directory.
        - If Path: A [pathlib.Path][] to the caching directory.
        - If Memory: Object with the [joblib.Memory][] interface.

    verbose: int, default=0
        Verbosity level of the class. Choose from:

        - 0 to not print anything.
        - 1 to print basic information.
        - 2 to print detailed information.

    warnings: bool or str, default=False
        - If True: Default warning action (equal to "once").
        - If False: Suppress all warnings (equal to "ignore").
        - If str: One of python's [warnings filters][warnings].

        Changing this parameter affects the `PYTHONWarnings` environment.
        ATOM can't manage warnings that go from C/C++ code to stdout.

    logger: str, Logger or None, default=None
        - If None: Logging isn't used.
        - If str: Name of the log file. Use "auto" for automatic name.
        - If Path: A [pathlib.Path][] to the log file.
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
    atom.api:ATOMRegressor

    Examples
    --------
    ```pycon
    from atom import ATOMForecaster
    from sktime.datasets import load_airline

    y = load_airline()

    # Initialize atom
    atom = ATOMForecaster(y, verbose=2)

    # Train models
    atom.run(models=["NF", "ES", "ETS"])

    # Analyze the results
    print(atom.results)

    print(atom.evaluate())
    ```

    """

    _goal = Goal.forecast

    def __init__(
        self,
        *arrays,
        y: YSelector = -1,
        ignore: ColumnSelector | None = None,
        sp: Seasonality | SPDict = None,
        n_rows: Scalar = 1,
        test_size: Scalar = 0.2,
        holdout_size: Scalar | None = None,
        n_jobs: NJobs = 1,
        device: str = "cpu",
        engine: Engine = None,
        backend: Backend = "loky",
        memory: Bool | str | Path | Memory = False,
        verbose: Verbose = 0,
        warnings: Bool | Warnings = False,
        logger: str | Path | Logger | None = None,
        experiment: str | None = None,
        random_state: IntLargerEqualZero | None = None,
    ):
        super().__init__(
            arrays=arrays,
            y=y,
            index=True,
            ignore=ignore,
            sp=sp,
            test_size=test_size,
            holdout_size=holdout_size,
            shuffle=False,
            stratify=False,
            n_rows=n_rows,
            n_jobs=n_jobs,
            device=device,
            engine=engine,
            backend=backend,
            memory=memory,
            verbose=verbose,
            warnings=warnings,
            logger=logger,
            experiment=experiment,
            random_state=random_state,
        )


@beartype
class ATOMRegressor(ATOM):
    """Main class for regression tasks.

    Apply all data transformations and model management provided by
    the package on a given dataset. Note that, contrary to sklearn's
    API, the instance contains the dataset on which to perform the
    analysis. Calling a method will automatically apply it on the
    dataset it contains.

    All [data cleaning][], [feature engineering][], [model training]
    [training] and [plotting][plots] functionality can be accessed
    from an instance of this class.

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

        **y: int, str, dict, sequence or dataframe**<br>
        Target column(s) corresponding to `X`.

        - If int: Position of the target column in `X`.
        - If str: Name of the target column in `X`.
        - If dict: Name of the target column and sequence of values.
        - If sequence: Target column with shape=(n_samples,) or
          sequence of column names or positions for multioutput tasks.
        - If dataframe: Target columns for multioutput tasks.

    y: int, str, dict, sequence or dataframe, default=-1
        Target column(s) corresponding to `X`.

        - If None: `y` is ignored.
        - If int: Position of the target column in `X`.
        - If str: Name of the target column in `X`.
        - If dict: Name of the target column and sequence of values.
        - If sequence: Target column with shape=(n_samples,) or
          sequence of column names or positions for multioutput tasks.
        - If dataframe: Target columns for multioutput tasks.

        This parameter is ignored if the target column is provided
        through `arrays`.

    index: bool, int, str or sequence, default=False
        Handle the index in the resulting dataframe.

        - If False: Reset to [RangeIndex][].
        - If True: Use the provided index.
        - If int: Position of the column to use as index.
        - If str: Name of the column to use as index.
        - If sequence: Array with shape=(n_samples,) to use as index.

    ignore: int, str, sequence or None, default=None
        Features in X to ignore during data transformations and model
        training. The features are still used in the remaining methods.

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
        Device on which to run the estimators. Use any string that
        follows the [SYCL_DEVICE_FILTER][] filter selector, e.g.
        `#!python device="gpu"` to use the GPU. Read more in the
        [user guide][gpu-acceleration].

    engine: str, dict or None, default=None
        Execution engine to use for [data][data-acceleration] and
        [estimators][estimator-acceleration]. The value should be
        one of the possible values to change one of the two engines,
        or a dictionary with keys `data` and `estimator`, with their
        corresponding choice as values to change both engines. If
        None, the default values are used. Choose from:

        - "data":

            - "numpy"
            - "pandas" (default)
            - "pandas-pyarrow"
            - "polars"
            - "polars-lazy"
            - "pyarrow"
            - "modin"
            - "dask"
            - "pyspark"
            - "pyspark-pandas"

        - "estimator":

            - "sklearn" (default)
            - "sklearnex"
            - "cuml"

    backend: str, default="loky"
        Parallelization backend. Read more in the
        [user guide][parallel-execution]. Choose from:

        - "loky": Single-node, process-based parallelism.
        - "multiprocessing": Legacy single-node, process-based
          parallelism. Less robust than `loky`.
        - "threading": Single-node, thread-based parallelism.
        - "ray": Multi-node, process-based parallelism.
        - "dask": Multi-node, process-based parallelism.

    memory: bool, str, Path or Memory, default=False
        Enables caching for memory optimization. Read more in the
        [user guide][memory-considerations].

        - If False: No caching is performed.
        - If True: A default temp directory is used.
        - If str: Path to the caching directory.
        - If Path: A [pathlib.Path][] to the caching directory.
        - If Memory: Object with the [joblib.Memory][] interface.

    verbose: int, default=0
        Verbosity level of the class. Choose from:

        - 0 to not print anything.
        - 1 to print basic information.
        - 2 to print detailed information.

    warnings: bool or str, default=False
        - If True: Default warning action (equal to "once").
        - If False: Suppress all warnings (equal to "ignore").
        - If str: One of python's [warnings filters][warnings].

        Changing this parameter affects the `PYTHONWarnings` environment.
        ATOM can't manage warnings that go from C/C++ code to stdout.

    logger: str, Logger or None, default=None
        - If None: Logging isn't used.
        - If str: Name of the log file. Use "auto" for automatic name.
        - If Path: A [pathlib.Path][] to the log file.
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
    atom.api:ATOMForecaster

    Examples
    --------
    ```pycon
    from atom import ATOMRegressor
    from sklearn.datasets import load_diabetes

    X, y = load_diabetes(return_X_y=True, as_frame=True)

    # Initialize atom
    atom = ATOMRegressor(X, y, verbose=2)

    # Apply data cleaning and feature engineering methods
    atom.scale()
    atom.feature_selection(strategy="rfecv", solver="xgb", n_features=12)

    # Train models
    atom.run(models=["OLS", "RF", "XGB"])

    # Analyze the results
    print(atom.results)

    print(atom.evaluate())
    ```

    """

    _goal = Goal.regression

    def __init__(
        self,
        *arrays,
        y: YSelector = -1,
        index: IndexSelector = False,
        ignore: ColumnSelector | None = None,
        shuffle: Bool = True,
        n_rows: Scalar = 1,
        test_size: Scalar = 0.2,
        holdout_size: Scalar | None = None,
        n_jobs: NJobs = 1,
        device: str = "cpu",
        engine: Engine = None,
        backend: Backend = "loky",
        memory: Bool | str | Path | Memory = False,
        verbose: Verbose = 0,
        warnings: Bool | Warnings = False,
        logger: str | Path | Logger | None = None,
        experiment: str | None = None,
        random_state: IntLargerEqualZero | None = None,
    ):
        super().__init__(
            arrays=arrays,
            y=y,
            index=index,
            ignore=ignore,
            test_size=test_size,
            holdout_size=holdout_size,
            shuffle=shuffle,
            stratify=False,
            n_rows=n_rows,
            n_jobs=n_jobs,
            device=device,
            engine=engine,
            backend=backend,
            memory=memory,
            verbose=verbose,
            warnings=warnings,
            logger=logger,
            experiment=experiment,
            random_state=random_state,
        )
