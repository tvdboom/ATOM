# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modeling (ATOM)
Author: Mavs
Description: Module containing the API classes.

"""

from __future__ import annotations

from logging import Logger
from pathlib import Path

from beartype import beartype
from joblib.memory import Memory
from sklearn.base import clone

from atom.atom import ATOM
from atom.basetransformer import BaseTransformer
from atom.utils.types import (
    Backend, Bool, Engine, IndexSelector, Int, Predictor, Scalar, Target,
    Verbose, Warnings,
)


@beartype
def ATOMModel(
    estimator: Predictor,
    name: str | None = None,
    *,
    acronym: str | None = None,
    needs_scaling: Bool = False,
    native_multilabel: Bool = False,
    native_multioutput: Bool = False,
    has_validation: str | None = None,
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
        If False and the task is multiouput, a multiotuput meta-estimator
        is wrapper around the estimator.

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
    if not callable(estimator):
        estimator = clone(estimator)

    if name:
        estimator.name = name
    if acronym:
        estimator.acronym = acronym
    estimator.needs_scaling = needs_scaling
    estimator.native_multioutput = native_multioutput
    estimator.native_multilabel = native_multilabel
    estimator.has_validation = has_validation

    return estimator


@beartype
class ATOMClassifier(BaseTransformer, ATOM):
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

        **y: int, str or sequence**<br>
        Target column corresponding to X.

        - If int: Position of the target column in X.
        - If str: Name of the target column in X.
        - If sequence: Target column with shape=(n_samples,) or
          sequence of column names or positions for multioutput tasks.
        - If dataframe: Target columns for multioutput tasks.

    y: int, str, dict, sequence or dataframe, default=-1
        Target column corresponding to X.

        - If int: Position of the target column in X.
        - If str: Name of the target column in X.
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

        For [multioutput tasks][], stratification is applied to the
        joint target columns.

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

    engine: dict, default={"data": "numpy", "estimator": "sklearn"}
        Execution engine to use for [data][data-acceleration] and
        [estimators][estimator-acceleration]. The value should be a
        dictionary with keys `data` and/or `estimator`, with their
        corresponding choice as values. Choose from:

        - "data":

            - "numpy"
            - "pyarrow"
            - "modin"

        - "estimator":

            - "sklearn"
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
        - If True: Default warning action (equal to "default").
        - If False: Suppress all warnings (equal to "ignore").
        - If str: One of python's [warnings filters][warnings].

        Changing this parameter affects the `PYTHONWarnings` environment.
        ATOM can't manage warnings that go from C/C++ code to stdout.

    logger: str, Logger or None, default=None
        - If None: Logging isn't used.
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

    def __init__(
        self,
        *arrays,
        y: Target = -1,
        index: IndexSelector = False,
        shuffle: Bool = True,
        stratify: IndexSelector = True,
        n_rows: Scalar = 1,
        test_size: Scalar = 0.2,
        holdout_size: Scalar | None = None,
        n_jobs: Int = 1,
        device: str = "cpu",
        engine: Engine = {"data": "numpy", "estimator": "sklearn"},
        backend: Backend = "loky",
        memory: Bool | str | Path | Memory = False,
        verbose: Verbose = 0,
        warnings: Bool | Warnings = False,
        logger: str | Logger | None = None,
        experiment: str | None = None,
        random_state: Int | None = None,
    ):
        super().__init__(
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


class ATOMForecaster(BaseTransformer, ATOM):
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
        Dataset containing exogeneous features and time series. Allowed
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
        Exogeneous feature set corresponding to y, with shape=(n_samples,
        n_features).

        **y: int, str or sequence**<br>
        Time series.

        - If int: Position of the target column in X.
        - If str: Name of the target column in X.
        - If sequence: Target column with shape=(n_samples,) or
          sequence of column names or positions for multioutput tasks.
        - If dataframe: Target columns for multioutput tasks.

    y: int, str, dict, sequence or dataframe, default=-1
        Time series.

        - If None: y is ignored.
        - If int: Position of the target column in X.
        - If str: Name of the target column in X.
        - If sequence: Target column with shape=(n_samples,) or
          sequence of column names or positions for multioutput tasks.
        - If dataframe: Target columns for multioutput tasks.

        This parameter is ignored if the time series is provided
        through `arrays`.

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

    engine: dict, default={"data": "numpy", "estimator": "sklearn"}
        Execution engine to use for [data][data-acceleration] and
        [estimators][estimator-acceleration]. The value should be a
        dictionary with keys `data` and/or `estimator`, with their
        corresponding choice as values. Choose from:

        - "data":

            - "numpy"
            - "pyarrow"
            - "modin"

        - "estimator":

            - "sklearn"
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
        - If True: Default warning action (equal to "default").
        - If False: Suppress all warnings (equal to "ignore").
        - If str: One of python's [warnings filters][warnings].

        Changing this parameter affects the `PYTHONWarnings` environment.
        ATOM can't manage warnings that go from C/C++ code to stdout.

    logger: str, Logger or None, default=None
        - If None: Logging isn't used.
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

    @beartype
    def __init__(
        self,
        *arrays,
        y: Target = -1,
        n_rows: Scalar = 1,
        test_size: Scalar = 0.2,
        holdout_size: Scalar | None = None,
        n_jobs: Int = 1,
        device: str = "cpu",
        engine: Engine = {"data": "numpy", "estimator": "sklearn"},
        backend: Backend = "loky",
        memory: Bool | str | Path | Memory = False,
        verbose: Verbose = 0,
        warnings: Bool | Warnings = False,
        logger: str | Logger | None = None,
        experiment: str | None = None,
        random_state: Int | None = None,
    ):
        super().__init__(
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

        self.goal = "fc"
        ATOM.__init__(
            self,
            arrays=arrays,
            y=y,
            index=True,
            test_size=test_size,
            holdout_size=holdout_size,
            shuffle=False,
            stratify=False,
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

        **y: int, str or sequence**<br>
        Target column corresponding to X.

        - If int: Position of the target column in X.
        - If str: Name of the target column in X.
        - If sequence: Target column with shape=(n_samples,) or
          sequence of column names or positions for multioutput tasks.
        - If dataframe: Target columns for multioutput tasks.

    y: int, str, dict, sequence or dataframe, default=-1
        Target column corresponding to X.

        - If None: y is ignored.
        - If int: Position of the target column in X.
        - If str: Name of the target column in X.
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

    engine: dict, default={"data": "numpy", "estimator": "sklearn"}
        Execution engine to use for [data][data-acceleration] and
        [estimators][estimator-acceleration]. The value should be a
        dictionary with keys `data` and/or `estimator`, with their
        corresponding choice as values. Choose from:

        - "data":

            - "numpy"
            - "pyarrow"
            - "modin"

        - "estimator":

            - "sklearn"
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
        - If True: Default warning action (equal to "default").
        - If False: Suppress all warnings (equal to "ignore").
        - If str: One of python's [warnings filters][warnings].

        Changing this parameter affects the `PYTHONWarnings` environment.
        ATOM can't manage warnings that go from C/C++ code to stdout.

    logger: str, Logger or None, default=None
        - If None: Logging isn't used.
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

    @beartype
    def __init__(
        self,
        *arrays,
        y: Target = -1,
        index: IndexSelector = False,
        shuffle: Bool = True,
        n_rows: Scalar = 1,
        test_size: Scalar = 0.2,
        holdout_size: Scalar | None = None,
        n_jobs: Int = 1,
        device: str = "cpu",
        engine: Engine = {"data": "numpy", "estimator": "sklearn"},
        backend: Backend = "loky",
        memory: Bool | str | Path | Memory = False,
        verbose: Verbose = 0,
        warnings: Bool | Warnings = False,
        logger: str | Logger | None = None,
        experiment: str | None = None,
        random_state: Int | None = None,
    ):
        super().__init__(
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
