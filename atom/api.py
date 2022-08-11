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

        X, train, test: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str or sequence
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
    ... from sklearn.datasets import load_breast_cancer

    >>> atom = ATOMClassifier(X, y)
    ... atom.scale()
    ... atom.run(["LR", "RF", "SGD"], metric="AP")
    ... atom.save("atom", save_data=False)  # Save atom to a pickle file

    # Load the class and add the data to the new instance
    >>> atom_2 = ATOMLoader("atom", data=(X, y), verbose=2)

    Transforming data for branch master:
    Scaling features...
    ATOMClassifier successfully loaded.

    >>> print(atom_2.results)

         metric_train  metric_test time_fit    time
    LR       0.998179     0.998570   0.016s  0.016s
    RF       1.000000     0.995568   0.141s  0.141s
    SGD      0.998773     0.994313   0.016s  0.016s

    ```

    """
    with open(filename, "rb") as f:
        cls = pickle.load(f)

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
                if not v1.pipeline.empty:
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
    fullname: Optional[str] = None,
    needs_scaling: bool = False,
) -> Predictor:
    """Convert an estimator to a model that can be ingested by atom.

    This function adds the relevant attributes to the estimator so
    that they can be used by atom. Note that only estimators that follow
    [sklearn's API](https://scikit-learn.org/stable/developers/develop.html)
    are compatible.

    Read more about using custom models in the [user guide][custom-models].

    Parameters
    ----------
    estimator: Predictor
        Custom estimator. Should implement a `fit` and `predict` method.

    acronym: str or None, default=None
        Model's acronym. Used to call the model from atom. If None, the
        capital letters in the estimator's \__name__ are used (only if
        two or more, else it uses the entire \__name__).

    fullname: str or None, default=None
        Full model's name. If None, the estimator's name is used.

    needs_scaling: bool, default=False
        Whether the model needs scaled features.

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
    ...      fullname="Random Sample Consensus",
    ...      needs_scaling=False,
    ...  )

    >>> atom = ATOMRegressor(X, y)
    >>> atom.run(ransac, verbose=2)

    Training ========================= >>
    Models: RANSAC
    Metric: r2

    Results for Random Sample Consensus:
    Fit ---------------------------------------------
    Train evaluation --> r2: -2.1784
    Test evaluation --> r2: -9.4592
    Time elapsed: 0.072s
    -------------------------------------------------
    Total time: 0.072s

    Final results ==================== >>
    Duration: 0.072s
    -------------------------------------
    Random Sample Consensus --> r2: -9.4592 ~

    ```

    """
    estimator = clone(estimator)

    if acronym:
        estimator.acronym = acronym
    if fullname:
        estimator.fullname = fullname
    estimator.needs_scaling = needs_scaling

    return estimator


# Classes ========================================================== >>

class ATOMClassifier(BaseTransformer, ATOM):
    """ATOM class for classification tasks.

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

        X, train, test: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str or sequence
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
            - If False: Reset to RangeIndex.
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
        set is provided through `arrays.

    n_rows: int or float, default=1
        Random subsample of the provided dataset to use. The default
        value selects all the rows.
        - If <=1: Select this fraction of the dataset.
        - If >1: Select this exact number of rows. Only if the input
                 doesn't already specify the data sets (i.e. X or X, y).

    n_jobs: int, default=1
        Number of cores to use for parallel processing.
            - If >0: Number of cores to use.
            - If -1: Use all available cores.
            - If <-1: Use number of cores - 1 + `n_jobs`.

    gpu: bool or str, default=False
        Train estimators on GPU (instead of CPU). Refer to the
        documentation to check which estimators are supported.
            - If False: Always use CPU implementation.
            - If True: Use GPU implementation if possible.
            - If "force": Force GPU implementation.

    verbose: int, default=0
        Verbosity level of the class. Choose from:
            - 0 to not print anything.
            - 1 to print basic information.
            - 2 to print detailed information.

    warnings: bool or str, default=False
        - If True: Default warning action (equal to "default").
        - If False: Suppress all warnings (equal to "ignore").
        - If str: One of the actions in python's warnings environment.

        Changing this parameter affects the `PYTHONWARNINGS` environment.
        ATOM can't manage warnings that go from C/C++ code to stdout.

    logger: str, Logger or None, default=None
        - If None: Doesn't save a logging file.
        - If str: Name of the log file. Use "auto" for automatic name.
        - Else: Python `logging.Logger` instance.

    experiment: str or None, default=None
        Name of the mlflow experiment to use for tracking. If None,
        no mlflow tracking is performed.

    random_state: int or None, default=None
        Seed used by the random number generator. If None, the random
        number generator is the `RandomState` used by `np.random`.

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
        gpu: Union[bool, str] = False,
        verbose: INT = 0,
        warnings: Union[bool, str] = False,
        logger: Optional[Union[str, Logger]] = None,
        experiment: Optional[str] = None,
        random_state: Optional[INT] = None,
    ):
        super().__init__(
            n_jobs=n_jobs,
            gpu=gpu,
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
    """ATOM class for regression tasks.

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

        X, train, test: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str or sequence
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
            - If False: Reset to RangeIndex.
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
        an unequal distribution of the target classes over the sets.

    n_rows: int or float, default=1
        Random subsample of the provided dataset to use. The default
        value selects all the rows.
        - If <=1: Select this fraction of the dataset.
        - If >1: Select this exact number of rows. Only if the input
                 doesn't already specify the data sets (i.e. X or X, y).

    n_jobs: int, default=1
        Number of cores to use for parallel processing.
            - If >0: Number of cores to use.
            - If -1: Use all available cores.
            - If <-1: Use number of cores - 1 + `n_jobs`.

    gpu: bool or str, default=False
        Train estimators on GPU (instead of CPU). Refer to the
        documentation to check which estimators are supported.
            - If False: Always use CPU implementation.
            - If True: Use GPU implementation if possible.
            - If "force": Force GPU implementation.

    verbose: int, default=0
        Verbosity level of the class. Choose from:
            - 0 to not print anything.
            - 1 to print basic information.
            - 2 to print detailed information.

    warnings: bool or str, default=False
        - If True: Default warning action (equal to "default").
        - If False: Suppress all warnings (equal to "ignore").
        - If str: One of the actions in python's warnings environment.

        Changing this parameter affects the `PYTHONWARNINGS` environment.
        ATOM can't manage warnings that go from C/C++ code to stdout.

    logger: str, Logger or None, default=None
        - If None: Doesn't save a logging file.
        - If str: Name of the log file. Use "auto" for automatic name.
        - Else: Python `logging.Logger` instance.

    experiment: str or None, default=None
        Name of the mlflow experiment to use for tracking. If None,
        no mlflow tracking is performed.

    random_state: int or None, default=None
        Seed used by the random number generator. If None, the random
        number generator is the `RandomState` used by `np.random`.

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
        gpu: Union[bool, str] = False,
        verbose: INT = 0,
        warnings: Union[bool, str] = False,
        logger: Optional[Union[str, Logger]] = None,
        experiment: Optional[str] = None,
        random_state: Optional[INT] = None,
    ):
        super().__init__(
            n_jobs=n_jobs,
            gpu=gpu,
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
