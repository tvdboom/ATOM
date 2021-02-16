# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing the API classes.

"""

# Standard packages
import pickle
from copy import copy
from typeguard import typechecked
from typing import Optional, Union

# Own modules
from .atom import ATOM
from .basetransformer import BaseTransformer
from .utils import SEQUENCE_TYPES, custom_transform


# Functions ======================================================== >>

@typechecked
def ATOMModel(
    estimator,
    acronym: str = None,
    fullname: str = None,
    needs_scaling: bool = False,
):
    """Convert an estimator to a model that can be ingested by ATOM.

    This function adds the relevant attributes to the estimator so
    that they can be used when initializing the CustomModel class.

    Parameters
    ----------
    estimator: class
        Model's estimator. Can be a class or an instance.

    acronym: str, optional (default=None)
        Model's acronym. Used to call the model from the trainer. If
        None and there are 2 or more capital letters in the estimator's
        __name__, those are used. Else, the full __name__ is used.

    fullname: str, optional (default=None)
        Full model's name. If None, the estimator's __name__ is used.

    needs_scaling: bool, optional (default=False)
        Whether the model needs scaled features. Can not be True for
        deep learning datasets.

    """
    if acronym:
        estimator.acronym = acronym
    if fullname:
        estimator.fullname = fullname
    estimator.needs_scaling = needs_scaling

    return estimator


@typechecked
def ATOMLoader(
    filename: str,
    data: Optional[SEQUENCE_TYPES] = None,
    transform_data: bool = True,
    verbose: Optional[int] = None,
):
    """Load a class instance from a pickle file.

    If the file is a trainer that was saved using `save_data=False`,
    you can load new data into it. For atom pickles, you can also
    apply all data transformations in the pipeline to the data.

    Parameters
    ----------
    filename: str
        Name of the pickle file to load.

    data: tuple of indexables or None, optional (default=None)
        Tuple containing the features and target. Only use this
        parameter if the file is a trainer that was saved using
        `save_data=False`. Allowed formats are:
            - X, y
            - train, test
            - X_train, X_test, y_train, y_test
            - (X_train, y_train), (X_test, y_test)

        X, train, test: dict, list, tuple, np.ndarray or pd.DataFrame
            Feature set with shape=(n_features, n_samples). If no
            y is provided, the last column is used as target.

        y: int, str or sequence
            - If int: Index of the target column in X.
            - If str: Name of the target column in X.
            - Else: Target column with shape=(n_samples,).

    transform_data: bool, optional (default=True)
        If False, the `data` is left as provided. If True, it is
        transformed through all the steps in the instance's
        pipeline. This parameter is ignored if the loaded file
        is not an atom pickle.

    verbose: int or None, optional (default=None)
        Verbosity level of the transformations applied on the new
        data. If None, use the verbosity from the loaded instance.
        This parameter is ignored if `transform_data=False`.

    """
    with open(filename, "rb") as f:
        cls = pickle.load(f)

    if data is not None:
        if not hasattr(cls, "_branches"):
            raise TypeError(
                "Data is provided but the class is not a "
                f"trainer, got {cls.__class__.__name__}."
            )
        elif cls.branch.data is not None:
            raise ValueError(
                f"The loaded {cls.__class__.__name__} instance already contains data!"
            )

        # Prepare the provided data
        data, idx = cls._get_data_and_idx(data, use_n_rows=transform_data)

        # Apply transformations per branch
        step = {}  # Current step in the pipeline per branch
        for b1, v1 in cls._branches.items():
            branch = cls._branches[b1]

            # Provide the input data if not already filled from another branch
            if branch.data is None:
                branch.data, branch.idx = data, idx

            if transform_data:
                if not v1.pipeline.empty:
                    cls.log(f"Transforming data for branch {b1}...", 1)

                for i, est1 in enumerate(v1.pipeline):
                    # Skip if the transformation was already applied
                    if step.get(b1, -1) < i:
                        custom_transform(est1, branch, verbose=verbose)

                        for b2, v2 in cls._branches.items():
                            try:  # Can fail if pipeline is shorter than i
                                if b1 != b2 and est1 is v2.pipeline.iloc[i]:
                                    # Update the data and step for the other branch
                                    cls._branches[b2].data = copy(branch.data)
                                    cls._branches[b2].idx = copy(branch.idx)
                                    step[b2] = i
                            except IndexError:
                                continue

    cls.log(f"{cls.__class__.__name__} loaded successfully!", 1)

    return cls


# Classes ========================================================== >>

class ATOMClassifier(BaseTransformer, ATOM):
    """ATOM class for classification tasks.

    Parameters
    ----------
    *arrays: sequence of indexables
        Dataset containing features and target. Allowed formats are:
            - X, y
            - train, test
            - X_train, X_test, y_train, y_test
            - (X_train, y_train), (X_test, y_test)

        X, train, test: dict, list, tuple, np.ndarray or pd.DataFrame
            Feature set with shape=(n_features, n_samples). If
            no y is provided, the last column is used as target.

        y: int, str or sequence
            - If int: Index of the target column in X.
            - If str: Name of the target column in X.
            - Else: Target column with shape=(n_samples,).

    n_rows: int or float, optional (default=1)
        - If <=1: Fraction of the dataset to use.
        - If >1: Number of rows to use (only if input is X, y).

    test_size: int, float, optional (default=0.2)
        - If <=1: Fraction of the dataset to include in the test set.
        - If >1: Number of rows to include in the test set.

        Is ignored if the train and test set are provided.

    n_jobs: int, optional (default=1)
        Number of cores to use for parallel processing.
            - If >0: Number of cores to use.
            - If -1: Use all available cores.
            - If <-1: Use number of cores - 1 - n_jobs.

        Beware that using multiple processes on the same machine may
        cause memory issues for large datasets.

    verbose: int, optional (default=0)
        Verbosity level of the class. Possible values are:
            - 0 to not print anything.
            - 1 to print basic information.
            - 2 to print detailed information.

    warnings: bool or str, optional (default=True)
        - If True: Default warning action (equal to "default").
        - If False: Suppress all warnings (equal to "ignore").
        - If str: One of the actions in python's warnings environment.

        Note that changing this parameter will affect the
        `PYTHONWARNINGS` environment.

        Note that ATOM can't manage warnings that go directly
        from C/C++ code to the stdout/stderr.

    logger: str, class or None, optional (default=None)
        - If None: Doesn't save a logging file.
        - If str: Name of the logging file. Use "auto" for default name.
        - If class: Python `Logger` object.

        The default name consists of the class' name followed by
        the timestamp of the logger's creation.

        Note that warnings will not be saved to the logger.

    random_state: int or None, optional (default=None)
        Seed used by the random number generator. If None, the random
        number generator is the `RandomState` used by `numpy.random`.

    """

    @typechecked
    def __init__(
        self,
        *arrays,
        n_rows: Union[int, float] = 1,
        test_size: float = 0.2,
        n_jobs: int = 1,
        verbose: int = 0,
        warnings: Union[bool, str] = True,
        logger: Optional[Union[str, callable]] = None,
        random_state: Optional[int] = None,
    ):
        super().__init__(
            n_jobs=n_jobs,
            verbose=verbose,
            warnings=warnings,
            logger=logger,
            random_state=random_state,
        )

        self.goal = "classification"
        ATOM.__init__(self, arrays, n_rows=n_rows, test_size=test_size)


class ATOMRegressor(BaseTransformer, ATOM):
    """ATOM class for regression tasks.

    Parameters
    ----------
    *arrays: sequence of indexables
        Dataset containing features and target. Allowed formats are:
            - X, y
            - train, test
            - X_train, X_test, y_train, y_test
            - (X_train, y_train), (X_test, y_test)

        X, train, test: dict, list, tuple, np.ndarray or pd.DataFrame
            Feature set with shape=(n_features, n_samples). If
            no y is provided, the last column is used as target.

        y: int, str or sequence
            - If int: Index of the target column in X.
            - If str: Name of the target column in X.
            - Else: Target column with shape=(n_samples,).

    n_rows: int or float, optional (default=1)
        - If <=1: Fraction of the dataset to use.
        - If >1: Number of rows to use (only if input is X, y).

    test_size: int, float, optional (default=0.2)
        - If <=1: Fraction of the dataset to include in the test set.
        - If >1: Number of rows to include in the test set.

        Is ignored if the train and test set are provided.

    n_jobs: int, optional (default=1)
        Number of cores to use for parallel processing.
            - If >0: Number of cores to use.
            - If -1: Use all available cores.
            - If <-1: Use number of cores - 1 - n_jobs.

        Beware that using multiple processes on the same machine may
        cause memory issues for large datasets.

    verbose: int, optional (default=0)
        Verbosity level of the class. Possible values are:
            - 0 to not print anything.
            - 1 to print basic information.
            - 2 to print detailed information.

    warnings: bool or str, optional (default=True)
        - If True: Default warning action (equal to "default").
        - If False: Suppress all warnings (equal to "ignore").
        - If str: One of the actions in python's warnings environment.

        Note that changing this parameter will affect the
        `PYTHONWARNINGS` environment.

        Note that ATOM can't manage warnings that go directly
        from C/C++ code to the stdout/stderr.

    logger: str, class or None, optional (default=None)
        - If None: Doesn't save a logging file.
        - If str: Name of the logging file. Use "auto" for default name.
        - If class: Python `Logger` object.

        The default name consists of the class' name followed by
        the timestamp of the logger's creation.

        Note that warnings will not be saved to the logger.

    random_state: int or None, optional (default=None)
        Seed used by the random number generator. If None, the random
        number generator is the `RandomState` used by `numpy.random`.

    """

    @typechecked
    def __init__(
        self,
        *arrays,
        n_rows: Union[int, float] = 1,
        test_size: float = 0.2,
        n_jobs: int = 1,
        verbose: int = 0,
        warnings: Union[bool, str] = True,
        logger: Optional[Union[str, callable]] = None,
        random_state: Optional[int] = None,
    ):
        super().__init__(
            n_jobs=n_jobs,
            verbose=verbose,
            warnings=warnings,
            logger=logger,
            random_state=random_state,
        )

        self.goal = "regression"
        ATOM.__init__(self, arrays, n_rows=n_rows, test_size=test_size)
