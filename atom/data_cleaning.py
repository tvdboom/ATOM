"""Automated Tool for Optimized Modeling (ATOM).

Author: Mavs
Description: Module containing the data cleaning transformers.

"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Hashable
from typing import Any, Literal, TypeVar, overload

import numpy as np
import pandas as pd
import sklearn
from beartype import beartype
from category_encoders import (
    BackwardDifferenceEncoder, BaseNEncoder, BinaryEncoder, CatBoostEncoder,
    HelmertEncoder, JamesSteinEncoder, MEstimateEncoder, OneHotEncoder,
    OrdinalEncoder, PolynomialEncoder, SumEncoder, TargetEncoder, WOEEncoder,
)
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import (
    ADASYN, SMOTE, SMOTEN, SMOTENC, SVMSMOTE, BorderlineSMOTE, KMeansSMOTE,
    RandomOverSampler,
)
from imblearn.under_sampling import (
    AllKNN, CondensedNearestNeighbour, EditedNearestNeighbours,
    InstanceHardnessThreshold, NearMiss, NeighbourhoodCleaningRule,
    OneSidedSelection, RandomUnderSampler, RepeatedEditedNearestNeighbours,
    TomekLinks,
)
from scipy.stats import zscore
from sklearn.base import (
    BaseEstimator, OneToOneFeatureMixin, _clone_parametrized,
)
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.utils.validation import _check_feature_names_in
from sktime.transformations.series.detrend import (
    ConditionalDeseasonalizer, Deseasonalizer, Detrender,
)
from sktime.transformations.series.impute import Imputer as SktimeImputer
from typing_extensions import Self

from atom.basetransformer import BaseTransformer
from atom.utils.constants import CAT_TYPES, DEFAULT_MISSING
from atom.utils.types import (
    Bins, Bool, CategoricalStrats, DiscretizerStrats, Engine,
    EngineDataOptions, EngineTuple, Estimator, FloatLargerZero, Int,
    IntLargerEqualZero, IntLargerTwo, IntLargerZero, NJobs, NormalizerStrats,
    NumericalStrats, Predictor, PrunerStrats, Scalar, ScalerStrats,
    SeasonalityModels, Sequence, Transformer, Verbose, XConstructor, XReturn,
    YConstructor, YReturn, sequence_t,
)
from atom.utils.utils import (
    Goal, check_is_fitted, get_col_names, get_col_order, get_cols, it, lst,
    make_sklearn, merge, n_cols, replace_missing, sign, to_df, to_series,
    to_tabular, variable_return,
)


T_Transformer = TypeVar("T_Transformer", bound=Transformer)


@beartype
class TransformerMixin(BaseEstimator, BaseTransformer):
    """Mixin class for all transformers in ATOM.

    Different from sklearn in the following ways:

    - Accounts for the transformation of y.
    - Always add a fit method.
    - Wraps the fit method with attributes and a data check.
    - Wraps transforming methods a data check.
    - Maintains internal attributes when cloned.

    """

    def __repr__(self, N_CHAR_MAX: Int = 700) -> str:
        """Drop named tuples if default parameters from string representation."""
        out = super().__repr__(N_CHAR_MAX)

        # Remove default engine for cleaner representation
        if hasattr(self, "engine") and sklearn.get_config()["print_changed_only"]:
            if self.engine.data == EngineTuple().data:
                out = re.sub(f"'data': '{self.engine.data}'", "", out)
            if self.engine.estimator == EngineTuple().estimator:
                out = re.sub(f", 'estimator': '{self.engine.estimator}'", "", out)
            out = re.sub("engine={}", "", out)
            out = re.sub(r"((?<=[{(]),\s|,\s(?=[})])|,\s(?=,\s))", "", out)  # Drop comma-spaces

        return out

    def __sklearn_clone__(self: T_Transformer) -> T_Transformer:
        """Wrap cloning method to attach internal attributes."""
        cloned = _clone_parametrized(self)

        for attr in ("_cols", "_train_only"):
            if hasattr(self, attr):
                setattr(cloned, attr, getattr(self, attr))

        return cloned

    def fit(
        self,
        X: XConstructor | None = None,
        y: YConstructor | None = None,
        **fit_params,
    ) -> Self:
        """Do nothing.

        Implemented for continuity of the API.

        Parameters
        ----------
        X: dataframe-like or None, default=None
            Feature set with shape=(n_samples, n_features). If None,
            `X` is ignored.

        y: sequence, dataframe-like or None, default=None
            Target column(s) corresponding to `X`. If None, `y` is
            ignored.

        **fit_params
            Additional keyword arguments for the fit method.

        Returns
        -------
        self
            Estimator instance.

        """
        Xt = to_df(X)

        self._check_feature_names(Xt, reset=True)
        self._check_n_features(Xt, reset=True)

        self._log(f"Fitting {self.__class__.__name__}...", 1)

        return self

    @overload
    def fit_transform(
        self,
        X: Literal[None],
        y: YConstructor,
        **fit_params,
    ) -> YReturn: ...

    @overload
    def fit_transform(
        self,
        X: XConstructor,
        y: Literal[None] = ...,
        **fit_params,
    ) -> XReturn: ...

    @overload
    def fit_transform(
        self,
        X: XConstructor,
        y: YConstructor,
        **fit_params,
    ) -> tuple[XReturn, YReturn]: ...

    def fit_transform(
        self,
        X: XConstructor | None = None,
        y: YConstructor | None = None,
        **fit_params,
    ) -> YReturn | tuple[XReturn, YReturn]:
        """Fit to data, then transform it.

        Parameters
        ----------
        X: dataframe-like or None, default=None
            Feature set with shape=(n_samples, n_features). If None,
            `X` is ignored.

        y: sequence, dataframe-like or None, default=None
            Target column(s) corresponding to `X`. If None, `y` is
            ignored.

        **fit_params
            Additional keyword arguments for the fit method.

        Returns
        -------
        dataframe
            Transformed feature set. Only returned if provided.

        series or dataframe
            Transformed target column. Only returned if provided.

        """
        return self.fit(X, y, **fit_params).transform(X, y)

    @overload
    def inverse_transform(
        self,
        X: Literal[None],
        y: YConstructor,
        **fit_params,
    ) -> YReturn: ...

    @overload
    def inverse_transform(
        self,
        X: XConstructor,
        y: Literal[None] = ...,
        **fit_params,
    ) -> XReturn: ...

    @overload
    def inverse_transform(
        self,
        X: XConstructor,
        y: YConstructor,
        **fit_params,
    ) -> tuple[XReturn, YReturn]: ...

    def inverse_transform(
        self,
        X: XConstructor | None = None,
        y: YConstructor | None = None,
        **fit_params,
    ) -> YReturn | tuple[XReturn, YReturn]:
        """Do nothing.

        Returns the input unchanged. Implemented for continuity of the
        API.

        Parameters
        ----------
        X: dataframe-like or None, default=None
            Feature set with shape=(n_samples, n_features). If None,
            `X` is ignored.

        y: sequence, dataframe-like or None, default=None
            Target column(s) corresponding to `X`. If None, `y` is
            ignored.

        Returns
        -------
        dataframe
            Feature set. Only returned if provided.

        series or dataframe
            Target column(s). Only returned if provided.

        """
        check_is_fitted(self)

        Xt = to_df(X)
        yt = to_tabular(y, index=getattr(Xt, "index", None))

        return variable_return(self._convert(Xt), self._convert(yt))

    def set_output(self, *, transform: EngineDataOptions | None = None) -> Self:
        """Set output container.

        See sklearn's [user guide][set_output] on how to use the
        `set_output` API. See [here][data-engines] a description
        of the choices.

        Parameters
        ----------
        transform: str or None, default=None
            Configure the output of the `transform`, `fit_transform`,
            and `inverse_transform` method. If None, the configuration
            is not changed. Choose from:

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

        Returns
        -------
        Self
            Estimator instance.

        """
        if not hasattr(self, "_engine"):
            self.engine = EngineTuple()

        if transform is not None:
            self.engine = EngineTuple(estimator=self.engine.estimator, data=transform)

        return self


@beartype
class Balancer(TransformerMixin, OneToOneFeatureMixin):
    """Balance the number of samples per class in the target column.

    When oversampling, the newly created samples have an increasing
    integer index for numerical indices, and an index of the form
    [estimator]_N for non-numerical indices, where N stands for the
    N-th sample in the data set. Use only for classification tasks.

    This class can be accessed from atom through the [balance]
    [atomclassifier-balance] method. Read more in the [user guide]
    [balancing-the-data].

    !!! warning
         * The [clustercentroids][] estimator is unavailable because of
           incompatibilities of the APIs.
         * The Balancer class does not support [multioutput tasks][].

    Parameters
    ----------
    strategy: str or transformer, default="ADASYN"
        Type of algorithm with which to balance the dataset. Choose
        from the name of any estimator in the imbalanced-learn package
        or provide a custom instance of such.

    n_jobs: int, default=1
        Number of cores to use for parallel processing.

        - If >0: Number of cores to use.
        - If -1: Use all available cores.
        - If <-1: Use number of cores - 1 - value.

    verbose: int, default=0
        Verbosity level of the class. Choose from:

        - 0 to not print anything.
        - 1 to print basic information.
        - 2 to print detailed information.

    random_state: int or None, default=None
        Seed used by the random number generator. If None, the random
        number generator is the `RandomState` used by `np.random`.

    **kwargs
        Additional keyword arguments for the `strategy` estimator.

    Attributes
    ----------
    [strategy]_: imblearn estimator
        Object (lowercase strategy) used to balance the data,
        e.g., `balancer.adasyn_` for the default strategy.

    mapping_: dict
        Target values mapped to their respective encoded integers.

    feature_names_in_: np.ndarray
        Names of features seen during `fit`.

    target_names_in_: np.ndarray
        Names of the target column seen during `fit`.

    n_features_in_: int
        Number of features seen during `fit`.

    See Also
    --------
    atom.data_cleaning:Encoder
    atom.data_cleaning:Imputer
    atom.data_cleaning:Pruner

    Examples
    --------
    === "atom"
        ```python
        from atom import ATOMClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        atom = ATOMClassifier(X, y, random_state=1)
        print(atom.train)

        atom.balance(strategy="smote", verbose=2)

        # Note that the number of rows has increased
        print(atom.train)
        ```

    === "stand-alone"
        ```python
        from atom.data_cleaning import Balancer
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)
        print(X)

        balancer = Balancer(strategy="smote", verbose=2)
        X, y = balancer.fit_transform(X, y)

        # Note that the number of rows has increased
        print(X)
        ```

    """

    _train_only = True

    def __init__(
        self,
        strategy: str | Estimator = "ADASYN",
        *,
        n_jobs: NJobs = 1,
        verbose: Verbose = 0,
        random_state: IntLargerEqualZero | None = None,
        **kwargs,
    ):
        super().__init__(n_jobs=n_jobs, verbose=verbose, random_state=random_state)
        self.strategy = strategy
        self.kwargs = kwargs

    def _more_tags(self):
        """Overwrite some default sklearn tags."""
        return {"requires_y": True, "X_types": ["2darray", "sparse"]}

    def _log_changes(self, y: pd.Series):
        """Print the changes per target class.

        Parameters
        ----------
        y: pd.Series
            Target column.

        """
        for key, value in self.mapping_.items():
            diff = self._counts[key] - np.sum(y == value)
            if diff > 0:
                self._log(f" --> Removing {diff} samples from class {key}.", 2)
            elif diff < 0:
                self._log(f" --> Adding {-diff} samples to class {key}.", 2)

    def fit(self, X: XConstructor, y: YConstructor) -> Self:
        """Fit to data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: sequence
            Target column corresponding to `X`.

        Returns
        -------
        Self
            Estimator instance.

        """
        Xt = to_df(X)
        yt = to_tabular(y, index=Xt.index)

        self._check_feature_names(Xt, reset=True)
        self._check_n_features(Xt, reset=True)

        if isinstance(yt, pd.Series):
            self.target_names_in_ = np.array([yt.name])
        else:
            raise ValueError("The Balancer class does not support multioutput tasks.")

        # ClusterCentroids is unavailable since it has no sample_indices_
        strategies = {
            "condensednearestneighbour": CondensedNearestNeighbour,
            "editednearestneighborus": EditedNearestNeighbours,
            "repeatededitednearestneighbours": RepeatedEditedNearestNeighbours,
            "allknn": AllKNN,
            "instancehardnessthreshold": InstanceHardnessThreshold,
            "nearmiss": NearMiss,
            "neighbourhoodcleaningrule": NeighbourhoodCleaningRule,
            "onesidedselection": OneSidedSelection,
            "randomundersampler": RandomUnderSampler,
            "tomeklinks": TomekLinks,
            "randomoversampler": RandomOverSampler,
            "smote": SMOTE,
            "smotenc": SMOTENC,
            "smoten": SMOTEN,
            "adasyn": ADASYN,
            "borderlinesmote": BorderlineSMOTE,
            "kmeanssmote": KMeansSMOTE,
            "svmsmote": SVMSMOTE,
            "smoteenn": SMOTEENN,
            "smotetomek": SMOTETomek,
        }

        if isinstance(self.strategy, str):
            if self.strategy.lower() not in strategies:
                raise ValueError(
                    f"Invalid value for the strategy parameter, got {self.strategy}. "
                    f"Choose from: {', '.join(strategies)}."
                )
            est_class = strategies[self.strategy.lower()]
            estimator = self._inherit(est_class(**self.kwargs), fixed=tuple(self.kwargs))
        elif not hasattr(self.strategy, "fit_resample"):
            raise TypeError(
                "Invalid type for the strategy parameter. A "
                "custom estimator must have a fit_resample method."
            )
        elif callable(self.strategy):
            estimator = self._inherit(self.strategy(**self.kwargs), fixed=tuple(self.kwargs))
        else:
            estimator = self.strategy

        # Create dict of class counts in y
        if not hasattr(self, "mapping_"):
            self.mapping_ = {str(v): v for v in yt.sort_values().unique()}

        self._counts = {}
        for key, value in self.mapping_.items():
            self._counts[key] = np.sum(yt == value)

        # Fit only checks input and sampling strategy
        self._estimator = estimator.fit(Xt, yt)

        # Add the estimator as attribute to the instance
        setattr(self, f"{estimator.__class__.__name__.lower()}_", self._estimator)

        return self

    def transform(self, X: XConstructor, y: YConstructor) -> tuple[XReturn, YReturn]:
        """Balance the data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: sequence
            Target column corresponding to `X`.

        Returns
        -------
        dataframe
            Balanced dataframe.

        series
            Transformed target column.

        """
        check_is_fitted(self)

        Xt = to_df(X, columns=self.feature_names_in_)
        yt = to_series(y, index=Xt.index, name=self.target_names_in_[0])  # type: ignore[arg-type]

        if "over_sampling" in self._estimator.__module__:
            self._log(f"Oversampling with {self._estimator.__class__.__name__}...", 1)

            index = Xt.index  # Save indices for later reassignment
            Xt, yt = self._estimator.fit_resample(Xt, yt)

            # Create indices for the new samples
            n_idx: list[int | str]
            if index.dtype.kind in "ifu":
                n_idx = list(range(max(index) + 1, max(index) + len(Xt) - len(index) + 1))
            else:
                n_idx = [
                    f"{self._estimator.__class__.__name__.lower()}_{i}"
                    for i in range(1, len(Xt) - len(index) + 1)
                ]

            # Assign the old + new indices
            Xt.index = pd.Index(list(index) + n_idx)
            yt.index = pd.Index(list(index) + n_idx)

            self._log_changes(yt)

        elif "under_sampling" in self._estimator.__module__:
            self._log(f"Undersampling with {self._estimator.__class__.__name__}...", 1)

            self._estimator.fit_resample(Xt, yt)

            # Select chosen rows (imblearn doesn't return them in order)
            samples = np.asarray(sorted(self._estimator.sample_indices_))
            Xt, yt = Xt.iloc[samples], yt.iloc[samples]

            self._log_changes(yt)

        elif "combine" in self._estimator.__module__:
            self._log(f"Balancing with {self._estimator.__class__.__name__}...", 1)

            index = Xt.index
            X_new, y_new = self._estimator.fit_resample(Xt, yt)

            # Select rows kept by the undersampler
            if self._estimator.__class__.__name__ == "SMOTEENN":
                samples = np.asarray(sorted(self._estimator.enn_.sample_indices_))
            elif self._estimator.__class__.__name__ == "SMOTETomek":
                samples = np.asarray(sorted(self._estimator.tomek_.sample_indices_))

            # Select the remaining samples from the old dataframe
            o_samples = [s for s in samples if s < len(Xt)]
            Xt, yt = Xt.iloc[o_samples], yt.iloc[o_samples]  # type: ignore[call-overload]

            # Create indices for the new samples
            if index.dtype.kind in "ifu":
                n_idx = list(range(max(index) + 1, max(index) + len(X_new) - len(Xt) + 1))
            else:
                n_idx = [
                    f"{self._estimator.__class__.__name__.lower()}_{i}"
                    for i in range(1, len(X_new) - len(Xt) + 1)
                ]

            # Select the new samples and assign the new indices
            X_new = X_new.iloc[-len(X_new) + len(o_samples):]
            X_new.index = pd.Index(n_idx)
            y_new = y_new.iloc[-len(y_new) + len(o_samples):]
            y_new.index = pd.Index(n_idx)

            # First, output the samples created
            for key, value in self.mapping_.items():
                if (diff := np.sum(y_new == value)) > 0:
                    self._log(f" --> Adding {diff} samples to class: {key}.", 2)

            # Then, output the samples dropped
            for key, value in self.mapping_.items():
                if (diff := self._counts[key] - np.sum(yt == value)) > 0:
                    self._log(f" --> Removing {diff} samples from class: {key}.", 2)

            # Add the new samples to the old dataframe
            Xt, yt = pd.concat([Xt, X_new]), pd.concat([yt, y_new])

        return self._convert(Xt), self._convert(yt)


@beartype
class Cleaner(TransformerMixin):
    """Applies standard data cleaning steps on a dataset.

    Use the parameters to choose which transformations to perform.
    The available steps are:

    - Convert dtypes to the best possible types.
    - Drop columns with specific data types.
    - Remove characters from column names.
    - Strip categorical features from spaces.
    - Drop duplicate rows.
    - Drop rows with missing values in the target column.
    - Encode the target column.

    This class can be accessed from atom through the [clean]
    [atomclassifier-clean] method. Read more in the [user guide]
    [standard-data-cleaning].

    Parameters
    ----------
    convert_dtypes: bool, default=True
        Convert the column's data types to the best possible types
        that support `pd.NA`.

    drop_dtypes: str, sequence or None, default=None
        Columns with these data types are dropped from the dataset.

    drop_chars: str or None, default=None
        Remove the specified regex pattern from column names, e.g.
        `[^A-Za-z0-9]+` to remove all non-alphanumerical characters.

    strip_categorical: bool, default=True
        Whether to strip spaces from categorical columns.

    drop_duplicates: bool, default=False
        Whether to drop duplicate rows. Only the first occurrence of
        every duplicated row is kept.

    drop_missing_target: bool, default=True
        Whether to drop rows with missing values in the target column.
        This transformation is ignored if `y` is not provided.

    encode_target: bool, default=True
        Whether to encode the target column(s). This includes
        converting categorical columns to numerical, and binarizing
        [multilabel][] columns. This transformation is ignored if `y`
        is not provided.

    device: str, default="cpu"
        Device on which to run the estimators. Use any string that
        follows the [SYCL_DEVICE_FILTER][] filter selector, e.g.
        `#!python device="gpu"` to use the GPU. Read more in the
        [user guide][gpu-acceleration].

    engine: str or None, default=None
        Execution engine to use for [estimators][estimator-acceleration].
        If None, the default value is used. Choose from:

        - "sklearn" (default)
        - "cuml"

    verbose: int, default=0
        Verbosity level of the class. Choose from:

        - 0 to not print anything.
        - 1 to print basic information.
        - 2 to print detailed information.

    Attributes
    ----------
    missing_: list
        Values that are considered "missing". Default values are: None,
        NaN, NA, NaT, +inf, -inf, "", "?", "NA", "nan", "NaN", "NaT",
        "none", "None", "inf", "-inf". Note that None, NaN, NA, +inf and
        -inf are always considered missing since they are incompatible
        with sklearn estimators.

    mapping_: dict
        Target values mapped to their respective encoded integers. Only
        available if encode_target=True.

    feature_names_in_: np.ndarray
        Names of features seen during `fit`.

    target_names_in_: np.ndarray
        Names of the target column(s) seen during `fit`.

    n_features_in_: int
        Number of features seen during `fit`.

    See Also
    --------
    atom.data_cleaning:Encoder
    atom.data_cleaning:Discretizer
    atom.data_cleaning:Scaler

    Examples
    --------
    === "atom"
        ```python
        from atom import ATOMClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)
        y = ["a" if i else "b" for i in y]

        atom = ATOMClassifier(X, y, random_state=1)
        print(atom.y)

        atom.clean(verbose=2)

        print(atom.y)
        ```

    === "stand-alone"
        ```python
        from atom.data_cleaning import Cleaner
        from numpy.random import randint

        y = ["a" if i else "b" for i in range(randint(100))]

        cleaner = Cleaner(verbose=2)
        y = cleaner.fit_transform(y=y)

        print(y)
        ```

    """

    def __init__(
        self,
        *,
        convert_dtypes: Bool = True,
        drop_dtypes: str | Sequence[str] | None = None,
        drop_chars: str | None = None,
        strip_categorical: Bool = True,
        drop_duplicates: Bool = False,
        drop_missing_target: Bool = True,
        encode_target: Bool = True,
        device: str = "cpu",
        engine: Engine = None,
        verbose: Verbose = 0,
    ):
        super().__init__(device=device, engine=engine, verbose=verbose)
        self.convert_dtypes = convert_dtypes
        self.drop_dtypes = drop_dtypes
        self.drop_chars = drop_chars
        self.strip_categorical = strip_categorical
        self.drop_duplicates = drop_duplicates
        self.drop_missing_target = drop_missing_target
        self.encode_target = encode_target

    def _more_tags(self):
        """Overwrite some default sklearn tags."""
        return {"X_types": ["2darray", "sparse"]}

    def fit(self, X: XConstructor | None = None, y: YConstructor | None = None) -> Self:
        """Fit to data.

        Parameters
        ----------
        X: dataframe-like or None, default=None
            Feature set with shape=(n_samples, n_features). If None,
            `X` is ignored.

        y: sequence, dataframe-like or None, default=None
            Target column(s) corresponding to `X`.

        Returns
        -------
        Self
            Estimator instance.

        """
        Xt = to_df(X)
        yt = to_tabular(y, index=getattr(Xt, "index", None))

        self._check_feature_names(Xt, reset=True)
        self._check_n_features(Xt, reset=True)

        self.mapping_: dict[str, Any] = {}
        self.target_names_in_ = np.array([])
        self._drop_cols = []
        self._estimators = {}

        if not hasattr(self, "missing_"):
            self.missing_ = DEFAULT_MISSING

        self._log("Fitting Cleaner...", 1)

        if Xt is not None and self.drop_dtypes is not None:
            self._drop_cols = list(Xt.select_dtypes(include=lst(self.drop_dtypes)).columns)

        if yt is not None:
            self.target_names_in_ = np.array(get_col_names(yt))

            if self.drop_chars:
                if isinstance(yt, pd.DataFrame):
                    yt = yt.rename(lambda x: re.sub(self.drop_chars, "", str(x)), axis=1)
                else:
                    yt.name = re.sub(self.drop_chars, "", str(yt.name))

            if self.drop_missing_target:
                yt = replace_missing(yt, self.missing_).dropna(axis=0)

            if self.encode_target:
                for col in get_cols(yt):
                    if isinstance(col.iloc[0], sequence_t):  # Multilabel
                        MultiLabelBinarizer = self._get_est_class(
                            name="MultiLabelBinarizer",
                            module="preprocessing",
                        )
                        self._estimators[col.name] = MultiLabelBinarizer().fit(col)
                    elif list(uq := np.unique(col)) != list(range(col.nunique())):
                        LabelEncoder = self._get_est_class("LabelEncoder", "preprocessing")
                        self._estimators[col.name] = LabelEncoder().fit(col)
                        self.mapping_.update(
                            {str(col.name): {str(it(v)): i for i, v in enumerate(uq)}}
                        )

        return self

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features: sequence or None, default=None
            Only used to validate feature names with the names seen in
            `fit`.

        Returns
        -------
        np.ndarray
            Transformed feature names.

        """
        check_is_fitted(self, attributes="feature_names_in_")
        _check_feature_names_in(self, input_features)

        columns = [col for col in self.feature_names_in_ if col not in self._drop_cols]

        if self.drop_chars:
            # Drop prohibited chars from column names
            columns = [re.sub(self.drop_chars, "", str(c)) for c in columns]

        return np.array(columns)

    def transform(
        self,
        X: XConstructor | None = None,
        y: YConstructor | None = None,
    ) -> YReturn | tuple[XReturn, YReturn]:
        """Apply the data cleaning steps to the data.

        Parameters
        ----------
        X: dataframe-like or None, default=None
            Feature set with shape=(n_samples, n_features). If None,
            `X` is ignored.

        y: sequence, dataframe-like or None, default=None
            Target column(s) corresponding to `X`.

        Returns
        -------
        dataframe
            Transformed feature set. Only returned if provided.

        series or dataframe
            Transformed target column. Only returned if provided.

        """
        check_is_fitted(self)

        Xt = to_df(X, columns=getattr(self, "feature_names_in_", None))
        yt = to_tabular(y, index=getattr(Xt, "index", None), columns=self.target_names_in_)

        self._log("Cleaning the data...", 1)

        if Xt is not None:
            # Unify all missing values
            Xt = replace_missing(Xt, self.missing_)

            for name, column in Xt.items():
                # Drop features with an invalid data type
                if name in self._drop_cols:
                    self._log(
                        f" --> Dropping feature {name} for "
                        f"having type: {column.dtype.name}.", 2,
                    )
                    Xt = Xt.drop(columns=name)

                elif column.dtype.name in CAT_TYPES:
                    if self.strip_categorical:
                        # Strip strings from blank spaces
                        Xt[name] = column.apply(
                            lambda val: val.strip() if isinstance(val, str) else val
                        )

            # Drop prohibited chars from column names
            if self.drop_chars:
                Xt = Xt.rename(columns=lambda x: re.sub(self.drop_chars, "", str(x)))

            # Drop duplicate samples
            if self.drop_duplicates:
                Xt = Xt.drop_duplicates(ignore_index=True)

            if self.convert_dtypes:
                Xt = Xt.convert_dtypes()

        if yt is not None:
            if self.drop_chars:
                if isinstance(y, pd.Series):
                    yt.name = re.sub(self.drop_chars, "", str(yt.name))
                else:
                    yt = yt.rename(lambda x: re.sub(self.drop_chars, "", str(x)), axis=1)

            # Delete samples with missing values in target
            if self.drop_missing_target:
                length = len(yt)  # Save original length to count deleted rows later
                yt = replace_missing(yt, self.missing_).dropna()

                if Xt is not None:
                    Xt = Xt[Xt.index.isin(yt.index)]  # Select only indices that remain

                if (d := length - len(yt)) > 0:
                    self._log(f" --> Dropping {d} rows with missing values in target.", 2)

            if self.encode_target and self._estimators:
                y_new = yt.__class__(dtype="object")
                for col in get_cols(yt):
                    if est := self._estimators.get(col.name):
                        if n_cols(out := est.transform(col)) == 1:
                            self._log(f" --> Label-encoding column {col.name}.", 2)
                            out = to_series(out, yt.index, str(col.name))
                        else:
                            self._log(f" --> Label-binarizing column {col.name}.", 2)
                            out = to_df(
                                data=out,
                                index=yt.index,
                                columns=[f"{col.name}_{c}" for c in est.classes_],
                            )

                        # Replace target with encoded column(s)
                        if isinstance(yt, pd.Series):
                            y_new = out
                        else:
                            y_new = merge(y_new, out)

                    else:  # Add unchanged column
                        y_new = merge(y_new, col)

                yt = y_new

            if self.convert_dtypes:
                yt = yt.convert_dtypes()

        return variable_return(self._convert(Xt), self._convert(yt))

    def inverse_transform(
        self,
        X: XConstructor | None = None,
        y: YConstructor | None = None,
    ) -> YReturn | tuple[XReturn, YReturn]:
        """Inversely transform the label encoding.

        This method only inversely transforms the target encoding.
        The rest of the transformations can't be inverted. If
        `encode_target=False`, the data is returned as is.

        Parameters
        ----------
        X: dataframe-like or None, default=None
            Do nothing. Implemented for continuity of the API.

        y: sequence, dataframe-like or None, default=None
            Target column(s) corresponding to `X`.

        Returns
        -------
        dataframe
            Unchanged feature set. Only returned if provided.

        series or dataframe
            Original target column. Only returned if provided.

        """
        check_is_fitted(self)

        Xt = to_df(X, columns=getattr(self, "feature_names_in_", None))
        yt = to_tabular(y, index=getattr(Xt, "index", None))

        self._log("Inversely cleaning the data...", 1)

        if yt is not None and self._estimators:
            y_new = yt.__class__(dtype="object")
            for col in self.target_names_in_:
                if est := self._estimators.get(col):
                    if est.__class__.__name__ == "LabelEncoder":
                        self._log(f" --> Inversely label-encoding column {col}.", 2)
                        out = est.inverse_transform(pd.DataFrame(yt)[col])

                    elif isinstance(yt, pd.DataFrame):
                        self._log(f" --> Inversely label-binarizing column {col}.", 2)
                        out = est.inverse_transform(
                            yt.loc[:, yt.columns.str.startswith(f"{col}_")].to_numpy()
                        )

                    # Replace encoded columns with target column
                    if isinstance(yt, pd.Series):
                        y_new = to_series(out, yt.index, col)
                    else:
                        y_new = merge(y_new, to_series(out, yt.index, col))

                else:  # Add unchanged column
                    y_new = merge(y_new, pd.DataFrame(yt)[col])

            yt = y_new

        return variable_return(self._convert(Xt), self._convert(yt))


@beartype
class Decomposer(TransformerMixin, OneToOneFeatureMixin):
    """Detrend and deseasonalize the time series.

    This class does two things:

    - Remove the trend from every column, returning the in-sample
      residuals of the model's predicted values.
    - Remove the seasonal component from every column, subject to
      a seasonaility test.

    Categorical columns are ignored.

    This class can be accessed from atom through the [decompose]
    [atomforecaster-decompose] method. Read more in the [user guide]
    [time-series-decomposition].

    !!! note
        When using this class from atom, the `trend_model`, `sp` and
        `seasonal_model` parameters are set automatically based on the
        `atom.sp` attribute.

    Parameters
    ----------
    model: str, predictor or None, default=None
        The forecasting model to remove the trend with. It must be
        a model that supports the forecast task. If None,
        [PolynomialTrend][](degree=1) is used.

    trend_model: str, default="additive"
        Mode of the trend decomposition. Choose from:

        - "additive": The `model.transform` subtracts the trend, i.e.,
          `transform(X)` returns `X - model.predict(fh=X.index)`.
        - "multiplicative": The `model.transform` divides by the trend,
          i.e., `transform(X)` returns `X / model.predict(fh=X.index)`.

    test_seasonality: bool, default=True

        - If True, it fits a 90% autocorrelation seasonality test, and
          if the passed time series has a seasonal component, it
          applies seasonal decomposition. If the test is negative,
          deseasonalization is skipped.
        - If False, always performs deseasonalization.

    sp: int or None, default=None
        Seasonality period of the time series. If None, there's no
        seasonality.

    seasonal_model: str, default="additive"
        Mode of the seasonal decomposition. Choose from:

        - "additive": Assumes the components have a linear relation,
          i.e., y(t) = level + trend + seasonality + noise.
        - "multiplicative": Assumes the components have a nonlinear
          relation, i.e., y(t) = level * trend * seasonality * noise.

    n_jobs: int, default=1
        Number of cores to use for parallel processing.

        - If >0: Number of cores to use.
        - If -1: Use all available cores.
        - If <-1: Use number of cores - 1 + `n_jobs`.

    verbose: int, default=0
        Verbosity level of the class. Choose from:

        - 0 to not print anything.
        - 1 to print basic information.
        - 2 to print detailed information.

    random_state: int or None, default=None
        Seed used by the random number generator. If None, the random
        number generator is the `RandomState` used by `np.random`.

    Attributes
    ----------
    feature_names_in_: np.ndarray
        Names of features seen during `fit`.

    n_features_in_: int
        Number of features seen during `fit`.

    See Also
    --------
    atom.data_cleaning:Encoder
    atom.data_cleaning:Discretizer
    atom.data_cleaning:Scaler

    Examples
    --------
    === "atom"
        ```python
        from atom import ATOMForecaster
        from sktime.datasets import load_airline

        y = load_airline()

        atom = ATOMForecaster(y, random_state=1)
        print(atom.y)

        atom.decompose(columns=-1, verbose=2)

        print(atom.y)
        ```

    === "stand-alone"
        ```python
        from atom.data_cleaning import Decomposer
        from sktime.datasets import load_longley

        X, _ = load_longley()

        decomposer = Decomposer(verbose=2)
        X = decomposer.fit_transform(X)

        print(X)
        ```

    """

    def __init__(
        self,
        *,
        model: str | Predictor | None = None,
        trend_model: SeasonalityModels = "additive",
        test_seasonality: Bool = True,
        sp: IntLargerZero | None = None,
        seasonal_model: SeasonalityModels = "additive",
        n_jobs: NJobs = 1,
        verbose: Verbose = 0,
        random_state: IntLargerEqualZero | None = None,
    ):
        super().__init__(n_jobs=n_jobs, verbose=verbose, random_state=random_state)
        self.model = model
        self.trend_model = trend_model
        self.test_seasonality = test_seasonality
        self.sp = sp
        self.seasonal_model = seasonal_model

    def fit(self, X: XConstructor, y: YConstructor | None = None) -> Self:
        """Fit to data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: sequence, dataframe-like or None, default=None
            Do nothing. Implemented for continuity of the API.

        Returns
        -------
        Self
            Estimator instance.

        """
        from atom.models import MODELS

        Xt = to_df(X)

        self._check_feature_names(Xt, reset=True)
        self._check_n_features(Xt, reset=True)

        if isinstance(self.model, str):
            if self.model in MODELS:
                model = MODELS[self.model](
                    goal=Goal.forecast,
                    **{x: getattr(self, x) for x in BaseTransformer.attrs if hasattr(self, x)},
                )
                forecaster = model._get_est({})
            else:
                raise ValueError(
                    "Invalid value for the model parameter. Unknown "
                    f"model: {self.model}. Available models are:\n" +
                    "\n".join(
                        [
                            f" --> {m.__name__} ({m.acronym})"
                            for m in MODELS
                            if "forecast" in m._estimators
                        ]
                    )
                )
        elif callable(self.model):
            forecaster = self._inherit(self.model())
        else:
            forecaster = self.model

        self._log("Fitting Decomposer...", 1)

        self._estimators: dict[Hashable, tuple[Transformer, Transformer]] = {}
        for name, column in Xt.select_dtypes(include="number").items():
            trend = Detrender(
                forecaster=forecaster,
                model=self.trend_model,
            ).fit(column)

            if self.test_seasonality:
                season = ConditionalDeseasonalizer(
                    sp=self.sp or 1,
                    model=self.seasonal_model,
                ).fit(trend.transform(column))
            else:
                season = Deseasonalizer(
                    sp=self.sp or 1,
                    model=self.seasonal_model,
                ).fit(trend.transform(column))

            self._estimators[name] = (trend, season)

        return self

    def transform(self, X: XConstructor, y: YConstructor | None = None) -> XReturn:
        """Decompose the data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: sequence, dataframe-like or None, default=None
            Do nothing. Implemented for continuity of the API.

        Returns
        -------
        dataframe
            Transformed feature set.

        """
        check_is_fitted(self)

        Xt = to_df(X, columns=self.feature_names_in_)

        self._log("Decomposing the data...", 1)

        for col, (trend, season) in self._estimators.items():
            Xt[col] = season.transform(trend.transform(Xt[col]))

        return self._convert(Xt)

    def inverse_transform(self, X: XConstructor, y: YConstructor | None = None) -> XReturn:
        """Inversely transform the data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: sequence, dataframe-like or None, default=None
            Do nothing. Implemented for continuity of the API.

        Returns
        -------
        dataframe
            Original feature set.

        """
        check_is_fitted(self)

        Xt = to_df(X, columns=self.feature_names_in_)

        self._log("Inversely decomposing the data...", 1)

        for col, (trend, season) in self._estimators.items():
            Xt[col] = trend.inverse_transform(season.inverse_transform(Xt[col]))

        return self._convert(Xt)


@beartype
class Discretizer(TransformerMixin, OneToOneFeatureMixin):
    """Bin continuous data into intervals.

    For each feature, the bin edges are computed during fit and,
    together with the number of bins, they define the intervals.
    Ignores categorical columns.

    This class can be accessed from atom through the [discretize]
    [atomclassifier-discretize] method. Read more in the [user guide]
    [binning-numerical-features].

    !!! tip
        The transformation returns categorical columns. Use the
        [Encoder][] class to convert them back to numerical types.

    Parameters
    ----------
    strategy: str, default="quantile"
        Strategy used to define the widths of the bins. Choose from:

        - "uniform": All bins have identical widths.
        - "quantile": All bins have the same number of points.
        - "kmeans": Values in each bin have the same nearest center of
          a 1D k-means cluster.
        - "custom": Use custom bin edges provided through `bins`.

    bins: int, sequence or dict, default=5
        Bin number or bin edges in which to split every column.

        - If int: Number of bins to produce for all columns. Only for
          strategy!="custom".
        - If sequence:

            - For strategy!="custom": Number of bins per column. The
              n-th value corresponds to the n-th column that is
              transformed. Categorical columns are ignored.
            - For strategy="custom": Bin edges with length=n_bins - 1.
              The outermost edges are always `-inf` and `+inf`, e.g.,
              bins `[1, 2]` indicate `(-inf, 1], (1, 2], (2, inf]`.

        - If dict: One of the aforementioned options per column, where
          the key is the column's name. Columns that are not in the
          dictionary are not transformed.

    labels: sequence, dict or None, default=None
        Label names with which to replace the binned intervals.

        - If None: Use default labels of the form `(min_edge, max_edge]`.
        - If sequence: Labels to use for all columns.
        - If dict: Labels per column, where the key is the column's name.
          Columns that are not in the dictionary use the default labels.

    device: str, default="cpu"
        Device on which to run the estimators. Use any string that
        follows the [SYCL_DEVICE_FILTER][] filter selector, e.g.
        `#!python device="gpu"` to use the GPU. Read more in the
        [user guide][gpu-acceleration].

    engine: str or None, default=None
        Execution engine to use for [estimators][estimator-acceleration].
        If None, the default value is used. Choose from:

        - "sklearn" (default)
        - "cuml"

    verbose: int, default=0
        Verbosity level of the class. Choose from:

        - 0 to not print anything.
        - 1 to print basic information.
        - 2 to print detailed information.

    random_state: int or None, default=None
        Seed used by the random number generator. If None, the random
        number generator is the `RandomState` used by `np.random`. Only
        for strategy="quantile".

    Attributes
    ----------
    feature_names_in_: np.ndarray
        Names of features seen during `fit`.

    n_features_in_: int
        Number of features seen during `fit`.

    See Also
    --------
    atom.data_cleaning:Encoder
    atom.data_cleaning:Imputer
    atom.data_cleaning:Normalizer

    Examples
    --------
    === "atom"
        ```python
        from atom import ATOMClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        atom = ATOMClassifier(X, y, random_state=1)
        print(atom["mean radius"])

        atom.discretize(
            strategy="custom",
            bins=[13, 18],
            labels=["small", "medium", "large"],
            verbose=2,
            columns="mean radius",
        )

        print(atom["mean radius"])
        ```

    === "stand-alone"
        ```python
        from atom.data_cleaning import Discretizer
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)
        print(X["mean radius"])

        discretizer = Discretizer(
            strategy="custom",
            bins={"mean radius": [13, 18]},
            labels=["small", "medium", "large"],
            verbose=2,
        )
        X = discretizer.fit_transform(X)

        print(X["mean radius"])
        ```

    """

    def __init__(
        self,
        strategy: DiscretizerStrats = "quantile",
        *,
        bins: Bins = 5,
        labels: Sequence[str] | dict[str, Sequence[str]] | None = None,
        device: str = "cpu",
        engine: Engine = None,
        verbose: Verbose = 0,
        random_state: IntLargerEqualZero | None = None,
    ):
        super().__init__(
            device=device,
            engine=engine,
            verbose=verbose,
            random_state=random_state,
        )
        self.strategy = strategy
        self.bins = bins
        self.labels = labels

    def fit(self, X: XConstructor, y: YConstructor | None = None) -> Self:
        """Fit to data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: sequence, dataframe-like or None, default=None
            Do nothing. Implemented for continuity of the API.

        Returns
        -------
        Self
            Estimator instance.

        """

        def get_labels(col: str, bins: Sequence[Scalar]) -> tuple[str, ...]:
            """Get labels for the specified bins.

            Parameters
            ----------
            col: str
                Name of the column.

            bins: sequence
                Bin edges.

            Returns
            -------
            tuple
                Labels for the column.

            """
            default = [
                f"({np.round(bins[i], 2)}, {np.round(bins[i + 1], 1)}]"
                for i in range(len(bins[:-1]))
            ]

            if self.labels is None:
                labels = tuple(default)
            elif isinstance(self.labels, dict):
                labels = tuple(self.labels.get(col, default))
            else:
                labels = tuple(self.labels)

            if len(bins) - 1 != len(labels):
                raise ValueError(
                    "Invalid value for the labels parameter. The length of "
                    "the bins does not match the length of the labels, got "
                    f"len(bins)={len(bins) - 1} and len(labels)={len(labels)}."
                )

            return labels

        Xt = to_df(X)

        self._check_feature_names(Xt, reset=True)
        self._check_n_features(Xt, reset=True)

        self._estimators: dict[Hashable, Estimator] = {}
        self._labels: dict[Hashable, Sequence[str]] = {}

        self._log("Fitting Discretizer...", 1)

        for i, col in enumerate(Xt.select_dtypes(include="number")):
            # Assign bins per column
            if isinstance(self.bins, dict):
                if col in self.bins:
                    bins_c = self.bins[str(col)]
                else:
                    continue  # Ignore existing column not specified in dict
            else:
                bins_c = self.bins

            if self.strategy != "custom":
                if isinstance(bins_c, sequence_t):
                    try:
                        bins_x = bins_c[i]  # Fetch the i-th bin for the i-th column
                    except IndexError:
                        raise ValueError(
                            "Invalid value for the bins parameter. The length of the "
                            "bins does not match the length of the columns, got len"
                            f"(bins)={len(bins_c)} and len(columns)={Xt.shape[1]}."
                        ) from None
                else:
                    bins_x = bins_c

                KBinsDiscretizer = self._get_est_class("KBinsDiscretizer", "preprocessing")

                # cuML implementation has no subsample and random_state
                kwargs: dict[str, Any] = {}
                if "subsample" in sign(KBinsDiscretizer):
                    kwargs["subsample"] = 200000
                    kwargs["random_state"] = self.random_state

                self._estimators[col] = KBinsDiscretizer(
                    n_bins=bins_x,
                    encode="ordinal",
                    strategy=self.strategy,
                    **kwargs,
                ).fit(Xt[[col]])

                # Save labels for transform method
                self._labels[col] = get_labels(
                    col=str(col),
                    bins=self._estimators[col].bin_edges_[0],
                )

            else:
                if not isinstance(bins_c, sequence_t):
                    raise TypeError(
                        f"Invalid type for the bins parameter, got {bins_c}. Only "
                        "a sequence of bin edges is accepted when strategy='custom'."
                    )
                else:
                    bins_c = [-np.inf, *bins_c, np.inf]

                FunctionTransformer = self._get_est_class(
                    name="FunctionTransformer",
                    module="preprocessing",
                )

                # Make of cut a transformer
                self._estimators[col] = FunctionTransformer(
                    func=pd.cut,
                    kw_args={"bins": bins_c, "labels": get_labels(str(col), bins_c)},
                ).fit(Xt[[col]])

        return self

    def transform(self, X: XConstructor, y: YConstructor | None = None) -> XReturn:
        """Bin the data into intervals.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: sequence, dataframe-like or None, default=None
            Do nothing. Implemented for continuity of the API.

        Returns
        -------
        dataframe
            Transformed feature set.

        """
        check_is_fitted(self)

        Xt = to_df(X, columns=self.feature_names_in_)

        self._log("Binning the features...", 1)

        for col in self._estimators:
            if self.strategy == "custom":
                Xt[col] = self._estimators[col].transform(Xt[col])
            else:
                Xt[col] = self._estimators[col].transform(Xt[[col]]).iloc[:, 0]

                # Replace cluster values with labels
                for i, label in enumerate(self._labels[col]):
                    Xt[col] = Xt[col].replace(i, label)

            self._log(f" --> Discretizing feature {col} in {Xt[col].nunique()} bins.", 2)

        return self._convert(Xt)


@beartype
class Encoder(TransformerMixin):
    """Perform encoding of categorical features.

    The encoding type depends on the number of classes in the column:

    - If n_classes=2 or ordinal feature, use Ordinal-encoding.
    - If 2 < n_classes <= `max_onehot`, use OneHot-encoding.
    - If n_classes > `max_onehot`, use `strategy`-encoding.

    Missing values are propagated to the output column. Unknown
    classes encountered during transforming are imputed according
    to the selected strategy. Infrequent classes can be replaced with
    a value in order to prevent too high cardinality.

    This class can be accessed from atom through the [encode]
    [atomclassifier-encode] method. Read more in the [user guide]
    [encoding-categorical-features].

    !!! warning
        Three category-encoders estimators are unavailable:

        * [OneHotEncoder][]: Use the max_onehot parameter.
        * [HashingEncoder][]: Incompatibility of APIs.
        * [LeaveOneOutEncoder][]: Incompatibility of APIs.

    Parameters
    ----------
    strategy: str or transformer, default="Target"
        Type of encoding to use for high cardinality features. Choose
        from any of the estimators in the category-encoders package
        or provide a custom one.

    max_onehot: int or None, default=10
        Maximum number of unique values in a feature to perform
        one-hot encoding. If None, `strategy`-encoding is always
        used for columns with more than two classes.

    ordinal: dict or None, default=None
        Order of ordinal features, where the dict key is the feature's
        name and the value is the class order, e.g., `{"salary": ["low",
        "medium", "high"]}`.

    infrequent_to_value: int, float or None, default=None
        Replaces infrequent class occurrences in categorical columns
        with the string in parameter `value`. This transformation is
        done before the encoding of the column.

        - If None: Skip this step.
        - If int: Minimum number of occurrences in a class.
        - If float: Minimum fraction of occurrences in a class.

    value: str, default="infrequent"
        Value with which to replace rare classes. This parameter is
        ignored if `infrequent_to_value=None`.

    n_jobs: int, default=1
        Number of cores to use for parallel processing.

        - If >0: Number of cores to use.
        - If -1: Use all available cores.
        - If <-1: Use number of cores - 1 - value.

    verbose: int, default=0
        Verbosity level of the class. Choose from:

        - 0 to not print anything.
        - 1 to print basic information.
        - 2 to print detailed information.

    **kwargs
        Additional keyword arguments for the `strategy` estimator.

    Attributes
    ----------
    mapping_: dict of dicts
        Encoded values and their respective mapping. The column name is
        the key to its mapping dictionary. Only for ordinal encoding.

    feature_names_in_: np.ndarray
        Names of features seen during `fit`.

    n_features_in_: int
        Number of features seen during `fit`.

    See Also
    --------
    atom.data_cleaning:Cleaner
    atom.data_cleaning:Imputer
    atom.data_cleaning:Pruner

    Examples
    --------
    === "atom"
        ```python
        from atom import ATOMClassifier
        from sklearn.datasets import load_breast_cancer
        from numpy.random import randint

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)
        X["cat_feature_1"] = [f"x{i}" for i in randint(0, 2, len(X))]
        X["cat_feature_2"] = [f"x{i}" for i in randint(0, 3, len(X))]
        X["cat_feature_3"] = [f"x{i}" for i in randint(0, 20, len(X))]

        atom = ATOMClassifier(X, y, random_state=1)
        print(atom.X)

        atom.encode(strategy="target", max_onehot=10, verbose=2)

        # Note the one-hot encoded column with name [feature]_[class]
        print(atom.X)
        ```

    === "stand-alone"
        ```python
        from atom.data_cleaning import Encoder
        from sklearn.datasets import load_breast_cancer
        from numpy.random import randint

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)
        X["cat_feature_1"] = [f"x{i}" for i in randint(0, 2, len(X))]
        X["cat_feature_2"] = [f"x{i}" for i in randint(0, 3, len(X))]
        X["cat_feature_3"] = [f"x{i}" for i in randint(0, 20, len(X))]
        print(X)

        encoder = Encoder(strategy="target", max_onehot=10, verbose=2)
        X = encoder.fit_transform(X, y)

        # Note the one-hot encoded column with name [feature]_[class]
        print(X)
        ```

    """

    def __init__(
        self,
        strategy: str | Transformer = "Target",
        *,
        max_onehot: IntLargerTwo | None = 10,
        ordinal: dict[str, Sequence[Any]] | None = None,
        infrequent_to_value: FloatLargerZero | None = None,
        value: str = "infrequent",
        n_jobs: NJobs = 1,
        verbose: Verbose = 0,
        **kwargs,
    ):
        super().__init__(n_jobs=n_jobs, verbose=verbose)
        self.strategy = strategy
        self.max_onehot = max_onehot
        self.ordinal = ordinal
        self.infrequent_to_value = infrequent_to_value
        self.value = value
        self.kwargs = kwargs

    def fit(self, X: XConstructor, y: YConstructor | None = None) -> Self:
        """Fit to data.

        Note that leaving y=None can lead to errors if the `strategy`
        encoder requires target values. For multioutput tasks, only
        the first target column is used to fit the encoder.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: sequence or dataframe-like
            Target column(s) corresponding to `X`.

        Returns
        -------
        Self
            Estimator instance.

        """
        self.mapping_ = {}
        self._to_value = {}
        self._categories = {}

        strategies = {
            "backwarddifference": BackwardDifferenceEncoder,
            "basen": BaseNEncoder,
            "binary": BinaryEncoder,
            "catboost": CatBoostEncoder,
            "helmert": HelmertEncoder,
            "jamesstein": JamesSteinEncoder,
            "mestimate": MEstimateEncoder,
            "ordinal": OrdinalEncoder,
            "polynomial": PolynomialEncoder,
            "sum": SumEncoder,
            "target": TargetEncoder,
            "woe": WOEEncoder,
        }

        Xt = to_df(X)
        yt = to_tabular(y, index=Xt.index)

        self._check_feature_names(Xt, reset=True)
        self._check_n_features(Xt, reset=True)

        if isinstance(self.strategy, str):
            if self.strategy.lower().endswith("encoder"):
                self.strategy = self.strategy[:-7]  # Remove 'Encoder' at the end
            if self.strategy.lower() not in strategies:
                raise ValueError(
                    f"Invalid value for the strategy parameter, got {self.strategy}. "
                    f"Choose from: {', '.join(strategies)}."
                )
            estimator = strategies[self.strategy.lower()]
        elif callable(self.strategy):
            estimator = self.strategy
        else:
            raise TypeError(
                f"Invalid type for the strategy parameter, got {self.strategy}. "
                "For customs estimators, a class is expected, but got an instance."
            )

        if self.max_onehot is None:
            max_onehot = 0
        else:
            max_onehot = int(self.max_onehot)

        if self.infrequent_to_value:
            if self.infrequent_to_value < 1:
                infrequent_to_value = int(self.infrequent_to_value * len(Xt))
            else:
                infrequent_to_value = int(self.infrequent_to_value)

        self._log("Fitting Encoder...", 1)

        encoders: dict[str, list[str]] = defaultdict(list)

        for name, column in Xt.select_dtypes(include=CAT_TYPES).items():
            # Replace infrequent classes with the string in `value`
            if self.infrequent_to_value:
                values = column.value_counts()
                self._to_value[name] = values[values <= infrequent_to_value].index.tolist()
                Xt[name] = column.replace(self._to_value[name], self.value)

            # Get the unique categories before fitting
            self._categories[name] = column.dropna().sort_values().unique().tolist()

            # Perform encoding type dependent on number of unique values
            ordinal = self.ordinal or {}
            if name in ordinal or len(self._categories[name]) == 2:
                # Check that provided classes match those of column
                ordinal_c = ordinal.get(str(name), self._categories[name])
                if column.nunique(dropna=True) != len(ordinal_c):
                    self._log(
                        f" --> The number of classes passed to feature {name} in the "
                        f"ordinal parameter ({len(ordinal_c)}) don't match the number "
                        f"of classes in the data ({column.nunique(dropna=True)}).", 1,
                        severity="warning",
                    )

                # Create custom mapping from 0 to N - 1
                mapping: dict[Hashable, Scalar] = {v: i for i, v in enumerate(ordinal_c)}
                mapping.setdefault(np.nan, -1)  # Encoder always needs mapping of NaN
                self.mapping_[str(name)] = mapping

                encoders["ordinal"].append(str(name))
            elif 2 < len(self._categories[name]) <= max_onehot:
                encoders["onehot"].append(str(name))
            else:
                encoders["rest"].append(str(name))

        ordinal_enc = OrdinalEncoder(
            mapping=[{"col": c, "mapping": self.mapping_[c]} for c in encoders["ordinal"]],
            cols=encoders["ordinal"],
            handle_missing="return_nan",
            handle_unknown="value",
        )

        onehot_enc = OneHotEncoder(
            cols=encoders["onehot"],
            use_cat_names=True,
            handle_missing="return_nan",
            handle_unknown="value",
        )

        rest_enc = estimator(
            cols=encoders["rest"],
            handle_missing="return_nan",
            handle_unknown="value",
            **self.kwargs,
        )

        self._estimator = ColumnTransformer(
            transformers=[
                ("ordinal", ordinal_enc, encoders["ordinal"]),
                ("onehot", onehot_enc, encoders["onehot"]),
                ("rest", rest_enc, encoders["rest"]),
            ],
            remainder="passthrough",
            n_jobs=self.n_jobs,
            verbose_feature_names_out=False,
        ).fit(Xt, yt)

        return self

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features: sequence or None, default=None
            Only used to validate feature names with the names seen in
            `fit`.

        Returns
        -------
        np.ndarray
            Transformed feature names.

        """
        check_is_fitted(self, attributes="feature_names_in_")
        _check_feature_names_in(self, input_features)

        # Drop _nan columns (since missing values are propagated)
        cols = [c for c in self._estimator.get_feature_names_out() if not c.endswith("_nan")]

        return get_col_order(cols, self.feature_names_in_, self._estimator.feature_names_in_)

    def transform(self, X: XConstructor, y: YConstructor | None = None) -> XReturn:
        """Encode the data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: sequence, dataframe-like or None, default=None
            Do nothing. Implemented for continuity of the API.

        Returns
        -------
        dataframe
            Encoded dataframe.

        """
        check_is_fitted(self)

        Xt = to_df(X, columns=self.feature_names_in_)

        self._log("Encoding categorical columns...", 1)

        # Convert infrequent classes to value
        Xt = Xt.replace(self._to_value, self.value)

        for name, categories in self._categories.items():
            if name in self._estimator.transformers_[0][2]:
                estimator = self._estimator.transformers_[0][1]
            elif name in self._estimator.transformers_[1][2]:
                estimator = self._estimator.transformers_[1][1]
            else:
                estimator = self._estimator.transformers_[2][1]

            self._log(
                f" --> {estimator.__class__.__name__[:-7]}-encoding feature "
                f"{name}. Contains {Xt[name].nunique()} classes.", 2,
            )

            # Count the propagated missing values
            if n_nans := Xt[name].isna().sum():
                self._log(f"   --> Propagating {n_nans} missing values.", 2)

            # Check for unknown classes
            if uc := len(Xt[name].dropna()[~Xt[name].isin(categories)]):
                self._log(f"   --> Handling {uc} unknown classes.", 2)

        Xt = self._estimator.transform(Xt)

        return self._convert(Xt[self.get_feature_names_out()])


@beartype
class Imputer(TransformerMixin):
    """Handle missing values in the data.

    Impute or remove missing values according to the selected strategy.
    Also removes rows and columns with too many missing values. Use
    the `missing_` attribute to customize what are considered "missing
    values".

    This class can be accessed from atom through the [impute]
    [atomclassifier-impute] method. Read more in the [user guide]
    [imputing-missing-values].

    Parameters
    ----------
    strat_num: str, int or float, default="mean"
        Imputing strategy for numerical columns. Choose from:

        - "drop": Drop rows containing missing values.
        - "mean": Impute with mean of column.
        - "median": Impute with median of column.
        - "most_frequent": Impute with the most frequent value.
        - "knn": Impute using a K-Nearest Neighbors approach.
        - "iterative": Impute using a multivariate imputer.
        - "drift": Impute values using a [PolynomialTrend][] model.
        - "linear": Impute using linear interpolation.
        - "nearest": Impute with nearest value.
        - "bfill": Impute by using the next valid observation to fill
           the gap.
        - "ffill": Impute by propagating the last valid observation
          to next valid.
        - "random": Impute with random values between the min and max
           of column.
        - int or float: Impute with provided numerical value.

    strat_cat: str, default="most_frequent"
        Imputing strategy for categorical columns. Choose from:

        - "drop": Drop rows containing missing values.
        - "most_frequent": Impute with the most frequent value.
        - str: Impute with provided string.

    max_nan_rows: int, float or None, default=None
        Maximum number or fraction of missing values in a row
        (if more, the row is removed). If None, ignore this step.

    max_nan_cols: int, float or None, default=None
        Maximum number or fraction of missing values in a column
        (if more, the column is removed). If None, ignore this step.

    n_jobs: int, default=1
        Number of cores to use for parallel processing.

        - If >0: Number of cores to use.
        - If -1: Use all available cores.
        - If <-1: Use number of cores - 1 - value.

    device: str, default="cpu"
        Device on which to run the estimators. Use any string that
        follows the [SYCL_DEVICE_FILTER][] filter selector, e.g.
        `#!python device="gpu"` to use the GPU. Read more in the
        [user guide][gpu-acceleration].

    engine: str or None, default=None
        Execution engine to use for [estimators][estimator-acceleration].
        If None, the default value is used. Choose from:

        - "sklearn" (default)
        - "cuml"

    verbose: int, default=0
        Verbosity level of the class. Choose from:

        - 0 to not print anything.
        - 1 to print basic information.
        - 2 to print detailed information.

    random_state: int or None, default=None
        Seed used by the random number generator. If None, the random
        number generator is the `RandomState` used by `np.random`. Only
        used when strat_num="iterative".

    Attributes
    ----------
    missing_: list
        Values that are considered "missing". Default values are: None,
        NaN, NA, NaT, +inf, -inf, "", "?", "NA", "nan", "NaN", "NaT",
        "none", "None", "inf", "-inf". Note that None, NaN, NA, +inf and
        -inf are always considered missing since they are incompatible
        with sklearn estimators.

    feature_names_in_: np.ndarray
        Names of features seen during `fit`.

    n_features_in_: int
        Number of features seen during `fit`.

    See Also
    --------
    atom.data_cleaning:Balancer
    atom.data_cleaning:Discretizer
    atom.data_cleaning:Encoder

    Examples
    --------
    === "atom"
        ```python
        import numpy as np
        from atom import ATOMClassifier
        from numpy.random import randint
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        # Add some random missing values to the data
        for i, j in zip(randint(0, X.shape[0], 600), randint(0, 4, 600)):
            X.iloc[i, j] = np.NaN

        atom = ATOMClassifier(X, y, random_state=1)
        print(atom.nans)

        atom.impute(strat_num="median", max_nan_rows=0.1, verbose=2)

        print(atom.n_nans)
        ```

    === "stand-alone"
        ```python
        import numpy as np
        from atom.data_cleaning import Imputer
        from numpy.random import randint
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        # Add some random missing values to the data
        for i, j in zip(randint(0, X.shape[0], 600), randint(0, 4, 600)):
            X.iloc[i, j] = np.nan

        imputer = Imputer(strat_num="median", max_nan_rows=0.1, verbose=2)
        X, y = imputer.fit_transform(X, y)

        print(X)
        ```

    """

    def __init__(
        self,
        strat_num: Scalar | NumericalStrats = "mean",
        strat_cat: str | CategoricalStrats = "most_frequent",
        *,
        max_nan_rows: FloatLargerZero | None = None,
        max_nan_cols: FloatLargerZero | None = None,
        n_jobs: NJobs = 1,
        device: str = "cpu",
        engine: Engine = None,
        verbose: Verbose = 0,
        random_state: IntLargerEqualZero | None = None,
    ):
        super().__init__(
            n_jobs=n_jobs,
            device=device,
            engine=engine,
            verbose=verbose,
            random_state=random_state,
        )
        self.strat_num = strat_num
        self.strat_cat = strat_cat
        self.max_nan_rows = max_nan_rows
        self.max_nan_cols = max_nan_cols

    def fit(self, X: XConstructor, y: YConstructor | None = None) -> Self:
        """Fit to data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: sequence, dataframe-like or None, default=None
            Do nothing. Implemented for continuity of the API.

        Returns
        -------
        Self
            Estimator instance.

        """
        Xt = to_df(X)

        self._check_feature_names(Xt, reset=True)
        self._check_n_features(Xt, reset=True)

        if not hasattr(self, "missing_"):
            self.missing_ = DEFAULT_MISSING

        self._log("Fitting Imputer...", 1)

        # Unify all values to impute
        Xt = replace_missing(Xt, self.missing_)

        if self.max_nan_rows is not None:
            if self.max_nan_rows <= 1:
                self._max_nan_rows = int(Xt.shape[1] * self.max_nan_rows)
            else:
                self._max_nan_rows = int(self.max_nan_rows)

            Xt = Xt.dropna(axis=0, thresh=Xt.shape[1] - self._max_nan_rows)
            if Xt.empty:
                raise ValueError(
                    "Invalid value for the max_nan_rows parameter, got "
                    f"{self.max_nan_rows}. All rows contain more than "
                    f"{self._max_nan_rows} missing values. Choose a "
                    f"larger value or set the parameter to None."
                )

        if self.max_nan_cols is not None:
            if self.max_nan_cols <= 1:
                max_nan_cols = int(Xt.shape[0] * self.max_nan_cols)
            else:
                max_nan_cols = int(self.max_nan_cols)

            Xt = Xt.drop(columns=Xt.columns[Xt.isna().sum() > max_nan_cols])

        # Load the imputer class from sklearn or cuml (note the different modules)
        SimpleImputer = self._get_est_class(
            name="SimpleImputer",
            module="preprocessing" if self.engine.estimator == "cuml" else "impute",
        )

        # Note missing_values=pd.NA also imputes np.NaN
        num_imputer: Estimator | Literal["passthrough"]
        if isinstance(self.strat_num, str):
            if self.strat_num in ("mean", "median", "most_frequent"):
                num_imputer = SimpleImputer(missing_values=pd.NA, strategy=self.strat_num)
            elif self.strat_num == "knn":
                num_imputer = KNNImputer()
            elif self.strat_num == "iterative":
                num_imputer = IterativeImputer(random_state=self.random_state)
            elif self.strat_num == "drop":
                num_imputer = "passthrough"
            else:
                num_imputer = make_sklearn(SktimeImputer)(
                    method=self.strat_num,
                    missing_values=[pd.NA],
                    random_state=self.random_state,
                )
        else:
            num_imputer = SimpleImputer(
                missing_values=pd.NA,
                strategy="constant",
                fill_value=self.strat_num,
            )

        cat_imputer: Estimator | Literal["passthrough"]
        if self.strat_cat == "most_frequent":
            cat_imputer = SimpleImputer(missing_values=pd.NA, strategy=self.strat_cat)
        elif self.strat_cat == "drop":
            cat_imputer = "passthrough"
        else:
            cat_imputer = SimpleImputer(
                missing_values=pd.NA,
                strategy="constant",
                fill_value=self.strat_cat,
            )

        ColumnTransformer = self._get_est_class("ColumnTransformer", "compose")

        self._estimator = ColumnTransformer(
            transformers=[
                ("num_imputer", num_imputer, list(Xt.select_dtypes(include="number"))),
                ("cat_imputer", cat_imputer, list(Xt.select_dtypes(include=CAT_TYPES))),
            ],
            remainder="passthrough",
            n_jobs=self.n_jobs,
            verbose_feature_names_out=False,
        ).fit(Xt)

        return self

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features: sequence or None, default=None
            Only used to validate feature names with the names seen in
            `fit`.

        Returns
        -------
        np.ndarray
            Transformed feature names.

        """
        check_is_fitted(self, attributes="feature_names_in_")
        _check_feature_names_in(self, input_features)

        return np.array(
            [c for c in self.feature_names_in_ if c in self._estimator.get_feature_names_out()]
        )

    def transform(
        self,
        X: XConstructor,
        y: YConstructor | None = None,
    ) -> YReturn | tuple[XReturn, YReturn]:
        """Impute the missing values.

        Note that leaving y=None can lead to inconsistencies in
        data length between X and y if rows are dropped during
        the transformation.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: sequence, dataframe-like or None, default=None
            Target column(s) corresponding to `X`.

        Returns
        -------
        dataframe
            Imputed dataframe.

        series or dataframe
            Transformed target column. Only returned if provided.

        """
        check_is_fitted(self)

        Xt = to_df(X, columns=self.feature_names_in_)
        yt = to_tabular(y, index=Xt.index)

        num_imputer = self._estimator.named_transformers_["num_imputer"]
        cat_imputer = self._estimator.named_transformers_["cat_imputer"]

        get_stat = lambda est, n: est.statistics_[est.feature_names_in_.tolist().index(n)]

        self._log("Imputing missing values...", 1)

        # Unify all values to impute
        Xt = replace_missing(Xt, self.missing_)

        # Drop rows with too many missing values
        if self.max_nan_rows is not None:
            length = len(Xt)
            Xt = Xt.dropna(axis=0, thresh=Xt.shape[1] - self._max_nan_rows)
            if diff := length - len(Xt):
                self._log(
                    f" --> Dropping {diff} samples for containing more "
                    f"than {self._max_nan_rows} missing values.", 2,
                )

        if self.strat_num == "drop":
            length = len(Xt)
            Xt = Xt.dropna(subset=self._estimator.transformers_[0][2])
            if diff := length - len(Xt):
                self._log(
                    f" --> Dropping {diff} samples for containing "
                    f"missing values in numerical columns.", 2,
                )

        if self.strat_cat == "drop":
            length = len(Xt)
            Xt = Xt.dropna(subset=self._estimator.transformers_[1][2])
            if diff := length - len(Xt):
                self._log(
                    f" --> Dropping {diff} samples for containing "
                    f"missing values in categorical columns.", 2,
                )

        # Print imputation information per feature
        for name, column in Xt.items():
            if nans := column.isna().sum():
                # Drop columns with too many missing values
                if name not in self._estimator.feature_names_in_:
                    self._log(
                        f" --> Dropping feature {name}. Contains {nans} "
                        f"({nans * 100 // len(Xt)}%) missing values.", 2,
                    )
                    Xt = Xt.drop(columns=name)
                    continue

                if name in getattr(num_imputer, "feature_names_in_", []):
                    if not isinstance(self.strat_num, str):
                        self._log(
                            f" --> Imputing {nans} missing values with "
                            f"number '{self.strat_num}' in column {name}.", 2,
                        )
                    elif self.strat_num in ("knn", "iterative"):
                        self._log(
                            f" --> Imputing {nans} missing values using "
                            f"the {self.strat_num} imputer in column {name}.", 2,
                        )
                    elif self.strat_num in ("mean", "median", "most_frequent"):
                        self._log(
                            f" --> Imputing {nans} missing values with {self.strat_num} "
                            f"({np.round(get_stat(num_imputer, name), 2)}) in column {name}.", 2,
                        )
                    else:
                        self._log(
                            f" --> Imputing {nans} missing values with {self.strat_num} "
                            f"in column {name}.", 2,
                        )
                elif name in getattr(cat_imputer, "feature_names_in_", []):
                    if self.strat_cat == "most_frequent":
                        self._log(
                            f" --> Imputing {nans} missing values with most_frequent "
                            f"({get_stat(cat_imputer, name)}) in column {name}.", 2,
                        )
                    elif self.strat_cat != "drop":
                        self._log(
                            f" --> Imputing {nans} missing values with value "
                            f"'{self.strat_cat}' in column {name}.", 2,
                        )

        Xt = self._estimator.transform(Xt)

        # Make y consistent with X
        if yt is not None:
            yt = yt.loc[yt.index.isin(Xt.index)]

        # Reorder columns to original order
        Xt = Xt[self.get_feature_names_out()]

        return variable_return(self._convert(Xt), self._convert(yt))


@beartype
class Normalizer(TransformerMixin, OneToOneFeatureMixin):
    """Transform the data to follow a Normal/Gaussian distribution.

    This transformation is useful for modeling issues related to
    heteroscedasticity (non-constant variance), or other situations
    where normality is desired. Missing values are disregarded in
    fit and maintained in transform. Categorical columns are ignored.

    This class can be accessed from atom through the [normalize]
    [atomclassifier-normalize] method. Read more in the [user guide]
    [normalizing-the-feature-set].

    !!! warning
        The quantile strategy performs a non-linear transformation.
        This may distort linear correlations between variables measured
        at the same scale but renders variables measured at different
        scales more directly comparable.

    !!! note
        The yeojohnson and boxcox strategies scale the data after
        transforming. Use the `kwargs` to change this behavior.

    Parameters
    ----------
    strategy: str, default="yeojohnson"
        The transforming strategy. Choose from:

        - "[yeojohnson][]"
        - "[boxcox][]" (only works with strictly positive values)
        - "[quantile][]": Transform features using quantiles information.

    device: str, default="cpu"
        Device on which to run the estimators. Use any string that
        follows the [SYCL_DEVICE_FILTER][] filter selector, e.g.
        `#!python device="gpu"` to use the GPU. Read more in the
        [user guide][gpu-acceleration].

    engine: str or None, default=None
        Execution engine to use for [estimators][estimator-acceleration].
        If None, the default value is used. Choose from:

        - "sklearn" (default)
        - "cuml"

    verbose: int, default=0
        Verbosity level of the class. Choose from:

        - 0 to not print anything.
        - 1 to print basic information.

    random_state: int or None, default=None
        Seed used by the quantile strategy. If None, the random
        number generator is the `RandomState` used by `np.random`.

    **kwargs
        Additional keyword arguments for the `strategy` estimator.

    Attributes
    ----------
    [strategy]_: sklearn transformer
        Object with which the data is transformed, e.g.,
        `normalizer.yeojohnson` for the default strategy.

    feature_names_in_: np.ndarray
        Names of features seen during `fit`.

    n_features_in_: int
        Number of features seen during `fit`.

    See Also
    --------
    atom.data_cleaning:Cleaner
    atom.data_cleaning:Pruner
    atom.data_cleaning:Scaler

    Examples
    --------
    === "atom"
        ```python
        from atom import ATOMClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        atom = ATOMClassifier(X, y, random_state=1)
        print(atom.dataset)

        atom.plot_distribution(columns=0)

        atom.normalize(verbose=2)

        print(atom.dataset)

        atom.plot_distribution(columns=0)
        ```

    === "stand-alone"
        ```python
        from atom.data_cleaning import Normalizer
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        normalizer = Normalizer(verbose=2)
        X = normalizer.fit_transform(X)

        print(X)
        ```

    """

    def __init__(
        self,
        strategy: NormalizerStrats = "yeojohnson",
        *,
        device: str = "cpu",
        engine: Engine = None,
        verbose: Verbose = 0,
        random_state: IntLargerEqualZero | None = None,
        **kwargs,
    ):
        super().__init__(
            device=device,
            engine=engine,
            verbose=verbose,
            random_state=random_state,
        )
        self.strategy = strategy
        self.kwargs = kwargs

    def fit(self, X: XConstructor, y: YConstructor | None = None) -> Self:
        """Fit to data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: sequence, dataframe-like or None, default=None
            Do nothing. Implemented for continuity of the API.

        Returns
        -------
        Self
            Estimator instance.

        """
        strategies = {
            "yeojohnson": "PowerTransformer",
            "boxcox": "PowerTransformer",
            "quantile": "QuantileTransformer",
        }

        Xt = to_df(X)

        self._check_feature_names(Xt, reset=True)
        self._check_n_features(Xt, reset=True)

        if self.strategy in ("yeojohnson", "boxcox"):
            estimator = self._get_est_class(strategies[self.strategy], "preprocessing")
            self._estimator = estimator(
                method=self.strategy[:3] + "-" + self.strategy[3:],
                **self.kwargs,
            )
        elif self.strategy == "quantile":
            kwargs = self.kwargs.copy()
            estimator = self._get_est_class(strategies[self.strategy], "preprocessing")
            self._estimator = estimator(
                output_distribution=kwargs.pop("output_distribution", "normal"),
                random_state=kwargs.pop("random_state", self.random_state),
                **kwargs,
            )

        num_cols = Xt.select_dtypes(include="number")

        if num_cols.empty:
            raise ValueError(
                "The Normalizer class encountered no columns during fit. "
                "Make sure X contains numerical columns."
            )

        self._log("Fitting Normalizer...", 1)
        self._estimator.fit(num_cols)

        # Add the estimator as attribute to the instance
        setattr(self, f"{self.strategy}_", self._estimator)

        return self

    def transform(self, X: XConstructor, y: YConstructor | None = None) -> XReturn:
        """Apply the transformations to the data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: sequence, dataframe-like or None, default=None
            Do nothing. Implemented for continuity of the API.

        Returns
        -------
        dataframe
            Normalized dataframe.

        """
        check_is_fitted(self)

        Xt = to_df(X, columns=self.feature_names_in_)

        self._log("Normalizing features...", 1)

        Xt.update(self._estimator.transform(Xt[self._estimator.feature_names_in_]))

        return self._convert(Xt)

    def inverse_transform(self, X: XConstructor, y: YConstructor | None = None) -> XReturn:
        """Apply the inverse transformation to the data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: sequence, dataframe-like or None, default=None
            Do nothing. Implemented for continuity of the API.

        Returns
        -------
        dataframe
            Original dataframe.

        """
        check_is_fitted(self)

        Xt = to_df(X, columns=self.feature_names_in_)

        self._log("Inversely normalizing features...", 1)

        out: np.ndarray = self._estimator.inverse_transform(Xt[self._estimator.feature_names_in_])

        Xt.update(to_df(out, index=Xt.index, columns=self._estimator.feature_names_in_))

        return self._convert(Xt)


@beartype
class Pruner(TransformerMixin, OneToOneFeatureMixin):
    """Prune outliers from the data.

    Replace or remove outliers. The definition of outlier depends
    on the selected strategy and can greatly differ from one another.
    Ignores categorical columns.

    This class can be accessed from atom through the [prune]
    [atomclassifier-prune] method. Read more in the [user guide]
    [handling-outliers].

    !!! info
        The "sklearnex" and "cuml" engines are only supported for
        strategy="dbscan".

    Parameters
    ----------
    strategy: str or sequence, default="zscore"
        Strategy with which to select the outliers. If sequence of
        strategies, only samples marked as outliers by all chosen
        strategies are dropped. Choose from:

        - "zscore": Z-score of each data value.
        - "[iforest][]": Isolation Forest.
        - "[ee][]": Elliptic Envelope.
        - "[lof][]": Local Outlier Factor.
        - "[svm][]": One-class SVM.
        - "[dbscan][]": Density-Based Spatial Clustering.
        - "[hdbscan][]": Hierarchical Density-Based Spatial Clustering.
        - "[optics][]": DBSCAN-like clustering approach.

    method: int, float or str, default="drop"
        Method to apply on the outliers. Only the zscore strategy
        accepts another method than "drop". Choose from:

        - "drop": Drop any sample with outlier values.
        - "minmax": Replace outlier with the min/max of the column.
        - Any numerical value with which to replace the outliers.

    max_sigma: int or float, default=3
        Maximum allowed standard deviations from the mean of the
        column. If more, it is considered an outlier. Only if
        strategy="zscore".

    include_target: bool, default=False
        Whether to include the target column in the search for
        outliers. This can be useful for regression tasks. Only
        if strategy="zscore".

    device: str, default="cpu"
        Device on which to run the estimators. Use any string that
        follows the [SYCL_DEVICE_FILTER][] filter selector, e.g.
        `#!python device="gpu"` to use the GPU. Read more in the
        [user guide][gpu-acceleration].

    engine: str or None, default=None
        Execution engine to use for [estimators][estimator-acceleration].
        If None, the default value is used. Choose from:

        - "sklearn" (default)
        - "cuml"

    verbose: int, default=0
        Verbosity level of the class. Choose from:

        - 0 to not print anything.
        - 1 to print basic information.
        - 2 to print detailed information.

    **kwargs
        Additional keyword arguments for the `strategy` estimator. If
        sequence of strategies, the params should be provided in a dict
        with the strategy's name as key.

    Attributes
    ----------
    [strategy]_: sklearn estimator
        Object used to prune the data, e.g., `pruner.iforest` for the
        isolation forest strategy. Not available for strategy="zscore".

    feature_names_in_: np.ndarray
        Names of features seen during `fit`.

    n_features_in_: int
        Number of features seen during `fit`.

    See Also
    --------
    atom.data_cleaning:Balancer
    atom.data_cleaning:Normalizer
    atom.data_cleaning:Scaler

    Examples
    --------
    === "atom"
        ```python
        from atom import ATOMClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        atom = ATOMClassifier(X, y, random_state=1)
        print(atom.dataset)

        atom.prune(stratgey="iforest", verbose=2)

        # Note the reduced number of rows
        print(atom.dataset)

        atom.plot_distribution(columns=0)
        ```

    === "stand-alone"
        ```python
        from atom.data_cleaning import Normalizer
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        normalizer = Normalizer(verbose=2)
        X = normalizer.fit_transform(X)

        # Note the reduced number of rows
        print(X)
        ```

    """

    _train_only = True

    def __init__(
        self,
        strategy: PrunerStrats | Sequence[PrunerStrats] = "zscore",
        *,
        method: Scalar | Literal["drop", "minmax"] = "drop",
        max_sigma: FloatLargerZero = 3,
        include_target: Bool = False,
        device: str = "cpu",
        engine: Engine = None,
        verbose: Verbose = 0,
        **kwargs,
    ):
        super().__init__(device=device, engine=engine, verbose=verbose)
        self.strategy = strategy
        self.method = method
        self.max_sigma = max_sigma
        self.include_target = include_target
        self.kwargs = kwargs

    def transform(
        self,
        X: XConstructor,
        y: YConstructor | None = None,
    ) -> YReturn | tuple[XReturn, YReturn]:
        """Apply the outlier strategy on the data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: sequence, dataframe-like or None, default=None
            Target column(s) corresponding to `X`.

        Returns
        -------
        dataframe
            Transformed feature set.

        series or dataframe
            Transformed target column. Only returned if provided.

        """
        Xt = to_df(X, columns=getattr(self, "feature_names_in_", None))
        yt = to_tabular(y, index=Xt.index)

        # Estimators with their modules
        strategies = {
            "iforest": ["IsolationForest", "ensemble"],
            "ee": ["EllipticEnvelope", "covariance"],
            "lof": ["LocalOutlierFactor", "neighbors"],
            "svm": ["OneClassSVM", "svm"],
            "dbscan": ["DBSCAN", "cluster"],
            "hdbscan": ["HDBSCAN", "cluster"],
            "optics": ["OPTICS", "cluster"],
        }

        for strat in lst(self.strategy):
            if strat != "zscore" and str(self.method) != "drop":
                raise ValueError(
                    "Invalid value for the method parameter. Only the zscore "
                    f"strategy accepts another method than 'drop', got {self.method}."
                )

        # Allocate kwargs to every estimator
        kwargs: dict[PrunerStrats, dict[str, Any]] = {}
        for strat in lst(self.strategy):
            kwargs[strat] = {}
            for key, value in self.kwargs.items():
                # Parameters for this estimator only
                if key == strat:
                    kwargs[strat].update(value)
                # Parameters for all estimators
                elif key not in lst(self.strategy):
                    kwargs[strat].update({key: value})

        self._log("Pruning outliers...", 1)

        # Prepare dataset (merge with y and exclude categorical columns)
        objective = merge(Xt, yt) if self.include_target and yt is not None else Xt
        objective = objective.select_dtypes(include=["number"])

        outliers = []
        for strat in lst(self.strategy):
            if strat == "zscore":
                # stats.zscore only works with np types, therefore, convert
                z_scores = zscore(objective.values.astype(float), nan_policy="propagate")

                if not isinstance(self.method, str):
                    cond = np.abs(z_scores) > self.max_sigma
                    objective = objective.mask(cond, self.method)
                    self._log(
                        f" --> Replacing {cond.sum()} outlier values with {self.method}.", 2,
                    )

                elif self.method.lower() == "minmax":
                    counts = 0
                    for i, col in enumerate(objective):
                        # Replace outliers with NaN and after that with max,
                        # so that the max is not calculated with the outliers in it
                        cond1 = z_scores[:, i] > self.max_sigma
                        mask = objective[col].mask(cond1, np.nan)
                        objective[col] = mask.replace(np.nan, mask.max(skipna=True))

                        # Replace outliers with the minimum
                        cond2 = z_scores[:, i] < -self.max_sigma
                        mask = objective[col].mask(cond2, np.nan)
                        objective[col] = mask.replace(np.nan, mask.min(skipna=True))

                        # Sum number of replacements
                        counts += cond1.sum() + cond2.sum()

                    self._log(
                        f" --> Replacing {counts} outlier values "
                        "with the min or max of the column.", 2,
                    )

                elif self.method.lower() == "drop":
                    mask = (np.abs(zscore(z_scores)) <= self.max_sigma).all(axis=1)
                    outliers.append(mask)
                    if len(lst(self.strategy)) > 1:
                        self._log(
                            f" --> The zscore strategy detected "
                            f"{len(mask) - sum(mask)} outliers.", 2,
                        )

            else:
                estimator = self._get_est_class(*strategies[strat])(**kwargs[strat])
                mask = estimator.fit_predict(objective) >= 0
                outliers.append(mask)
                if len(lst(self.strategy)) > 1:
                    self._log(
                        f" --> The {estimator.__class__.__name__} "
                        f"detected {len(mask) - sum(mask)} outliers.", 2,
                    )

                # Add the estimator as attribute to the instance
                setattr(self, f"{strat}_", estimator)

        if outliers:
            # Select outliers from intersection of strategies
            outlier_rows = [any(strats) for strats in zip(*outliers, strict=True)]
            self._log(f" --> Dropping {len(outlier_rows) - sum(outlier_rows)} outliers.", 2)

            # Keep only the non-outliers from the data
            Xt = Xt[outlier_rows]
            if yt is not None:
                yt = yt[outlier_rows]

        else:
            # Replace the columns in X with the new values from objective
            Xt.update(objective)

        return variable_return(self._convert(Xt), self._convert(yt))


@beartype
class Scaler(TransformerMixin, OneToOneFeatureMixin):
    """Scale the data.

    Apply one of sklearn's scaling strategies. Categorical columns
    are ignored.

    This class can be accessed from atom through the [scale]
    [atomclassifier-scale] method. Read more in the [user guide]
    [scaling-the-feature-set].

    Parameters
    ----------
    strategy: str, default="standard"
        Strategy with which to scale the data. Choose from:

        - "[standard][]": Remove mean and scale to unit variance.
        - "[minmax][]": Scale features to a given range.
        - "[maxabs][]": Scale features by their maximum absolute value.
        - "[robust][]": Scale using statistics that are robust to outliers.

    include_binary: bool, default=False
        Whether to scale binary columns (only 0s and 1s).

    device: str, default="cpu"
        Device on which to run the estimators. Use any string that
        follows the [SYCL_DEVICE_FILTER][] filter selector, e.g.
        `#!python device="gpu"` to use the GPU. Read more in the
        [user guide][gpu-acceleration].

    engine: str or None, default=None
        Execution engine to use for [estimators][estimator-acceleration].
        If None, the default value is used. Choose from:

        - "sklearn" (default)
        - "cuml"

    verbose: int, default=0
        Verbosity level of the class. Choose from:

        - 0 to not print anything.
        - 1 to print basic information.

    **kwargs
        Additional keyword arguments for the `strategy` estimator.

    Attributes
    ----------
    [strategy]_: sklearn transformer
        Object with which the data is scaled, e.g.,
        `scaler.standard` for the default strategy.

    feature_names_in_: np.ndarray
        Names of features seen during `fit`.

    n_features_in_: int
        Number of features seen during `fit`.

    See Also
    --------
    atom.data_cleaning:Balancer
    atom.data_cleaning:Normalizer
    atom.data_cleaning:Scaler

    Examples
    --------
    === "atom"
        ```python
        from atom import ATOMClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        atom = ATOMClassifier(X, y, random_state=1)
        print(atom.dataset)

        atom.scale(verbose=2)

        # Note the reduced number of rows
        print(atom.dataset)
        ```

    === "stand-alone"
        ```python
        from atom.data_cleaning import Scaler
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        scaler = Scaler(verbose=2)
        X = scaler.fit_transform(X)

        # Note the reduced number of rows
        print(X)
        ```

    """

    def __init__(
        self,
        strategy: ScalerStrats = "standard",
        *,
        include_binary: Bool = False,
        device: str = "cpu",
        engine: Engine = None,
        verbose: Verbose = 0,
        **kwargs,
    ):
        super().__init__(device=device, engine=engine, verbose=verbose)
        self.strategy = strategy
        self.include_binary = include_binary
        self.kwargs = kwargs

    def fit(
        self,
        X: XConstructor,
        y: YConstructor | None = None,
        sample_weight: Sequence[Scalar] | None = None,
    ) -> Self:
        """Fit to data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: sequence, dataframe-like or None, default=None
            Do nothing. Implemented for continuity of the API.

        sample_weight: sequence or None, default=None
            Sample weights with shape=(n_samples,).

        Returns
        -------
        Self
            Estimator instance.

        """
        strategies = {
            "standard": "StandardScaler",
            "minmax": "MinMaxScaler",
            "maxabs": "MaxAbsScaler",
            "robust": "RobustScaler",
        }

        Xt = to_df(X)

        self._check_feature_names(Xt, reset=True)
        self._check_n_features(Xt, reset=True)

        num_cols = Xt.select_dtypes(include="number")

        if not self.include_binary:
            num_cols = Xt[
                [n for n, c in num_cols.items() if ~np.isin(c.dropna().unique(), [0, 1]).all()]
            ]

        if num_cols.empty:
            raise ValueError(
                "The Scaler class encountered no columns during fit. Make "
                "sure X contains numerical columns or check if there are "
                "non-binary columns when include_binary=False."
            )

        self._log("Fitting Scaler...", 1)

        estimator = self._get_est_class(strategies[self.strategy], "preprocessing")
        self._estimator = estimator(**self.kwargs)

        self._estimator.fit(num_cols)

        # Add the estimator as attribute to the instance
        setattr(self, f"{self.strategy}_", self._estimator)

        return self

    def transform(self, X: XConstructor, y: YConstructor | None = None) -> XReturn:
        """Perform standardization by centering and scaling.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: sequence, dataframe-like or None, default=None
            Do nothing. Implemented for continuity of the API.

        Returns
        -------
        dataframe
            Scaled dataframe.

        """
        check_is_fitted(self)

        Xt = to_df(X, columns=self.feature_names_in_)

        self._log("Scaling features...", 1)

        Xt.update(self._estimator.transform(Xt[self._estimator.feature_names_in_]))

        return self._convert(Xt)

    def inverse_transform(self, X: XConstructor, y: YConstructor | None = None) -> XReturn:
        """Apply the inverse transformation to the data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: sequence, dataframe-like or None, default=None
            Do nothing. Implemented for continuity of the API.

        Returns
        -------
        dataframe
            Scaled dataframe.

        """
        check_is_fitted(self)

        Xt = to_df(X, columns=self.feature_names_in_)

        self._log("Inversely scaling features...", 1)

        out: np.ndarray = self._estimator.inverse_transform(Xt[self._estimator.feature_names_in_])

        Xt.update(to_df(out, index=Xt.index, columns=self._estimator.feature_names_in_))

        return self._convert(Xt)
