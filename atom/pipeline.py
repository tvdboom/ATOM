"""Automated Tool for Optimized Modeling (ATOM).

Author: Mavs
Description: Module containing the ATOM's custom sklearn-like pipeline.

"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal

import numpy as np
from joblib import Memory
from sklearn.base import clone
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.pipeline import _final_estimator_has
from sklearn.utils import _print_elapsed_time
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_memory
from typing_extensions import Self

from atom.utils.types import (
    Bool, DataFrame, Estimator, Float, Pandas, Scalar, Sequence, Verbose,
    XSelector, YSelector,
)
from atom.utils.utils import (
    NotFittedError, adjust_verbosity, check_is_fitted, fit_one,
    fit_transform_one, transform_one, variable_return,
)


class Pipeline(SkPipeline):
    """Pipeline of transforms with a final estimator.

    Sequentially apply a list of transforms and a final estimator.
    Intermediate steps of the pipeline must be transformsers, that
    is, they must implement `fit` and `transform` methods. The final
    estimator only needs to implement `fit`. The transformers in the
    pipeline can be cached using the `memory` parameter.

    The purpose of the pipeline is to assemble several steps that can
    be cross-validated together while setting different parameters. For
    this, it enables setting parameters of the various steps using their
    names and the parameter name separated by `__`, as in the example
    below. A step's estimator may be replaced entirely by setting the
    parameter with its name to another estimator, or a transformer
    removed by setting it to `passthrough` or `None`.

    Read more in sklearn's the [user guide][pipelinedocs].

    !!! info
        This class behaves similarly to sklearn's [pipeline][skpipeline],
        and additionally:

        - Works with an empty pipeline.
        - Accepts transformers that drop rows.
        - Accepts transformers that only are fitted on a subset of the
          provided dataset.
        - Accepts transformers that apply only on the target column.
        - Uses transformers that are only applied on the training set
          to fit the pipeline, not to make predictions on new data.
        - The instance is considered fitted at initialization if all
          the underlying transformers/estimator in the pipeline are.
        - It returns attributes from the final estimator if they are
          not of the Pipeline.
        - The last transformer is also cached.

    !!! warning
        This Pipeline only works with estimators whose parameters
        for fit, transform, predict, etc... are named `X` and/or `y`.

    Parameters
    ----------
    steps: list of tuple
        List of (name, transform) tuples (implementing `fit`/`transform`)
        that are chained in sequential order.

    memory: str, [Memory][joblibmemory] or None, default=None
        Used to cache the fitted transformers of the pipeline. Enabling
        caching triggers a clone of the transformers before fitting.
        Therefore, the transformer instance given to the pipeline cannot
        be inspected directly. Use the attribute `named_steps` or `steps`
        to inspect estimators within the pipeline. Caching the
        transformers is advantageous when fitting is time-consuming.

    verbose: int or None, default=0
        Verbosity level of the transformers in the pipeline. If None,
        it leaves them to their original verbosity. If >0, the time
        elapsed while fitting each step is printed.

    Attributes
    ----------
    named_steps: [Bunch][]
        Dictionary-like object, with the following attributes. Read-only
        attribute to access any step parameter by user given name. Keys
        are step names and values are steps parameters.

    classes_: np.ndarray of shape (n_classes,)
        The class' labels. Only exist if the last step of the pipeline
        is a classifier.

    feature_names_in_: np.ndarray
        Names of features seen during first step `fit` method.

    n_features_in_: int
        Number of features seen during first step `fit` method.

    Examples
    --------
    ```pycon
    from atom import ATOMClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    # Initialize atom
    atom = ATOMClassifier(X, y, verbose=2)

    # Apply data cleaning and feature engineering methods
    atom.scale()
    atom.balance(strategy="smote")
    atom.feature_selection(strategy="rfe", solver="lr", n_features=22)

    # Train models
    atom.run(models="LR")

    # Get the pipeline and make predictions
    pl = atom.lr.export_pipeline()
    print(pl.predict(X))
    ```

    """

    def __init__(
        self,
        steps: list[tuple[str, Estimator]],
        *,
        memory: str | Memory | None = None,
        verbose: Verbose | None = 0,
    ):
        super().__init__(steps, memory=memory, verbose=verbose)

    def __bool__(self):
        """Whether the pipeline has at least one estimator."""
        return len(self.steps) > 0

    def __contains__(self, item: str | Any):
        """Whether the name or estimator is in the pipeline."""
        if isinstance(item, str):
            return item in self.named_steps
        else:
            return item in self.named_steps.values()

    def __getattr__(self, item: str):
        """Get the attribute from the final estimator."""
        try:
            return getattr(self._final_estimator, item)
        except (AttributeError, IndexError):
            raise AttributeError(f"'Pipeline' object has no attribute '{item}'.") from None

    def __sklearn_is_fitted__(self):
        """Whether the pipeline has been fitted."""
        try:
            # check if the last step of the pipeline is fitted
            # we only check the last step since if the last step is fit, it
            # means the previous steps should also be fit. This is faster than
            # checking if every step of the pipeline is fit.
            check_is_fitted(self.steps[-1][1])
            return True
        except (NotFittedError, IndexError):
            return False

    @property
    def memory(self) -> Memory:
        """Get the internal memory object."""
        return self._memory

    @memory.setter
    def memory(self, value: str | Memory | None):
        """Create a new internal memory object."""
        self._memory = check_memory(value)
        self._mem_fit = self._memory.cache(fit_transform_one)
        self._mem_transform = self._memory.cache(transform_one)

    @property
    def _final_estimator(self) -> Literal["passthrough"] | Estimator | None:
        """Return the last estimator in the pipeline.

        If the pipeline is empty, return None. If the estimator is
        None, return "passthrough".

        """
        try:
            estimator = self.steps[-1][1]
            return "passthrough" if estimator is None else estimator
        except (ValueError, AttributeError, TypeError, IndexError):
            # This condition happens when the pipeline is empty or a call
            # to a method is first calling `_available_if` and `fit` did
            # not validate `steps` yet.
            return None

    def _can_transform(self) -> bool:
        """Check if the pipeline can use the transform method."""
        return (
            self._final_estimator is None
            or self._final_estimator == "passthrough"
            or hasattr(self._final_estimator, "transform")
        )

    def _can_inverse_transform(self) -> bool:
        """Check if the pipeline can use the transform method."""
        return all(
            est is None or est == "passthrough" or hasattr(est, "inverse_transform")
            for _, _, est in self._iter()
        )

    def _iter(
        self,
        *,
        with_final: Bool = True,
        filter_passthrough: Bool = True,
        filter_train_only: Bool = True,
    ) -> Iterator[tuple[int, str, Estimator]]:
        """Generate (idx, name, estimator) tuples from self.steps.

        By default, estimators that are only applied on the training
        set are filtered out for predictions.

        Parameters
        ----------
        with_final: bool, default=True
            Whether to include the final estimator.

        filter_passthrough: bool, default=True
            Whether to exclude `passthrough` elements.

        filter_passthrough: bool, default=True
            Whether to exclude estimators that should only be used for
            training (have the `_train_only` attribute).

        Returns
        -------
        int
            Index position in the pipeline.

        str
            Name of the estimator.

        Estimator
            Transformer or predictor instance.

        """
        it = super()._iter(with_final=with_final, filter_passthrough=filter_passthrough)
        if filter_train_only:
            return (x for x in it if not getattr(x[-1], "_train_only", False))
        else:
            return it

    def _fit(
        self,
        X: XSelector | None = None,
        y: YSelector | None = None,
        **fit_params_steps,
    ) -> tuple[DataFrame | None, Pandas | None]:
        """Get data transformed through the pipeline.

        Parameters
        ----------
        X: dataframe-like or None, default=None
            Feature set with shape=(n_samples, n_features). If None,
            X is ignored. None if the pipeline only uses y.

        y: int, str, dict, sequence or None, default=None
            Target column corresponding to X.

            - If None: y is ignored.
            - If int: Position of the target column in X.
            - If str: Name of the target column in X.
            - Else: Array with shape=(n_samples,) to use as target.

        **fit_params
            Additional keyword arguments for the fit method.

        Returns
        -------
        dataframe or None
            Transformed feature set.

        series, dataframe or None
            Transformed target column.

        """
        self.steps: list[tuple[str, Estimator]] = list(self.steps)
        self._validate_steps()

        for step, name, transformer in self._iter(
            with_final=False, filter_passthrough=False, filter_train_only=False
        ):
            if transformer is None or transformer == "passthrough":
                with _print_elapsed_time("Pipeline", self._log_message(step)):
                    continue

            # Don't clone when caching is disabled to preserve backward compatibility
            if self.memory.location is None:
                cloned = transformer
            else:
                cloned = clone(transformer)

                # Attach internal attrs otherwise wiped by clone
                for attr in ("_cols", "_train_only"):
                    if hasattr(transformer, attr):
                        setattr(cloned, attr, getattr(transformer, attr))

            with adjust_verbosity(cloned, self.verbose):
                # Fit or load the current estimator from cache
                X, y, fitted_transformer = self._mem_fit(
                    transformer=cloned,
                    X=X,
                    y=y,
                    message=self._log_message(step),
                    **fit_params_steps.get(name, {}),
                )

            # Replace the estimator of the step with the fitted
            # estimator (necessary when loading from cache)
            self.steps[step] = (name, fitted_transformer)

        return X, y

    def fit(
        self,
        X: XSelector | None = None,
        y: YSelector | None = None,
        **fit_params,
    ) -> Self:
        """Fit the pipeline.

        Parameters
        ----------
        X: dataframe-like or None, default=None
            Feature set with shape=(n_samples, n_features). If None,
            X is ignored.

        y: int, str, dict, sequence or None, default=None
            Target column corresponding to X.

            - If None: y is ignored.
            - If int: Position of the target column in X.
            - If str: Name of the target column in X.
            - Else: Array with shape=(n_samples,) to use as target.

        **fit_params
            Additional keyword arguments for the fit method.

        Returns
        -------
        self
            Estimator instance.

        """
        fit_params_steps = self._check_fit_params(**fit_params)
        X, y = self._fit(X, y, **fit_params_steps)
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            last_step = self._final_estimator
            if last_step is not None and last_step != "passthrough":
                with adjust_verbosity(last_step, self.verbose):
                    fit_params_last_step = fit_params_steps[self.steps[-1][0]]
                    fit_one(last_step, X, y, **fit_params_last_step)

        return self

    @available_if(_can_transform)
    def fit_transform(
        self,
        X: XSelector | None = None,
        y: YSelector | None = None,
        **fit_params,
    ) -> Pandas | tuple[DataFrame, Pandas]:
        """Fit the pipeline and transform the data.

        Call `fit` followed by `transform` on each transformer in the
        pipeline. The transformed data are finally passed to the final
        estimator that calls the `transform` method. Only valid if the
        final estimator implements `transform`. This also works when the
        final estimator is `None`, in which case all prior
        transformations are applied.

        Parameters
        ----------
        X: dataframe-like or None, default=None
            Feature set with shape=(n_samples, n_features). If None,
            X is ignored. None
            if the estimator only uses y.

        y: int, str, dict, sequence, dataframe or None, default=None
            Target column corresponding to X.

            - If None: y is ignored.
            - If int: Position of the target column in X.
            - If str: Name of the target column in X.
            - If dict: Name of the target column and sequence of values.
            - If sequence: Target column with shape=(n_samples,) or
              sequence of column names or positions for multioutput tasks.
            - If dataframe: Target columns for multioutput tasks.

        **fit_params
            Additional keyword arguments for the fit method.

        Returns
        -------
        dataframe
            Transformed feature set. Only returned if provided.

        series or dataframe
            Transformed target column. Only returned if provided.

        """
        fit_params_steps = self._check_fit_params(**fit_params)
        X, y = self._fit(X, y, **fit_params_steps)

        # Don't clone when caching is disabled to preserve backward compatibility
        if self.memory.location is None:
            last_step = self._final_estimator
        else:
            last_step = clone(self._final_estimator)

        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if last_step is None or last_step == "passthrough":
                return variable_return(X, y)

            with adjust_verbosity(last_step, self.verbose):
                fit_params_last_step = fit_params_steps[self.steps[-1][0]]
                X, y, _ = self._mem_fit(last_step, X, y, **fit_params_last_step)

        return variable_return(X, y)

    @available_if(_can_transform)
    def transform(
        self,
        X: XSelector | None = None,
        y: YSelector | None = None,
        **kwargs,
    ) -> Pandas | tuple[DataFrame, Pandas]:
        """Transform the data.

        Call `transform` on each transformer in the pipeline. The
        transformed data are finally passed to the final estimator
        that calls the `transform` method. Only valid if the final
        estimator implements `transform`. This also works when the
        final estimator is `None`, in which case all prior
        transformations are applied.

        Parameters
        ----------
        X: dataframe-like or None, default=None
            Feature set with shape=(n_samples, n_features). If None,
            X is ignored. None if the pipeline only uses y.

        y: int, str, dict, sequence or None, default=None
            Target column corresponding to X.

            - If None: y is ignored.
            - If int: Position of the target column in X.
            - If str: Name of the target column in X.
            - Else: Array with shape=(n_samples,) to use as target.

        **kwargs
            Additional keyword arguments for the `_iter` inner method.

        Returns
        -------
        dataframe
            Transformed feature set. Only returned if provided.

        series or dataframe
            Transformed target column. Only returned if provided.

        """
        for _, _, transformer in self._iter(**kwargs):
            with adjust_verbosity(transformer, self.verbose):
                X, y = self._mem_transform(transformer, X, y)

        return variable_return(X, y)

    @available_if(_can_inverse_transform)
    def inverse_transform(
        self,
        X: XSelector | None = None,
        y: YSelector | None = None,
    ) -> Pandas | tuple[DataFrame, Pandas]:
        """Inverse transform for each step in a reverse order.

        All estimators in the pipeline must implement the
        `inverse_transform` method.

        Parameters
        ----------
        X: dataframe-like or None, default=None
            Feature set with shape=(n_samples, n_features). If None,
            X is ignored. None if the pipeline only uses y.

        y: int, str, dict, sequence, dataframe or None, default=None
            Target column corresponding to X.

            - If None: y is ignored.
            - If int: Position of the target column in X.
            - If str: Name of the target column in X.
            - If dict: Name of the target column and sequence of values.
            - If sequence: Target column with shape=(n_samples,) or
              sequence of column names or positions for multioutput tasks.
            - If dataframe: Target columns for multioutput tasks.

        Returns
        -------
        dataframe
            Transformed feature set. Only returned if provided.

        series or dataframe
            Transformed target column. Only returned if provided.

        """
        for _, _, transformer in reversed(list(self._iter())):
            with adjust_verbosity(transformer, self.verbose):
                X, y = self._mem_transform(transformer, X, y, method="inverse_transform")

        return variable_return(X, y)

    @available_if(_final_estimator_has("predict"))
    def predict(self, X: XSelector, **predict_params) -> np.ndarray:
        """Transform, then predict of the final estimator.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        **predict_params
            Additional keyword arguments for the predict method. Note
            that while this may be used to return uncertainties from
            some models with return_std or return_cov, uncertainties
            that are generated by the transformations in the pipeline
            are not propagated to the final estimator.

        Returns
        -------
        np.ndarray
            Predicted classes with shape=(n_samples,).

        """
        for _, _, transformer in self._iter(with_final=False):
            with adjust_verbosity(transformer, self.verbose):
                X, _ = self._mem_transform(transformer, X)

        return self.steps[-1][-1].predict(X, **predict_params)

    @available_if(_final_estimator_has("predict_proba"))
    def predict_proba(self, X: XSelector) -> np.ndarray:
        """Transform, then predict_proba of the final estimator.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted class probabilities.

        """
        for _, _, transformer in self._iter(with_final=False):
            with adjust_verbosity(transformer, self.verbose):
                X, _ = self._mem_transform(transformer, X)

        return self.steps[-1][-1].predict_proba(X)

    @available_if(_final_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X: XSelector) -> np.ndarray:
        """Transform, then predict_log_proba of the final estimator.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted class log-probabilities.

        """
        for _, _, transformer in self._iter(with_final=False):
            with adjust_verbosity(transformer, self.verbose):
                X, _ = self._mem_transform(transformer, X)

        return self.steps[-1][-1].predict_log_proba(X)

    @available_if(_final_estimator_has("decision_function"))
    def decision_function(self, X: XSelector) -> np.ndarray:
        """Transform, then decision_function of the final estimator.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted confidence scores.

        """
        for _, _, transformer in self._iter(with_final=False):
            with adjust_verbosity(transformer, self.verbose):
                X, _ = self._mem_transform(transformer, X)

        return self.steps[-1][-1].decision_function(X)

    @available_if(_final_estimator_has("score"))
    def score(
        self,
        X: XSelector,
        y: YSelector,
        sample_weight: Sequence[Scalar] | None = None,
    ) -> Float:
        """Transform, then score of the final estimator.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str, dict, sequence
            - If int: Position of the target column in X.
            - If str: Name of the target column in X.
            - Else: Array with shape=(n_samples,) to use as target.

        sample_weight: sequence or None, default=None
            Sample weights corresponding to y.

        Returns
        -------
        float
            Mean accuracy or r2 of self.predict(X) with respect to y.

        """
        for _, _, transformer in self._iter(with_final=False):
            with adjust_verbosity(transformer, self.verbose):
                X, y = self._mem_transform(transformer, X, y)

        return self.steps[-1][-1].score(X, y, sample_weight=sample_weight)
