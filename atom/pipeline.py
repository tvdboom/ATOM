# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing the ATOM's custom sklearn-like pipeline.

"""

from __future__ import annotations

from typing import Generator

import numpy as np
from joblib import Memory
from sklearn.base import clone
from sklearn.pipeline import Pipeline as skPipeline
from sklearn.pipeline import _final_estimator_has
from sklearn.utils import _print_elapsed_time
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_memory


from atom.utils.types import (
    BOOL, DATAFRAME, ESTIMATOR, FEATURES, FLOAT, INT, PANDAS, SEQUENCE, SERIES,
    TARGET,
)
from atom.utils.utils import (
    check_is_fitted, fit_one, fit_transform_one, transform_one,
    variable_return,
)


class Pipeline(skPipeline):
    """Custom Pipeline class.

    This class behaves as a sklearn pipeline, and additionally:

    - Accepts transformers that drop rows.
    - Accepts transformers that only are fitted on a subset
      of the provided dataset.
    - Accepts transformers that apply only on the target column.
    - Uses transformers that are only applied on the training set
      to fit the pipeline, not to make predictions on new data.
    - The instance is considered fitted at initialization if all
      the underlying transformers/estimator in the pipeline are.
    - It returns attributes from the final estimator if they are
      not of the Pipeline.

    Note: This Pipeline only works with estimators whose parameters
    for fit, transform, predict, etcare named X and/or y.

    See sklearn's [Pipeline][] for a description of the parameters
    and attributes.

    """

    def __init__(
        self,
        steps: list[tuple[str, ESTIMATOR]],
        *,
        memory: str | Memory | None = None,
        verbose: BOOL = False,
    ):
        super().__init__(steps, memory=memory, verbose=verbose)

        # If all estimators are fitted, the Pipeline is fitted
        self._is_fitted = False
        if all(check_is_fitted(est[2], False) for est in self._iter(True, True, False)):
            self._is_fitted = True

    def __getattr__(self, item: str):
        try:
            return getattr(self._final_estimator, item)
        except AttributeError:
            raise AttributeError(f"'Pipeline' object has no attribute '{item}'.")

    @property
    def memory(self) -> Memory:
        """Get the internal memory object."""
        return self._memory

    @memory.setter
    def memory(self, value: str | Memory | None):
        """Create a new internal memory object."""
        self._memory = check_memory(value)
        self._memory_fit = self._memory.cache(fit_transform_one)
        self._memory_transform = self._memory.cache(transform_one)

    def _can_transform(self) -> BOOL:
        """Check if the pipeline can use the transform method."""
        return (
            self._final_estimator is None or self._final_estimator == "passthrough"
            or hasattr(self._final_estimator, "transform")
        )

    def _can_inverse_transform(self) -> BOOL:
        """Check if the pipeline can use the transform method."""
        return all(
            est is None or est == "passthrough" or hasattr(est, "inverse_transform")
            for _, _, est in self._iter()
        )

    def _iter(
        self,
        with_final: BOOL = True,
        filter_passthrough: BOOL = True,
        filter_train_only: BOOL = True,
    ) -> Generator[INT, str, ESTIMATOR]:
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

        Yields
        ------
        int
            Index position in the pipeline.

        str
            Name of the estimator.

        Estimator
            Transformer or predictor instance.

        """
        it = super()._iter(with_final, filter_passthrough)
        if filter_train_only:
            no_train_only = filter(lambda x: not getattr(x[-1], "_train_only", False), it)
            return (x for x in no_train_only)
        else:
            return it

    def _fit(
        self,
        X: FEATURES | None = None,
        y: TARGET | None = None,
        **fit_params_steps,
    ):
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

        series or None
            Transformed target column.

        """
        self.steps = list(self.steps)
        self._validate_steps()

        for (step_idx, name, transformer) in self._iter(False, False, False):
            if transformer is None or transformer == "passthrough":
                with _print_elapsed_time("Pipeline", self._log_message(step_idx)):
                    continue

            if hasattr(transformer, "transform"):
                # Don't clone when caching is disabled to preserve backward compatibility
                if self._memory_fit.__class__.__name__ == "NotMemorizedFunc":
                    cloned = transformer
                else:
                    cloned = clone(transformer)

                    # Attach internal attrs otherwise wiped by clone
                    for attr in ("_cols", "_train_only"):
                        if hasattr(transformer, attr):
                            setattr(cloned, attr, getattr(transformer, attr))

                # Fit or load the current estimator from cache
                X, y, fitted_transformer = self._memory_fit(
                    transformer=cloned,
                    X=X,
                    y=y,
                    message=self._log_message(step_idx),
                    **fit_params_steps[name],
                )

            # Replace the estimator of the step with the fitted
            # estimator (necessary when loading from cache)
            self.steps[step_idx] = (name, fitted_transformer)

        return X, y

    def fit(
        self,
        X: FEATURES | None = None,
        y: TARGET | None = None,
        **fit_params,
    ) -> Pipeline:
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
            if self._final_estimator != "passthrough":
                fit_params_last_step = fit_params_steps[self.steps[-1][0]]
                fit_one(self._final_estimator, X, y, **fit_params_last_step)

        self._is_fitted = True
        return self

    @available_if(_can_transform)
    def transform(
        self,
        X: FEATURES | None = None,
        y: TARGET | None = None,
    ) -> DATAFRAME | SERIES | tuple[DATAFRAME, PANDAS]:
        """Transform the data.

        Call `transform` on each transformer in the pipeline. The
        transformed data are finally passed to the final estimator
        that calls the `transform` method. Only valid if the final
        estimator implements `transform`. This also works where final
        estimator is `None` in which case all prior transformations
        are applied.

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

        Returns
        -------
        dataframe
            Transformed feature set. Only returned if provided.

        series
            Transformed target column. Only returned if provided.

        """
        for _, _, transformer in self._iter():
            X, y = self._memory_transform(transformer, X, y)

        return variable_return(X, y)

    def fit_transform(
        self,
        X: FEATURES | None = None,
        y: TARGET | None = None,
        **fit_params,
    ) -> DATAFRAME | SERIES | tuple[DATAFRAME, PANDAS]:
        """Fit the pipeline and transform the data.

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
            - If sequence: Target array with shape=(n_samples,) or
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

        last_step = self._final_estimator
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if last_step == "passthrough":
                return variable_return(X, y)

            fit_params_last_step = fit_params_steps[self.steps[-1][0]]
            X, y, _ = fit_transform_one(last_step, X, y, **fit_params_last_step)

        return variable_return(X, y)

    @available_if(_can_inverse_transform)
    def inverse_transform(
        self,
        X: FEATURES | None = None,
        y: TARGET | None = None,
    ) -> DATAFRAME | SERIES | tuple[DATAFRAME, PANDAS]:
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
            - If sequence: Target array with shape=(n_samples,) or
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
            X, y = self._memory_transform(transformer, X, y, method="inverse_transform")

        return variable_return(X, y)

    @available_if(_final_estimator_has("predict"))
    def predict(self, X: FEATURES, **predict_params) -> np.ndarray:
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
        np.array
            Predicted classes with shape=(n_samples,).

        """
        for _, name, transformer in self._iter(with_final=False):
            X, _ = self._memory_transform(transformer, X)

        return self.steps[-1][-1].predict(X, **predict_params)

    @available_if(_final_estimator_has("predict_proba"))
    def predict_proba(self, X: FEATURES) -> np.ndarray:
        """Transform, then predict_proba of the final estimator.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        Returns
        -------
        np.array
            Predicted class probabilities.

        """
        for _, _, transformer in self._iter(with_final=False):
            X, _ = self._memory_transform(transformer, X)

        return self.steps[-1][-1].predict_proba(X)

    @available_if(_final_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X: FEATURES) -> np.ndarray:
        """Transform, then predict_log_proba of the final estimator.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        Returns
        -------
        np.array
            Predicted class log-probabilities.

        """
        for _, _, transformer in self._iter(with_final=False):
            X, _ = self._memory_transform(transformer, X)

        return self.steps[-1][-1].predict_log_proba(X)

    @available_if(_final_estimator_has("decision_function"))
    def decision_function(self, X: FEATURES) -> np.ndarray:
        """Transform, then decision_function of the final estimator.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        Returns
        -------
        np.array
            Predicted confidence scores.

        """
        for _, _, transformer in self._iter(with_final=False):
            X, _ = self._memory_transform(transformer, X)

        return self.steps[-1][-1].decision_function(X)

    @available_if(_final_estimator_has("score"))
    def score(
        self,
        X: FEATURES,
        y: TARGET,
        sample_weight: SEQUENCE | None = None,
    ) -> FLOAT:
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
            X, y = self._memory_transform(transformer, X, y)

        return self.steps[-1][-1].score(X, y, sample_weight=sample_weight)
