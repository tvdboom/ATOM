# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: Mavs
Description: Module containing the ATOM's custom sklearn-like pipeline.

"""

# Standard packages
from sklearn import pipeline
from sklearn.base import clone
from sklearn.utils import _print_elapsed_time
from sklearn.utils.validation import check_memory
from sklearn.utils.metaestimators import if_delegate_has_method

# Own modules
from .utils import variable_return, fit_one, transform_one, fit_transform_one


class Pipeline(pipeline.Pipeline):
    """Custom Pipeline class.

    This class behaves as a sklearn pipeline, and additionally:
        - Accepts transformers that change the target column.
        - Accepts transformers that drop rows.
        - Accepts transformers that only are fitted on a subset
          of the provided dataset.
        - Always outputs pandas objects.
        - Uses transformers that are only applied on the training set
          to fit the pipeline, not to make predictions on unseen data.

    Partially from https://github.com/scikit-learn-contrib/imbalanced
    -learn/blob/master/imblearn/pipeline.py.

    """

    def _iter(self, with_final=True, filter_passthrough=True, filter_train_only=True):
        """Generate (idx, (name, trans)) tuples from self.steps.

        By default, estimators that are only applied on the training
        set are filtered out for predictions.

        """
        it = super()._iter(with_final, filter_passthrough)
        if filter_train_only:
            return filter(lambda x: not getattr(x[-1], "_train_only", False), it)
        else:
            return it

    def _fit(self, X=None, y=None, **fit_params_steps):
        self.steps = list(self.steps)
        self._validate_steps()

        # Setup the memory
        memory = check_memory(self.memory).cache(fit_transform_one)

        for (step_idx, name, transformer) in self._iter(False, False, False):
            if transformer is None or transformer == "passthrough":
                with _print_elapsed_time("Pipeline", self._log_message(step_idx)):
                    continue

            if hasattr(transformer, "transform"):
                if getattr(memory, "location", "") is None:
                    # Don't clone when caching is disabled to
                    # preserve backward compatibility
                    cloned = transformer
                else:
                    cloned = clone(transformer)

                # Fit or load the current estimator from cache
                X, y, fitted_transformer = memory(
                    transformer=cloned,
                    X=X,
                    y=y,
                    message=self._log_message(step_idx),
                    **fit_params_steps[name],
                )

            # Replace the estimator of the step with the fitted
            # estimator (necessary when loading from the cache)
            self.steps[step_idx] = (name, fitted_transformer)

        return X, y

    def fit(self, X=None, y=None, **fit_params):
        """Fit the pipeline.

        Parameters
        ----------
        X: dict, list, tuple, np.ndarray, pd.DataFrame, optional (default=None)
            Feature set with shape=(n_samples, n_features). None
            if the estimator only uses y.

        y: int, str, sequence or None, optional (default=None)
            - If None: y is ignored.
            - If int: Index of the target column in X.
            - If str: Name of the target column in X.
            - Else: Target column with shape=(n_samples,).

        **fit_params
            Additional keyword arguments for the fit method.

        Returns
        -------
        self: Pipeline
            Fitted instance of self.

        """
        fit_params_steps = self._check_fit_params(**fit_params)
        X, y = self._fit(X, y, **fit_params_steps)
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator != "passthrough":
                fit_params_last_step = fit_params_steps[self.steps[-1][0]]
                fit_one(self._final_estimator, X, y, **fit_params_last_step)

        return self

    def fit_transform(self, X=None, y=None, **fit_params):
        """Fit the pipeline and transform the data.

        Parameters
        ----------
        X: dict, list, tuple, np.ndarray, pd.DataFrame, optional (default=None)
            Feature set with shape=(n_samples, n_features). None
            if the estimator only uses y.

        y: int, str, sequence or None, optional (default=None)
            - If None: y is ignored.
            - If int: Index of the target column in X.
            - If str: Name of the target column in X.
            - Else: Target column with shape=(n_samples,).

        **fit_params
            Additional keyword arguments for the fit method.

        Returns
        -------
        X: np.ndarray
            Transformed dataset.

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

    @if_delegate_has_method(delegate="_final_estimator")
    def predict(self, X, **predict_params):
        """Transform, then predict of the final estimator.

        Parameters
        ----------
        X: dict, list, tuple, np.array, sps.matrix or pd.DataFrame
            Feature set with shape=(n_samples, n_features).

        **predict_params
            Additional keyword arguments for the predict method. Note
            that while this may be used to return uncertainties from
            some models with return_std or return_cov, uncertainties
            that are generated by the transformations in the pipeline
            are not propagated to the final estimator.

        Returns
        -------
        y_pred: np.ndarray
            Predicted target with shape=(n_samples,).

        """
        for _, name, transformer in self._iter(with_final=False):
            X, _ = transform_one(transformer, X)

        return self.steps[-1][-1].predict(X, **predict_params)

    @if_delegate_has_method(delegate="_final_estimator")
    def predict_proba(self, X):
        """Transform, then predict_proba of the final estimator.

        Parameters
        ----------
        X: dict, list, tuple, np.array, sps.matrix or pd.DataFrame
            Feature set with shape=(n_samples, n_features).

        Returns
        -------
        y_pred: np.ndarray
            Predicted class probabilities.

        """
        for _, _, transformer in self._iter(with_final=False):
            X, _ = transform_one(transformer, X)

        return self.steps[-1][-1].predict_proba(X)

    @if_delegate_has_method(delegate="_final_estimator")
    def predict_log_proba(self, X):
        """Transform, then predict_log_proba of the final estimator.

        Parameters
        ----------
        X: dict, list, tuple, np.array, sps.matrix or pd.DataFrame
            Feature set with shape=(n_samples, n_features).

        Returns
        -------
        y_pred: np.ndarray
            Predicted class log-probabilities.

        """
        for _, _, transformer in self._iter(with_final=False):
            X, _ = transform_one(transformer, X)

        return self.steps[-1][-1].predict_log_proba(X)

    @if_delegate_has_method(delegate="_final_estimator")
    def decision_function(self, X):
        """Transform, then decision_function of the final estimator.

        Parameters
        ----------
        X: dict, list, tuple, np.array, sps.matrix or pd.DataFrame
            Feature set with shape=(n_samples, n_features).

        Returns
        -------
        y_pred: np.ndarray
            Predicted confidence scores.

        """
        for _, _, transformer in self._iter(with_final=False):
            X, _ = transform_one(transformer, X)

        return self.steps[-1][-1].decision_function(X)

    @if_delegate_has_method(delegate="_final_estimator")
    def score(self, X, y, sample_weight=None):
        """Transform, then score of the final estimator.

        Parameters
        ----------
        X: dict, list, tuple, np.array, sps.matrix or pd.DataFrame
            Feature set with shape=(n_samples, n_features).

        y: int, str, sequence
            - If int: Index of the target column in X.
            - If str: Name of the target column in X.
            - Else: Target column with shape=(n_samples,).

        sample_weight: sequence or None, optional (default=None)
            Sample weights corresponding to y.

        Returns
        -------
        y_pred: float
            Mean accuracy or r2 of self.predict(X) with respect to y.

        """
        for _, _, transformer in self._iter(with_final=False):
            X, y = transform_one(transformer, X, y)

        return self.steps[-1][-1].score(X, y, sample_weight=sample_weight)

    @property
    def transform(self):
        """Transform the data.

        This also works where final estimator is None: all prior
        transformations are applied.

        """
        # _final_estimator is None or has transform, otherwise
        # attribute error handling the None case means we can't
        # use if_delegate_has_method
        return self._custom_transform

    def _custom_transform(self, X=None, y=None):
        for _, _, transformer in self._iter():
            X, y = transform_one(transformer, X, y)

        return variable_return(X, y)
