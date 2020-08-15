# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing the BasePredictor class.

"""

# Standard packages
import numpy as np
from typing import Union, Optional, Sequence
from typeguard import typechecked

# Own modules
from .utils import (
    X_TYPES, Y_TYPES, METRIC_ACRONYMS, flt, check_is_fitted, get_best_score,
    get_model_name, clear, method_to_log, composed, crash
    )


class BasePredictor(object):
    """Data properties and shared methods fot he ATOM and training classes."""

    # Utility properties ==================================================== >>

    @property
    def metric(self):
        """Return a list of the model subclasses."""
        return flt([metric.name for metric in self.metric_])

    @property
    def models_(self):
        """Return a list of the model subclasses."""
        return [getattr(self, model) for model in self.models]

    @property
    def results(self):
        """Return the _results dataframe without empty columns."""
        return self._results.dropna(axis=1, how='all')

    @property
    def winner(self):
        """Return the model subclass that performed best."""
        if self.models:  # Returns None if not fitted
            return self.models_[np.argmax([get_best_score(m) for m in self.models_])]

    @property
    def shape(self):
        return self._data.shape

    @property
    def columns(self):
        return self._data.columns

    @property
    def target(self):
        return self._data.columns[-1]

    # Data properties=== ==================================================== >>

    @property
    def dataset(self):
        return self._data

    @property
    def train(self):
        return self._data[:self._idx[0]]

    @property
    def test(self):
        return self._data[-self._idx[1]:]

    @property
    def X(self):
        return self._data.drop(self.target, axis=1)

    @property
    def y(self):
        return self._data[self.target]

    @property
    def X_train(self):
        return self.train.drop(self.target, axis=1)

    @property
    def X_test(self):
        return self.test.drop(self.target, axis=1)

    @property
    def y_train(self):
        return self.train[self.target]

    @property
    def y_test(self):
        return self.test[self.target]

    # Methods =============================================================== >>

    @composed(crash, method_to_log)
    def calibrate(self, **kwargs):
        """Calibrate the winning model."""
        check_is_fitted(self, 'results')
        return self.winner.calibrate(**kwargs)

    @composed(crash, method_to_log, typechecked)
    def predict(self, X: X_TYPES, **kwargs):
        """Get predictions on new data."""
        check_is_fitted(self, 'results')
        return self.winner.predict(X, **kwargs)

    @composed(crash, method_to_log, typechecked)
    def predict_proba(self, X: X_TYPES, **kwargs):
        """Get probability predictions on new data."""
        check_is_fitted(self, 'results')
        return self.winner.predict_proba(X, **kwargs)

    @composed(crash, method_to_log, typechecked)
    def predict_log_proba(self, X: X_TYPES, **kwargs):
        """Get log probability predictions on new data."""
        check_is_fitted(self, 'results')
        return self.winner.predict_log_proba(X, **kwargs)

    @composed(crash, method_to_log, typechecked)
    def decision_function(self, X: X_TYPES, **kwargs):
        """Get the decision function on new data."""
        check_is_fitted(self, 'results')
        return self.winner.decision_function(X, **kwargs)

    @composed(crash, method_to_log, typechecked)
    def score(self, X: X_TYPES, y: Y_TYPES, **kwargs):
        """Get the score function on new data."""
        check_is_fitted(self, 'results')
        return self.winner.score(X, y, **kwargs)

    @composed(crash, method_to_log, typechecked)
    def scoring(self, metric: Optional[str] = None):
        """Print the trainer's final scoring for a specific metric_.

        If a model shows a `XXX`, it means the metric_ failed for that specific
        model. This can happen if either the metric_ is unavailable for the task
        or if the model does not have a `predict_proba` method while the metric_
        requires it.

        Parameters
        ----------
        metric: string or None, optional (default=None)
            String of one of sklearn's predefined scorers. If None, the metric_(s)
            used to fit the trainer is selected and the bagging results will be
            showed (if used).

        """
        check_is_fitted(self, 'results')

        # If a metric_ acronym is used, assign the correct name
        if metric and metric.lower() in METRIC_ACRONYMS:
            metric = METRIC_ACRONYMS[metric.lower()]

        # Get max length of the model names
        maxlen = max([len(m.longname) for m in self.models_])

        # Get list of scores
        all_scores = [m.scoring(metric) for m in self.models_]

        # Raise an error if the metric_ was invalid for all models
        if metric and all([isinstance(score, str) for score in all_scores]):
            raise ValueError("Invalid metric_ selected!")

        self.log("Results ===================== >>", -2)

        for m in self.models_:
            if not metric:
                out = f"{m.longname:{maxlen}s} --> {m._final_output()}"
            else:
                score = m.scoring(metric)

                # Create string of the score (if wrong metric_ for model -> XXX)
                if isinstance(score, str):
                    out = f"{m.longname:{maxlen}s} --> XXX"
                else:
                    out = f"{m.longname:{maxlen}s} --> {metric}: {round(score, 3)}"

            self.log(out, -2)  # Always print

    @composed(crash, method_to_log, typechecked)
    def clear(self, models: Union[str, Sequence[str]] = 'all'):
        """Clear models from the trainer.

        Removes all traces of a model in the trainer's pipeline (except for the
        errors attribute). This includes the models and results attributes, and
        the model subclass. If all models in the pipeline are removed, the
        metric is reset.

        Parameters
        ----------
        models: str, or sequence, optional (default='all')
            Name of the models to clear from the pipeline. If 'all', clear
            all models.

        """
        # Prepare the models parameter
        if models == 'all':
            keyword = 'Pipeline'
            models = self.models.copy()
        elif isinstance(models, str):
            models = [get_model_name(models)]
            keyword = 'Model ' + models[0]
        else:
            models = [get_model_name(m) for m in models]
            keyword = 'Models ' + ', '.join(models) + ' were'

        clear(self, models)

        # If called from atom, clear also all traces from the trainer
        if hasattr(self, 'trainer'):
            clear(self.trainer, [m for m in models if m in self.trainer.models])
            if not self.models:
                self.trainer = None

        self.log(f"{keyword} cleared successfully!", 1)
