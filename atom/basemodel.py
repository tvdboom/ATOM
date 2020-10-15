# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing the parent class for all model subclasses.

"""

# Standard packages
import pickle
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
from joblib import Parallel, delayed
from typeguard import typechecked
from typing import Optional

# Sklearn
from sklearn.utils import resample
from sklearn.metrics import SCORERS, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV

# Others
from skopt.utils import use_named_args
from skopt.callbacks import DeadlineStopper, DeltaXStopper, DeltaYStopper
from skopt.optimizer import (
    base_minimize, gp_minimize, forest_minimize, gbrt_minimize
)

# Own modules
from .utils import (
    X_TYPES, Y_TYPES, METRIC_ACRONYMS, flt, lst, merge, check_scaling,
    time_to_string, catch_return, transform, composed, get_best_score,
    crash, method_to_log, PlotCallback
)
from .plots import SuccessiveHalvingPlotter, TrainSizingPlotter


# Classes =================================================================== >>

class BaseModel(SuccessiveHalvingPlotter, TrainSizingPlotter):
    """Parent class of all model subclasses.

   Parameters
    ----------
    data: dict
        Dictionary of the data used for this model (train and test).

    T: class
        Class from which the model is called. To avoid having to pass
        attributes through params.

    """

    def __init__(self, **kwargs):
        # Set attributes from ATOM to the model's parent class
        self.__dict__.update(kwargs)
        self.name, self.longname = None, None

        # BO attributes
        self._iter = 0
        self._init_bo = None
        self._pbar = None
        self._cv = 5  # Default value
        self._early_stopping = None
        self._stopped = None
        self.bo = pd.DataFrame(
            columns=['params', 'model', 'score', 'time_iteration', 'time'])
        self.bo.index.name = 'call'

        # BaseModel attributes
        self.best_params = None
        self.estimator = None
        self.time_fit = None
        self.metric_bo = None
        self.time_bo = None
        self.metric_train = None
        self.metric_test = None
        self.metric_bagging = []
        self.mean_bagging = None
        self.std_bagging = None
        self.time_bagging = None
        self.reset_prediction_attributes()

        # Results
        self._results = pd.DataFrame(
            columns=['metric_bo', 'time_bo', 'metric_train',
                     'metric_test', 'time_fit', 'mean_bagging',
                     'std_bagging', 'time_bagging', 'time'])
        self._results.index.name = 'model'

    def __repr__(self):
        repr_ = (
            f"{self.longname}"
            f"\n --> Estimator: {self.estimator.__class__.__name__}"
            f"\n --> Score: {get_best_score(self)}"
        )
        return repr_

    def get_params(self, x):
        """Return a dictionary of the model´s hyperparameters."""
        params = {}
        for i, (key, value) in enumerate(self.params.items()):
            if value[1]:  # If it has decimals...
                params[key] = round(x[i], value[1])
            else:
                params[key] = x[i]

        return params

    def get_init_values(self):
        """Return the default values of the model's hyperparameters."""
        return [value[0] for value in self.params.values()]

    @composed(crash, method_to_log, typechecked)
    def bayesian_optimization(self,
                              n_calls: int = 15,
                              n_initial_points: int = 5,
                              est_params: dict = {},
                              bo_params: dict = {}):
        """Run the bayesian optimization algorithm.

        Search for the best combination of hyperparameters. The function to
        optimize is evaluated either with a K-fold cross-validation on the
        training set or using a different validation set every iteration.

        Parameters
        ----------
        n_calls: int or sequence, optional (default=15)
            Maximum number of iterations of the BO (including `n_initial_points`).
            If 0, skip the BO and fit the model on its default Parameters.
            If sequence, the n-th value will apply to the n-th model in the
            pipeline.

        n_initial_points: int or sequence, optional (default=5)
            Initial number of random tests of the BO before fitting the
            surrogate function. If equal to `n_calls`, the optimizer will
            technically be performing a random search. If sequence, the n-th
            value will apply to the n-th model in the pipeline.

        est_params: dict, optional (default={})
            Additional parameters for the estimator.

        bo_params: dict, optional (default={})
            Additional parameters to for the BO. These can include:
                - base_estimator: str, optional (default='GP')
                    Surrogate model to use. Choose from:
                        - 'GP' for Gaussian Process
                        - 'RF' for Random Forest
                        - 'ET' for Extra-Trees
                        - 'GBRT' for Gradient Boosted Regression Trees
                - max_time: int, optional (default=np.inf)
                    Stop the optimization after `max_time` seconds.
                - delta_x: int or float, optional (default=0)
                    Stop the optimization when `|x1 - x2| < delta_x`.
                - delta_y: int or float, optional (default=0)
                    Stop the optimization if the 5 minima are within `delta_y`.
                - early stopping: int, float or None, optional (default=None)
                    Training will stop if the model didn't improve in last
                    `early_stopping` rounds. If <1, fraction of rounds from the
                    total. If None, no early stopping is performed. Only available
                    for models that allow in-training evaluation.
                - cv: int, optional (default=5)
                    Number of folds for the cross-validation. If 1, the
                    training set will be randomly split in a subtrain and
                    validation set.
                - callbacks: callable or list of callables, optional (default=None)
                    Callbacks for the BO.
                - dimensions: dict, array or None, optional (default=None)
                    Custom hyperparameter space for the bayesian optimization.
                    Can be an array to share the same dimensions across models
                    or a dictionary with the model names as key. If None, ATOM's
                    predefined dimensions are used.
                - plot_bo: bool, optional (default=False)
                    Whether to plot the BO's progress as it runs. Creates a canvas
                    with two plots: the first plot shows the score of every trial
                    and the second shows the distance between the last consecutive
                    steps. Don't forget to call `%matplotlib` at the start of the
                    cell if you are using an interactive notebook!
                - Any other parameter for skopt's optimizer.

        """
        def optimize(**params):
            """Optimization function for the bayesian optimization algorithm.

            Parameters
            ----------
            params: dict
               Model's hyperparameters to be used for this iteration of the BO.

            Returns
            -------
            score: float
                Score achieved by the model.

            """
            def fit_model(train_idx, val_idx, estimator):
                """Fit the model. Function for parallelization.

                Divide the training set in a (sub)train and validation set for this
                fit. Fit the model on custom_fit if exists, else normally. Return
                the score on the validation set.

                Parameters
                ----------
                train_idx: list
                    Indices for the subtrain set.

                val_idx: list
                    Indices for the validation set.

                estimator: class
                    Estimator's instance.

                Returns
                -------
                score: float
                    Score of the fitted model on the validation set.

                """
                X_subtrain = self.X_train.loc[train_idx]
                y_subtrain = self.y_train.loc[train_idx]
                X_val = self.X_train.loc[val_idx]
                y_val = self.y_train.loc[val_idx]

                if hasattr(self, 'custom_fit'):
                    train, test = (X_subtrain, y_subtrain), (X_val, y_val)
                    self.custom_fit(estimator, train, test)

                    # Alert if early stopping was applied
                    if self._cv == 1 and self._stopped:
                        self.T.log("Early stop at iteration {} of {}."
                                   .format(self._stopped[0], self._stopped[1]), 2)
                else:
                    estimator.fit(X_subtrain, y_subtrain)

                # Calculate metrics on the validation set
                return [metric(estimator, X_val, y_val) for metric in self.T.metric_]

            t_iter = time()  # Get current time for start of the iteration

            # Print iteration and time
            self._iter += 1
            if self._iter > n_initial_points:
                call = f'Iteration {self._iter}'
            else:
                call = f'Initial point {self._iter}'

            if self._pbar:
                self._pbar.set_description(call)
            len_ = '-' * (48 - len(call))
            self.T.log(f"{call} {len_}", 2)
            self.T.log(f"Parameters --> {params}", 2)

            estimator = self.get_estimator(params)
            estimator.set_params(**est_params)

            # Same splits per model, but different for every iteration of the BO
            rs = self.T.random_state + self._iter if self.T.random_state else None

            if self._cv == 1:
                # Select test_size from ATOM or use default of 0.2
                t_size = self.T._test_size if hasattr(self.T, '_test_size') else 0.2
                kwargs = dict(test_size=t_size, random_state=rs)
                if self.T.goal.startswith('class'):
                    # Folds are made preserving the % of samples for each class
                    split = StratifiedShuffleSplit(1, **kwargs)
                else:
                    split = ShuffleSplit(1, **kwargs)

                train_idx, val_idx = next(split.split(self.X_train, self.y_train))
                scores = fit_model(train_idx, val_idx, estimator)

            else:  # Use cross validation to get the score
                kwargs = dict(n_splits=self._cv, shuffle=True, random_state=rs)
                if self.T.goal.startswith('class'):
                    # Folds are made preserving the % of samples for each class
                    k_fold = StratifiedKFold(**kwargs)
                else:
                    k_fold = KFold(**kwargs)

                # Parallel loop over fit_model
                jobs = Parallel(self.T.n_jobs)(
                           delayed(fit_model)(i, j, estimator)
                           for i, j in k_fold.split(self.X_train, self.y_train))
                scores = list(np.mean(jobs, axis=0))

            # Append row to the bo attribute
            t = time_to_string(t_iter)
            t_tot = time_to_string(self._init_bo)
            self.bo.loc[call] = {
                'params': params,
                'estimator': estimator,
                'score': flt(scores),
                'time_iteration': t,
                'time': t_tot
            }

            # Update the progress bar
            if self._pbar:
                self._pbar.update(1)

            # Print output of the BO
            out = [f"{m.name}: {scores[i]:.4f}  Best {m.name}: " +
                   f"{max([lst(s)[i] for s in self.bo.score]):.4f}"
                   for i, m in enumerate(self.T.metric_)]
            self.T.log(f"Evaluation --> {'   '.join(out)}", 2)
            self.T.log(f"Time iteration: {t}   Total time: {t_tot}", 2)

            return -scores[0]  # Negative since skopt tries to minimize

        # Running optimization ============================================== >>

        # Check parameters
        if n_initial_points < 1:
            raise ValueError("Invalid value for the n_initial_points parameter. "
                             f"Value should be >0, got {n_initial_points}.")
        if n_calls < n_initial_points:
            raise ValueError("Invalid value for the n_calls parameter. Value "
                             f"should be >n_initial_points, got {n_calls}.")

        self.T.log(f"\n\nRunning BO for {self.longname}...", 1)

        self._init_bo = time()
        if self.T.verbose == 1:
            self._pbar = tqdm(total=n_calls, desc="Random start 1")

        # Prepare callbacks
        callbacks = []
        if bo_params.get('callbacks'):
            if not isinstance(bo_params['callbacks'], (list, tuple)):
                callbacks = [bo_params['callbacks']]
            else:
                callbacks = bo_params['callbacks']
            bo_params.pop('callbacks')

        if bo_params.get('max_time'):
            if bo_params['max_time'] <= 0:
                raise ValueError("Invalid value for the max_time parameter. "
                                 f"Value should be >0, got {bo_params['max_time']}.")
            callbacks.append(DeadlineStopper(bo_params['max_time']))
            bo_params.pop('max_time')

        if bo_params.get('delta_x'):
            if bo_params['delta_x'] < 0:
                raise ValueError("Invalid value for the delta_x parameter. "
                                 f"Value should be >=0, got {bo_params['delta_x']}.")
            callbacks.append(DeltaXStopper(bo_params['delta_x']))
            bo_params.pop('delta_x')

        if bo_params.get('delta_y'):
            if bo_params['delta_y'] < 0:
                raise ValueError("Invalid value for the delta_y parameter. "
                                 f"Value should be >=0, got {bo_params['delta_y']}.")
            callbacks.append(DeltaYStopper(bo_params['delta_y'], n_best=5))
            bo_params.pop('delta_y')

        if 'plot_bo' in bo_params:
            if bo_params['plot_bo']:
                callbacks.append(PlotCallback(self))
            bo_params.pop('plot_bo')

        # Prepare additional arguments
        if bo_params.get('cv'):
            if bo_params['cv'] <= 0:
                raise ValueError("Invalid value for the max_time parameter. "
                                 f"Value should be >=0, got {bo_params['cv']}.")
            self._cv = bo_params['cv']
            bo_params.pop('cv')

        if bo_params.get('early_stopping'):
            if bo_params['early_stopping'] <= 0:
                raise ValueError(
                    "Invalid value for the early_stopping parameter. "
                    f"Value should be >=0, got {bo_params['early_stopping']}.")
            self._early_stopping = bo_params['early_stopping']
            bo_params.pop('early_stopping')

        # Drop dimensions from BO if already in est_params
        for param in est_params:
            if param not in self.get_estimator().get_params().keys():
                raise ValueError(
                    f"Invalid value for the est_params parameter. Got {param} "
                    f"for estimator {self.get_estimator().__class__.__name__}.")
            elif param in self.params:
                self.params.pop(param)

        # Specify model dimensions
        def pre_defined_hyperparameters(x):
            return optimize(**self.get_params(x))

        dimensions = self.get_dimensions()
        func = pre_defined_hyperparameters  # Default optimization func
        if bo_params.get('dimensions'):
            if bo_params['dimensions'].get(self.name):
                dimensions = bo_params.get('dimensions')[self.name]

                @use_named_args(dimensions)
                def custom_hyperparameters(**x):
                    return optimize(**x)
                func = custom_hyperparameters  # Use custom hyperparameters
            bo_params.pop('dimensions')

        # If only 1 initial point, use the model's default parameters
        if n_initial_points == 1:
            bo_params['x0'] = self.get_init_values()

        # Choose base estimator (GP is chosen as default)
        base = bo_params.pop('base_estimator', 'GP')

        # Prepare keyword arguments for the optimizer
        kwargs = dict(
            func=func,
            dimensions=dimensions,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            callback=callbacks,
            n_jobs=self.T.n_jobs,
            random_state=self.T.random_state
        )
        kwargs.update(**bo_params)

        if isinstance(base, str):
            if base.lower() == 'gp':
                optimizer = gp_minimize(**kwargs)
            elif base.lower() == 'et':
                optimizer = forest_minimize(base_estimator='ET', **kwargs)
            elif base.lower() == 'rf':
                optimizer = forest_minimize(base_estimator='RF', **kwargs)
            elif base.lower() == 'gbrt':
                optimizer = gbrt_minimize(**kwargs)
            else:
                raise ValueError(
                    f"Invalid value for the base_estimator parameter, got {base}. "
                    "Value should be one of: 'GP', 'ET', 'RF', 'GBRT'.")
        else:
            optimizer = base_minimize(base_estimator=base, **kwargs)

        if self._pbar:
            self._pbar.close()

        # Optimal parameters found by the BO
        # Return from skopt wrapper to get dict of custom hyperparameter space
        if func is pre_defined_hyperparameters:
            self.best_params = self.get_params(optimizer.x)
        else:
            @use_named_args(dimensions)
            def get_custom_params(**x):
                return x

            self.best_params = get_custom_params(optimizer.x)

        # Optimal score found by the BO
        self.metric_bo = self.bo.score.max(axis=0)

        # Save best model (not yet fitted)
        self.estimator = self.get_estimator(self.best_params)
        self.estimator.set_params(**est_params)

        # Get the BO duration
        self.time_bo = time_to_string(self._init_bo)

        # Print results
        self.T.log(f"\nResults for {self.longname}:{' ':9s}", 1)
        self.T.log("Bayesian Optimization ---------------------------", 1)
        self.T.log(f"Best parameters --> {self.best_params}", 1)
        out = [f"{m.name}: {lst(self.metric_bo)[i]:.4f}"
               for i, m in enumerate(self.T.metric_)]
        self.T.log(f"Best evaluation --> {'   '.join(out)}", 1)
        self.T.log(f"Time elapsed: {self.time_bo}", 1)

    @composed(crash, method_to_log)
    def fit(self, est_params: dict = {}):
        """Fit to the complete training set and get the score on the test set.

        Parameters
        ----------
        est_params: dict, optional (default={})
            Additional parameters for the estimator.

        """
        t_init = time()

        # In case the bayesian_optimization method wasn't called
        if self.estimator is None:
            self.estimator = self.get_estimator()
            self.estimator.set_params(**est_params)

        # Fit the selected model on the complete training set
        if hasattr(self, 'custom_fit'):
            train, test = (self.X_train, self.y_train), (self.X_test, self.y_test)
            self.custom_fit(self.estimator, train, test)
        else:
            self.estimator.fit(self.X_train, self.y_train)

        # Save metric scores on complete training and test set
        self.metric_train = flt([metric(self.estimator, self.X_train, self.y_train)
                                 for metric in self.T.metric_])
        self.metric_test = flt([metric(self.estimator, self.X_test, self.y_test)
                                for metric in self.T.metric_])

        # Print stats ======================================================= >>

        if self.bo.empty:
            self.T.log('\n', 1)  # Print 2 extra lines
            self.T.log(f"Results for {self.longname}:{' ':9s}", 1)
        self.T.log("Fit ---------------------------------------------", 1)
        if self._stopped:
            self.T.log("Early stop at iteration {} of {}."
                       .format(self._stopped[0], self._stopped[1]), 1)
        self.T.log("Score on the train set --> {}"
                   .format('   '.join([f"{m.name}: {lst(self.metric_train)[i]:.4f}"
                                       for i, m in enumerate(self.T.metric_)])), 1)
        self.T.log("Score on the test set  --> {}"
                   .format('   '.join([f"{m.name}: {lst(self.metric_test)[i]:.4f}"
                                       for i, m in enumerate(self.T.metric_)])), 1)

        # Get duration and print to log
        self.time_fit = time_to_string(t_init)
        self.T.log(f"Time elapsed: {self.time_fit}", 1)

    @composed(crash, method_to_log, typechecked)
    def bagging(self, bagging: Optional[int] = 5):
        """Apply a bagging algorithm on the model.

        Take bootstrap samples from the training set and test them on the test
        set to get a distribution of the model's results.

        Parameters
        ----------
        bagging: int or None, optional (default=5)
            Number of data sets (bootstrapped from the training set) to use in
            the bagging algorithm. If None or 0, no bagging is performed.

        """
        t_init = time()

        if bagging < 0:
            raise ValueError("Invalid value for the bagging parameter." +
                             f"Value should be >=0, got {bagging}.")

        self.metric_bagging = []
        for _ in range(bagging):
            # Create samples with replacement
            sample_x, sample_y = resample(self.X_train, self.y_train)

            # Fit on bootstrapped set and predict on the independent test set
            algorithm = self.estimator.fit(sample_x, sample_y)
            scores = flt([metric(algorithm, self.X_test, self.y_test)
                          for metric in self.T.metric_])

            # Append metric result to list
            self.metric_bagging.append(scores)

        # Numpy array for mean and std
        # Separate for multi-metric to transform numpy types in python types
        if len(self.T.metric_) == 1:
            self.mean_bagging = np.mean(self.metric_bagging, axis=0).item()
            self.std_bagging = np.std(self.metric_bagging, axis=0).item()
        else:
            self.mean_bagging = np.mean(self.metric_bagging, axis=0).tolist()
            self.std_bagging = np.std(self.metric_bagging, axis=0).tolist()

        self.T.log("Bagging -----------------------------------------", 1)
        out = [u"{}: {:.4f} \u00B1 {:.4f}"
               .format(m.name, lst(self.mean_bagging)[i], lst(self.std_bagging)[i])
               for i, m in enumerate(self.T.metric_)]
        self.T.log("Score --> " + '   '.join(out), 1)

        # Get duration and print to log
        self.time_bagging = time_to_string(t_init)
        self.T.log(f"Time elapsed: {self.time_bagging}", 1)

    # Prediction methods ==================================================== >>

    def _prediction_methods(self, X, y=None, method='predict', **kwargs):
        """Apply prediction methods on new data.

        First transform the new data and apply the attribute on the best model.
        The model has to have the provided attribute.

        Parameters
        ----------
        X: dict, sequence, np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        y: int, str, sequence, np.array, pd.Series, optional (default=None)
            - If None, the target column is not used in the attribute
            - If int: index of the column of X which is selected as target
            - If string: name of the target column in X
            - Else: data target column with shape=(n_samples,)

        method: str, optional (default='predict')
            Method of the model to be applied.

        **kwargs
            Keyword arguments for the transform method.

        Returns
        -------
        np.array
            Return of the attribute.

        """
        if not hasattr(self.estimator, method):
            raise AttributeError(f"The {self.estimator.__class__.__name__} "
                                 f"estimator doesn't have a {method} method!")

        # When called from the ATOM class, apply all data transformations first
        if hasattr(self, 'pipeline'):
            if kwargs.get('verbose') is None:
                kwargs['verbose'] = self.T.verbose
            X, y = catch_return(transform(self.pipeline, X, y, **kwargs))

        # Scale the data if needed
        if self.need_scaling and not check_scaling(X):
            X = self.T.scaler.transform(X)

        if y is None:
            return getattr(self.estimator, method)(X)
        else:
            return getattr(self.estimator, method)(X, y)

    @composed(crash, method_to_log, typechecked)
    def predict(self, X: X_TYPES, **kwargs):
        """Get predictions on new data."""
        return self._prediction_methods(X, method='predict', **kwargs)

    @composed(crash, method_to_log, typechecked)
    def predict_proba(self, X: X_TYPES, **kwargs):
        """Get probability predictions on new data."""
        return self._prediction_methods(X, method='predict_proba', **kwargs)

    @composed(crash, method_to_log, typechecked)
    def predict_log_proba(self, X: X_TYPES, **kwargs):
        """Get log probability predictions on new data."""
        return self._prediction_methods(X, method='predict_log_proba', **kwargs)

    @composed(crash, method_to_log, typechecked)
    def decision_function(self, X: X_TYPES, **kwargs):
        """Get the decision function on new data."""
        return self._prediction_methods(X, method='decision_function', **kwargs)

    @composed(crash, method_to_log, typechecked)
    def score(self, X: X_TYPES, y: Y_TYPES, **kwargs):
        """Get the score function on new data."""
        return self._prediction_methods(X, y, method='score', **kwargs)

    # Prediction properties ================================================= >>

    def reset_prediction_attributes(self):
        """Clear all the prediction attributes."""
        self._predict_train, self._predict_test = None, None
        self._predict_proba_train, self._predict_proba_test = None, None
        self._predict_log_proba_train, self._predict_log_proba_test = None, None
        self._dec_func_train, self._dec_func_test = None, None
        self._score_train, self._score_test = None, None

    @property
    def predict_train(self):
        if self._predict_train is None:
            self._predict_train = self.estimator.predict(self.X_train)
        return self._predict_train

    @property
    def predict_test(self):
        if self._predict_test is None:
            self._predict_test = self.estimator.predict(self.X_test)
        return self._predict_test

    @property
    def predict_proba_train(self):
        if self._predict_proba_train is None:
            self._predict_proba_train = self.estimator.predict_proba(self.X_train)
        return self._predict_proba_train

    @property
    def predict_proba_test(self):
        if self._predict_proba_test is None:
            self._predict_proba_test = self.estimator.predict_proba(self.X_test)
        return self._predict_proba_test

    @property
    def predict_log_proba_train(self):
        if self._predict_log_proba_train is None:
            self._predict_log_proba_train = np.log(self.predict_proba_train)
        return self._predict_log_proba_train

    @property
    def predict_log_proba_test(self):
        if self._predict_log_proba_test is None:
            self._predict_log_proba_test = np.log(self.predict_proba_test)
        return self._predict_log_proba_test

    @property
    def decision_function_train(self):
        if self._dec_func_train is None:
            self._dec_func_train = self.estimator.decision_function(self.X_train)
        return self._dec_func_train

    @property
    def decision_function_test(self):
        if self._dec_func_test is None:
            self._dec_func_test = self.estimator.decision_function(self.X_test)
        return self._dec_func_test

    @property
    def score_train(self):
        if self._score_train is None:
            self._score_train = self.estimator.score(self.X_train, self.y_train)
        return self._score_train

    @property
    def score_test(self):
        if self._score_test is None:
            self._score_test = self.estimator.score(self.X_test, self.y_test)
        return self._score_test

    # Properties ==================================================== >>

    @property
    def results(self):
        """Return results without empty columns."""
        return self._results.dropna(axis=1, how='all')

    @property
    def dataset(self):
        if self.need_scaling and not check_scaling(self.T.X):
            return merge(self.T.scaler.transform(self.T.X), self.y)
        else:
            return self.T.dataset

    @property
    def train(self):
        if self.need_scaling and not check_scaling(self.T.X):
            return merge(self.T.scaler.transform(self.T.X_train), self.T.y_train)
        else:
            return self.T.train

    @property
    def test(self):
        if self.need_scaling and not check_scaling(self.T.X):
            return merge(self.T.scaler.transform(self.T.X_test), self.y_test)
        else:
            return self.T.test

    @property
    def X(self):
        if self.need_scaling and not check_scaling(self.T.X):
            return self.T.scaler.transform(self.T.X)
        else:
            return self.T.X

    @property
    def y(self):
        return self.T.y

    @property
    def X_train(self):
        if self.need_scaling and not check_scaling(self.T.X):
            return self.T.scaler.transform(self.T.X_train)
        else:
            return self.T.X_train

    @property
    def X_test(self):
        if self.need_scaling and not check_scaling(self.T.X):
            return self.T.scaler.transform(self.T.X_test)
        else:
            return self.T.X_test

    @property
    def y_train(self):
        return self.T.y_train

    @property
    def y_test(self):
        return self.T.y_test

    @property
    def shape(self):
        return self.T.shape

    @property
    def columns(self):
        return self.T.columns

    @property
    def target(self):
        return self.T.target

    @property
    def categories(self):
        return self.T.categories

    @property
    def n_categories(self):
        return self.T.n_categories

    # Utility methods ======================================================= >>

    def _final_output(self):
        """Returns the model's final output as a string."""
        # If bagging was used, we use a different format
        if self.mean_bagging is None:
            out = '   '.join([f"{m.name}: {lst(self.metric_test)[i]:.3f}"
                              for i, m, in enumerate(self.T.metric_)])
        else:
            out = '   '.join([f"{m.name}: {lst(self.mean_bagging)[i]:.3f}"
                              u" \u00B1 "
                              f"{lst(self.std_bagging)[i]:.3f}"
                              for i, m in enumerate(self.T.metric_)])

        # Annotate if model overfitted when train 20% > test
        metric_train = lst(self.metric_train)
        metric_test = lst(self.metric_test)
        if metric_train[0] - 0.2 * metric_train[0] > metric_test[0]:
            out += ' ~'

        return out

    @composed(crash)
    def calibrate(self, **kwargs):
        """Calibrate the model.

        Applies probability calibration on the winning model. The calibration is
        done with the CalibratedClassifierCV class from sklearn. The estimator will
        be trained via cross-validation on a subset of the training data, using the
        rest to fit the calibrator. The new classifier will replace the `estimator`
        attribute. All prediction attributes will reset.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments for the CalibratedClassifierCV instance.
            Using cv='prefit' will use the trained model and fit the calibrator on
            the test set. Note that doing this will result in data leakage in the
            test set. Use this only if you have another, independent set for testing.

        """
        if self.T.goal.startswith('reg'):
            raise PermissionError(
                "The calibrate method is only available for classification tasks!")

        cal = CalibratedClassifierCV(self.estimator, **kwargs)
        if kwargs.get('cv') != 'prefit':
            self.estimator = cal.fit(self.X_train, self.y_train)
        else:
            self.estimator = cal.fit(self.X_test, self.y_test)

        # Reset all prediction properties since we changed the model attribute
        self.reset_prediction_attributes()

    @composed(crash, typechecked)
    def scoring(self, metric: Optional[str] = None, dataset: str = 'test'):
        """Get the scoring of a specific metric on the test set.

        Parameters
        ----------
        metric: str, optional (default=None)
            Name of the metric to calculate. Choose from any of sklearn's SCORERS or
            one of the following custom metrics: 'cm', 'tn', 'fp', 'fn', 'tp',
            'lift', 'fpr', 'tpr' or 'sup'. If None, returns ATOM's final results for
            the model (ignores `dataset`).

        dataset: str, optional (default='test')
            Data set on which to calculate the metric. Options are 'train' or 'test'.

        """
        metric_opts = ['cm', 'confusion_matrix', 'tn', 'fp', 'fn',
                       'tp', 'lift', 'fpr', 'tpr', 'sup'] + list(SCORERS)

        # Check metric parameter
        if metric is None:
            return self._final_output()
        elif metric.lower() in METRIC_ACRONYMS:
            metric = METRIC_ACRONYMS[metric.lower()]
        elif metric.lower() not in metric_opts:
            raise ValueError("Unknown value for the metric parameter, "
                             f"got {metric}. Try one of {', '.join(metric_opts)}.")

        # Check set parameter
        dataset = dataset.lower()
        if dataset not in ('train', 'test'):
            raise ValueError("Unknown value for the dataset parameter. "
                             "Choose between 'train' or 'test'.")

        try:
            if metric.lower() in ('cm', 'confusion_matrix'):
                return confusion_matrix(getattr(self, f'y_{dataset}'),
                                        getattr(self, f'predict_{dataset}'))
            elif metric.lower() == 'tn':
                return int(self.scoring('confusion_matrix').ravel()[0])
            elif metric.lower() == 'fp':
                return int(self.scoring('confusion_matrix').ravel()[1])
            elif metric.lower() == 'fn':
                return int(self.scoring('confusion_matrix').ravel()[2])
            elif metric.lower() == 'tp':
                return int(self.scoring('confusion_matrix').ravel()[3])
            elif metric.lower() == 'lift':
                tn, fp, fn, tp = self.scoring('confusion_matrix').ravel()
                return float((tp / (tp + fp)) / ((tp + fn) / (tp + tn + fp + fn)))
            elif metric.lower() == 'fpr':
                tn, fp, _, _ = self.scoring('confusion_matrix').ravel()
                return float(fp / (fp + tn))
            elif metric.lower() == 'tpr':
                _, _, fn, tp = self.scoring('confusion_matrix').ravel()
                return float(tp / (tp + fn))
            elif metric.lower() == 'sup':
                tn, fp, fn, tp = self.scoring('confusion_matrix').ravel()
                return float((tp + fp) / (tp + fp + fn + tn))

            # Calculate the scorer via _score_func to use the prediction properties
            if type(SCORERS[metric]).__name__ == '_ThresholdScorer':
                if self.T.task.startswith('reg'):
                    y_pred = getattr(self, f'predict_{dataset}')
                elif hasattr(self.estimator, 'decision_function'):
                    y_pred = getattr(self, f'decision_function_{dataset}')
                else:
                    y_pred = getattr(self, f'predict_proba_{dataset}')
                    if self.T.task.startswith('bin'):
                        y_pred = y_pred[:, 1]
            elif type(SCORERS[metric]).__name__ == '_ProbaScorer':
                if hasattr(self.estimator, 'predict_proba'):
                    y_pred = getattr(self, f'predict_proba_{dataset}')
                    if self.T.task.startswith('bin'):
                        y_pred = y_pred[:, 1]
                else:
                    y_pred = getattr(self, f'decision_function_{dataset}')
            else:
                y_pred = getattr(self, f'predict_{dataset}')

            # Calculate metric on the test set
            return SCORERS[metric]._sign * SCORERS[metric]._score_func(
                getattr(self, f'y_{dataset}'), y_pred, **SCORERS[metric]._kwargs)

        except (ValueError, TypeError):
            return f"Invalid metric for a {self.name} model with {self.T.task} task!"

    @composed(crash, method_to_log, typechecked)
    def save_estimator(self, filename: Optional[str] = None):
        """Save the estimator to a pickle file.

        Parameters
        ----------
        filename: str, optional (default=None)
            Name of the file to save. If None or 'auto', the default name is used.

        """
        if not filename:
            filename = self.estimator.__class__.__name__
        elif filename == 'auto' or filename.endswith('/auto'):
            filename = filename.replace('auto', self.estimator.__class__.__name__)

        with open(filename, 'wb') as file:
            pickle.dump(self.estimator, file)
        self.T.log(f"{self.longname} estimator saved successfully!", 1)
