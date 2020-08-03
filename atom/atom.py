# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing the main ATOM class.

"""


# Standard packages
import numpy as np
import pandas as pd
from typeguard import typechecked
from typing import Union, Optional, Sequence, List, Tuple
from pandas_profiling import ProfileReport

# Own modules
from .basepredictor import BasePredictor
from .basetrainer import BaseTrainer
from .data_cleaning import (
    StandardCleaner, Scaler, Imputer, Encoder, Outliers, Balancer
    )
from .feature_engineering import FeatureGenerator, FeatureSelector
from .training import (
    TrainerClassifier, TrainerRegressor,
    SuccessiveHalvingClassifier, SuccessiveHalvingRegressor,
    TrainSizingClassifier, TrainSizingRegressor
    )
from .plots import ATOMPlotter
from .utils import (
    CAL, X_TYPES, Y_TYPES, TRAIN_TYPES, composed, crash, method_to_log, merge,
    check_property, check_scaling, infer_task, catch_return, variable_return, clear
    )


class ATOM(BasePredictor, ATOMPlotter):
    """ATOM base class.

    The ATOM class (parent for the ATOMClassifier and ATOMRegressor classes) is a
    convenient wrapper for all data_cleaning, feature_engineering and training methods
    in this package. Provide the dataset to the class, and apply all transformations
    and model management from here.

    Parameters
    ----------
    X: dict, sequence, np.array or pd.DataFrame
        Dataset containing the features, with shape=(n_samples, n_features)

    y: int, str, sequence, np.array or pd.Series
        - If int: Index of the column of X which is selected as target.
        - If str: Name of the target column in X.
        - Else: Data target column with shape=(n_samples,).

    n_rows: int or float
        - If <=1: Fraction of the data to use.
        - If >1: Number of rows to use.

    test_size: float
        Split fraction for the training and test set.

    """

    @composed(crash, method_to_log)
    def __init__(self, X, y, n_rows, test_size):
        """Prepares input, runs StandardCleaner and splits in train and test sets."""
        # Data attributes
        self._idx = [None, None]  # Train and test set sizes
        self._sizes = None

        # Method attributes
        self.profile = None
        self.scaler = None
        self.imputer = None
        self.encoder = None
        self.outlier = None
        self.balancer = None
        self.feature_generator = None
        self.feature_selector = None

        # Training attributes
        self.trainer = None
        self.models = []
        self.metric = None
        self.errors = {}
        self._results = pd.DataFrame(
            columns=['name', 'score_bo', 'time_bo',
                     'score_train', 'score_test', 'time_fit',
                     'mean_bagging', 'std_bagging', 'time_bagging', 'time'])
        self._results.index.name = 'model'

        # Check input Parameters ============================================ >>

        if n_rows <= 0:
            raise ValueError("Invalid value for the n_rows parameter. " +
                             f"Value should be >0, got {n_rows}.")
        elif n_rows > len(X):
            n_rows = len(X)

        if test_size <= 0 or test_size >= 1:
            raise ValueError("Invalid value for the test_size parameter. " +
                             f"Value should be between 0 and 1, got {test_size}.")
        else:
            self._test_size = test_size

        self.log("<< ================== ATOM ================== >>", 1)

        X, y = self._prepare_input(X, y)

        # Assign the algorithm's task
        self.task = infer_task(y, goal=self.goal)
        self.log(f"Algorithm task: {self.task}.", 1)
        if self.n_jobs > 1:
            self.log(f"Parallel processing with {self.n_jobs} cores.", 1)

        # List of data types ATOM can't handle
        prohibited_types = ['datetime64', 'datetime64[ns]', 'timedelta[ns]']

        # Whether to map the target column to numerical values
        map_target = False if self.goal.startswith('reg') else True

        # Apply the standard cleaning steps
        self.standard_cleaner = \
            StandardCleaner(prohibited_types=prohibited_types,
                            map_target=map_target,
                            verbose=self.verbose,
                            logger=self.logger)
        X_y = merge(*self.standard_cleaner.transform(X, y))
        self.mapping = self.standard_cleaner.mapping

        # Get number of rows, shuffle the dataset and reset indices
        kwargs = {'frac': n_rows} if n_rows <= 1 else {'n': int(n_rows)}
        self._data = X_y.sample(random_state=self.random_state, **kwargs)
        self._data.reset_index(drop=True, inplace=True)
        self._idx[1] = int(self._test_size * len(self.dataset))
        self._idx[0] = len(self.dataset) - self._idx[1]
        self.stats(1)  # Print data stats

    # Utility methods ======================================================= >>

    @composed(crash, method_to_log)
    def stats(self, _vb: int = -2):
        """Print some information about the dataset.

        Parameters
        ----------
        _vb: int, optional (default=-2)
            Internal parameter to always print if the user calls this method.

        """
        self.log("\nDataset stats ================= >>", _vb)
        self.log(f"Shape: {self.dataset.shape}", _vb)

        nans = self.dataset.isna().sum().sum()
        if nans > 0:
            self.log(f"Missing values: {nans}", _vb)
        categorical = self.X.select_dtypes(include=['category', 'object']).shape[1]
        if categorical > 0:
            self.log(f"Categorical columns: {categorical}", _vb)

        self.log(f"Scaled: {check_scaling(self.X)}", _vb)
        self.log("----------------------------------", _vb)
        self.log(f"Size of training set: {len(self.train)}", _vb)
        self.log(f"Size of test set: {len(self.test)}", _vb)

        # Print count of target classes
        if self.task != 'regression':
            # Create dataframe with stats per target class
            index = []
            for key, value in self.mapping.items():
                try:
                    list_ = list(map(int, self.mapping))
                    if list_ != list(self.mapping.values()):
                        index.append(str(value) + ': ' + key)
                    else:
                        index.append(value)
                except ValueError:
                    index.append(str(value) + ': ' + key)

            stats = pd.DataFrame(columns=[' total', ' train_set', ' test_set'])

            # Count number of occurrences in all sets
            uq_train, c_train = np.unique(self.y_train, return_counts=True)
            uq_test, c_test = np.unique(self.y_test, return_counts=True)
            keys, values = '', []
            for key, value in self.mapping.items():
                # If set has 0 instances of that class the array is empty
                idx_train = np.where(uq_train == value)[0]
                train = c_train[idx_train[0]] if len(idx_train) != 0 else 0
                idx_test = np.where(uq_test == value)[0]
                test = c_test[idx_test[0]] if len(idx_test) != 0 else 0
                stats = stats.append({' total': train + test,
                                      ' train_set': train,
                                      ' test_set': test}, ignore_index=True)

                keys += key + ':'
                values.append(train + test)

            stats.set_index(pd.Index(index), inplace=True)

            string = ''  # Class balance string for values
            for i in values:
                string += str(round(i/(train+test), 1)) + ':'

            self.log("----------------------------------", _vb + 1)
            if len(self.mapping) < 5:  # Gets ugly for too many classes
                # [-1] to remove last colon
                self.log(f"Class balance: {keys[:-1]} <==> {string[:-1]}", _vb + 1)
            self.log(f"Instances in {self.target} per class:", _vb + 1)
            self.log(stats.to_markdown(), _vb + 1)

        self.log('', _vb)  # Add always an empty line at the end

    @composed(crash, method_to_log, typechecked)
    def report(self,
               df: str = 'dataset',
               n_rows: Optional[Union[int, float]] = None,  # float for 1e3...
               filename: Optional[str] = None):
        """Create an extensive profile analysis of the data.

        The profile report is rendered in HTML5 and CSS3. Note that this
        method can be slow for rows>10k.

        Parameters
        ----------
        df: str, optional(default='dataset')
            Name of the data class property to get the report from.

        n_rows: int or None, optional(default=None)
            Number of (randomly picked) rows to process. None for all rows.

        filename: str or None, optional (default=None)
            Name of the file when saved (as .html). None to not save anything.

        """
        # If rows=None, select all rows in the dataframe
        rows = getattr(self, df).shape[0] if n_rows is None else int(n_rows)

        self.log("Creating profile report...", 1)

        self.profile = ProfileReport(getattr(self, df).sample(rows))
        try:  # Render if possible (for notebooks)
            from IPython.display import display
            display(self.profile)
        except ModuleNotFoundError:
            pass

        if filename:
            if not filename.endswith('.html'):
                filename = filename + '.html'
            self.profile.to_file(filename)
            self.log("Report saved successfully!", 1)

    @composed(crash, method_to_log, typechecked)
    def transform(self,
                  X: X_TYPES,
                  y: Y_TYPES = None,
                  standard_cleaner: bool = True,
                  scale: bool = True,
                  impute: bool = True,
                  encode: bool = True,
                  outliers: bool = False,
                  balance: bool = False,
                  feature_generation: bool = True,
                  feature_selection: bool = True,
                  verbose: int = None):
        """Apply all data transformations in ATOM's pipeline to new data.

        Outliers and balance are False by default since they should only be used for
        the training set.

        Parameters
        ----------
        X: dict, sequence, np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        y: int, str, sequence, np.array or pd.Series, optional (default=None)
            - If None: y is not used in the transformation.
            - If int: Index of the column of X which is selected as target.
            - If str: Name of the target column in X.
            - Else: Data target column with shape=(n_samples,).

        standard_cleaner: bool, optional (default=True)
            Whether to apply the standard cleaning step in the transformer.

        scale: bool, optional (default=True)
            Whether to apply the scaler step in the transformer.

        impute: bool, optional (default=True)
            Whether to apply the imputer step in the transformer.

        encode: bool, optional (default=True)
            Whether to apply the encoder step in the transformer.

        outliers: bool, optional (default=False)
            Whether to apply the outlier step in the transformer.

        balance: bool, optional (default=False)
            Whether to apply the balancer step in the transformer.

        feature_generation: bool, optional (default=True)
            Whether to apply the feature generator step in the transformer.

        feature_selection: bool, optional (default=True)
            Whether to apply the feature selector step in the transformer.

        verbose: int, optional (default=None)
            Verbosity level of the output. If None, it uses ATOM's verbosity.

        Returns
        -------
        X: pd.DataFrame
            Transformed dataset.

        y: pd.Series
            Transformed target column. Only returned if provided.

        """
        # Dict of all data cleaning and feature selection classes and their
        # respective attributes in the ATOM class
        steps = dict(standard_cleaner='standard_cleaner',
                     scale='scaler',
                     impute='imputer',
                     encode='encoder',
                     outliers='outlier',
                     balance='balancer',
                     feature_generation='feature_generator',
                     feature_selection='feature_selector')

        # Check verbose parameter
        if verbose is not None and (verbose < 0 or verbose > 2):
            raise ValueError("Invalid value for the verbose parameter." +
                             f"Value should be between 0 and 2, got {verbose}.")

        for key, value in steps.items():
            if eval(key) and getattr(self, value, None):
                # If verbose is specified, change the class verbosity
                if verbose is not None:
                    getattr(self, value).verbose = verbose

                # Some transformers return no y, but we need the original
                X, y_returned = catch_return(getattr(self, value).transform(X, y))
                y = y if y_returned is None else y_returned

        return variable_return(X, y)

    # Data cleaning methods ================================================= >>

    @composed(crash, method_to_log)
    def scale(self):
        """Scale the features.

        Scale the features in the dataset to mean=0 and std=1. The scaler is
        fitted only on the training set to avoid data leakage.

        """
        self.scaler = Scaler(verbose=self.verbose, logger=self.logger)
        self.scaler.fit(self.X_train)

        self.X = self.scaler.transform(self.X)

    @composed(crash, method_to_log, typechecked)
    def impute(self,
               strat_num: Union[int, float, str] = 'drop',
               strat_cat: str = 'drop',
               min_frac_rows: float = 0.5,
               min_frac_cols: float = 0.5,
               missing: Optional[Union[int, float, str, list]] = None):
        """Handle missing values in the dataset.

        Impute or remove missing values according to the selected strategy.
        Also removes rows and columns with too many missing values. The imputer
        is fitted only on the training set to avoid data leakage.

        See the data_cleaning.py module for a description of the parameters.

        """
        self.imputer = Imputer(strat_num=strat_num,
                               strat_cat=strat_cat,
                               min_frac_rows=min_frac_rows,
                               min_frac_cols=min_frac_cols,
                               missing=missing,
                               verbose=self.verbose,
                               logger=self.logger)
        self.imputer.fit(self.X_train, self.y_train)

        X, y = self.imputer.transform(self.X, self.y)
        self.dataset = merge(X, y)
        self.dataset.reset_index(drop=True, inplace=True)

    @composed(crash, method_to_log, typechecked)
    def encode(self,
               max_onehot: Optional[int] = 10,
               encode_type: str = 'Target',
               frac_to_other: Optional[float] = None,
               **kwargs):
        """Perform encoding of categorical features.

        The encoding type depends on the number of unique values in the column:
            - If n_unique=2, use label-encoding.
            - If 2 < n_unique <= max_onehot, use one-hot-encoding.
            - If n_unique > max_onehot, use 'encode_type'.

        Also replaces classes with low occurrences with the value 'other' in
        order to prevent too high cardinality. Categorical features are defined as
        all columns whose dtype.kind not in 'ifu'. The encoder is fitted only on
        the training set to avoid data leakage.

        See the data_cleaning.py module for a description of the parameters.

        """
        self.encoder = Encoder(max_onehot=max_onehot,
                               encode_type=encode_type,
                               frac_to_other=frac_to_other,
                               verbose=self.verbose,
                               logger=self.logger,
                               **kwargs)
        self.encoder.fit(self.X_train, self.y_train)

        self.X = self.encoder.transform(self.X)

    @composed(crash, method_to_log, typechecked)
    def outliers(self,
                 strategy: Union[int, float, str] = 'drop',
                 max_sigma: Union[int, float] = 3,
                 include_target: bool = False):
        """Remove or replace outliers in the training set.

        Outliers are defined as values that lie further than
        `max_sigma` * standard_deviation away from the mean of the column. Only
        outliers from the training set are removed to maintain an original
        sample of target values in the test set.

        See the data_cleaning.py module for a description of the parameters.

        """
        self.outlier = Outliers(strategy=strategy,
                                max_sigma=max_sigma,
                                include_target=include_target,
                                verbose=self.verbose,
                                logger=self.logger)

        X_train, y_train = self.outlier.transform(self.X_train, self.y_train)
        self.train = merge(X_train, y_train)
        self.dataset.reset_index(drop=True, inplace=True)

    @composed(crash, method_to_log, typechecked)
    def balance(self,
                oversample: Optional[Union[int, float, str]] = None,
                undersample: Optional[Union[int, float, str]] = None,
                n_neighbors: int = 5):
        """Balance the target categories in the training set.

        Balance the number of instances per target category in the training set.
        Using oversample and undersample at the same time or not using any will
        raise an exception. Only the training set is balanced in order to maintain
        the original distribution of target categories in the test set. Use only for
        classification tasks.

        See the data_cleaning.py module for a description of the parameters.

        """
        if not self.goal.startswith('class'):
            raise PermissionError(
                "The balance method is only available for classification tasks!")

        self.balancer = Balancer(oversample=oversample,
                                 undersample=undersample,
                                 n_neighbors=n_neighbors,
                                 n_jobs=self.n_jobs,
                                 verbose=self.verbose,
                                 logger=self.logger,
                                 random_state=self.random_state)

        # Add mapping from ATOM to balancer for cleaner printing
        self.balancer.mapping = self.mapping

        X_train, y_train = self.balancer.transform(self.X_train, self.y_train)
        self.train = merge(X_train, y_train)
        self.dataset.reset_index(drop=True, inplace=True)

    # Feature engineering methods =========================================== >>

    @composed(crash, method_to_log, typechecked)
    def feature_generation(self,
                           strategy: str = 'DFS',
                           n_features: Optional[int] = None,
                           generations: int = 20,
                           population: int = 500,
                           operators: Optional[Union[str, Sequence[str]]] = None):
        """Create new non-linear features.

        Use Deep feature Synthesis or a genetic algorithm to create new combinations
        of existing features to capture the non-linear relations between the original
        features.

        See the feature_engineering.py module for a description of the parameters.

        """
        self.feature_generator = \
            FeatureGenerator(strategy=strategy,
                             n_features=n_features,
                             generations=generations,
                             population=population,
                             operators=operators,
                             n_jobs=self.n_jobs,
                             verbose=self.verbose,
                             logger=self.logger,
                             random_state=self.random_state)
        self.feature_generator.fit(self.X_train, self.y_train)

        self.X = self.feature_generator.transform(self.X)

        # Attach attributes to the ATOM class
        if strategy.lower() in ('gfg', 'genetic'):
            for attr in ['symbolic_transformer', 'genetic_features']:
                setattr(self, attr, getattr(self.feature_generator, attr))

    @composed(crash, method_to_log, typechecked)
    def feature_selection(self,
                          strategy: Optional[str] = None,
                          solver: Optional[Union[str, callable]] = None,
                          n_features: Optional[Union[int, float]] = None,
                          max_frac_repeated: Optional[Union[int, float]] = 1.,
                          max_correlation: Optional[float] = 0.98,
                          **kwargs):
        """Apply feature selection techniques.

        Remove features according to the selected strategy. Ties between
        features with equal scores will be broken in an unspecified way.
        Also removes features with too low variance and finds pairs of
        collinear features based on the Pearson correlation coefficient. For
        each pair above the specified limit (in terms of absolute value), it
        removes one of the two.

        Note that the RFE and RFECV strategies don't work when the solver is a
        CatBoost model due to incompatibility of the APIs. If the run method has
        already been called before running RFECV, the scoring parameter will be set
        to the selected metric (if not explicitly provided).

        The sklearn objects are attached as attributes to the ATOM instance under
        the names: univariate, pca, sfm, rfe and rfecv.

        After running the method, the FeatureSelector's plot methods can be
        called directly from ATOM.

        See the feature_engineering.py module for a description of the parameters.

        """
        if isinstance(strategy, str):
            if strategy.lower() == 'univariate' and solver is None:
                if self.goal.startswith('reg'):
                    solver = 'f_regression'
                else:
                    solver = 'f_classif'
            elif strategy.lower() in ['sfm', 'rfe', 'rfecv']:
                if solver is None and self.winner:
                    solver = self.winner.model
                elif isinstance(solver, str):
                    # In case the user already filled the task...
                    if not solver.endswith('_class') and not solver.endswith('_reg'):
                        solver += '_reg' if self.task.startswith('reg') else '_class'

            # If the run method was called before, use the main metric for RFECV
            if strategy.lower() == 'rfecv':
                if self.metric and 'scoring' not in kwargs:
                    kwargs['scoring'] = self.metric[0]

        self.feature_selector = \
            FeatureSelector(strategy=strategy,
                            solver=solver,
                            n_features=n_features,
                            max_frac_repeated=max_frac_repeated,
                            max_correlation=max_correlation,
                            n_jobs=self.n_jobs,
                            verbose=self.verbose,
                            logger=self.logger,
                            random_state=self.random_state,
                            **kwargs)
        self.feature_selector.fit(self.X_train, self.y_train)

        # Attach used attributes to the ATOM class
        attrs = ['collinear', 'univariate', 'pca', 'sfm', 'rfe', 'rfecv']
        for attr in attrs:
            if getattr(self.feature_selector, attr) is not None:
                setattr(self, attr, getattr(self.feature_selector, attr))

        self.X = self.feature_selector.transform(self.X)

    # Training methods ====================================================== >>

    def _run(self):
        """Run the trainer.

        If all models failed, catch the errors and pass them to the ATOM instance
        before raising the exception. If successful run, update all relevant
        attributes and methods. Only the run method allows for subsequent calls
        without erasing previous information.

        """
        try:
            self.trainer._data = self._data
            self.trainer._idx = self._idx
            self.trainer.run()
        except ValueError as exception:
            raise exception
        else:
            # Attach mapping from ATOM to the trainer instance (for plots)
            self.trainer.mapping = self.mapping

            # Update attributes
            self.models += [m for m in self.trainer.models if m not in self.models]
            self.metric = self.trainer.metric

            # If SuccessiveHalving or TrainSizing, replace the results
            if isinstance(self.trainer.results.index, pd.MultiIndex):
                self._results = self.trainer._results
            else:  # If Trainer, add them row by row
                for idx, row in self.trainer._results.iterrows():
                    self._results.loc[idx] = row

            for model in self.trainer.models:
                self.errors.pop(model, None)  # Remove model from errors if there
                setattr(self, model, getattr(self.trainer, model))
                setattr(self, model.lower(), getattr(self.trainer, model.lower()))

                # Add transform method to model subclasses
                setattr(getattr(self, model), 'transform', self.transform)
                setattr(getattr(self, model.lower()), 'transform', self.transform)

        finally:
            # Catch errors and pass them to ATOM's attribute
            for model, error in self.trainer.errors.items():
                self.errors[model] = error

    @composed(crash, method_to_log, typechecked)
    def run(self,
            models: Union[str, List[str], Tuple[str]],
            metric: Optional[Union[CAL, Sequence[CAL]]] = None,
            greater_is_better: Union[bool, Sequence[bool]] = True,
            needs_proba: Union[bool, Sequence[bool]] = False,
            needs_threshold: Union[bool, Sequence[bool]] = False,
            n_calls: Union[int, Sequence[int]] = 0,
            n_random_starts: Union[int, Sequence[int]] = 5,
            bo_params: dict = {},
            bagging: Optional[Union[int, Sequence[int]]] = None):
        """Fit the models to the dataset in a direct fashion.

        Contrary to the Trainer class in training.py, this method allows
        subsequent runs and stores all results as attributes (only the model
        subclasses are overwritten if the same model is rerun).

        See the basetrainer.py module for a description of the parameters.

        """
        # If this is the first direct run, clear all previous results
        if not type(self.trainer).__name__.startswith('Trainer'):
            clear(self, self.models[:])

        elif metric is None:  # Assign the existing metric
            metric = self.metric

        else:  # Check that the selected metric is the same as previous run
            metric = BaseTrainer._prepare_metric(
                metric, greater_is_better, needs_proba, needs_threshold)

            for metric1, metric2 in zip(metric, self.metric):
                if metric1.name != metric2.name:
                    raise ValueError(
                        f"Invalid metric parameter! Metric {metric2.name} " +
                        f"is already in use, got {metric1.name}. Use the clear " +
                        "method before selecting a new metric.")

        params = (models, metric, greater_is_better, needs_proba, needs_threshold,
                  n_calls, n_random_starts, bo_params, bagging, self.n_jobs,
                  self.verbose, self.warnings, self.logger, self.random_state)

        if self.goal.startswith('class'):
            self.trainer = TrainerClassifier(*params)
        else:
            self.trainer = TrainerRegressor(*params)

        self._run()

    @composed(crash, method_to_log, typechecked)
    def successive_halving(
            self,
            models: Union[str, List[str], Tuple[str]],
            metric: Optional[Union[CAL, Sequence[CAL]]] = None,
            greater_is_better: Union[bool, Sequence[bool]] = True,
            needs_proba: Union[bool, Sequence[bool]] = False,
            needs_threshold: Union[bool, Sequence[bool]] = False,
            skip_iter: int = 0,
            n_calls: Union[int, Sequence[int]] = 0,
            n_random_starts: Union[int, Sequence[int]] = 5,
            bo_params: dict = {},
            bagging: Optional[Union[int, Sequence[int]]] = None):
        """Fit the models to the dataset in a successive halving fashion.

        If you want to compare similar models, you can choose to use a successive
        halving approach to run the pipeline. This technique is a bandit-based
        algorithm that fits N models to 1/N of the data. The best half are selected
        to go to the next iteration where the process is repeated. This continues
        until only one model remains, which is fitted on the complete dataset.
        Beware that a model's performance can depend greatly on the amount of data
        on which it is trained. For this reason, we recommend only to use this
        technique with similar models, e.g. only using tree-based models.

        See the basetrainer.py module for a description of the parameters.

        """
        clear(self, self.models[:])

        params = (models, metric, greater_is_better, needs_proba, needs_threshold,
                  skip_iter, n_calls, n_random_starts, bo_params, bagging,
                  self.n_jobs, self.verbose, self.warnings, self.logger,
                  self.random_state)

        if self.goal.startswith('class'):
            self.trainer = SuccessiveHalvingClassifier(*params)
        else:
            self.trainer = SuccessiveHalvingRegressor(*params)

        self._run()

    @composed(crash, method_to_log, typechecked)
    def train_sizing(self,
                     models: Union[str, List[str], Tuple[str]],
                     metric: Optional[Union[CAL, Sequence[CAL]]] = None,
                     greater_is_better: Union[bool, Sequence[bool]] = True,
                     needs_proba: Union[bool, Sequence[bool]] = False,
                     needs_threshold: Union[bool, Sequence[bool]] = False,
                     train_sizes: TRAIN_TYPES = np.linspace(0.2, 1.0, 5),
                     n_calls: Union[int, Sequence[int]] = 0,
                     n_random_starts: Union[int, Sequence[int]] = 5,
                     bo_params: dict = {},
                     bagging: Optional[Union[int, Sequence[int]]] = None):
        """Fit the models to the dataset in a training sizing fashion.

        If you want to compare how different models perform when training on
        varying dataset sizes, you can choose to use a train_sizing approach
        to run the pipeline. This technique fits the models on increasingly large
        training sets.

        See the basetrainer.py module for a description of the parameters.

        """
        clear(self, self.models[:])

        params = (models, metric, greater_is_better, needs_proba, needs_threshold,
                  train_sizes, n_calls, n_random_starts, bo_params, bagging,
                  self.n_jobs, self.verbose, self.warnings, self.logger,
                  self.random_state)

        if self.goal.startswith('class'):
            self.trainer = TrainSizingClassifier(*params)
        else:
            self.trainer = TrainSizingRegressor(*params)

        self._run()
        self._sizes = self.trainer._sizes

    # Properties ============================================================ >>

    def _update_trainer(self):
        """Update the trainer's data when changing data properties from ATOM."""
        if self.trainer is not None:
            self.trainer._data = self._data
            self.trainer._idx = self._idx

    @BasePredictor.dataset.setter
    @typechecked
    def dataset(self, dataset: Optional[X_TYPES]):
        # None is also possible because of the save method
        self._data = None if dataset is None else check_property(dataset, 'dataset')
        self._update_trainer()

    @BasePredictor.train.setter
    @typechecked
    def train(self, train: X_TYPES):
        df = check_property(train, 'train', under=self.test, under_name='test')
        self._data = pd.concat([df, self.test])
        self._idx[0] = len(df)
        self._update_trainer()

    @BasePredictor.test.setter
    @typechecked
    def test(self, test: X_TYPES):
        df = check_property(test, 'test', under=self.train, under_name='train')
        self._data = pd.concat([self.train, df])
        self._idx[1] = len(df)
        self._update_trainer()

    @BasePredictor.X.setter
    @typechecked
    def X(self, X: X_TYPES):
        df = check_property(X, 'X', side=self.y, side_name='y')
        self._data = merge(df, self.y)
        self._update_trainer()

    @BasePredictor.y.setter
    @typechecked
    def y(self, y: Union[list, tuple, dict, np.ndarray, pd.Series]):
        series = check_property(y, 'y', side=self.X, side_name='X')
        self._data = merge(self._data.drop(self.target, axis=1), series)
        self._update_trainer()

    @BasePredictor.X_train.setter
    @typechecked
    def X_train(self, X_train: X_TYPES):
        df = check_property(X_train, 'X_train',
                            side=self.y_train, side_name='y_train',
                            under=self.X_test, under_name='X_test')
        self._data = pd.concat([merge(df, self.train[self.target]), self.test])
        self._update_trainer()

    @BasePredictor.X_test.setter
    @typechecked
    def X_test(self, X_test: X_TYPES):
        df = check_property(X_test, 'X_test',
                            side=self.y_test, side_name='y_test',
                            under=self.X_train, under_name='X_train')
        self._data = pd.concat([self.train, merge(df, self.test[self.target])])
        self._update_trainer()

    @BasePredictor.y_train.setter
    @typechecked
    def y_train(self, y_train: Union[list, tuple, dict, np.ndarray, pd.Series]):
        series = check_property(y_train, 'y_train',
                                side=self.X_train, side_name='X_train',
                                under=self.y_test, under_name='y_test')
        self._data = pd.concat([merge(self.X_train, series), self.test])
        self._update_trainer()

    @BasePredictor.y_test.setter
    @typechecked
    def y_test(self, y_test: Union[list, tuple, dict, np.ndarray, pd.Series]):
        series = check_property(y_test, 'y_test',
                                side=self.X_test, side_name='X_test',
                                under=self.y_train, under_name='y_train')
        self._data = pd.concat([self.train, merge(self.X_test, series)])
        self._update_trainer()
