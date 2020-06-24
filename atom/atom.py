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

# Sklearn
from sklearn.model_selection import train_test_split

# Own modules
from .models import get_model_name
from .data_cleaning import (
    StandardCleaner, Scaler, Imputer, Encoder, Outliers, Balancer
    )
from .feature_selection import FeatureGenerator, FeatureSelector
from .training import (
    TrainerClassifier, TrainerRegressor,
    SuccessiveHalvingClassifier, SuccessiveHalvingRegressor,
    TrainSizingClassifier, TrainSizingRegressor
    )
from .plots import BasePlotter, plot_correlation
from .utils import (
    X_TYPES, Y_TYPES, TRAIN_TYPES, composed, crash, params_to_log, merge,
    check_property, check_scaling, catch_return,  variable_return, get_metric,
    get_metric_name, get_default_metric, get_best_score, attach_methods, infer_task,
    clear, save
    )


class ATOM(BasePlotter):
    """ATOM base class.

    The ATOM class (parent for the ATOMClassifier and ATOMRegressor classes) is a
    convenient wrapper for all data_cleaning, feature_selection and training methods
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

    @composed(crash, params_to_log)
    def __init__(self, X, y, n_rows, test_size):
        """Prepares input, runs StandardCleaner and splits in train and test sets."""
        # Data attributes
        self._train_idx = None
        self._test_idx = None

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
        self.winner = None
        self._results = pd.DataFrame(
            columns=['name', 'score_train', 'score_test', 'time_fit',
                     'mean_bagging', 'std_bagging', 'time_bagging', 'time'])

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
        self.task = infer_task(y, self.goal)
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
        self.dataset = X_y.sample(random_state=self.random_state, **kwargs)
        self.dataset.reset_index(drop=True, inplace=True)

        # Get train and test indices
        train, test = train_test_split(self.dataset,
                                       test_size=self._test_size,
                                       shuffle=False)
        self._train_idx = train.index
        self._test_idx = test.index
        self.stats(1)  # Print data stats

    # Properties ============================================================ >>

    @property
    def results(self):
        """Return df without the bagging columns if they are empty."""
        return self._results.dropna(axis=1, how='all')

    @property
    def dataset(self):
        return self._data

    @dataset.setter
    @typechecked
    def dataset(self, dataset: X_TYPES):
        df = check_property(dataset, 'dataset')
        self._data = df

    @property
    def train(self):
        return self._data[self._data.index.isin(self._train_idx)]

    @train.setter
    @typechecked
    def train(self, train: X_TYPES):
        df = check_property(train, 'train', under=self.test, under_name='test')
        self._train_idx = df.index
        self._data = pd.concat([df, self.test])

    @property
    def test(self):
        return self._data[self._data.index.isin(self._test_idx)]

    @test.setter
    @typechecked
    def test(self, test: X_TYPES):
        df = check_property(test, 'test', under=self.train, under_name='train')
        self._test_idx = df.index
        self._data = pd.concat([self.train, df])

    @property
    def X(self):
        return self._data.drop(self.target, axis=1)

    @X.setter
    @typechecked
    def X(self, X: X_TYPES):
        df = check_property(X, 'X', side=self.y, side_name='y')
        self._data = merge(df, self.y)

    @property
    def y(self):
        return self._data[self.target]

    @y.setter
    @typechecked
    def y(self, y: Y_TYPES):
        series = check_property(y, 'y', side=self.X, side_name='X')
        self._data = merge(self._data.drop(self.target, axis=1), series)

    @property
    def X_train(self):
        return self.train.drop(self.target, axis=1)

    @X_train.setter
    def X_train(self, X_train: X_TYPES):
        df = check_property(X_train, 'X_train',
                            side=self.y_train, side_name='y_train',
                            under=self.X_test, under_name='X_test')
        self._data = pd.concat([merge(df, self.train[self.target]), self.test])

    @property
    def X_test(self):
        return self.test.drop(self.target, axis=1)

    @X_test.setter
    @typechecked
    def X_test(self, X_test: X_TYPES):
        df = check_property(X_test, 'X_test',
                            side=self.y_test, side_name='y_test',
                            under=self.X_train, under_name='X_train')
        self._data = pd.concat([self.train, merge(df, self.test[self.target])])

    @property
    def y_train(self):
        return self.train[self.target]

    @y_train.setter
    @typechecked
    def y_train(self, y_train: Y_TYPES):
        series = check_property(y_train, 'y_train',
                                side=self.X_train, side_name='X_train',
                                under=self.y_test, under_name='y_test')
        self._data = pd.concat([merge(self.X_train, series), self.test])

    @property
    def y_test(self):
        return self.test[self.target]

    @y_test.setter
    @typechecked
    def y_test(self, y_test: Y_TYPES):
        series = check_property(y_test, 'y_test',
                                side=self.X_test, side_name='X_test',
                                under=self.y_train, under_name='y_train')
        self._data = pd.concat([self.train, merge(self.X_test, series)])

    @property
    def target(self):
        return self._data.columns[-1]

    # Utility methods ======================================================= >>

    @crash
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
        categ = self.X.select_dtypes(include=['category', 'object']).shape[1]
        if categ > 0:
            self.log(f"Categorical columns: {categ}", _vb)

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
                self.log("Class balance: {} <==> {}"  # [-1] to remove last :
                         .format(keys[:-1], string[:-1]), _vb + 1)
            self.log(f"Instances in {self.target} per class:", _vb + 1)
            self.log(stats.to_markdown(), _vb + 1)

        self.log('', 1)  # Insert an empty row

    @composed(crash, params_to_log, typechecked)
    def report(self,
               df: str = 'dataset',
               n_rows: Optional[Union[int, float]] = None,  # float for 1e3...
               filename: Optional[str] = None):
        """Create an extensive profile analysis of the data.

        The profile report is rendered in HTML5 and CSS3. Note that this
        method can be slow for rows>10k. Dependency: pandas-profiling.

        Parameters
        ----------
        df: str, optional(default='dataset')
            Name of the data class property to get the report from.

        n_rows: int or None, optional(default=None)
            Number of (randomly picked) rows to process. None for all rows.

        filename: str or None, optional (default=None)
            Name of the file when saved (as .html). None to not save anything.

        """
        try:
            from pandas_profiling import ProfileReport
        except ImportError:
            raise ModuleNotFoundError(
                "Failed to import the pandas-profiling package. Install it" +
                "before using the report method.")

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

    @composed(crash, params_to_log, typechecked)
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
            if eval(key) and getattr(self, value):
                # If verbose is specified, change the class verbosity
                if verbose is not None:
                    getattr(self, value).verbose = verbose
                X, y = catch_return(getattr(self, value).transform(X, y))

        return variable_return(X, y)

    @composed(crash, typechecked)
    def plot_correlation(self,
                         title: Optional[str] = None,
                         figsize: Tuple[int, int] = (10, 10),
                         filename: Optional[str] = None,
                         display: bool = True):
        """Plot the data's correlation matrix.

        See the function in the plots.py module for the parameter descriptions.

        """
        plot_correlation(self, title, figsize, filename, display)

    @composed(crash, params_to_log, typechecked)
    def clear(self, models: Union[str, Sequence[str]] = 'all'):
        """Clear model from the pipeline.

        This functions removes all traces of a model in the ATOM pipeline (except
        for the errors attribute). This includes the models and results attributes,
        and the model subclass. If the winning model is removed. The next best model
        (through score_test or mean_bagging if available) is selected as winner.
        If all models in the pipeline are removed, the metric attribute is reset.

        Parameters
        ----------
        models: str or sequence, optional (default='all')
            Names of the models to clear. If 'all', clear all models.

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

        if not self.models:
            self.trainer = None  # Clear the trainer attribute

        self.log(keyword + " cleared successfully!", 1)

    @composed(crash, params_to_log, typechecked)
    def save(self, filename: Optional[str] = None):
        """Save the class to a pickle file.

        Parameters
        ----------
        filename: str or None, optional (default=None)
            Name to save the file with. None to save with classes' name.

        """
        save(self, self.__class__.__name__ if filename is None else filename)
        self.log(self.__class__.__name__ + " saved successfully!", 1)

    # Data cleaning methods ================================================= >>

    @composed(crash, params_to_log)
    def scale(self):
        """Scale the features.

        Scale the features in the dataset to mean=0 and std=1. The scaler is
        fitted only on the training set to avoid data leakage.

        """
        self.scaler = Scaler(verbose=self.verbose, logger=self.logger)
        self.scaler.fit(self.X_train)

        self.X = self.scaler.transform(self.X)

    @composed(crash, params_to_log, typechecked)
    def impute(self,
               strat_num: Union[int, float, str] = 'remove',
               strat_cat: str = 'remove',
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

    @composed(crash, params_to_log, typechecked)
    def encode(self,
               max_onehot: Optional[int] = 10,
               encode_type: str = 'Target',
               frac_to_other: float = 0,
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

    @composed(crash, params_to_log, typechecked)
    def outliers(self,
                 strategy: Union[int, float, str] = 'remove',
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

    @composed(crash, params_to_log, typechecked)
    def balance(self,
                oversample: Optional[Union[int, float, str]] = None,
                undersample: Optional[Union[int, float, str]] = None,
                n_neighbors: int = 5):
        """Balance the target categories in the training set.

        Balance the number of instances per target category in the training set.
        Using oversample and undersample at the same time or not using any will
        raise an exception. Only the training set is balanced in order to maintain
        the original distribution of target categories in the test set. Use only for
        classification tasks. Dependency: imbalanced-learn.

        See the data_cleaning.py module for a description of the parameters.

        """
        try:
            import imblearn
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Failed to import the imbalanced-learn package. Install it " +
                "before using the balance method.")

        if not self.goal.startswith('class'):
            raise RuntimeError("The balance only works for classification tasks!")

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

    # Feature selection methods ============================================= >>

    @composed(crash, params_to_log, typechecked)
    def feature_generation(self,
                           n_features: int = 2,
                           generations: int = 20,
                           population: int = 500):
        """Create new non-linear features.

        Use a genetic algorithm to create new combinations of existing
        features and add them to the original dataset in order to capture
        the non-linear relations between the original features. A dataframe
        containing the description of the newly generated features and their
        scores can be accessed through the `genetic_features` attribute. It is
        recommended to only use this method when fitting linear models.
        Dependency: gplearn.

        See the feature_selection.py module for a description of the parameters.

        """
        try:
            import gplearn
        except ImportError:
            raise ModuleNotFoundError(
                "Failed to import the gplearn package. Install it before " +
                "using the feature_generation method.")

        self.feature_generator = \
            FeatureGenerator(n_features=n_features,
                             generations=generations,
                             population=population,
                             n_jobs=self.n_jobs,
                             verbose=self.verbose,
                             logger=self.logger,
                             random_state=self.random_state)
        self.feature_generator.fit(self.X_train, self.y_train)

        self.X = self.feature_generator.transform(self.X)

        # Attach attributes to the ATOM class
        for attr in ['symbolic_transformer', 'genetic_features']:
            setattr(self, attr, getattr(self.feature_generator, attr))

    @composed(crash, params_to_log, typechecked)
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

        After running the method, the FeatureSelector's plot methods can be
        called directly from ATOM.

        See the feature_selection.py module for a description of the parameters.

        """
        if isinstance(strategy, str):
            if strategy.lower() == 'univariate' and solver is None:
                if self.goal.startswith('reg'):
                    solver = 'f_regression'
                else:
                    solver = 'f_classif'
            elif strategy.lower() in ['sfm', 'rfe', 'rfecv']:
                if solver is None and self.winner:
                    solver = self.winner.best_model_fit
                elif isinstance(solver, str):
                    # In case the user already filled the task...
                    if not solver.endswith('_class') or not solver.endswith('_reg'):
                        solver += '_reg' if self.task.startswith('reg') else '_class'

            # If the run method was called before, use the selected metric for RFECV
            if strategy.lower() == 'rfecv':
                if self.metric and 'scoring' not in kwargs:
                    kwargs['scoring'] = self.metric

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
        attrs = ['collinear', 'univariate', 'scaler', 'pca', 'sfm', 'rfe', 'rfecv']
        for attr in attrs:
            if getattr(self.feature_selector, attr) is not None:
                setattr(self, attr, getattr(self.feature_selector, attr))

        # Attach plot methods to the ATOM instance
        attach_methods(self, self.feature_selector.__class__, 'plot')

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
            self.trainer.run(self.X_train, self.X_test, self.y_train, self.y_test)
        except ValueError as exception:
            # Catch errors and pass them to ATOM's attribute
            for model, error in self.trainer.errors.items():
                self.errors[model] = error
            raise ValueError(exception)
        else:
            # Attach plot and transformation methods to the ATOM class
            methods = ['plot', '_pipe', 'predict', 'decision', 'score', 'outcome']
            attach_methods(self, self.trainer.__class__, methods)

            # Attach mapping from ATOM to the trainer instance (for plots)
            self.trainer.mapping = self.mapping

            # Update attributes
            self.models += [m for m in self.trainer.models if m not in self.models]
            self.metric = self.trainer.metric

            # If SuccessiveHalving or TrainSizing, replace the results
            if isinstance(self.trainer.results.index, pd.MultiIndex):
                self._results = self.trainer._results
            else:  # If Trainer, add them row by row
                for idx, row in self.trainer.results.iterrows():
                    self._results.loc[idx] = row

            for model in self.trainer.models:
                self.errors.pop(model, None)  # Remove model from errors if there
                setattr(self, model, getattr(self.trainer, model))
                setattr(self, model.lower(), getattr(self.trainer, model.lower()))

                # Add transform method to model subclasses
                setattr(getattr(self, model), 'transform', self.transform)
                setattr(getattr(self, model.lower()), 'transform', self.transform)

            # Assign winning model
            best_row = self.results.apply(lambda row: get_best_score(row), axis=1)
            best_model = best_row.idxmax()
            best_model = best_model if isinstance(best_model, str) else best_model[1]
            self.winner = getattr(self, best_model)

            for model, error in self.trainer.errors.items():
                self.errors[model] = error

    @composed(crash, params_to_log, typechecked)
    def run(self,
            models: Union[str, List[str], Tuple[str]],
            metric: Optional[Union[str, callable]] = None,
            greater_is_better: bool = True,
            needs_proba: bool = False,
            needs_threshold: bool = False,
            n_calls: Union[int, Sequence[int]] = 0,
            n_random_starts: Union[int, Sequence[int]] = 5,
            bo_params : dict = {},
            bagging: Optional[Union[int, Sequence[int]]] = None):
        """Fit the models to the dataset in a direct fashion.

        Contrary to the Trainer class in training.py, this method allows
        subsequent runs and stores all results as attributes (only the model
        subclasses are overwritten if the same model is rerun).

        See the training.py module for a description of the parameters.

        """
        # If this is the first direct run, clear all previous results
        if not type(self.trainer).__name__.startswith('Trainer'):
            clear(self, models=self.models)
            if metric is None:  # If no metric, assign the default option
                metric = get_default_metric(self.task)

        elif metric is None:  # Assign the existing metric
            metric = self.metric

        else:  # Check that the selected metric is the same as previous run
            metric = get_metric(
                metric, greater_is_better, needs_proba, needs_threshold)
            metric_name = get_metric_name(metric)
            if metric_name != self.metric.name:
                raise ValueError(
                    f"Invalid metric parameter! Metric {self.metric.name} is " +
                    f"already in use, got {metric_name}. Use the clear method " +
                    "before selecting a new metric.")

        params = (models, metric, greater_is_better, needs_proba, needs_threshold,
                  n_calls, n_random_starts, bo_params, bagging, self.n_jobs,
                  self.verbose, self.warnings, self.logger, self.random_state)

        if self.goal.startswith('class'):
            self.trainer = TrainerClassifier(*params)
        else:
            self.trainer = TrainerRegressor(*params)

        self._run()

    @composed(crash, params_to_log, typechecked)
    def successive_halving(
            self,
            models: Union[str, List[str], Tuple[str]],
            metric: Optional[Union[str, callable]] = None,
            greater_is_better: bool = True,
            needs_proba: bool = False,
            needs_threshold: bool = False,
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

        See the training.py module for a description of the parameters.

        """
        clear(self, models=self.models)
        if metric is None:  # If no metric, assign the default option
            metric = get_default_metric(self.task)

        params = (models, metric, greater_is_better, needs_proba, needs_threshold,
                  skip_iter, n_calls, n_random_starts, bo_params, bagging,
                  self.n_jobs, self.verbose, self.warnings, self.logger,
                  self.random_state)

        if self.goal.startswith('class'):
            self.trainer = SuccessiveHalvingClassifier(*params)
        else:
            self.trainer = SuccessiveHalvingRegressor(*params)

        self._run()

    @composed(crash, params_to_log, typechecked)
    def train_sizing(self,
                     models: Union[str, List[str], Tuple[str]],
                     metric: Optional[Union[str, callable]] = None,
                     greater_is_better: bool = True,
                     needs_proba: bool = False,
                     needs_threshold: bool = False,
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

        See the training.py module for a description of the parameters.

        """
        clear(self, models=self.models)
        if metric is None:  # If no metric, assign the default option
            metric = get_default_metric(self.task)

        params = (models, metric, greater_is_better, needs_proba, needs_threshold,
                  train_sizes, n_calls, n_random_starts, bo_params, bagging,
                  self.n_jobs, self.verbose, self.warnings, self.logger,
                  self.random_state)

        if self.goal.startswith('class'):
            self.trainer = TrainSizingClassifier(*params)
        else:
            self.trainer = TrainSizingRegressor(*params)

        self._run()
