# -*- coding: utf-8 -*-

'''
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom

'''

# << ============ Import Packages ============ >>

# Standard packages
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
from datetime import datetime
import multiprocessing

# Sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, VarianceThreshold, SelectFromModel
    )
from sklearn.model_selection import train_test_split

# Others
try:
    import xgboost
except ImportError:
    xgb_import = False
else:
    xgb_import = True

# Plotting
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style='darkgrid', palette="GnBu_d")


# << ============ Functions ============ >>

def convert_to_df(X):
    ''' Convert X to pd.Dataframe '''

    if not isinstance(X, pd.DataFrame):
        columns = ['Feature ' + str(i) for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=columns)

    return X


def prlog(string, cl, level=0, time=False):

    '''
    DESCRIPTION -----------------------------------

    Print and save output to log file.

    PARAMETERS -------------------------------------

    cl     --> class of the element
    string --> string to output
    level  --> minimum verbosity level to print
    time   --> Wether to add the timestamp to the log

    '''

    if cl.verbose > level:
        print(string)

    if cl.log is not None:
        with open(cl.log, 'a+') as file:
            if time:
                # Datetime object containing current date and time
                now = datetime.now()
                date = now.strftime("%d/%m/%Y %H:%M:%S")
                file.write(date + '\n' + string + '\n')
            else:
                file.write(string + '\n')


# << ============ Classes ============ >>

class ATOM(object):

    def __init__(self, models=None, metric=None,
                 successive_halving=False, skip_steps=0,
                 impute='median', strategy=None, solver=None, max_features=0.9,
                 ratio=0.3, max_iter=15, max_time=np.inf, eps=1e-08,
                 batch_size=1, init_points=5, plot_bo=False,
                 cross_validation=True, n_splits=4,
                 log=None, n_jobs=1, verbose=0):

        '''
        DESCRIPTION -----------------------------------

        Initialize class.

        PARAMETERS -------------------------------------

        models             --> list of models to use
        metric             --> metric to perform evaluation on
        successive_halving --> wether to perform successive halving
        skip_steps         --> skip last n steps of successive halving
        impute             --> imputation strategy (if None, no imputing)
        strategy           --> feature selection strategy to use
        solver             --> solver or model for feature selection
        max_features       --> number of features to select (None for all)
        ratio              --> train/test split ratio
        max_iter           --> maximum number of iterations of the BO
        max_time           --> maximum time for the BO (in seconds)
        eps                --> minimum distance between two consecutive x's
        batch_size         --> batch size in which the objective is evaluated
        init_points        --> initial number of random tests of the BO
        plot_bo            --> boolean to plot the BO's progress
        cross_validation   --> perform kfold cross-validation
        n_splits           --> number of splits for the stratified kfold
        log                --> keep log file
        n_jobs             --> number of cores to use for parallel processing
        verbose            --> verbosity level (0, 1 or 2)

        '''

        # Set attributes to class (set to default if input is invalid)
        self.models = models
        self.metric = metric
        self.successive_halving = bool(successive_halving)
        self.skip_steps = int(skip_steps) if skip_steps > 0 else 0
        self.impute = impute
        self.strategy = strategy
        self.solver = solver
        self.max_features = max_features
        self.ratio = ratio if 0 < ratio < 1 else 0.3
        self.max_iter = int(max_iter) if max_iter > 0 else 15
        self.max_time = float(max_time) if max_time > 0 else np.inf
        self.eps = float(eps) if eps >= 0 else 1e-08
        self.batch_size = int(batch_size) if batch_size > 0 else 1
        self.init_points = int(init_points) if init_points > 0 else 5
        self.plot_bo = bool(plot_bo)
        self.cross_validation = bool(cross_validation)
        self.n_splits = int(n_splits) if n_splits > 0 else 4
        self.n_jobs = int(n_jobs)  # Gets checked later
        self.verbose = verbose if verbose in (0, 1, 2, 3) else 0

        # Set log file name
        self.log = log if log is None or log.endswith('.txt') else log + '.txt'

        # Save model erros (if any) in dictionary
        self.errors = {}

        # Save the cross-validation's results in array of dataframes
        # Only for successive halving
        self.results = []

    def check_features(self, X, output=False):
        ''' Remove non-numeric features with unhashable type '''

        if output:
            prlog('Checking feature types...', self, 1)

        for column in X.columns:
            dtype = str(X[column].dtype)
            if dtype in ('datetime64', 'timedelta[ns]', 'category'):
                if output:
                    prlog(' --> Dropping feature {}. Unhashable type: {}.'
                          .format(column, dtype), self, 2)
                X.drop(column, axis=1)
            elif dtype == 'object':  # Strip str features from blank spaces
                X[column] = X[column].astype(str).str.strip()

        return X

    def imputer(self, X, strategy='median', max_frac_missing=0.5,
                missing=[np.inf, -np.inf, '', '?', 'NA', 'nan', 'NaN', None]):

        '''
        DESCRIPTION -----------------------------------

        Impute missing values in feature df. Non-numeric
        features are always imputed with the most frequent
        strategy. Removes columns with too many missing values.

        PARAMETERS -------------------------------------

        X                  --> data features: pd.Dataframe or array
        strategy           --> impute strategy. Choose from:
                                   remove: remove column if any missing value
                                   mean: fill with mean of column
                                   median: fill with median of column
                                   most_frequent: fill with most frequent value
        max_frac_missing   --> maximum fraction of NaN values in column
        missing            --> list of values to impute, besides NaN and None

        RETURNS ----------------------------------------

        Imputed dataframe.

        '''

        strats = ['remove', 'mean', 'median', 'most_frequent']
        if strategy.lower() not in strats:
            raise ValueError(f'Unkwown impute strategy. Try one of {strats}.')

        # Convert array to dataframe (can be called independent of fit method)
        X = convert_to_df(X)
        X = self.check_features(X)  # Needed if ran alone

        prlog('Imputing missing values...', self, 1)

        # Make list for enumeration
        if not isinstance(missing, list):
            missing = [missing]

        # None must always be imputed
        if None not in missing:
            missing.append(None)

        X = X.replace(missing, np.NaN)  # Replace missing values with NaN

        imp_numeric = SimpleImputer(strategy=strategy.lower())
        imp_nonnumeric = SimpleImputer(strategy='most_frequent')
        for i in X.columns:
            if X[i].isna().any():  # Check if any value is missing in column
                if strategy.lower() == 'remove':
                    prlog(f' --> Feature {i} was removed since it contained ' +
                          'missing values.', self, 2)
                    X.drop(i, axis=1, inplace=True)
                    continue

                # Drop columns with too many NaN values
                nans = X[i].isna().sum()
                pnans = int(nans/len(X[i])*100)
                if nans > max_frac_missing * len(X[i]):
                    prlog(f' --> Feature {i} was removed since it contained ' +
                          f'{nans} ({pnans}%) missing values.', self, 2)
                    X.drop(i, axis=1, inplace=True)
                    continue

                # Check if column is numeric (int, float, unsigned int)
                imp = X[i].values.reshape(-1, 1)
                if X[i].dtype.kind in 'ifu':
                    X[i] = imp_numeric.fit_transform(imp)
                else:
                    X[i] = imp_nonnumeric.fit_transform(imp)

        return X

    def encoder(self, X, max_number_onehot=10):

        '''
        DESCRIPTION -----------------------------------

        Perform encoding on categorical features. The encoding
        type depends on the number of unique values in the column.
        Also removes columns with only one unique category.

        PARAMETERS -------------------------------------

        X                 --> data features: pd.Dataframe or array
        max_number_onehot --> threshold between onehot and label encoding

        RETURNS ----------------------------------------

        Encoded dataframe.

        '''

        # Convert array to dataframe (can be called independent of fit method)
        X = convert_to_df(X)
        X = self.check_features(X)  # Needed if ran alone

        prlog('Encoding categorical features...', self, 1)

        for i in X.columns:
            # Check if column is non-numeric (thus categorical)
            if X[i].dtype.kind not in 'ifu':
                # Count number of unique values in the column
                n_unique = len(np.unique(X[i]))

                # Perform encoding type dependent on number of unique values
                if 2 < n_unique <= max_number_onehot:
                    prlog(' --> One-hot-encoding feature {}. Contains {} '
                          .format(i, n_unique) + 'unique categories.', self, 2)
                    X = pd.concat([X, pd.get_dummies(X[i], prefix=i)], axis=1)
                    X.drop(i, axis=1, inplace=True)
                else:
                    prlog(' --> Label-encoding feature {}. Contains {} '
                          .format(i, n_unique) + 'unique categories.', self, 2)
                    X[i] = LabelEncoder().fit_transform(X[i])

        return X

    def feature_selection(self, X, Y,
                          strategy='univariate',
                          solver=None,
                          max_features=0.9,
                          threshold=-np.inf,
                          frac_variance=1.,
                          max_correlation=0.98):

        '''
        DESCRIPTION -----------------------------------

        Select best features according to a univariate F-test or with a
        recursive feature selector (RFS). Ties between features with equal
        scores will be broken in an unspecified way. Also removes features
        with too low variance and too high collinearity.

        PARAMETERS -------------------------------------

        X            --> data features: pd.Dataframe or array
        Y            --> data targets: pd.Series or array
        strategy     --> strategy for feature selection. Choose from:
                             'univariate': perform a univariate F-test
                             'PCA': perform principal component analysis
                             'RFS': perform recursive feature selection
        solver       --> solver or model class for the strategy
        max_features --> if < 1: fraction of features to select
                         if >= 1: number of features to select
                         None to select all (only for RFS)
        threshold    --> threshold value to use for selection. Only for model.
                         Choose from: float, 'mean', 'median'.
                            float
        frac_variance   --> minimum value of the Pearson correlation
                            cofficient to identify correlated features
        max_correlation --> remove features with constant instances in at
                            least this fraction of the total

        RETURNS ----------------------------------------

        Dataframe of the selected features.

        '''

        def remove_collinear(X, limit):

            '''
            DESCRIPTION -----------------------------------

            Finds pairs of collinear features based on the Pearson
            correlation coefficient. For each pair above the specified
            limit (in terms of absolute value), it removes one of the two.
            Using code adapted from:
                https://chrisalbon.com/machine_learning/
                    feature_selection/drop_highly_correlated_features/

            PARAMETERS -------------------------------------

            X     --> data features: pd.Dataframe or array
            limit --> minimum value of the Pearson correlation cofficient
                      to identify correlated features

            RETURNS ----------------------------------------

            Dataframe after removing one of the two correlated features.

            '''

            mtx = X.corr()  # Pearson correlation coefficient matrix

            # Extract the upper triangle of the correlation matrix
            upper = mtx.where(np.triu(np.ones(mtx.shape).astype(np.bool), k=1))

            # Select the features with correlations above the threshold
            to_drop = [i for i in upper.columns if any(abs(upper[i]) > limit)]

            # Dataframe to hold correlated pairs
            columns = ['drop_feature', 'corr_feature', 'corr_value']
            self.collinear = pd.DataFrame(columns=columns)

            # Iterate to record pairs of correlated features
            for column in to_drop:

                # Find the correlated features
                corr_features = list(upper.index[abs(upper[column]) > limit])

                # Find the correlated values
                corr_values = list(upper[column][abs(upper[column]) > limit])
                drop_features = [column for _ in corr_features]

                # Add to class attribute
                self.collinear = \
                    self.collinear.append({'drop_feature': drop_features,
                                           'corr_feature': corr_features,
                                           'corr_value': corr_values},
                                          ignore_index=True)

                prlog(f' --> Feature {column} was removed due to ' +
                      'collinearity with another feature.', self, 2)

            return X.drop(to_drop, axis=1)

        def remove_low_variance(X, frac_variance):

            '''
            DESCRIPTION -----------------------------------

            Removes featrues with too low variance.

            PARAMETERS -------------------------------------

            X             --> data features: pd.Dataframe or array
            frac_variance --> remove features with constant instances in at
                              least this fraction of the total

            RETURNS ----------------------------------------

            Dataframe after removing the low variance features.

            '''

            threshold = frac_variance * (1. - frac_variance)
            var = VarianceThreshold(threshold=threshold).fit(X)
            mask = var.get_support()  # Get boolean mask of selected features

            for n, column in enumerate(X.columns):
                if not mask[n]:
                    prlog(f' --> Feature {column} was removed due to' +
                          f' low variance: {var.variances_[n]:.2f}.', self, 2)
                    X.drop(column, axis=1, inplace=True)

            return X

        prlog('Performing feature selection...', self, 1)

        # First, drop features with too high correlation
        X = remove_collinear(X, limit=max_correlation)
        # Then, remove features with too low variance
        X = remove_low_variance(X, frac_variance=frac_variance)

        if max_features is not None and max_features < 1:
            max_features = int(max_features * X.shape[1])

        # Perform selection based on strategy
        if strategy.lower() == 'univariate':
            if max_features is None:
                max_features = X.shape[1]
            if solver is None:   # Set function dependent on goal
                func = f_classif if self.goal != 'regression' else f_regression
            else:
                func = solver
            self.univariate = SelectKBest(func, k=max_features).fit(X, Y)
            mask = self.univariate.get_support()
            for n, column in enumerate(X.columns):
                if not mask[n]:
                    prlog(f' --> Feature {column} was removed after the ' +
                          'univariate test (score: {:.2f}  p-value: {:.2f}).'
                          .format(self.univariate.scores_[n],
                                  self.univariate.pvalues_[n]), self, 2)
                    X.drop(column, axis=1, inplace=True)

        elif strategy.lower() == 'pca':
            prlog(f' --> Applying Principal Component Analysis... ', self, 2)
            X = StandardScaler().fit_transform(X)  # Scale features first
            solver = 'auto' if solver is None else solver
            self.PCA = PCA(n_components=max_features, svd_solver=solver)
            X = self.PCA.fit_transform(X, Y)
            X = convert_to_df(X)

        elif strategy.lower() == 'rfs':
            if solver is None:
                raise ValueError('Select a model class for the RFS solver!')
            self.RFS = SelectFromModel(estimator=solver,
                                       threshold=threshold,
                                       max_features=max_features).fit(X, Y)
            mask = self.RFS.get_support()
            for n, column in enumerate(X.columns):
                if not mask[n]:
                    prlog(f' --> Feature {column} was removed by the ' +
                          'recursive feature eliminator.', self, 2)
                    X.drop(column, axis=1, inplace=True)

        elif strategy is not None:
            raise ValueError('Invalid feature selection strategy selected.'
                             "Choose one of: ['univariate', 'PCA', 'RFS']")

        return X

    def fit(self, X, Y, percentage=100):

        '''
        DESCRIPTION -----------------------------------

        Run the pipeline.

        PARAMETERS -------------------------------------

        X          --> data features: pd.Dataframe or array
        Y          --> data targets: pd.Series or array
        percentage --> percentage of data to use

        '''

        # << ============ Inner Function ============ >>

        def run_iteration(self):
            ''' Core iterations, multiple needed for successive halving '''

            # In case there is no cross-validation
            results = 'No cross-validation performed'
            # If verbose=1, use tqdm to evaluate process
            if self.verbose == 1:
                loop = tqdm(self.final_models)
            else:
                loop = self.final_models

            # Loop over every independent model
            for model in loop:
                # Set model class
                setattr(self, model, eval(model)(self.data,
                                                 self.metric,
                                                 self.goal,
                                                 self.log,
                                                 self.verbose))

                try:  # If errors occure, just skip the model
                    # GNB and GP have no hyperparameters to tune
                    if model not in ('GNB', 'GP'):
                        getattr(self, model).BayesianOpt(self.max_iter,
                                                         self.max_time,
                                                         self.eps,
                                                         self.batch_size,
                                                         self.init_points,
                                                         self.plot_bo,
                                                         self.n_jobs)
                    if self.cross_validation:
                        getattr(self, model).cross_val_evaluation(
                                                    self.n_splits, self.n_jobs)

                except Exception as ex:
                    prlog('Exception encountered while running '
                          + f'the {model} model. Removing model from pipeline.'
                          + f'\n{type(ex).__name__}: {ex}', self, 1, True)

                    # Save the exception to model attribute
                    exception = type(ex).__name__ + ': ' + str(ex)
                    getattr(self, model).error = exception

                    # Append exception to ATOM errors dictionary
                    self.errors[model] = exception

                    # Replace model with value X for later removal
                    # Can't remove at once to not disturb list order
                    self.final_models[self.final_models.index(model)] = 'X'

                # Set model attributes for lowercase as well
                setattr(self, model.lower(), getattr(self, model))

            # Remove faulty models (replaced with X)
            while 'X' in self.final_models:
                self.final_models.remove('X')

            if self.cross_validation:
                try:  # Check that at least one model worked
                    lenx = max([len(getattr(self, m).name)
                                for m in self.final_models])
                    max_mean = max([getattr(self, m).results.mean()
                                    for m in self.final_models])
                except ValueError:
                    raise ValueError('It appears all models failed to run...')

                # Print final results (summary of cross-validation)
                t = time() - t_init  # Total time in seconds
                h = int(t/3600.)
                m = int(t/60.) - h*60
                s = int(t - h*3600 - m*60)
                prlog('\n\nFinal stats ================>>', self)
                prlog(f'Duration: {h:02}h:{m:02}m:{s:02}s', self)
                prlog(f'Target metric: {self.metric}', self)
                prlog('--------------------------------', self)

                # Create dataframe with final results
                results = pd.DataFrame(columns=['model', 'cv_mean', 'cv_std'])

                for m in self.final_models:
                    name = getattr(self, m).name
                    shortname = getattr(self, m).shortname
                    cv_mean = getattr(self, m).results.mean()
                    cv_std = getattr(self, m).results.std()
                    results = results.append({'model': shortname,
                                              'cv_mean': cv_mean,
                                              'cv_std': cv_std},
                                             ignore_index=True)

                    # Highlight best score (if more than one)
                    if cv_mean == max_mean and len(self.final_models) > 1:
                        prlog(u'{0:{1}s} --> {2:.3f} \u00B1 {3:.3f} !!'
                              .format(name, lenx, cv_mean, cv_std), self)
                    else:
                        prlog(u'{0:{1}s} --> {2:.3f} \u00B1 {3:.3f}'
                              .format(name, lenx, cv_mean, cv_std), self)

            return results

        def data_preparation(self, X, Y, percentage=100):
            ''' Make a dct of data (complete, train, test and scaled) '''

            data = {}
            data['X'] = X.head(int(len(X)*percentage/100))
            data['Y'] = Y.head(int(len(X)*percentage/100))

            # Split train and test for the BO on percentage of data
            data['X_train'], data['X_test'], data['Y_train'], data['Y_test'] \
                = train_test_split(data['X'],
                                   data['Y'],
                                   test_size=self.ratio,
                                   random_state=1)

            # List of models that need scaling
            scaling_models = ['LinReg', 'LogReg', 'KNN', 'XGB',
                              'lSVM', 'kSVM', 'PA', 'SGD', 'MLP']
            # Check if any scaling models in final_models
            scale = any(model in self.final_models for model in scaling_models)
            # If PCA was performed, features are already scaled
            # Made string in case it is None
            if scale and str(self.strategy).lower() != 'pca':
                # Normalize features to mean=0, std=1
                data['X_scaled'] = StandardScaler().fit_transform(data['X'])
                scaler = StandardScaler().fit(data['X_train'])
                data['X_train_scaled'] = scaler.transform(data['X_train'])
                data['X_test_scaled'] = scaler.transform(data['X_test'])

            return data

        def print_data_stats(self):
            prlog('\nData stats =====================>', self, 1)
            prlog('Number of features: {}\nNumber of instances: {}'
                  .format(self.data['X'].shape[1],
                          self.data['X'].shape[0]), self, 1)
            prlog('Size of training set: {}\nSize of validation set: {}'
                  .format(len(self.data['X_train']),
                          len(self.data['X_test'])), self, 1)

            # Print count of target values
            if self.goal != 'regression':
                _, counts = np.unique(self.data['Y'], return_counts=True)
                lenx = max(max([len(str(i)) for i in self.unique]),
                           len(self.data['Y'].name))
                prlog('Number of instances per target class:', self, 2)
                prlog(f"{self.data['Y'].name:{lenx}} --> Count", self, 2)
                for i in range(len(self.unique)):
                    prlog(f'{self.unique[i]:<{lenx}} --> {counts[i]}', self, 2)

        def not_regression(final_models):
            ''' Remove classification-only models from pipeline '''

            class_models = ['BNB', 'GNB', 'MNB', 'LogReg', 'LDA', 'QDA']
            for model in class_models:
                if model in final_models:
                    prlog(f"{model} can't perform regression tasks."
                          + " Removing model from pipeline.", self)
                    final_models.remove(model)

            return final_models

        def shuffle(X, Y):
            ''' Shuffles pd.Dataframe X and pd.Series Y in unison '''

            if len(X) != len(Y):
                raise ValueError("X and Y don't have the same number of rows!")
            p = np.random.permutation(len(X))
            return X.ix[p], Y[p]  # Difference in calling because of data type

        # << ============ Initialize ============ >>

        t_init = time()  # To measure the time the whole pipeline takes
        prlog('\n<================ ATOM ================>\n', self, 0, True)

        # << ============ Handle input ============ >>

        # Convert array to dataframe
        X = convert_to_df(X)

        # Convert target column to pandas series
        if not isinstance(Y, pd.Series):
            Y = pd.Series(Y, name='target')

        # Delete rows with NaN in target
        idx = pd.isnull(Y).any().nonzero()
        X.drop(X.index[idx], inplace=True)
        Y.drop(Y.index[idx], inplace=True)

        # Get percentage of data
        X, Y = shuffle(X, Y)  # Shuffle before selecting percentage
        X = X.head(int(len(X)*percentage/100))
        Y = Y.head(int(len(Y)*percentage/100))

        # << ============ Parameters tests ============ >>

        # Check number of cores for multiprocessing
        n_cores = multiprocessing.cpu_count()

        if self.n_jobs > n_cores:
            prlog('\nWarning! No {} cores available. n_jobs reduced to {}.'
                  .format(self.n_jobs, n_cores), self)
            self.n_jobs = n_cores

        elif self.n_jobs == 0:
            prlog("\nWarning! Value of n_jobs can't be {}. Using 1 core."
                  .format(self.n_jobs), self)
            self.n_jobs = 1

        else:
            if self.n_jobs <= -1:
                self.n_jobs = n_cores + 1 + self.n_jobs

            # Final check
            if self.n_jobs < 1 or self.n_jobs > n_cores:
                raise ValueError('Invalid value for n_jobs!')
            elif self.n_jobs != 1:
                prlog(f'Parallel processing with {self.n_jobs} cores.', self)

        # Set algorithm goal (regression, binaryclass or multiclass)
        classes = set(Y)
        if len(classes) < 2:
            raise ValueError(f'Only found one target value: {classes}!')
        elif len(classes) == 2:
            prlog('Algorithm set to binary classification.', self)
            self.goal = 'binary classification'
        elif 2 < len(classes) < 0.1*len(Y):
            prlog('Algorithm set to multiclass classification.' +
                  f' Number of classes: {len(classes)}', self)
            self.goal = 'multiclass classification'
        else:
            prlog('Algorithm set to regression.', self)
            self.goal = 'regression'

        # Check validity models
        model_list = ['BNB', 'GNB', 'MNB', 'GP', 'LinReg', 'LogReg', 'LDA',
                      'QDA', 'KNN', 'Tree', 'ET', 'RF', 'AdaBoost', 'GBM',
                      'XGB', 'lSVM', 'kSVM', 'PA', 'SGD', 'MLP']

        # Final list of models to be used
        # Class attribute because needed for boxplot
        self.final_models = []
        if self.models is None:  # Use all possible models (default)
            self.final_models = model_list.copy()
        else:
            # If only one model, make list for enumeration
            if not isinstance(self.models, list):
                self.models = [self.models]

            # Remove duplicates keeping same order
            # Use and on the None output of set.add to call the function
            self.models = [not set().add(x.lower()) and x
                           for x in self.models if x.lower() not in set()]

            # Set models to right name
            for m in self.models:
                # Compare strings case insensitive
                if m.lower() not in map(str.lower, model_list):
                    prlog(f"Unknown model {m}. Removed from pipeline.", self)
                else:
                    for n in model_list:
                        if m.lower() == n.lower():
                            self.final_models.append(n)
                            break

        # Check if XGBoost is available
        if 'XGB' in self.final_models and not xgb_import:
            prlog("Unable to import XGBoost. Model removed from pipeline.",
                  self)
            self.final_models.remove('XGB')

        # Linear regression can't perform classification
        if 'LinReg' in self.final_models and self.goal != 'regression':
            prlog("Linear Regression can't perform classification tasks."
                  + " Removing model from pipeline.", self)
            self.final_models.remove('LinReg')

        # Remove classification-only models from pipeline
        if self.goal == 'regression':
            self.final_models = not_regression(self.final_models)

        # Check if there are still valid models
        if len(self.final_models) == 0:
            raise ValueError(f"No models found in pipeline. Try {model_list}")

        if not self.successive_halving:
            prlog(f'Models in pipeline: {self.final_models}', self)

        # Set default metric
        if self.metric is None and self.goal == 'binary classification':
            self.metric = 'F1'
        elif self.metric is None:
            self.metric = 'MSE'

        # Check validity metric
        metric_class = ['Precision', 'Recall', 'Accuracy', 'F1', 'AUC',
                        'LogLoss', 'Jaccard']
        mreg = ['R2', 'max_error', 'MAE', 'MSE', 'MSLE']
        for m in metric_class + mreg:
            # Compare strings case insensitive
            if self.metric.lower() == m.lower():
                self.metric = m

        if self.metric not in metric_class + mreg:
            raise ValueError('Unknown metric. Try one of {}.'
                             .format(metric_class if self.goal ==
                                     'binary classification' else mreg))
        elif self.metric not in mreg and self.goal != 'binary classification':
            raise ValueError("{} is an invalid metric for {}. Try one of {}."
                             .format(self.metric, self.goal, mreg))

        # << ============ Save ATOM's parameters to log ============ >>

        params = 'Parameters: {metric: ' + str(self.metric) + \
                 ', successive_halving: ' + str(self.successive_halving) + \
                 ', skip_steps: ' + str(self.skip_steps) + \
                 ', impute: ' + str(self.impute) + \
                 ', strategy: ' + str(self.strategy) + \
                 ', solver: ' + str(self.solver) + \
                 ', max_features: ' + str(self.max_features) + \
                 ', ratio: ' + str(self.ratio) + \
                 ', max_iter: ' + str(self.max_iter) + \
                 ', max_time: ' + str(self.max_time) + \
                 ', eps: ' + str(self.eps) + \
                 ', batch_size: ' + str(self.batch_size) + \
                 ', init_points: ' + str(self.init_points) + \
                 ', plot_bo: ' + str(self.plot_bo) + \
                 ', cross_validation: ' + str(self.cross_validation) + \
                 ', n_splits: ' + str(self.n_splits) + \
                 ', n_jobs: ' + str(self.n_jobs) + \
                 ', verbose: ' + str(self.verbose) + '}'

        prlog(params, self, 5)  # Never print (only write to log)

        # << ============ Data preprocessing ============ >>

        prlog('\nData preprocessing =============>', self, 1)

        X = self.check_features(X, output=True)

        # Impute values
        if self.impute is not None:
            X = self.imputer(X, strategy=self.impute)

        X = self.encoder(X)  # Perform encoding on features

        # Perform feature selection
        if self.strategy is not None:
            X = self.feature_selection(X, Y,
                                       strategy=self.strategy,
                                       solver=self.solver,
                                       max_features=self.max_features)

        # Get unqiue target values before encoding (for later print)
        self.unique = np.unique(Y)

        # Make sure the target categories are numerical
        if Y.dtype.kind not in 'ifu':
            Y = pd.Series(LabelEncoder().fit_transform(Y), name=Y.name)

        self.data = data_preparation(self, X, Y)  # Creates dct of data

        # Save data to class attribute for later use or for the user
        self.X, self.Y = X, Y
        self.X_train, self.Y_train = self.data['X_train'], self.data['Y_train']
        self.X_test, self.Y_test = self.data['X_test'], self.data['Y_test']
        self.dataset = X.merge(Y.to_frame(), left_index=True, right_index=True)

        # << =================== Core ==================== >>

        if self.successive_halving:
            prlog('\n\nRunning successive halving =================>>', self)
            iteration = 0
            while len(self.final_models) > 2**self.skip_steps - 1:
                # Select percentage of data to use for this iteration
                pct = 100./len(self.final_models)  # Use 1/N of the data
                self.data = data_preparation(self, X, Y, pct)
                prlog('\n\n<<================ Iteration {} ================>>'
                      .format(iteration), self)
                prlog(f'Models in pipeline: {self.final_models}', self)
                print_data_stats(self)

                # Run iteration
                results = run_iteration(self)
                self.results.append(results)

                # Select best models for halving
                lx = results.nlargest(n=int(len(self.final_models)/2),
                                      columns='cv_mean',
                                      keep='all')

                # Keep the models in the same order
                n = []  # List of new models
                [n.append(m) for m in self.final_models if m in list(lx.model)]
                self.final_models = n.copy()
                iteration += 1

        else:
            print_data_stats(self)
            prlog('\n\nRunning pipeline =================>', self)
            self.results = run_iteration(self)

        # <====================== End fit function ======================>

    def boxplot(self, i=-1, figsize=None, filename=None):

        '''
        DESCRIPTION -----------------------------------

        Plot a boxplot of the found metric results.

        PARAMETERS -------------------------------------

        i        --> iteration of the successive halving to plot (default last)
        figsize  --> figure size: format as (x, y)
        filename --> name of the file to save

        '''

        results, names = [], []
        try:  # Can't make plot before running fit!
            df = self.results[i] if self.successive_halving else self.results
            for m in df.model:
                results.append(getattr(self, m).results)
                names.append(getattr(self, m).shortname)
        except (IndexError, AttributeError):
            raise Exception('You need to fit the class before plotting!')

        if figsize is None:
            figsize = (int(8+len(names)/2), 6)

        sns.set_style('darkgrid')
        fig, ax = plt.subplots(figsize=figsize)
        plt.boxplot(results)
        ax.set_xticklabels(names)
        plt.xlabel('Model', fontsize=16, labelpad=12)
        plt.ylabel(self.metric, fontsize=16, labelpad=12)
        plt.title('Model comparison', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        plt.show()

    def plot_successive_halving(self, figsize=(10, 6), filename=None):

        '''
        DESCRIPTION -----------------------------------

        Plot a boxplot of the found metric results.

        PARAMETERS -------------------------------------

        figsize  --> figure size: format as (x, y)
        filename --> name of the file to save

        '''

        if not self.successive_halving:
            raise ValueError('This plot is only available if the class was ' +
                             'fitted using a successive halving approach!')

        models = self.results[0].model  # List of models in first iteration
        linx = [[] for m in models]
        liny = [[] for m in models]
        try:  # Can't make plot before running fit!
            for m, df in enumerate(self.results):
                for n, model in enumerate(models):
                    if model in df.model.values:  # If model in iteration
                        linx[n].append(m)
                        liny[n].append(df.cv_mean[df.model == model].values[0])
        except (IndexError, AttributeError):
            raise Exception('You need to fit the class before plotting!')

        sns.set_style('darkgrid')
        fig, ax = plt.subplots(figsize=figsize)
        for x, y, label in zip(linx, liny, models):
            plt.plot(x, y, lw=2, marker='o', label=label)
        plt.xlim(-0.1, len(self.results)-0.9)
        plt.xlabel('Iteration', fontsize=16, labelpad=12)
        plt.ylabel(self.metric, fontsize=16, labelpad=12)
        plt.title('Successive halving scores', fontsize=16)
        plt.legend(frameon=False, fontsize=14)
        ax.set_xticks(range(len(self.results)))
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        plt.show()

    def plot_correlation(self, X=None, Y=None,
                         figsize=(10, 10), filename=None):

        '''
        DESCRIPTION -----------------------------------

        Plot the feature's correlation matrix. Ignores non-numeric columns.

        PARAMETERS -------------------------------------

        X        --> feature's dataframe (None if ATOM is already fitted)
        Y        --> target (None if ATOM is already fitted)
        figsize  --> figure size: format as (x, y)
        filename --> name of the file to save

        '''

        if X is None and Y is None:
            try:
                df = self.dataset
            except AttributeError:
                raise ValueError('Provide data for the plot or fit the class!')
        elif X is None:
            df = self.data['X']
        else:
            df = convert_to_df(X)

        if Y is not None:
            if not isinstance(Y, pd.Series):
                Y = pd.Series(Y, name='target')
            df = df.merge(Y.to_frame(), left_index=True, right_index=True)

        # Compute the correlation matrix
        corr = X.corr()

        # Generate a mask for the upper triangle
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        sns.set_style('white')
        fig, ax = plt.subplots(figsize=figsize)

        # Draw the heatmap with the mask and correct aspect ratio
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.title('Feature correlation matrix', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        fig.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        plt.show()
