# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing the ATOM class.

"""

# Standard packages
import numpy as np
from typeguard import typechecked
from typing import Union, Optional
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pandas_profiling import ProfileReport

# Own modules
from .branch import Branch
from .basepredictor import BasePredictor
from .basetrainer import BaseTrainer
from .basetransformer import BaseTransformer
from .data_cleaning import (
    Cleaner,
    Scaler,
    Imputer,
    Encoder,
    Outliers,
    Balancer,
)
from .feature_engineering import FeatureGenerator, FeatureSelector
from .training import (
    DirectClassifier,
    DirectRegressor,
    SuccessiveHalvingClassifier,
    SuccessiveHalvingRegressor,
    TrainSizingClassifier,
    TrainSizingRegressor,
)
from .models import CustomModel
from .plots import ATOMPlotter
from .utils import (
    SEQUENCE_TYPES, X_TYPES, Y_TYPES, TRAIN_TYPES, flt, lst,
    infer_task, check_method, check_scaling, check_deep,
    names_from_estimator, add_transformer, transform,
    method_to_log, composed, crash, CustomDict,
)


class ATOM(BasePredictor, ATOMPlotter):
    """ATOM base class.

    The ATOM class is a convenient wrapper for all data cleaning,
    feature engineering and trainer estimators in this package.
    Provide the dataset to the class, and apply all transformations
    and model management from here.

    Warning: This class should not be called directly. Use descendant
    classes instead.

    """

    @composed(crash, method_to_log)
    def __init__(self, arrays, n_rows, test_size):
        self.n_rows = n_rows
        self.test_size = test_size
        self.missing = ["", "?", "NA", "nan", "NaN", "None", "inf"]

        # Branching attributes
        self._current = "master"  # Current pipeline
        self._branches = {self._current: Branch(self, self._current)}

        # Training attributes
        self._models = CustomDict()
        self._metric = CustomDict()
        self.errors = {}

        self.log("<< ================== ATOM ================== >>", 1)

        # Prepare the provided data
        self.branch.data, self.branch.idx = self._get_data_and_idx(arrays)

        # Save the test_size fraction for later use
        self._test_size = self.branch.idx[1] / len(self.dataset)

        self.task = infer_task(self.y, goal=self.goal)
        self.log(f"Algorithm task: {self.task}.", 1)
        if self.n_jobs > 1:
            self.log(f"Parallel processing with {self.n_jobs} cores.", 1)

        # Assign mapping
        try:  # Can fail if str and NaN in target column
            classes = sorted(self.y.unique())
        except TypeError:
            classes = self.y.unique()
        self.branch.mapping = {str(value): value for value in classes}

        self.log('', 1)  # Add empty rows around stats for cleaner look
        self.stats(1)
        self.log('', 1)

    def __repr__(self):
        out = f"{self.__class__.__name__}"
        out += f"\n --> Branches: {', '.join(list(self._branches.keys()))}"
        out += f"\n --> Models: {', '.join(lst(self.models)) if self.models else None}"
        out += f"\n --> Metric: {', '.join(lst(self.metric)) if self.metric else None}"
        out += f"\n --> Errors: {len(self.errors)}"

        return out

    def __iter__(self):
        yield from self.pipeline.iteritems()

    def __len__(self):
        return len(self.pipeline)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.pipeline[item]
        elif isinstance(item, str):
            return self.pipeline.loc[item]
        else:
            return self.pipeline.iloc[item]

    # Utility properties =========================================== >>

    @BasePredictor.branch.setter
    @typechecked
    def branch(self, branch: str):
        if not branch:
            raise ValueError("Can't create a branch with an empty name!")
        elif branch.lower() in self._branches:
            self._current = branch.lower()
            self.log(f"Switched to branch '{branch}'.", 1)
        else:
            # Branch can be created from current or another
            if "_from_" in branch:
                new_branch, from_branch = branch.lower().split("_from_")
            else:
                new_branch, from_branch = branch.lower(), self._current

            if from_branch not in self._branches:
                raise ValueError(
                    "The selected branch to split from does not exist! Print "
                    "atom.branch for an overview of the branches in the pipeline."
                )

            self._branches[new_branch] = Branch(self, new_branch, parent=from_branch)
            self._current = new_branch
            self.log(f"New branch '{self._current}' successfully created!", 1)

    @property
    def nans(self):
        """Returns columns with number of missing values."""
        nans = self.dataset.replace(self.missing, np.NaN).isna().sum()
        return nans[nans > 0]

    @property
    def n_nans(self):
        """Returns the total number of missing values in the dataset."""
        return self.nans.sum()

    @property
    def categorical(self):
        """Returns the names of categorical columns in the dataset."""
        if not check_deep(self.X):
            return list(self.X.select_dtypes(include=["category", "object"]).columns)
        else:
            return []

    @property
    def n_categorical(self):
        """Returns the number of categorical columns in the dataset."""
        return len(self.categorical)

    @property
    def scaled(self):
        """Returns whether the feature set is scaled."""
        return check_scaling(self.X)

    # Utility methods =============================================== >>

    @composed(crash, method_to_log)
    def stats(self, _vb: int = -2):
        """Print basic information about the dataset.

        Parameters
        ----------
        _vb: int, optional (default=-2)
            Parameter to always print if the user calls this method.

        """
        self.log("Dataset stats ================== >>", _vb)
        self.log(f"Shape: {self.shape}", _vb)

        if self.n_nans:
            self.log(f"Missing values: {self.n_nans}", _vb)
        if self.n_categorical:
            self.log(f"Categorical columns: {self.n_categorical}", _vb)

        self.log(f"Scaled: {self.scaled}", _vb)
        self.log("-----------------------------------", _vb)
        self.log(f"Train set size: {len(self.train)}", _vb)
        self.log(f"Test set size: {len(self.test)}", _vb)

        # Print count and balance of classes for classification tasks
        if self.task != "regression":
            balance = {}
            keys = ":".join(self.mapping.keys())
            for set_ in ("train", "test"):
                balance[set_] = ":".join(
                    round(self.classes[set_] / min(self.classes[set_]), 1).astype(str)
                )
            self.log("-----------------------------------", _vb + 1)
            if balance["train"] == balance["test"]:
                self.log(f"Dataset balance: {keys} <==> {balance['train']}", _vb + 1)
            else:
                self.log(f"Train set balance: {keys} <==> {balance['train']}", _vb + 1)
                self.log(f"Test set balance: {keys} <==> {balance['test']}", _vb + 1)
            self.log("-----------------------------------", _vb + 1)
            self.log(f"Distribution of classes:", _vb + 1)
            self.log(self.classes.to_markdown(), _vb + 1)

    @composed(crash, method_to_log, typechecked)
    def report(
        self,
        dataset: str = "dataset",
        n_rows: Optional[Union[int, float]] = None,  # float for 1e3...
        filename: Optional[str] = None,
    ):
        """Create an extensive profile analysis report of the data.

        The profile report is rendered in HTML5 and CSS3. Note that
        this method can be slow for rows>10k.

        Parameters
        ----------
        dataset: str, optional(default="dataset")
            Data set to get the report from.

        n_rows: int or None, optional(default=None)
            Number of (randomly picked) rows in to process. None for
            all rows.

        filename: str or None, optional (default=None)
            Name to save the file with (as .html). None to not save
            anything.

        Returns
        -------
        profile: ProfileReport
            Created profile object.

        """
        # If rows=None, select all rows in the dataframe
        n_rows = getattr(self, dataset).shape[0] if n_rows is None else int(n_rows)

        self.log("Creating profile report...", 1)

        profile = ProfileReport(getattr(self, dataset).sample(n_rows))
        try:  # Render if possible (for notebooks)
            from IPython.display import display

            display(profile)
        except ModuleNotFoundError:
            pass

        if filename:
            if not filename.endswith(".html"):
                filename = filename + ".html"
            profile.to_file(filename)
            self.log("Report saved successfully!", 1)

        return profile

    @composed(crash, method_to_log, typechecked)
    def transform(
        self, X: X_TYPES, y: Y_TYPES = None, verbose: Optional[int] = None, **kwargs
    ):
        """Transform new data through all transformers in a branch.

        The outliers and balance transformations are not included by
        default since they should only be applied on the training set.

        Parameters
        ----------
        X: dict, list, tuple, np.ndarray or pd.DataFrame
            Feature set with shape=(n_samples, n_features).

        y: int, str, sequence or None, optional (default=None)
            - If None: y is ignored in the transformers.
            - If int: Index of the target column in X.
            - If str: Name of the target column in X.
            - Else: Target column with shape=(n_samples,).

        verbose: int or None, optional (default=None)
            Verbosity level of the transformers. If None, it uses
            the atom's verbosity.

        **kwargs
            Additional keyword arguments to customize which transforming
            methods to apply. You can either select them via their index,
            e.g. pipeline = [0, 1, 4] or include/exclude them via every
            individual transformer, e.g. outliers=True, encode=False.

        Returns
        -------
        X: pd.DataFrame
            Transformed feature set.

        y: pd.Series
            Transformed target column. Only returned if provided.

        """
        if verbose is None:
            verbose = self.verbose

        return transform(self.pipeline, X, y, verbose, **kwargs)

    @composed(crash, method_to_log, typechecked)
    def save_data(self, filename: str = None, dataset: str = "dataset"):
        """Save the data in the current branch to a csv file.

        Parameters
        ----------
        filename: str or None, optional (default=None)
            Name to save the file with. None or "auto" for default name.

        dataset: str, optional (default="dataset")
            Data set to save.

        """
        if not filename:
            filename = f"{self.__class__.__name__}_{dataset}"
        elif filename == "auto" or filename.endswith("/auto"):
            filename = filename.replace("auto", f"{self.__class__.__name__}_{dataset}")

        if not filename.endswith(".csv"):
            filename += ".csv"
        getattr(self, dataset).to_csv(filename, index=False)
        self.log("Data set saved successfully!", 1)

    @composed(crash, typechecked)
    def export_pipeline(self, model: Optional[str] = None):
        """Export atom's pipeline to a sklearn's Pipeline object.

        Optionally, you can add a model as final estimator. If the
        model needs feature scaling and there is no scaler in the
        pipeline, a StandardScaler will be added. The returned
        pipeline is already fitted.

        Parameters
        ----------
        model: str or None, optional (default=None)
            Name of the model to add as a final estimator to the
            pipeline. If None, no model is added.

        Returns
        -------
        pipeline: Pipeline
            Pipeline in the current branch as a sklearn object.

        """
        if len(self.pipeline) == 0:
            raise PermissionError("The pipeline seems to be empty!")

        steps = [item for item in self.pipeline.iteritems()]

        if model:
            model = getattr(self, self._get_model_name(model)[0])
            has_scaling = any(self.pipeline.index.str.contains("scale"))
            if model.needs_scaling and not has_scaling:
                scaler = StandardScaler().fit(self.X_train, self.y_train)
                steps += [("standardscaler", scaler)]
            steps += [(model.name, model.estimator)]

        return Pipeline(steps)

    # Transformer methods ========================================== >>

    def _prepare_kwargs(self, kwargs, params=None):
        """Return kwargs with atom's values if not specified."""
        for attr in BaseTransformer.attrs:
            if (not params or attr in params) and attr not in kwargs:
                kwargs[attr] = getattr(self, attr)

        return kwargs

    @composed(crash, method_to_log, typechecked)
    def add(self, transformer, name: Optional[str] = None, train_only: bool = False):
        """Add a transformer to the current branch.

        If the transformer is not fitted, it is fitted on the
        complete training set. Afterwards, the data set is
        transformed and the transformer is added to atom's
        pipeline. If the transformer is a sklearn Pipeline,
        every step will be transformed independently on the
        complete data set.

        If the transformer has a n_jobs and/or random_state
        parameter and it is left to its default value, it
        adopts atom's value.

        Parameters
        ----------
        transformer: class
            Transformer to add to the pipeline. Should implement a
            `transform` method.

        name: str or None, optional (default=None)
            Name of the transformation step. If None, it defaults to
            the __name__ of the transformer's class (lower case).

        train_only: bool, optional (default=False)
            Whether to apply the transformer only on the train set or
            on the complete dataset.

        """
        if isinstance(transformer, Pipeline):
            # Recursively add all transformers to the pipeline
            for name, est in transformer.named_steps.items():
                add_transformer(self, est, name, train_only)
        else:
            add_transformer(self, transformer, name, train_only)
            self.log(
                f"Transformer {self.pipeline[-1].__class__.__name__} "
                f"successfully added to the pipeline.", 1
            )

    @composed(crash, method_to_log)
    def scale(self, **kwargs):
        """Scale features to mean=0 and std=1.

        This class is equal to sklearn's StandardScaler except that it
        returns a dataframe when provided and it ignores non-numerical
        columns (instead of raising an exception). The scaler is fitted
        only on the training set to avoid data leakage.

        """
        check_method(self, "scale")
        kwargs = self._prepare_kwargs(kwargs, Scaler().get_params())
        add_transformer(self, Scaler(**kwargs), name="scale")

    @composed(crash, method_to_log, typechecked)
    def clean(
        self,
        prohibited_types: Optional[Union[str, SEQUENCE_TYPES]] = None,
        strip_categorical: bool = True,
        maximum_cardinality: bool = True,
        minimum_cardinality: bool = True,
        missing_target: bool = True,
        encode_target: bool = True,
        **kwargs,
    ):
        """Applies standard data cleaning steps on the dataset.

         These steps can include:
            - Strip categorical features from white spaces.
            - Removing columns with prohibited data types.
            - Removing categorical columns with maximal cardinality.
            - Removing columns with minimum cardinality.
            - Removing rows with missing values in the target column.
            - Encode the target column.

        See the data_cleaning.py module for a description of the parameters.

        """
        check_method(self, "clean")
        kwargs = self._prepare_kwargs(kwargs, Cleaner().get_params())
        cleaner = Cleaner(
            prohibited_types=prohibited_types,
            strip_categorical=strip_categorical,
            maximum_cardinality=maximum_cardinality,
            minimum_cardinality=minimum_cardinality,
            missing_target=missing_target,
            encode_target=encode_target,
            **kwargs,
        )
        # Pass atom's missing values to the cleaner before transforming
        cleaner.missing = self.missing

        add_transformer(self, cleaner, name="clean")

        # Assign mapping (if it changed)
        if cleaner.mapping:
            self.branch.mapping = cleaner.mapping

    @composed(crash, method_to_log, typechecked)
    def impute(
        self,
        strat_num: Union[int, float, str] = "drop",
        strat_cat: str = "drop",
        min_frac_rows: float = 0.5,
        min_frac_cols: float = 0.5,
        **kwargs,
    ):
        """Handle missing values in the dataset.

        Impute or remove missing values according to the selected strategy.
        Also removes rows and columns with too many missing values. Use
        the `missing` attribute to customize what are considered "missing
        values".

        See the data_cleaning.py module for a description of the parameters.

        """
        check_method(self, "impute")
        kwargs = self._prepare_kwargs(kwargs, Imputer().get_params())
        imputer = Imputer(
            strat_num=strat_num,
            strat_cat=strat_cat,
            min_frac_rows=min_frac_rows,
            min_frac_cols=min_frac_cols,
            **kwargs,
        )
        # Pass atom's missing values to the imputer before transforming
        imputer.missing = self.missing

        add_transformer(self, imputer, name="impute")

    @composed(crash, method_to_log, typechecked)
    def encode(
        self,
        strategy: str = "LeaveOneOut",
        max_onehot: Optional[int] = 10,
        frac_to_other: Optional[float] = None,
        **kwargs,
    ):
        """Perform encoding of categorical features.

        The encoding type depends on the number of unique values in the column:
            - If n_unique=2, use Label-encoding.
            - If 2 < n_unique <= max_onehot, use OneHot-encoding.
            - If n_unique > max_onehot, use `strategy`-encoding.

        Also replaces classes with low occurrences with the value `other`
        in order to prevent too high cardinality. Categorical features are
        defined as all columns whose dtype.kind not in `ifu`. Will raise
        an error if it encounters missing values or unknown classes when
        transforming.

        See the data_cleaning.py module for a description of the parameters.

        """
        check_method(self, "encode")
        kwargs = self._prepare_kwargs(kwargs, Encoder().get_params())
        encoder = Encoder(
            strategy=strategy,
            max_onehot=max_onehot,
            frac_to_other=frac_to_other,
            **kwargs,
        )

        add_transformer(self, encoder, name="encode")

    @composed(crash, method_to_log, typechecked)
    def outliers(
        self,
        strategy: Union[int, float, str] = "drop",
        max_sigma: Union[int, float] = 3,
        include_target: bool = False,
        **kwargs,
    ):
        """Remove or replace outliers in the training set.

        Outliers are values that lie further than `max_sigma` * std
        away from the mean of the column. Ignores categorical columns.
        Only outliers from the training set are removed to maintain the
        original distribution of target values in the test set. Ignores
        categorical columns.

        See the data_cleaning.py module for a description of the parameters.

        """
        check_method(self, "outliers")
        kwargs = self._prepare_kwargs(kwargs, Outliers().get_params())
        outliers = Outliers(
            strategy=strategy,
            max_sigma=max_sigma,
            include_target=include_target,
            **kwargs,
        )

        add_transformer(self, outliers, name="outliers", train_only=True)

    @composed(crash, method_to_log, typechecked)
    def balance(self, strategy: str = "ADASYN", **kwargs):
        """Balance the number of rows per class in the target column.

        Only the training set is balanced in order to maintain the
        original distribution of target classes in the test set. Use
        only for classification tasks.

        See the data_cleaning.py module for a description of the parameters.

        """
        check_method(self, "balance")
        if not self.goal.startswith("class"):
            raise PermissionError(
                "The balance method is only available for classification tasks!"
            )

        kwargs = self._prepare_kwargs(kwargs, Balancer().get_params())
        balancer = Balancer(strategy=strategy, **kwargs)

        # Add mapping from atom to balancer for cleaner printing
        balancer.mapping = self.mapping

        add_transformer(self, balancer, name="balance", train_only=True)

        # Attach the estimator attribute to atom's branch
        setattr(self.branch, strategy.lower(), getattr(balancer, strategy.lower()))

    @composed(crash, method_to_log, typechecked)
    def feature_generation(
        self,
        strategy: str = "DFS",
        n_features: Optional[int] = None,
        generations: int = 20,
        population: int = 500,
        operators: Optional[Union[str, SEQUENCE_TYPES]] = None,
        **kwargs,
    ):
        """Apply automated feature engineering.

        Use Deep feature Synthesis or a genetic algorithm to create
        new combinations of existing features to capture the non-linear
        relations between the original features. Attributes created by
        the class are attached to atom.

        See the feature_engineering.py module for a description of the parameters.

        """
        check_method(self, "feature_generation")
        kwargs = self._prepare_kwargs(kwargs, FeatureGenerator().get_params())
        feature_generator = FeatureGenerator(
            strategy=strategy,
            n_features=n_features,
            generations=generations,
            population=population,
            operators=operators,
            **kwargs,
        )

        add_transformer(self, feature_generator, name="feature_generation")

        # Attach used attributes to atom's branch
        if strategy.lower() in ("gfg", "genetic"):
            for attr in ["symbolic_transformer", "genetic_features"]:
                setattr(self.branch, attr, getattr(feature_generator, attr))

    @composed(crash, method_to_log, typechecked)
    def feature_selection(
        self,
        strategy: Optional[str] = None,
        solver: Optional[Union[str, callable]] = None,
        n_features: Optional[Union[int, float]] = None,
        max_frac_repeated: Optional[Union[int, float]] = 1.0,
        max_correlation: Optional[float] = 1.0,
        **kwargs,
    ):
        """Apply feature selection techniques.

        Remove features according to the selected strategy. Ties
        between features with equal scores will be broken in an
        unspecified way. Additionally, removes features with too low
        variance and finds pairs of collinear features based on the
        Pearson correlation coefficient. For each pair above the
        specified limit (in terms of absolute value), it removes one
        of the two. Plotting methods and attributes created by the
        class are attached to atom.

        See the feature_engineering.py module for a description of the parameters.

        """
        check_method(self, "feature_selection")
        if isinstance(strategy, str):
            if strategy.lower() == "univariate" and solver is None:
                if self.goal.startswith("reg"):
                    solver = "f_regression"
                else:
                    solver = "f_classif"
            elif strategy.lower() in ["sfm", "rfe", "rfecv", "sfs"]:
                if solver is None and self.winner:
                    solver = self.winner.estimator
                elif isinstance(solver, str):
                    # In case the user already filled the task...
                    if not solver.endswith("_class") and not solver.endswith("_reg"):
                        solver += "_reg" if self.task.startswith("reg") else "_class"

            # If the run method was called before, use the main metric
            if strategy.lower() in ("rfecv", "sfs"):
                if self._metric and "scoring" not in kwargs:
                    kwargs["scoring"] = self._metric[0]

        kwargs = self._prepare_kwargs(kwargs, FeatureSelector().get_params())
        feature_selector = FeatureSelector(
            strategy=strategy,
            solver=solver,
            n_features=n_features,
            max_frac_repeated=max_frac_repeated,
            max_correlation=max_correlation,
            **kwargs,
        )

        add_transformer(self, feature_selector, name="feature_selection")

        # Attach used attributes to atom's branch
        attrs = (
            "collinear",
            "univariate",
            "pca",
            "sfm",
            "rfe",
            "rfecv",
            "sfs",
            "feature_importance",
        )
        for attr in attrs:
            if getattr(feature_selector, attr) is not None:
                setattr(self.branch, attr, getattr(feature_selector, attr))

    def automl(self, **kwargs):
        """Use AutoML to search for an optimized pipeline.

        Uses the TPOT package to perform an automated search of
        transformers and a final estimator that maximizes a metric
        on the dataset. The resulting transformations and estimator
        are merged with atom's pipeline. The tpot instance can be
        accessed through the `tpot` attribute.

        Parameters
        ----------
        **kwargs
            Keyword arguments for tpot's classifier/regressor.

        """
        from tpot import TPOTClassifier, TPOTRegressor

        check_method(self, "automl")
        # Define the scoring parameter
        if self._metric and not kwargs.get("scoring"):
            kwargs["scoring"] = self._metric[0]
        elif kwargs.get("scoring"):
            metric_ = BaseTrainer._prepare_metric([kwargs["scoring"]])
            if not self._metric:
                self._metric = metric_  # Update the pipeline's metric
            elif metric_[0].name != self.metric[0]:
                raise ValueError(
                    "Invalid value for the scoring parameter! The scoring "
                    "should be equal to the primary metric in the pipeline. "
                    f"Expected {self.metric[0]}, got {metric_[0].name}."
                )

        kwargs = dict(
            n_jobs=kwargs.pop("n_jobs", self.n_jobs),
            verbosity=kwargs.pop("verbosity", self.verbose),
            random_state=kwargs.pop("random_state", self.random_state),
            **kwargs,
        )
        if self.goal.startswith("class"):
            self.tpot = TPOTClassifier(**kwargs)
        else:
            self.tpot = TPOTRegressor(**kwargs)

        self.log("Fitting automl algorithm...", 1)

        self.tpot.fit(self.X_train, self.y_train)

        self.log("\nMerging automl results with atom...", 1)

        # A pipeline could consist of just a single estimator
        if len(self.tpot.fitted_pipeline_) > 1:
            for name, est in self.tpot.fitted_pipeline_[:-1].named_steps.items():
                add_transformer(self, est, name)
                self.log(
                    f" --> Transformer {self.pipeline[-1].__class__.__name__} "
                    f"successfully added to the pipeline.", 2
                )

        # Add the final estimator as a model to atom
        est = self.tpot.fitted_pipeline_[-1]
        est.acronym, est.fullname = names_from_estimator(self, est)
        model = CustomModel(self, estimator=est)
        model.estimator = model.est

        # Save metric scores on complete training and test set
        model.metric_train = flt([
            metric(model.estimator, self.X_train, self.y_train)
            for metric in self._metric
        ])
        model.metric_test = flt([
            metric(model.estimator, self.X_test, self.y_test)
            for metric in self._metric
        ])

        self._models.update({model.name: model})
        self.log(
            f" --> {model.fullname} ({model.name}) successfully added to the models.", 2
        )

    # Training methods ============================================= >>

    def _check(self, metric, gib, needs_proba, needs_threshold):
        """Check whether the provided metric is valid.

        Parameters
        ----------
        metric: str, sequence or callable
            Metric provided for the run.

        gib: bool or sequence
            Whether the metric is a score or a loss function.

        needs_proba: bool or sequence
            Whether the metric function requires probability estimates
            out of a classifier.

        needs_threshold: bool or sequence
            Whether the metric function takes a continuous decision
            certainty.

        Returns
        -------
        metric: str, function, scorer or sequence
            Metric for the run. Should be the same as previous run.

        """
        if self._metric:
            # If the metric is empty, assign the existing one
            if metric is None:
                metric = self._metric
            else:
                # If there's a metric, it should be the same as previous run
                _metric = BaseTrainer._prepare_metric(
                    metric=lst(metric),
                    greater_is_better=gib,
                    needs_proba=needs_proba,
                    needs_threshold=needs_threshold,
                )

                if list(_metric.keys()) != list(self._metric.keys()):
                    raise ValueError(
                        "Invalid value for the metric parameter! The metric "
                        "should be the same as previous run. Expected "
                        f"{self.metric}, got {flt([m.name for m in _metric])}."
                    )

        return metric

    def _run(self, trainer):
        """Run the trainer.

        If all models failed, catch the errors and pass them to the
        atom before raising the exception. If successful run, update
        all relevant attributes and methods.

        Parameters
        ----------
        trainer: class
            Trainer instance to run.

        """
        try:
            trainer._branches = {self._current: self.branch}
            trainer._current = self._current
            trainer.run()
        finally:
            # Catch errors and pass them to atom's attribute
            for model, error in trainer.errors.items():
                self.errors[model] = error
                self._models.pop(model.lower(), None)

        # Update attributes
        self._models.update(trainer._models)
        self._metric = trainer._metric

        for model in self._models:
            self.errors.pop(model.name, None)  # Remove model from errors (if there)
            model.T = self  # Change the model's parent class from trainer to atom

    @composed(crash, method_to_log, typechecked)
    def run(
        self,
        models: Union[str, callable, SEQUENCE_TYPES],
        metric: Optional[Union[str, callable, SEQUENCE_TYPES]] = None,
        greater_is_better: Union[bool, SEQUENCE_TYPES] = True,
        needs_proba: Union[bool, SEQUENCE_TYPES] = False,
        needs_threshold: Union[bool, SEQUENCE_TYPES] = False,
        n_calls: Union[int, SEQUENCE_TYPES] = 0,
        n_initial_points: Union[int, SEQUENCE_TYPES] = 5,
        est_params: Optional[dict] = None,
        bo_params: Optional[dict] = None,
        bagging: Union[int, SEQUENCE_TYPES] = 0,
        **kwargs,
    ):
        """Fit the models in a direct fashion.

        Fit and evaluate over the models. Contrary to SuccessiveHalving
        and TrainSizing, the direct approach only iterates once over the
        models, using the full dataset.

        See the basetrainer.py module for a description of the parameters.

        """
        metric = self._check(metric, greater_is_better, needs_proba, needs_threshold)

        params = (
            models, metric, greater_is_better, needs_proba, needs_threshold,
            n_calls, n_initial_points, est_params, bo_params, bagging
        )

        kwargs = self._prepare_kwargs(kwargs)
        if self.goal.startswith("class"):
            trainer = DirectClassifier(*params, **kwargs)
        else:
            trainer = DirectRegressor(*params, **kwargs)

        self._run(trainer)

    @composed(crash, method_to_log, typechecked)
    def successive_halving(
        self,
        models: Union[str, callable, SEQUENCE_TYPES],
        metric: Optional[Union[str, callable, SEQUENCE_TYPES]] = None,
        greater_is_better: Union[bool, SEQUENCE_TYPES] = True,
        needs_proba: Union[bool, SEQUENCE_TYPES] = False,
        needs_threshold: Union[bool, SEQUENCE_TYPES] = False,
        skip_runs: int = 0,
        n_calls: Union[int, SEQUENCE_TYPES] = 0,
        n_initial_points: Union[int, SEQUENCE_TYPES] = 5,
        est_params: Optional[dict] = None,
        bo_params: Optional[dict] = None,
        bagging: Union[int, SEQUENCE_TYPES] = 0,
        **kwargs,
    ):
        """Fit the models in a successive halving fashion.

        If you want to compare similar models, you can choose to use a
        successive halving approach to run the pipeline. This technique
        is a bandit-based algorithm that fits N models to 1/N of the data.
        The best half are selected to go to the next iteration where the
        process is repeated. This continues until only one model remains,
        which is fitted on the complete dataset. Beware that a model's
        performance can depend greatly on the amount of data on which it
        is trained. For this reason, we recommend only to use this
        technique with similar models, e.g. only using tree-based models.

        See the basetrainer.py module for a description of the parameters.

        """
        metric = self._check(metric, greater_is_better, needs_proba, needs_threshold)

        params = (
            models, metric, greater_is_better, needs_proba, needs_threshold,
            skip_runs, n_calls, n_initial_points, est_params, bo_params, bagging
        )

        kwargs = self._prepare_kwargs(kwargs)
        if self.goal.startswith("class"):
            trainer = SuccessiveHalvingClassifier(*params, **kwargs)
        else:
            trainer = SuccessiveHalvingRegressor(*params, **kwargs)

        self._run(trainer)

    @composed(crash, method_to_log, typechecked)
    def train_sizing(
        self,
        models: Union[str, callable, SEQUENCE_TYPES],
        metric: Optional[Union[str, callable, SEQUENCE_TYPES]] = None,
        greater_is_better: Union[bool, SEQUENCE_TYPES] = True,
        needs_proba: Union[bool, SEQUENCE_TYPES] = False,
        needs_threshold: Union[bool, SEQUENCE_TYPES] = False,
        train_sizes: TRAIN_TYPES = np.linspace(0.2, 1.0, 5),
        n_calls: Union[int, SEQUENCE_TYPES] = 0,
        n_initial_points: Union[int, SEQUENCE_TYPES] = 5,
        est_params: Optional[dict] = None,
        bo_params: Optional[dict] = None,
        bagging: Union[int, SEQUENCE_TYPES] = 0,
        **kwargs,
    ):
        """Fit the models in a train sizing fashion.

        When training models, there is usually a trade-off between model
        performance and computation time that is regulated by the number
        of samples in the training set. The TrainSizing class can be used
        to create insights in this trade-off and help determine the optimal
        size of the training set.

        See the basetrainer.py module for a description of the parameters.

        """
        metric = self._check(metric, greater_is_better, needs_proba, needs_threshold)

        params = (
            models, metric, greater_is_better, needs_proba, needs_threshold,
            train_sizes, n_calls, n_initial_points, est_params, bo_params, bagging
        )

        kwargs = self._prepare_kwargs(kwargs)
        if self.goal.startswith("class"):
            trainer = TrainSizingClassifier(*params, **kwargs)
        else:
            trainer = TrainSizingRegressor(*params, **kwargs)

        self._run(trainer)
