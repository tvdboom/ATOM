# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing the documentation rendering.

"""

from __future__ import annotations

import importlib
from inspect import (
    Parameter, getattr_static, getdoc, getmembers, getsourcelines, isclass,
    isfunction, isroutine, signature,
)
from typing import Any, Callable, Optional

import regex as re
import yaml


# Variables ======================================================== >>

# Mapping of keywords to urls
# Usage in docs: [anchor][key] or [key][] -> [anchor][value]
CUSTOM_URLS = dict(
    # API
    api="https://scikit-learn.org/stable/developers/develop.html",
    sycl_device_filter="https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md#sycl_device_filter",
    warnings="https://docs.python.org/3/library/warnings.html#the-warnings-filter",
    # ATOM
    rangeindex="https://pandas.pydata.org/docs/reference/api/pandas.RangeIndex.html",
    experiment="https://www.mlflow.org/docs/latest/tracking.html#organizing-runs-in-experiments",
    evalml="https://evalml.alteryx.com/en/stable/index.html",
    automlsearch="https://evalml.alteryx.com/en/stable/autoapi/evalml/automl/index.html#evalml.automl.AutoMLSearch",
    kstest="https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test",
    pipeline="https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html",
    profiling="https://github.com/ydataai/pandas-profiling",
    profilereport="https://pandas-profiling.github.io/pandas-profiling/docs/master/rtd/pages/api/_autosummary/pandas_profiling.profile_report.ProfileReport.html",
    # BaseModel
    study="https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html",
    optimize="https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize",
    trial="https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html",
    interface="https://gradio.app/docs/#interface",
    launch="https://gradio.app/docs/#launch-header",
    explainerdashboard_package="https://github.com/oegedijk/explainerdashboard",
    explainerdashboard="https://explainerdashboard.readthedocs.io/en/latest/dashboards.html#explainerdashboard-documentation",
    # Data cleaning
    clustercentroids="https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.ClusterCentroids.html",
    onehotencoder="https://contrib.scikit-learn.org/category_encoders/onehot.html",
    hashingencoder="https://contrib.scikit-learn.org/category_encoders/hashing.html",
    quantile="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html",
    boxcox="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html",
    yeojohnson="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html",
    iforest="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html",
    ee="https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EllipticEnvelope.html",
    lof="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html",
    svm="https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html",
    dbscan="https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html",
    optics="https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html",
    standard="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html",
    minmax="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html",
    maxabs="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html",
    robust="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html",
    # Ensembles
    votingclassifier="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html",
    votingregressor="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html",
    stackingclassifier="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html",
    stackingregressor="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html",
    # Feature engineering
    dfs="https://docs.featuretools.com/en/v0.16.0/automated_feature_engineering/afe.html#deep-feature-synthesis",
    gfg="https://gplearn.readthedocs.io/en/stable/reference.html#symbolic-transformer",
    symbolictransformer="https://gplearn.readthedocs.io/en/stable/reference.html#symbolic-transformer",
    selectkbest="https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html",
    pca="https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html",
    truncatedsvd="https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html",
    sfm="https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html",
    sfs="https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html",
    rfe="https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html",
    rfecv="https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html",
    pso="https://jaswinder9051998.github.io/zoofs/Particle%20Swarm%20Optimization%20Class/",
    hho="https://jaswinder9051998.github.io/zoofs/Harris%20Hawk%20Optimization/",
    gwo="https://jaswinder9051998.github.io/zoofs/Grey%20Wolf%20Optimization%20Class/",
    dfo="https://jaswinder9051998.github.io/zoofs/Dragon%20Fly%20Optimization%20Class/",
    go="https://jaswinder9051998.github.io/zoofs/Genetic%20Optimization%20Class/",
    pearson="https://en.wikipedia.org/wiki/Pearson_correlation_coefficient",
    f_classif="https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html",
    f_regression="https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html",
    mutual_info_classif="https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html",
    mutual_info_regression="https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html",
    chi2="https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html",
    # Models
    adaboostclassifier="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html",
    adaboostregressor="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html",
    adabdocs="https://scikit-learn.org/stable/modules/ensemble.html#adaboost",
    baggingclassifier="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html",
    baggingregressor="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html",
    bagdocs="https://scikit-learn.org/stable/modules/ensemble.html#bootstrapping",
    bernoullinbclass="https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html",
    bnbdocs="https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html",
    catboostclassifier="https://catboost.ai/docs/concepts/python-reference_catboostclassifier.html",
    catboostregressor="https://catboost.ai/docs/concepts/python-reference_catboostregressor.html",
    catbdocs="https://catboost.ai/",
    multinomialnbclass="https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html",
    mnbdocs="https://scikit-learn.org/stable/modules/naive_bayes.html#multinomial-naive-bayes",
    lgbmclassifier="https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html",
    lgbmregressor="https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html",
    lgb_gpu="https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html",
    logisticregression="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html",
    lrdocs="https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression",
    randomforestclassifier="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html",
    randomforestregressor="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html",
    cumlrf="https://docs.rapids.ai/api/cuml/stable/api.html#cuml.ensemble.RandomForestClassifier",
    rfdocs="https://scikit-learn.org/stable/modules/ensemble.html#random-forests",
    sgdclassifier="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html",
    sgdregressor="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html",
    sgddocs="https://scikit-learn.org/stable/modules/sgd.html",
    xgbclassifier="https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier",
    xgbregressor="https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor",
    xgbdocs="https://xgboost.readthedocs.io/en/latest/index.html",
    # NLP
    snowballstemmer="https://www.nltk.org/api/nltk.stem.snowball.html#nltk.stem.snowball.SnowballStemmer",
    bow="https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html",
    tfidf="https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html",
    hashing="https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html",
    # Plots
    style="https://seaborn.pydata.org/tutorial/aesthetics.html#seaborn-figure-styles",
    palette="https://seaborn.pydata.org/tutorial/color_palettes.html",
)


TYPES_CONVERSION = {
    "atom.utils.CustomDict": "dict",
    "atom.utils.Model": "[models][]",
    "atom.utils.Predictor": "Predictor",
    "List[str]": "list",
    "Tuple[int, int]": "tuple",
    "Optional[float]": "float or None",
    "Optional[pandas.core.frame.DataFrame]": "pd.DataFrame or None",
    "Optional[pandas.core.series.Series]": "pd.Series or None",
    "pandas.core.frame.DataFrame": "pd.DataFrame",
    "pandas.core.series.Series": "pd.Series",
    "Union[str, List[str]]": "str or list",
    "Union[float, List[float]]": "float or list",
    "Union[float, List[float], NoneType]": "float, list or None",
    "Union[pandas.core.series.Series, pandas.core.frame.DataFrame]": "pd.Series or pd.DataFrame",
    "Optional[Union[pandas.core.series.Series, pandas.core.frame.DataFrame]]": "pd.Series, pd.DataFrame or None",
    "Union[pandas.core.series.Series, pandas.core.frame.DataFrame, NoneType]": "pd.Series, pd.DataFrame or None",
    "optuna.study.study.Study": "[Study][]",
    "optuna.trial._trial.Trial": "[Trial][]",
}


# Classes ========================================================== >>

class DummyTrainer:
    """Dummy trainer class to call model instances."""

    def __init__(self, goal: str, device: str = "cpu", engine: str = "sklearn"):
        self.goal = goal
        self.device = device
        self.engine = engine


class AutoDocs:
    """Parses an object to documentation in markdown/html.

    The docstring should follow the numpydoc style[^1]. Blocks should
    start with `::`. The following blocks are accepted:

    - tags
    - head (summary + description)
    - summary (first line of docstring, required)
    - description (detailed explanation, can contain admonitions)
    - parameters
    - attributes
    - returns
    - raises
    - see also
    - notes
    - references
    - examples
    - hyperparameters
    - methods

    Parameters
    ----------
    obj: callable
        Class, method or function to parse.

    method: callable or None
        Method of the obj to parse.

    References
    ----------
    [1] https://numpydoc.readthedocs.io/en/latest/format.html

    """

    # Blocks that can be encountered in the object's docstring
    blocks = (
        "Parameters",
        "Attributes",
        "Returns",
        "Raises",
        "See Also",
        "Notes",
        "References",
        "Examples",
        "\Z",
    )

    def __init__(self, obj: Callable, method: Optional[Callable] = None):
        if method:
            self.obj = getattr(obj, method)
            self.method = method
            self._parent_anchor = obj.__name__.lower() + "-"
        else:
            self.obj = obj
            self.method = method
            self._parent_anchor = ""

        self.method = method
        self.doc = getdoc(self.obj)

    @staticmethod
    def get_obj(command: str) -> AutoDocs:
        """Get an AutoDocs object from a string.

        The provided string must be of the form module:object or
        module:object.method.

        Parameters
        ----------
        command: str
            Line with the module and object.

        Returns
        -------
        Autodocs
            New instance of the class.

        """
        module, name = command.split(":")
        if "." in name:
            name, method = name.split(".")
            cls = getattr(importlib.import_module(module), name)
            return AutoDocs(getattr(cls, method))
        else:
            return AutoDocs(getattr(importlib.import_module(module), name))

    @staticmethod
    def parse_body(body: str) -> str:
        """Parse a parameter's body to the right Markdown format.

        Allow lists to not have to start with a new line when there's
        no preceding line.

        Parameters
        ----------
        body: str
            A parameter's body.

        Returns
        -------
        str
            The body parsed to accept ATOM's docstring list format.

        """
        text = "\n"
        if any(body.lstrip().startswith(c) for c in ("- ", "* ", "+ ")):
            text += "\n"

        text += "".join([b if b == "\n" else b[4:] for b in body.splitlines(True)])

        return text + "\n"

    def get_tags(self) -> str:
        """Return the object's tags.

        Tags are obtained from class attributes.

        Returns
        -------
        str
            Object's tags.

        """
        text = ""
        text += f"[{self.obj.acronym}](../../../user_guide/models/#predefined-models){{ .md-tag }}"
        if self.obj.needs_scaling:
            text += "[needs scaling](../../../user_guide/training/#automated-feature-scaling){ .md-tag }"
        if self.obj.accepts_sparse:
            text += "[accept sparse](../../../user_guide/data_management/#sparse-datasets){ .md-tag }"
        if self.obj.has_validation:
            text += "[allows validation](../../../user_guide/training/#in-training-validation){ .md-tag }"
        if any(engine != "sklearn" for engine in self.obj.supports_engines):
            text += "[supports acceleration](../../../user_guide/accelerating/){ .md-tag }"

        return text + "<br><br>"

    def get_signature(self) -> str:
        """Return the object's signature.

        Returns
        -------
        str
            Object's signature.

        """
        # Assign object type
        params = signature(self.obj).parameters
        if isclass(self.obj):
            obj = "class"
        elif any(p in params for p in ("cls", "self")):
            obj = "method"
        else:
            obj = "function"

        # Get signature without self, cls and type hints
        sign = []
        for k, v in params.items():
            if k not in ("cls", "self"):
                if v.default == Parameter.empty:
                    if '**' in str(v):
                        sign.append(f"**{k}")  # Add ** to kwargs
                    elif '*' in str(v):
                        sign.append(f"*{k}")  # Add * to args
                    else:
                        sign.append(k)
                else:
                    if isinstance(v.default, str):
                        sign.append(f'{k}="{v.default}"')
                    else:
                        sign.append(f"{k}={v.default}")

        sign = f"({', '.join(sign)})"

        f = self.obj.__module__.replace('.', '/')  # Module and filename sep by /
        if "atom" in self.obj.__module__:
            url = f"https://github.com/tvdboom/ATOM/blob/master/{f}.py"
        elif "sklearn" in self.obj.__module__:
            url = f"https://github.com/scikit-learn/scikit-learn/blob/baf0ea25d/{f}.py"
        else:
            url = ""

        anchor = f"<a id='{self._parent_anchor}{self.obj.__name__}'></a>"
        module = self.obj.__module__ + '.' if obj != "method" else ""
        obj = f"<em>{obj}</em>"
        name = f"<strong style='color:#008AB8'>{self.obj.__name__}</strong>"
        if url:
            line = getsourcelines(self.obj)[1]
            url = f"<span style='float:right'><a href={url}#L{line}>[source]</a></span>"

        # \n\n in front of signature to break potential lists in markdown
        return f"\n\n{anchor}<div class='sign'>{obj} {module}{name}{sign}{url}</div>"

    def get_summary(self) -> str:
        """Return the object's summary.

        The summary is the first line of the docstring.

        Returns
        -------
        str
            Object's summary.

        """
        return next(filter(None, self.doc.splitlines()))  # Get first non-empty line

    def get_description(self) -> str:
        """Return the object's description.

        The description is the first part of the docstring where the
        object is explained (before any other block). The summary is
        excluded.

        Returns
        -------
        str
            Object's description.

        """
        pattern = f".*?(?={'|'.join(self.blocks)})"
        match = re.match(pattern, self.doc[len(self.get_summary()):], re.S)
        return match.group() if match else ""

    def get_see_also(self) -> str:
        """Return the object's See Also block.

        The block is rendered as an info admonition.

        Returns
        -------
        str
            Object's See Also block.

        """
        block = "<br>" + '\n!!! info "See Also"'
        for line in self.get_block("See Also").splitlines():
            if line:
                cls = self.get_obj(line)
                summary = f"<div style='margin: -1em 0 0 1.2em'>{cls.get_summary()}</div>"

                # If it's a class, refer to the page, else to the anchor
                if cls._parent_anchor:
                    link = f"{cls._parent_anchor}-{cls.obj.__name__}"
                else:
                    link = ""

                block += f"\n    [{cls.obj.__name__}][{link}]<br>    {summary}\n"

        return block

    def get_block(self, block: str) -> str:
        """Return a block from the docstring.

        Parameters
        ----------
        block: str
            Name of the block to retrieve.

        Returns
        -------
        str
            Block in docstring.

        """
        pattern = f"(?<={block}\n{'-' * len(block)}).*?(?={'|'.join(self.blocks)})"
        match = re.search(pattern, self.doc, re.S)
        return match.group() if match else ""

    def get_table(self, blocks: list) -> str:
        """Return a table from one or multiple blocks.

        Parameters
        ----------
        blocks: list
            Blocks to create the table from.

        Returns
        -------
        str
            Table in html format.

        """
        table = ""
        for block in blocks:
            if isinstance(block, str):
                name = block.capitalize()
                config = {}
            else:
                name = next(iter(block)).capitalize()
                config = block[name.lower()]

            # Get from config which attributes to display
            if config.get("include"):
                attrs = config["include"]
            else:
                attrs = [
                    m for m, _ in getmembers(self.obj, lambda x: not isroutine(x))
                    if not m.startswith("_")
                ]

            content = ""
            if not config.get("from_docstring", True):
                for attr in attrs:
                    if ":" in attr:
                        obj = AutoDocs.get_obj(attr).obj
                    else:
                        obj = getattr(self.obj, attr)

                    if isinstance(obj, property):
                        obj = obj.fget

                    output = str(signature(obj)).split(" -> ")[-1]
                    header = f"{obj.__name__}: {TYPES_CONVERSION.get(output, output)}"
                    text = f"<div markdown class='param'>{getdoc(obj)}</div>"

                    anchor = f"<a id='{self.obj.__name__.lower()}-{obj.__name__}'></a>"
                    content += f"{anchor}<strong>{header}</strong><br>{text}"

            elif match := self.get_block(name):
                # Headers start with letter, * or [ after new line
                for header in re.findall("^[\[*\w].*?$", match, re.M):
                    # Get the body corresponding to the header
                    pattern = f"(?<={re.escape(header)}\n).*?(?=\n\w|\n\*|\n\[|\Z)"
                    body = re.search(pattern, match, re.S | re.M).group()

                    header = header.replace("*", "\*")  # Use literal * for args/kwargs
                    text = f"<div markdown class='param'>{self.parse_body(body)}</div>"

                    obj_name = header.split(":")[0]
                    anchor = f"<a id='{self.obj.__name__.lower()}-{obj_name}'></a>"
                    content += f"{anchor}<strong>{header}</strong><br>{text}"

            if content:
                table += f"<tr><td class='td_title'><strong>{name}</strong></td>"
                table += f"<td class='td_params'>{content}</td></tr>"

        if table:
            table = f"<table markdown class='table_params'>{table}</table>"

        return table

    def get_hyperparameters(self) -> str:
        """Return the object's hyperparameters.

        Hyperparameters are obtained through the _get_distributions
        method.

        Returns
        -------
        str
            Object's hyperparameters.

        """

        def create_table(trainer: Any) -> str:
            """Create a table of hyperparameter distributions.

            This table is for only one combination of device and
            engine. It renders inside a tab.

            Parameters
            ----------
            trainer: Trainer
                Parent trainer from which the model is created.

            Returns
            -------
            str
                Table in html format.

            """
            # Create the model instance from the trainer
            instance = self.obj(trainer, fast_init=True)

            text = ""
            for name, dist in instance._get_distributions().items():
                anchor = f"<a id='{self.obj.__name__.lower()}-{name}'></a>"
                text += f"{anchor}<strong>{name}</strong><br>"
                text += f"<div markdown class='param'>{dist}</div>"

            table = f"<tr><td class='td_title'><strong>Parameters</strong></td>"
            table += f"<td class='td_params'>{text}</td></tr>"

            return f"<table markdown class='table_params'>{table}</table>"

        indent = content = ""
        for goal in self.obj._estimators:
            if len(self.obj._estimators) > 1:
                indent = " " * 4
                if goal == "class":
                    content += f'\n=== "classification"\n'
                elif goal == "reg":
                    content += f'\n=== "regression"\n'

            for engine in self.obj.supports_engines:
                sub_indent = indent
                if len(self.obj.supports_engines) > 1:
                    content += f'\n{indent}=== "{engine}"\n'
                    sub_indent += " " * 4

                # sklearnex can run on cpu or gpu
                if engine == "sklearnex":
                    for device in ("cpu", "gpu"):
                        content += f'\n{sub_indent}=== "{device}"\n'

                        trainer = DummyTrainer(goal, device, engine)
                        content += f"{sub_indent + ' ' * 4}{create_table(trainer)}\n\n"
                else:
                    trainer = DummyTrainer(
                        goal=goal,
                        device="cpu" if engine == "sklearn" else "gpu",
                        engine=engine,
                    )
                    content += f"{sub_indent}{create_table(trainer)}\n\n"

        return content + "<br><br>"

    def get_methods(self, config: dict) -> str:
        """Return an overview of the methods and their blocks.

        Parameters
        ----------
        config: dict
            Options to configure. Choose from:

            - toc_only: Whether to display only the toc.
            - include: Methods to include.
            - exclude: Methods to exclude.

        Returns
        -------
        str
            Toc and blocks for all selected methods.

        """
        toc_only = config.get('toc_only')

        if config.get("include"):
            methods = config["include"]
        else:
            methods = [
                m for m, _ in getmembers(self.obj, predicate=isfunction)
                if not m.startswith("_") and m not in config.get("exclude", [])
            ]

        # Create toc
        toc = "<table markdown style='font-size: 0.9em'>"
        for method in methods:
            func = AutoDocs(self.obj, method=method)

            name = f"[{method}][{'' if toc_only else func._parent_anchor}{method}]"
            summary = func.get_summary()
            toc += f"<tr><td>{name}</td><td>{summary}</td></tr>"

        toc += "</table>"

        # Create methods
        blocks = ""
        if not toc_only:
            for method in methods:
                func = AutoDocs(self.obj, method=method)

                blocks += "<br>" + func.get_signature()
                blocks += func.get_summary() + "\n"
                if func.obj.__module__.startswith("atom"):
                    if description := func.get_description():
                        blocks += "\n\n" + description + "\n"
                if table := func.get_table(["Parameters", "Returns"]):
                    blocks += table + "<br>"
                else:
                    # \n to exit markdown and <br> to insert space
                    blocks += "\n" + "<br>"

        return toc + blocks


# Functions ======================================================== >>

def render(markdown: str, **kwargs) -> str:
    """Render the markdown page.

    This function is the landing point for the mkdocs-simple-hooks
    plugin, called in mkdocs.yml.

    Parameters
    ----------
    markdown: str
        Markdown source text of page.

    **kwargs
        Additional keyword arguments of the hook.
            - page: Mkdocs Page instance.
            - config: Global configuration object.
            - files: Global files collection.

    Returns
    -------
    str
        Modified markdown/html source text of page.

    """
    while match := re.search("(:: )(\w.*?)(?=::|\n\n|\Z)", markdown, re.S):
        command = yaml.safe_load(match.group(2))

        # Commands should always be dicts with the configuration as a list in values
        if isinstance(command, str):
            if ":" in command:
                autodocs = AutoDocs.get_obj(command)
                markdown = markdown[:match.start()] + markdown[match.end():]
                continue
            else:
                command = {command: None}  # Has no options specified

        if "tags" in command:
            text = autodocs.get_tags()
        elif "signature" in command:
            text = autodocs.get_signature()
        elif "head" in command:
            text = autodocs.get_summary() + "\n\n" + autodocs.get_description()
        elif "summary" in command:
            text = autodocs.get_summary()
        elif "description" in command:
            text = autodocs.get_description()
        elif "table" in command:
            text = autodocs.get_table(command["table"])
        elif "see also" in command:
            text = autodocs.get_see_also()
        elif "notes" in command:
            text = autodocs.get_block("Notes")
        elif "references" in command:
            text = autodocs.get_block("References")
        elif "examples" in command:
            text = autodocs.get_block("Examples")
        elif "hyperparameters" in command:
            text = autodocs.get_hyperparameters()
        elif "methods" in command:
            text = autodocs.get_methods(command["methods"] or {})
        else:
            text = ""

        markdown = markdown[:match.start()] + text + markdown[match.end():]
        markdown = custom_autorefs(markdown, autodocs)

    return custom_autorefs(markdown)


def custom_autorefs(markdown: str, autodocs: Optional[AutoDocs] = None) -> str:
    """Custom handling of autorefs links.

    ATOM's documentation accepts some custom formatting for autorefs
    links in order to make the documentation cleaner and easier to
    write. The custom transformations are:

    - Replace keywords with full url (registered in CUSTOM_URLS).
    - Replace keyword `self` with the name of the class.
    - Replace spaces with dashes.
    - Convert all links to lower case.

    Parameters
    ----------
    markdown: str
        Markdown source text of page.

    autodocs: Autodocs
        Class for which the page is created.

    Returns
    -------
    str
        Modified source text of page.

    """
    result, start = "", 0
    for match in re.finditer("\[([\.`' \w_-]*?)\]\[([\w_-]*?)\]", markdown):
        anchor = match.group(1)
        link = match.group(2)

        text = match.group()
        if not link:
            # Only adapt when has form [anchor][]
            link = anchor.replace(' ', '-').lower()
            text = f"[{anchor}][{link}]"
        if link in CUSTOM_URLS:
            # Replace keyword with custom url
            text = f"[{anchor}]({CUSTOM_URLS[link]})"
        if "self" in link and autodocs:
            link = link.replace("self", autodocs.obj.__name__.lower())
            text = f"[{anchor}][{link}]"

        result += markdown[start:match.start()] + text
        start = match.end()

    return result + markdown[start:]
