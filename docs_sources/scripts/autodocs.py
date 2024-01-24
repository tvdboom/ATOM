"""Automated Tool for Optimized Modeling (ATOM).

Author: Mavs
Description: Module containing the documentation rendering.

"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from inspect import (
    Parameter,
    getdoc,
    getmembers,
    getsourcelines,
    isclass,
    isfunction,
    ismethod,
    isroutine,
    signature,
)
from typing import Any, Optional
from collections.abc import Callable

import regex as re
import yaml
from mkdocs.config.defaults import MkDocsConfig

from atom.utils.utils import Goal


# Variables ======================================================== >>

# Mapping of keywords to urls
# Usage in docs: [anchor][key] or [key][] -> [anchor][value]
CUSTOM_URLS = dict(
    # API
    api="https://scikit-learn.org/stable/developers/develop.html",
    metadata_routing="https://scikit-learn.org/stable/metadata_routing.html#metadata-routing",
    metadatarouter="https://scikit-learn.org/stable/modules/generated/sklearn.utils.metadata_routing.MetadataRouter.html",
    sycl_device_filter="https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md#sycl_device_filter",
    pathlibpath="https://docs.python.org/3/library/pathlib.html#pathlib.Path",
    joblibmemory="https://joblib.readthedocs.io/en/latest/generated/joblib.Memory.html",
    warnings="https://docs.python.org/3/library/warnings.html#the-warnings-filter",
    datetimeindex="https://pandas.pydata.org/docs/reference/api/pandas.DatetimeIndex.html",
    periodalias="https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#period-aliases",
    # ATOM
    rangeindex="https://pandas.pydata.org/docs/reference/api/pandas.RangeIndex.html",
    experiment="https://www.mlflow.org/docs/latest/tracking.html#organizing-runs-in-experiments",
    adf="https://en.wikipedia.org/wiki/Augmented_Dickey%E2%80%93Fuller_test",
    kpss="https://en.wikipedia.org/wiki/KPSS_test",
    lb="https://en.wikipedia.org/wiki/Ljung%E2%80%93Box_test",
    kstest="https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test",
    skpipeline="https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html",
    pipelinedocs="https://scikit-learn.org/stable/modules/compose.html#pipeline",
    bunch="https://scikit-learn.org/stable/modules/generated/sklearn.utils.Bunch.html",
    sweetviz="https://github.com/fbdesignpro/sweetviz",
    report="https://github.com/fbdesignpro/sweetviz/blob/master/sweetviz/dataframe_report.py#L23",
    to_csv="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html",
    # BaseModel
    mlflowrun="https://mlflow.org/docs/latest/python_api/mlflow.entities.html#mlflow.entities.Run",
    make_reduction="https://sktime-backup.readthedocs.io/en/v0.13.0/api_reference/auto_generated/sktime.forecasting.compose.make_reduction.html",
    study="https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html",
    optimize="https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize",
    trial="https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html",
    frozentrial="https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.FrozenTrial.html",
    normal="https://github.com/sktime/sktime/blob/b29e147b54959a53cc96e5be9c3f819717aa38e7/sktime/proba/normal.py#L13",
    forecastinghorizon="https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.forecasting.base.ForecastingHorizon.html#sktime.forecasting.base.ForecastingHorizon",
    interface="https://gradio.app/docs/#interface",
    launch="https://gradio.app/docs/#launch-header",
    sklearncrossvalidate="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html",
    sktimeevaluate="https://sktime-backup.readthedocs.io/en/latest/api_reference/auto_generated/sktime.forecasting.model_evaluation.evaluate.html",
    explainerdashboard_package="https://github.com/oegedijk/explainerdashboard",
    explainerdashboard="https://explainerdashboard.readthedocs.io/en/latest/dashboards.html#explainerdashboard-documentation",
    registry="https://www.mlflow.org/docs/latest/model-registry.html",
    ray="https://docs.ray.io/en/latest/cluster/getting-started.html",
    # BaseRunner
    stackingclassifier="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html",
    stackingregressor="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html",
    stackingforecaster="https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.forecasting.compose.StackingForecaster.html",
    votingclassifier="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html",
    votingregressor="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html",
    ensembleforecaster="https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.forecasting.compose.EnsembleForecaster.html",
    # Data cleaning
    clustercentroids="https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.ClusterCentroids.html",
    onehotencoder="https://contrib.scikit-learn.org/category_encoders/onehot.html",
    hashingencoder="https://contrib.scikit-learn.org/category_encoders/hashing.html",
    leaveoneoutencoder="https://contrib.scikit-learn.org/category_encoders/leaveoneout.html",
    quantile="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html",
    boxcox="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html",
    yeojohnson="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html",
    iforest="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html",
    ee="https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EllipticEnvelope.html",
    lof="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html",
    svm="https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html",
    dbscan="https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html",
    hdbscan="https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html",
    optics="https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html",
    standard="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html",
    minmax="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html",
    maxabs="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html",
    robust="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html",
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
    classifierchain="https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.ClassifierChain.html",
    regressorchain="https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.RegressorChain.html",
    multioutputclassifier="https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html",
    multioutputregressor="https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputRegressor.html",
    adaboostclassifier="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html",
    adaboostregressor="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html",
    adabdocs="https://scikit-learn.org/stable/modules/ensemble.html#adaboost",
    ardregression="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html",
    arddocs="https://scikit-learn.org/stable/modules/linear_model.html#automatic-relevance-determination-ard",
    arimaclass="https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.forecasting.arima.ARIMA.html",
    autoarimaclass="https://github.com/sktime/sktime/blob/c48883eb4f6d5336961c8413192dd22a65db1b77/sktime/forecasting/arima.py#L11",
    autoets="https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.forecasting.ets.AutoETS.html",
    baggingclassifier="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html",
    baggingregressor="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html",
    bagdocs="https://scikit-learn.org/stable/modules/ensemble.html#bootstrapping",
    batsclass="https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.forecasting.bats.BATS.html",
    bayesianridgeclass="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html",
    brdocs="https://scikit-learn.org/stable/modules/linear_model.html#bayesian-regression",
    bernoullinbclass="https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html",
    bnbdocs="https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html",
    catboostclassifier="https://catboost.ai/docs/concepts/python-reference_catboostclassifier.html",
    catboostregressor="https://catboost.ai/docs/concepts/python-reference_catboostregressor.html",
    catbdocs="https://catboost.ai/",
    categoricalnbclass="https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.CategoricalNB.html",
    catnbdocs="https://scikit-learn.org/stable/modules/naive_bayes.html#categorical-naive-bayes",
    complementnbclass="https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.CategoricalNB.html",
    cnbdocs="https://scikit-learn.org/stable/modules/naive_bayes.html#complement-naive-bayes",
    crostonclass="https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.forecasting.croston.Croston.html",
    decisiontreeclassifier="https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html",
    decisiontreeregressor="https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html",
    treedocs="https://scikit-learn.org/stable/modules/tree.html",
    dummyclassifier="https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html",
    dummyregressor="https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html",
    dummydocs="https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators",
    dynamicfactorclass="https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.forecasting.dynamic_factor.DynamicFactor.html",
    elasticnetreg="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html",
    endocs="https://scikit-learn.org/stable/modules/linear_model.html#elastic-net",
    esclass="https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.forecasting.exp_smoothing.ExponentialSmoothing.html",
    extratreeclassifier="https://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeClassifier.html",
    extratreeregressor="https://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeRegressor.html",
    extratreesclassifier="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html",
    extratreesregressor="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html",
    etdocs="https://scikit-learn.org/stable/modules/ensemble.html#extremely-randomized-trees",
    gaussiannbclass="https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html",
    gnbdocs="https://scikit-learn.org/stable/modules/naive_bayes.html#gaussian-naive-bayes",
    gaussianprocessclassifier="https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html",
    gaussianprocessregressor="https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html",
    gpdocs="https://scikit-learn.org/stable/modules/gaussian_process.html",
    gradientboostingclassifier="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html",
    gradientboostingregressor="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html",
    gbmdocs="https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting",
    histgradientboostingclassifier="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html",
    histgradientboostingregressor="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html",
    hgbmdocs="https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting",
    huberregressor="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html",
    huberdocs="https://scikit-learn.org/stable/modules/linear_model.html#huber-regression",
    kneighborsclassifier="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html",
    kneighborsregressor="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html",
    knndocs="https://scikit-learn.org/stable/modules/neighbors.html",
    multinomialnbclass="https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html",
    mnbdocs="https://scikit-learn.org/stable/modules/naive_bayes.html#multinomial-naive-bayes",
    statsforecastmstl="https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.forecasting.statsforecast.StatsForecastMSTL.html",
    lars="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lars.html",
    larsdocs="https://scikit-learn.org/stable/modules/linear_model.html#least-angle-regression",
    lassoreg="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html",
    lassodocs="https://scikit-learn.org/stable/modules/linear_model.html#lasso",
    ldaclassifier="https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html",
    ldadocs="https://scikit-learn.org/stable/modules/lda_qda.html#lda-qda",
    linearsvc="https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html",
    linearsvr="https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html",
    lgbmclassifier="https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html",
    lgbmregressor="https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html",
    lgbdocs="https://lightgbm.readthedocs.io/en/latest/index.html",
    lgb_gpu="https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html",
    logisticregression="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html",
    lrdocs="https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression",
    mlpclassifier="https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html",
    mlpregressor="https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html",
    mlpdocs="https://scikit-learn.org/stable/modules/neural_networks_supervised.html#neural-networks-supervised",
    linearregression="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html",
    olsdocs="https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares",
    orthogonalmatchingpursuit="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuit.html",
    ompdocs="https://scikit-learn.org/stable/modules/linear_model.html#orthogonal-matching-pursuit-omp",
    passiveaggressiveclassifier="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html",
    passiveaggressiveregressor="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveRegressor.html",
    padocs="https://scikit-learn.org/stable/modules/linear_model.html#passive-aggressive",
    percclassifier="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html",
    percdocs="https://scikit-learn.org/stable/modules/linear_model.html#perceptron",
    polynomialtrendforecaster="https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.forecasting.trend.PolynomialTrendForecaster.html",
    prophetclass="https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.forecasting.fbprophet.Prophet.html",
    qdaclassifier="https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html",
    radiusneighborsclassifier="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsClassifier.html",
    radiusneighborsregressor="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsRegressor.html",
    randomforestclassifier="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html",
    randomforestregressor="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html",
    ridgeclassifier="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html",
    ridgeregressor="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html",
    ridgedocs="https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression",
    cumlrf="https://docs.rapids.ai/api/cuml/stable/api.html#cuml.ensemble.RandomForestClassifier",
    rfdocs="https://scikit-learn.org/stable/modules/ensemble.html#random-forests",
    sarimaxclass="https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.forecasting.sarimax.SARIMAX.html",
    sgdclassifier="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html",
    sgdregressor="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html",
    sgddocs="https://scikit-learn.org/stable/modules/sgd.html",
    stlforecaster="https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.forecasting.trend.STLForecaster.html",
    svc="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html",
    svr="https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html",
    svmdocs="https://scikit-learn.org/stable/modules/svm.html",
    tbatsclass="https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.forecasting.tbats.TBATS.html",
    thetaforecaster="https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.forecasting.theta.ThetaForecaster.html",
    xgbclassifier="https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier",
    xgbregressor="https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor",
    xgbdocs="https://xgboost.readthedocs.io/en/latest/index.html",
    naiveforecasterclass="https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.forecasting.naive.NaiveForecaster.html",
    varclass="https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.forecasting.var.VAR.html",
    varmaxclass="https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.forecasting.varmax.VARMAX.html",
    # NLP
    snowballstemmer="https://www.nltk.org/api/nltk.stem.snowball.html#nltk.stem.snowball.SnowballStemmer",
    bow="https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html",
    tfidf="https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html",
    hashing="https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html",
    # Plots
    palette="https://plotly.com/python/discrete-color/",
    gofigure="https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html",
    pltfigure="https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html",
    update_layout="https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html#plotly.graph_objects.Figure.update_layout",
    update_traces="https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html#plotly.graph_objects.Figure.update_traces",
    fanova="https://optuna.readthedocs.io/en/stable/reference/generated/optuna.importance.FanovaImportanceEvaluator.html",
    kde="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html",
    wordcloud="https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html",
    calibration="https://scikit-learn.org/stable/modules/calibration.html",
    det="https://scikit-learn.org/stable/auto_examples/model_selection/plot_det.html",
    partial_dependence="https://scikit-learn.org/stable/modules/partial_dependence.html#partial-dependence-and-individual-conditional-expectation-plots",
    prc="https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html",
    roc="https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html",
    schemdraw="https://schemdraw.readthedocs.io/en/latest/index.html",
    plotly="https://plotly.com/python/",
    force="https://shap.readthedocs.io/en/latest/generated/shap.plots.force.html",
    # Training
    scorers="https://scikit-learn.org/stable/modules/model_evaluation.html",
)


# Classes ========================================================== >>

@dataclass
class DummyTrainer:
    """Dummy trainer class to call model instances."""

    goal: Goal
    device: str
    engine: str


class AutoDocs:
    """Parses an object to documentation in markdown/html.

    The docstring should follow the numpydoc style[^1]. Blocks should
    start with `::`. The following blocks are accepted:

    - toc
    - tags
    - head (summary + description)
    - summary (first line of docstring, required)
    - description (detailed explanation, can contain admonitions)
    - parameters
    - attributes
    - returns
    - yields
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
        "Parameters\n---------",
        "Attributes\n---------",
        "Returns\n-------",
        "Yields\n------",
        "Raises\n------",
        "See Also\n--------",
        "Notes\n-----",
        "References\n-------",
        "Examples\n--------",
        r"\Z",
    )

    def __init__(self, obj: Callable, method: Callable | None = None):
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

    def get_toc(self) -> str:
        """Return a toc of the objects in self.

        Note that object must be iterable.

        Returns
        -------
        str
            Toc of the objects.

        """
        toc = "<table markdown style='font-size: 0.9em'>"
        for obj in self.obj:
            func = AutoDocs(obj)

            name = f"[{obj.__name__}][] ({obj.acronym})"
            toc += f"<tr><td>{name}</td><td>{func.get_summary()}</td></tr>"

        toc += "</table>"
        return toc

    def get_tags(self) -> str:
        """Return the object's tags.

        Tags are obtained from class attributes.

        Returns
        -------
        str
            Object's tags.

        """
        text = f"[{self.obj.acronym}][predefined-models]{{ .md-tag }}"
        if getattr(self.obj, "needs_scaling", False):
            text += "&nbsp;&nbsp;[needs scaling][automated-feature-scaling]{ .md-tag }"
        if getattr(self.obj, "accepts_sparse", False):
            text += "&nbsp;&nbsp;[accept sparse][sparse-datasets]{ .md-tag }"
        if getattr(self.obj, "multiple_seasonality", False):
            text += "&nbsp;&nbsp;[multiple seasonality][seasonality]{ .md-tag }"
        if getattr(self.obj, "native_multilabel", False):
            text += "&nbsp;&nbsp;[native multilabel][multilabel]{ .md-tag }"
        if getattr(self.obj, "native_multioutput", False):
            text += "&nbsp;&nbsp;[native multioutput][multioutput-tasks]{ .md-tag }"
        if getattr(self.obj, "validation", None):
            text += "&nbsp;&nbsp;[in-training validation][]{ .md-tag }"
        if any(engine not in ("sklearn", "sktime") for engine in self.obj.supports_engines):
            text += "&nbsp;&nbsp;[supports acceleration][estimator-acceleration]{ .md-tag }"

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
            if k not in ("cls", "self") and not k.startswith("_"):
                if v.default == Parameter.empty:
                    if "**" in str(v):
                        sign.append(f"**{k}")  # Add ** to kwargs
                    elif "*" in str(v):
                        sign.append(f"*{k}")  # Add * to args
                    else:
                        sign.append(k)
                else:
                    if isinstance(v.default, str):
                        sign.append(f'{k}="{v.default}"')
                    else:
                        sign.append(f"{k}={v.default}")

        sign = f"({', '.join(sign)})"

        f = self.obj.__module__.replace(".", "/")  # Module and filename sep by /
        if "atom" in self.obj.__module__:
            url = f"https://github.com/tvdboom/ATOM/blob/master/{f}.py"
        elif "sklearn" in self.obj.__module__:
            url = f"https://github.com/scikit-learn/scikit-learn/blob/main/{f}.py"
        else:
            url = ""

        anchor = f"<a id='{self._parent_anchor}{self.obj.__name__}'></a>"
        module = self.obj.__module__ + "." if obj != "method" else ""
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
            if include := config.get("include"):
                attrs = include
            else:
                attrs = [
                    m for m, _ in getmembers(self.obj, lambda x: not isroutine(x))
                    if not m.startswith("_")
                    and not any(re.fullmatch(p, m) for p in config.get("exclude", []))
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
                    elif obj.__class__.__name__ == "cached_property":
                        obj = obj.func

                    # Get the return type. Sometimes it returns a string 'Pandas'
                    # and sometimes a class pandas.DataFrame. Unclear why
                    output = str(signature(obj).return_annotation)

                    header = f"{obj.__name__}: {types_conversion(output)}"
                    text = f"<div markdown class='param'>{getdoc(obj)}\n</div>"

                    anchor = f"<a id='{self.obj.__name__.lower()}-{obj.__name__}'></a>"
                    content += f"{anchor}<strong>{header}</strong><br>{text}"

            elif match := self.get_block(name):
                # Headers start with a letter, * or [ after new line
                for header in re.findall(r"^[\[a-zA-Z*].*?$", match, re.M):
                    # Check that the default value in docstring matches the real one
                    if default := re.search("(?<=default=).+?$", header):
                        try:
                            param = header.split(":")[0]
                            real = signature(self.obj).parameters[param]

                            # String representation uses single quotes
                            default = str(default.group()).replace('"', "'")

                            # Remove quotes for string values
                            if default.startswith("'") and default.endswith("'"):
                                default = default[1:-1]

                            if default != str(real.default):
                                raise ValueError(
                                    f"Default value {default} of parameter {param} "
                                    f"of object {self.obj} doesn't match the value "
                                    f"in the docstring: {real.default}."
                                )
                        except KeyError:
                            pass

                    # Get the body corresponding to the header
                    pattern = f"(?<={re.escape(header)}\n).*?(?=\n\\w|\n\\*|\n\\[|\\Z)"
                    body = re.search(pattern, match, re.S | re.M).group()

                    header = header.replace("*", r"\*")  # Use literal * for args/kwargs
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
            model = self.obj(goal=trainer.goal)

            text = ""
            for name, dist in model._get_distributions().items():
                anchor = f"<a id='{self.obj.__name__.lower()}-{name}'></a>"
                text += f"{anchor}<strong>{name}</strong><br>"
                text += f"<div markdown class='param'>{dist}</div>"

            table = "<tr><td class='td_title'><strong>Parameters</strong></td>"
            table += f"<td class='td_params'>{text}</td></tr>"

            return f"<table markdown class='table_params'>{table}</table>"

        indent = content = ""
        for goal in self.obj._estimators:
            if len(self.obj._estimators) > 1:
                indent = " " * 4
                content += f'\n=== "{goal}"\n'

            for engine in self.obj.supports_engines:
                sub_indent = indent
                if len(self.obj.supports_engines) > 1:
                    content += f'\n{indent}=== "{engine}"\n'
                    sub_indent += " " * 4

                # sklearnex can run on cpu or gpu
                if engine == "sklearnex":
                    for device in ("cpu", "gpu"):
                        content += f'\n{sub_indent}=== "{device}"\n'

                        trainer = DummyTrainer(Goal[goal], device, engine)
                        content += f"{sub_indent + ' ' * 4}{create_table(trainer)}\n\n"
                else:
                    trainer = DummyTrainer(
                        goal=Goal[goal],
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
            - solo_link: Whether the link comes from the parent.
            - include: Methods to include.
            - exclude: Methods to exclude.

        Returns
        -------
        str
            Toc and blocks for all selected methods.

        """
        toc_only = config.get("toc_only")
        solo_link = config.get("solo_link")
        include = config.get("include", [])
        exclude = config.get("exclude", [])

        predicate = lambda f: ismethod(f) or isfunction(f)

        if include:
            methods = include
        else:
            methods = [
                m
                for m, _ in getmembers(self.obj, predicate=predicate)
                if not m.startswith("_") and not any(re.fullmatch(p, m) for p in exclude)
            ]

        # Create toc
        toc = "<table markdown style='font-size: 0.9em'>"
        for method in methods:
            func = AutoDocs(self.obj, method=method)

            name = f"[{method}][{'' if solo_link else func._parent_anchor}{method}]"
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
                if table := func.get_table(["Parameters", "Returns", "Yields"]):
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
    autodocs = None
    while match := re.search("(:: )([a-z].*?)(?=::|\n\n|\\Z)", markdown, re.S):
        command = yaml.safe_load(match.group(2))

        # Commands should always be dicts with the configuration as a list in values
        if isinstance(command, str):
            if ":" in command:
                autodocs = AutoDocs.get_obj(command)
                markdown = markdown[:match.start()] + markdown[match.end():]
                continue
            else:
                command = {command: None}  # Has no options specified

        if "toc" in command:
            text = autodocs.get_toc()
        elif "tags" in command:
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
        elif "insert" in command:
            text = insert(command["insert"] or {})
        else:
            text = ""

        markdown = markdown[:match.start()] + text + markdown[match.end():]

        # Change the custom autorefs now to use [self-...][]
        markdown = custom_autorefs(markdown, autodocs)

    return custom_autorefs(markdown)


def types_conversion(dtype: str) -> str:
    """Convert data types to a clean representation.

    Parameters
    ----------
    dtype: str
        Type to convert.

    Returns
    -------
    str
        Converted type.

    """
    types = {
        "<class '": "",
        "'>": "",
        "typing.": "",  # For typing.Any
        "atom.pipeline.": "",  # To transform later both class and str
        "Study": "[Study][]",
        "FrozenTrial": "[FrozenTrial][]",
        "Model": "[model][models]",
        "Run": "[Run][mlflowrun]",
        "pandas.core.indexes.base.Index": "Index",
        "pandas.core.series.Series": "Series",
        "pandas.core.frame.DataFrame": "DataFrame",
        "atom.branch.branch.Branch": "[Branch][]",
        "Pipeline": "[Pipeline][]",
        "collections.abc.Hashable": "str",
        "Scalar": "int | float",
        "Pandas": "Series | DataFrame",
        "int | numpy.integer": "int",
        "float | numpy.floating": "float",
        "Series | modin.pandas.series.Series": "Series",
        "DataFrame | modin.pandas.dataframe.DataFrame": "DataFrame",
    }

    for k, v in types.items():
        dtype = dtype.replace(k, v)

    return dtype


def corrections(html: str, **kwargs) -> str:
    """Make last minute corrections to the page.

    This function adjusts the url to the download sources and changes
    the size of plotly plots to fit the screen's width.

    Parameters
    ----------
    html: str
        HTML source text of page.

    **kwargs
        Additional keyword arguments of the hook.
            - page: Mkdocs Page instance.
            - config: Global configuration object.
            - files: Global files collection.

    Returns
    -------
    str
        Modified html source text of page.

    """
    # Swap url to example datasets
    html = html.replace("./datasets/", "docs_source/examples/datasets/")

    # Correct sizes of the plot to adjust to frame
    html = re.sub(r'(?<=style="height:\d+?px; width:)\d+?px(?=;")', "100%", html)
    html = re.sub(r'(?<="showlegend":\w+?),"width":\d+?,"height":\d+?(?=[},])', "", html)

    return html


def clean_search(config: MkDocsConfig):
    """Clean the search index.

    Remove unnecessary plotly and css blocks (from mkdocs-jupyter) to
    keep the search index small.

    Parameters
    ----------
    config: MkdocsConfig
        Object containing the search index.

    """
    with open(f"{config.data['site_dir']}/search/search_index.json") as f:
        search = json.load(f)

    for elem in search["docs"]:
        # Remove plotly graphs
        elem["text"] = re.sub(r"window\.PLOTLYENV.*?\)\s*?}\s*?", "", elem["text"], flags=re.S)

        # Remove mkdocs-jupyter css
        elem["text"] = re.sub(
            r"\(function \(global, factory.*?(?=Example:)", "", elem["text"], flags=re.S
        )

    with open(f"{config.data['site_dir']}/search/search_index.json", "w") as f:
        json.dump(search, f)


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

    autodocs: Autodocs or None
        Class for which the page is created.

    Returns
    -------
    str
        Modified source text of page.

    """
    result, start = "", 0

    # Skip regex check for very long docs
    if len(markdown) < 1e5:
        for match in re.finditer(r"\[([\.`': \w_-]*?)\]\[([\w_:-]*?)\]", markdown):
            anchor = match.group(1)
            link = match.group(2)

            text = match.group()
            if not link:
                # Only adapt when has form [anchor][]
                link = anchor.replace(" ", "-").replace(".", "").lower()
                text = f"[{anchor}][{link}]"
            if link in CUSTOM_URLS:
                # Replace keyword with custom url
                text = f"[{anchor}]({CUSTOM_URLS[link]})"
            if "self" in link and autodocs:
                link = link.replace("self", autodocs.obj.__name__.lower())
                text = f"[{anchor}][{link}]"

            result += markdown[start : match.start()] + text
            start = match.end()

    return result + markdown[start:]
