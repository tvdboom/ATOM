# Dependencies
--------------

## Python & OS

As of the moment, ATOM supports the following Python versions:

* [Python 3.10](https://www.python.org/downloads/release/python-3100/)
* [Python 3.11](https://www.python.org/downloads/release/python-3110/)
* [Python 3.12](https://www.python.org/downloads/release/python-3120/)

And operating systems:

 * Linux (Ubuntu, Fedora, etc...)
 * Windows 8.1+
 * macOS (not tested)

<br><br>


## Packages

### Required

ATOM is built on top of several existing Python libraries. These
packages are necessary for its correct functioning.

* **[beartype](https://beartype.readthedocs.io/en/latest/)** (>=0.18.0)
* **[category-encoders](https://contrib.scikit-learn.org/categorical-encoding/index.html)** (>=2.6.3)
* **[dill](https://pypi.org/project/dill/)** (>=0.3.6)
* **[featuretools](https://www.featuretools.com/)** (>=1.28.0)
* **[gplearn](https://gplearn.readthedocs.io/en/stable/index.html)** (>=0.4.2)
* **[imbalanced-learn](https://imbalanced-learn.readthedocs.io/en/stable/api.html)** (>=0.12.0)
* **[ipython](https://ipython.readthedocs.io/en/stable/)** (>=8.9.0)
* **[ipywidgets](https://pypi.org/project/ipywidgets/)** (>=8.1.1)
* **[joblib](https://joblib.readthedocs.io/en/latest/)** (>=1.3.1)
* **[matplotlib](https://matplotlib.org/)** (>=3.7.2)
* **[mlflow](https://mlflow.org/)** (>=2.10.2)
* **[nltk](https://www.nltk.org/)** (>=3.8.1)
* **[numpy](https://numpy.org/)** (>=1.23.0)
* **[optuna](https://optuna.org/)** (>=3.6.0)
* **[pandas](https://pandas.pydata.org/)** (>=2.1.2)
* **[plotly](https://plotly.com/python/)** (>=5.18.0)
* **[scikit-learn](https://scikit-learn.org/stable/)** (>=1.4.1.post1)
* **[scipy](https://www.scipy.org/)** (>=1.10.1)
* **[shap](https://github.com/slundberg/shap/)** (>=0.43.0)
* **[sktime[forecasting]](http://www.sktime.net/en/latest/)** (>=0.26.0)
* **[statsmodels](https://www.statsmodels.org/stable/index.html)** (>=0.14.1)
* **[zoofs](https://jaswinder9051998.github.io/zoofs/)** (>=0.1.26)


### Optional

Some specific models, utility methods or plots require the installation of
additional libraries. You can install all the optional dependencies using
`pip install atom-ml[full]`. Doing so also installs the following libraries:

* **[botorch](https://botorch.org/docs/introduction)** (>=0.8.5)
* **[catboost](https://catboost.ai/docs/concepts/about.html)** (>=1.2)
* **[dagshub](https://github.com/DagsHub/client)** (>=0.3.8)
* **[dask[dataframe,distributed]](https://dask.org/)** (>=2024.2.0)
* **[explainerdashboard](https://explainerdashboard.readthedocs.io/en/latest/)** (>=0.4.3)
* **[gradio](https://github.com/gradio-app/gradio)** (>=3.44.4)
* **[lightgbm](https://lightgbm.readthedocs.io/en/latest/)** (>=4.1.0)
* **[modin[ray]](https://modin.readthedocs.io/en/stable/)** (>=0.25.0)
* **[optuna-integration](https://optuna-integration.readthedocs.io/en/latest/index.html)** (>=3.6.0)
* **[polars](https://pola.rs/)** (>=0.20.7)
* **[pyarrow](https://arrow.apache.org/docs/python/)** (>=15.0.0)
* **[pyspark](https://github.com/apache/spark/tree/master/python)** (>=3.5.0)
* **[requests](https://requests.readthedocs.io/en/latest/)** (>=2.31.0)
* **[ray[serve]](https://docs.ray.io/en/latest/)** (>=2.9.1)
* **[requests](https://requests.readthedocs.io/en/latest/)** (>=2.31.0)
* **[scikit-learn-intelex](https://github.com/intel/scikit-learn-intelex)** (>=2023.2.1)
* **[schemdraw](https://schemdraw.readthedocs.io/en/latest/index.html)** (>=0.16)
* **[statsforecast](https://github.com/Nixtla/statsforecast/)** (>=1.6.0)
* **[sweetviz](https://github.com/fbdesignpro/sweetviz)** (>=2.3.1)
* **[wordcloud](http://amueller.github.io/word_cloud/)** (>=1.9.2)
* **[xgboost](https://xgboost.readthedocs.io/en/latest/)** (>=2.0.0)


### Development

The development dependencies are not installed with the package, and are
not required for any of its functionalities. These libraries are only
necessary to [contribute][contributing] to the project. Install them
running `pdm install --dev` (remember to install [pdm](https://pdm-project.org/latest/) first with
`pip install -U pdm`).

**Linting**

* **[pre-commit](https://pre-commit.com/)** (>=3.6.2)

**Testing**

* **[nbmake](https://github.com/treebeardtech/nbmake)** (>=1.5.3)
* **[pytest](https://docs.pytest.org/en/latest/)** (>=8.1.1)
* **[pytest-cov](https://pytest-cov.readthedocs.io/en/latest/)** (>=4.1.0)
* **[pytest-mock](https://github.com/pytest-dev/pytest-mock/)** (>=3.12.0)
* **[pytest-xdist](https://github.com/pytest-dev/pytest-xdist)** (>=3.5.0)
* **[scikeras](https://github.com/adriangb/scikeras)** (>=0.11.0)
* **[tensorflow](https://www.tensorflow.org/learn)** (>=2.13.0)

**Documentation**

* **[jupyter-contrib-nbextensions](https://github.com/ipython-contrib/jupyter_contrib_nbextensions)** (>=0.7.0)
* **[kaleido](https://github.com/plotly/Kaleido)** (>=0.2.1)
* **[mike](https://github.com/jimporter/mike)** (>=2.0.0)
* **[mkdocs](https://www.mkdocs.org/)** (>=1.5.3)
* **[mkdocs-autorefs](https://mkdocstrings.github.io/autorefs/)** (>=1.0.1)
* **[mkdocs-git-committers-plugin-2](https://github.com/ojacques/mkdocs-git-committers-plugin-2/)** (>=2.3.0)
* **[mkdocs-git-revision-date-localized-plugin](https://github.com/timvink/mkdocs-git-revision-date-localized-plugin)** (>=1.2.4)
* **[mkdocs-jupyter](https://github.com/danielfrg/mkdocs-jupyter)** (>=0.24.6)
* **[mkdocs-material](https://squidfunk.github.io/mkdocs-material/)** (>=9.5.13)
* **[mkdocs-simple-hooks](https://github.com/aklajnert/mkdocs-simple-hooks)** (>=0.1.5)
* **[pymdown-extensions](https://github.com/facelessuser/pymdown-extensions)** (>=10.7.1)
* **[pyyaml](https://pyyaml.org/)** (>=6.0.1)
