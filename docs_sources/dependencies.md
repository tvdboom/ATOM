# Dependencies
--------------

## Python & OS

As of the moment, ATOM supports the following Python versions:

* [Python 3.10](https://www.python.org/downloads/release/python-3100/)
* [Python 3.11](https://www.python.org/downloads/release/python-3110/)

And operating systems:

 * Linux (Ubuntu, Fedora, etc...)
 * Windows 8.1+
 * macOS (not tested)

<br><br>


## Packages

### Required

ATOM is built on top of several existing Python libraries. These
packages are necessary for its correct functioning.

* **[beartype](https://beartype.readthedocs.io/en/latest/)** (>=0.16.4)
* **[category-encoders](https://contrib.scikit-learn.org/categorical-encoding/index.html)** (>=2.6.3)
* **[dagshub](https://github.com/DagsHub/client)** (>=0.3.8)
* **[dill](https://pypi.org/project/dill/)** (>=0.3.6)
* **[gplearn](https://gplearn.readthedocs.io/en/stable/index.html)** (>=0.4.2)
* **[imbalanced-learn](https://imbalanced-learn.readthedocs.io/en/stable/api.html)** (>=0.11.0)
* **[ipython](https://ipython.readthedocs.io/en/stable/)** (>=8.11.0)
* **[ipywidgets](https://pypi.org/project/ipywidgets/)** (>=8.1.1)
* **[featuretools](https://www.featuretools.com/)** (>=1.28.0)
* **[joblib](https://joblib.readthedocs.io/en/latest/)** (>=1.3.1)
* **[matplotlib](https://matplotlib.org/)** (>=3.7.2)
* **[mlflow](https://mlflow.org/)** (>=2.7.1)
* **[modin[ray]](https://modin.readthedocs.io/en/stable/)** (>=0.25.0)
* **[nltk](https://www.nltk.org/)** (>=3.8.1)
* **[numpy](https://numpy.org/)** (>=1.23.0)
* **[optuna](https://optuna.org/)** (>=3.4.0)
* **[pandas[parquet]](https://pandas.pydata.org/)** (>=2.1.2)
* **[pmdarima](http://alkaline-ml.com/pmdarima/)** (>=2.0.3)
* **[plotly](https://plotly.com/python/)** (>=5.15.0)
* **[ray[serve]](https://docs.ray.io/en/latest/)** (>=2.7.1)
* **[requests](https://requests.readthedocs.io/en/latest/)** (>=2.31.0)
* **[scikit-learn](https://scikit-learn.org/stable/)** (>=1.3.1)
* **[scikit-learn-intelex](https://github.com/intel/scikit-learn-intelex)** (>=2023.2.1)
* **[scipy](https://www.scipy.org/)** (>=1.10.1)
* **[shap](https://github.com/slundberg/shap/)** (>=0.43.0)
* **[sktime](http://www.sktime.net/en/latest/)** (>=0.24.0)
* **[zoofs](https://jaswinder9051998.github.io/zoofs/)** (>=0.1.26)


### Optional

Some specific models, utility methods or plots require the installation of
additional libraries. You can install all the optional dependencies using
`pip install atom-ml[full]`. Doing so also installs the following libraries:

* **[botorch](https://botorch.org/docs/introduction)** (>=0.8.5)
* **[catboost](https://catboost.ai/docs/concepts/about.html)** (>=1.2)
* **[explainerdashboard](https://explainerdashboard.readthedocs.io/en/latest/)** (>=0.4.3)
* **[gradio](https://github.com/gradio-app/gradio)** (>=3.44.4)
* **[lightgbm](https://lightgbm.readthedocs.io/en/latest/)** (>=4.1.0)
* **[schemdraw](https://schemdraw.readthedocs.io/en/latest/index.html)** (>=0.16)
* **[sweetviz](https://github.com/fbdesignpro/sweetviz)** (>=2.3.1)
* **[tbats](https://github.com/intive-DataScience/tbats)** (>=1.1.3)
* **[wordcloud](http://amueller.github.io/word_cloud/)** (>=1.9.2)
* **[xgboost](https://xgboost.readthedocs.io/en/latest/)** (>=2.0.0)


### Development

The development dependencies are not installed with the package, and are
not required for any of its functionalities. These libraries are only
necessary to [contribute][contributing] to the project. Install them
running `pdm install --dev` (remember to install [pdm](https://pdm-project.org/latest/) first with
`pip install -U pdm`).

**Linting**

* **[isort](https://pycqa.github.io/isort/)** (>=5.12.0)
* **[mypy](https://www.mypy-lang.org/)** (>=1.6.1)
* **[pandas_stubs](https://pypi.org/project/pandas-stubs/)** (>=2.1.1.230928)
* **[pre-commit](https://pre-commit.com/)** (>=3.5.0)
* **[ruff](https://docs.astral.sh/ruff/)** (>=0.1.7)
* **[types-requests](https://github.com/python/typeshed)** (>=2.31.0.10)

**Testing**

* **[nbmake](https://github.com/treebeardtech/nbmake)** (>=1.4.1)
* **[pytest](https://docs.pytest.org/en/latest/)** (>=7.2.1)
* **[pytest-cov](https://pytest-cov.readthedocs.io/en/latest/)** (>=4.0.0)
* **[pytest-mock](https://github.com/pytest-dev/pytest-mock/)** (>=3.12.0)
* **[pytest-xdist](https://github.com/pytest-dev/pytest-xdist)** (>=3.2.0)
* **[scikeras](https://github.com/adriangb/scikeras)** (>=0.11.0)
* **[tensorflow](https://www.tensorflow.org/learn)** (>=2.13.0)

**Documentation**

* **[jupyter-contrib-nbextensions](https://github.com/ipython-contrib/jupyter_contrib_nbextensions)** (>=0.7.0)
* **[mike](https://github.com/jimporter/mike)** (>=1.1.2)
* **[mkdocs](https://www.mkdocs.org/)** (>=1.5.3)
* **[mkdocs-autorefs](https://mkdocstrings.github.io/autorefs/)** (>=0.5.0)
* **[mkdocs-jupyter](https://github.com/danielfrg/mkdocs-jupyter)** (>=0.24.6)
* **[mkdocs-material](https://squidfunk.github.io/mkdocs-material/)** (>=9.4.7)
* **[mkdocs-simple-hooks](https://github.com/aklajnert/mkdocs-simple-hooks)** (>=0.1.5)
* **[pymdown-extensions](https://github.com/facelessuser/pymdown-extensions)** (>=10.3.1)
* **[pyyaml](https://pyyaml.org/)** (>=6.0)
