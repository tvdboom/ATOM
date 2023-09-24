# Dependencies
--------------

## Python & OS

As of the moment, ATOM supports the following Python versions:

* [Python 3.9](https://www.python.org/downloads/release/python-390/)
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

* **[beartype](https://beartype.readthedocs.io/en/latest/)** (>=0.15.0)
* **[category-encoders](https://contrib.scikit-learn.org/categorical-encoding/index.html)** (>=2.6.1)
* **[dagshub](https://github.com/DagsHub/client)** (>=0.2.10)
* **[dill](https://pypi.org/project/dill/)** (>=0.3.6)
* **[gplearn](https://gplearn.readthedocs.io/en/stable/index.html)** (>=0.4.2)
* **[imbalanced-learn](https://imbalanced-learn.readthedocs.io/en/stable/api.html)** (>=0.11.0)
* **[ipython](https://ipython.readthedocs.io/en/stable/)** (>=8.11.0)
* **[featuretools](https://www.featuretools.com/)** (>=1.27.0)
* **[joblib](https://joblib.readthedocs.io/en/latest/)** (>=1.3.1)
* **[matplotlib](https://matplotlib.org/)** (>=3.7.2)
* **[mlflow](https://mlflow.org/)** (>=2.5.0)
* **[modin[ray]](https://modin.readthedocs.io/en/stable/)** (>=0.23.0)
* **[nltk](https://www.nltk.org/)** (>=3.8.1)
* **[numpy](https://numpy.org/)** (>=1.23.0)
* **[optuna](https://optuna.org/)** (>=3.2.0)
* **[pandas[parquet]](https://pandas.pydata.org/)** (>=2.0.3)
* **[plotly](https://plotly.com/python/)** (>=5.15.0)
* **[ray[serve]](https://docs.ray.io/en/latest/)** (>=2.6.1)
* **[scikit-learn](https://scikit-learn.org/stable/)** (>=1.3.1)
* **[scikit-learn-intelex](https://github.com/intel/scikit-learn-intelex)** (>=2023.2.1)
* **[scipy](https://www.scipy.org/)** (>=1.10.1)
* **[shap](https://github.com/slundberg/shap/)** (>=0.42.1)
* **[sktime](http://www.sktime.net/en/latest/)** (>=0.20.1)
* **[zoofs](https://jaswinder9051998.github.io/zoofs/)** (>=0.1.26)


### Optional

Some specific models, utility methods or plots require the installation of
additional libraries. You can install all the optional dependencies using
`pip install atom-ml[full]`. Doing so also installs the following libraries:

* **[botorch](https://botorch.org/docs/introduction)** (>=0.8.5)
* **[catboost](https://catboost.ai/docs/concepts/about.html)** (>=1.2)
* **[explainerdashboard](https://explainerdashboard.readthedocs.io/en/latest/)** (>=0.4.3)
* **[evalml](https://evalml.alteryx.com/en/stable/)** (>=0.79.0)
* **[gradio](https://github.com/gradio-app/gradio)** (>=3.19.1)
* **[lightgbm](https://lightgbm.readthedocs.io/en/latest/)** (>=3.3.5)
* **[pmdarima](http://alkaline-ml.com/pmdarima/)** (>=2.0.3)
* **[schemdraw](https://schemdraw.readthedocs.io/en/latest/index.html)** (>=0.16)
* **[wordcloud](http://amueller.github.io/word_cloud/)** (>=1.9.2)
* **[xgboost](https://xgboost.readthedocs.io/en/latest/)** (>=2.0.0)
* **[ydata-profiling](https://github.com/ydataai/ydata-profiling)** (>=4.5.1)


### Development

The development dependencies are not installed with the package, and are
not required for any of its functionalities. These libraries are only
necessary to [contribute][contributing] to the project. Install them
using `pip install atom-ml[dev]`.

**Linting**

* **[isort](https://pycqa.github.io/isort/)** (>=5.12.0)
* **[flake8](https://github.com/pycqa/flake8)** (>=6.0.0)
* **[flake8-pyproject](https://github.com/john-hen/Flake8-pyproject)** (>=1.2.2)
* **[mypy](https://www.mypy-lang.org/)** (>=1.5.1)

**Testing**

* **[nbmake](https://github.com/treebeardtech/nbmake)** (>=1.4.1)
* **[pytest](https://docs.pytest.org/en/latest/)** (>=7.2.1)
* **[pytest-cov](https://pytest-cov.readthedocs.io/en/latest/)** (>=4.0.0)
* **[pytest-xdist](https://github.com/pytest-dev/pytest-xdist)** (>=3.2.0)
* **[scikeras](https://github.com/adriangb/scikeras)** (>=0.11.0)
* **[tensorflow](https://www.tensorflow.org/learn)** (>=2.13.0)

**Documentation**

* **[mike](https://github.com/jimporter/mike)** (>=1.1.2)
* **[mkdocs](https://www.mkdocs.org/)** (>=1.4.2)
* **[mkdocs-autorefs](https://mkdocstrings.github.io/autorefs/)** (>=0.5.0)
* **[mkdocs-jupyter](https://github.com/danielfrg/mkdocs-jupyter)** (>=0.22.0)
* **[mkdocs-material](https://squidfunk.github.io/mkdocs-material/)** (>=9.1.21)
* **[mkdocs-material-extensions](https://github.com/facelessuser/mkdocs-material-extensions)** (>=1.1.1)
* **[mkdocs-simple-hooks](https://github.com/aklajnert/mkdocs-simple-hooks)** (>=0.1.5)
* **[pymdown-extensions](https://github.com/facelessuser/pymdown-extensions)** (>=9.9.2)
* **[pyyaml](https://pyyaml.org/)** (>=6.0)
