# Dependencies
--------------

## Python & OS

As of the moment, ATOM supports the following Python versions:

* [Python 3.8](https://www.python.org/downloads/release/python-380/)
* [Python 3.9](https://www.python.org/downloads/release/python-390/)
* [Python 3.10](https://www.python.org/downloads/release/python-3100/)

And operating systems:

 * Linux (Ubuntu, Fedora, etc...)
 * Windows 8.1+
 * macOS (not tested)

<br><br>


## Packages

### Required

ATOM is built on top of several existing Python libraries. These
packages are necessary for its correct functioning.

* **[category-encoders](https://contrib.scikit-learn.org/categorical-encoding/index.html)** (>=2.5.1)
* **[dill](https://pypi.org/project/dill/)** (>=0.3.6)
* **[dagshub](https://github.com/DagsHub/client)** (>=0.2.12)
* **[gplearn](https://gplearn.readthedocs.io/en/stable/index.html)** (>=0.4.2)
* **[imbalanced-learn](https://imbalanced-learn.readthedocs.io/en/stable/api.html)** (>=0.10.1)
* **[ipython](https://ipython.org/)** (>=8.11.0)
* **[featuretools](https://www.featuretools.com/)** (>=1.23.0)
* **[joblib](https://joblib.readthedocs.io/en/latest/)** (>=1.2.0)
* **[matplotlib](https://matplotlib.org/)** (>=3.6.3)
* **[mlflow](https://mlflow.org/)** (>=2.2.0)
* **[modin[ray]](https://modin.readthedocs.io/en/stable/)** (>=0.18.1)
* **[nltk](https://www.nltk.org/)** (>=3.8.1)
* **[numpy](https://numpy.org/)** (>=1.23.5)
* **[optuna](https://optuna.org/)** (>=3.1.0)
* **[pandas](https://pandas.pydata.org/)** (>=1.5.3)
* **[plotly](https://plotly.com/python/)** (>=5.13.1)
* **[ray[serve]](https://docs.ray.io/en/latest/)** (>=2.3.0)
* **[shap](https://github.com/slundberg/shap/)** (>=0.41.0)
* **[scikit-learn](https://scikit-learn.org/stable/)** (>=1.2.1)
* **[scikit-learn-intelex](https://github.com/intel/scikit-learn-intelex)** (>=2023.0.1)
* **[scipy](https://www.scipy.org/)** (>=1.9.3)
* **[zoofs](https://jaswinder9051998.github.io/zoofs/)** (>=0.1.26)


### Optional

Some specific models, utility methods or plots require the installation of
additional libraries. You can install all the optional dependencies using
`pip install atom-ml[full]`. Doing so also installs the following libraries:

* **[catboost](https://catboost.ai/docs/concepts/about.html)** (>=1.1.1)
* **[explainerdashboard](https://explainerdashboard.readthedocs.io/en/latest/)** (>=0.4.2)
* **[evalml](https://evalml.alteryx.com/en/stable/)** (>=0.68.0)
* **[gradio](https://github.com/gradio-app/gradio)** (>=3.19.1)
* **[lightgbm](https://lightgbm.readthedocs.io/en/latest/)** (>=3.3.5)
* **[schemdraw](https://schemdraw.readthedocs.io/en/latest/index.html)** (>=0.15)
* **[wordcloud](http://amueller.github.io/word_cloud/)** (>=1.8.2)
* **[xgboost](https://xgboost.readthedocs.io/en/latest/)** (>=1.7.4)
* **[ydata-profiling](https://github.com/ydataai/ydata-profiling)** (>=4.0.0)


### Development

The development dependencies are not installed with the package, and are
not required for any of its functionalities. These libraries are only
necessary to [contribute][contributing] to the project. Install them
using `pip install atom-ml[dev]`.

**Linting**

* **[isort](https://pycqa.github.io/isort/)** (>=5.12.0)
* **[flake8](https://github.com/pycqa/flake8)** (>=6.0.0)
* **[flake8-pyproject](https://github.com/john-hen/Flake8-pyproject)** (>=1.2.2)

**Testing**

* **[nbmake](https://github.com/treebeardtech/nbmake)** (>=1.4.1)
* **[pytest](https://docs.pytest.org/en/latest/)** (>=7.2.1)
* **[pytest-cov](https://pytest-cov.readthedocs.io/en/latest/)** (>=4.0.0)
* **[pytest-xdist](https://github.com/pytest-dev/pytest-xdist)** (>=3.2.0)
* **[scikeras](https://github.com/adriangb/scikeras)** (>=0.10.0)
* **[tensorflow](https://pypi.org/project/tensorflow/#history)** (>=2.11.0)

**Documentation**

* **[mike](https://github.com/jimporter/mike)** (>=1.1.2)
* **[mkdocs](https://www.mkdocs.org/)** (>=1.4.2)
* **[mkdocs-autorefs](https://mkdocstrings.github.io/autorefs/)** (>=0.4.1)
* **[mkdocs-jupyter](https://github.com/danielfrg/mkdocs-jupyter)** (>=0.22.0)
* **[mkdocs-material](https://squidfunk.github.io/mkdocs-material/)** (>=9.1.0)
* **[mkdocs-simple-hooks](https://github.com/aklajnert/mkdocs-simple-hooks)** (>=0.1.5)
* **[pymdown-extensions](https://github.com/facelessuser/pymdown-extensions)** (>=9.9.2)
* **[pyyaml](https://pyyaml.org/)** (>=6.0)
