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

* **[category-encoders](https://contrib.scikit-learn.org/categorical-encoding/index.html)** (>=2.4.1)
* **[explainerdashboard](https://explainerdashboard.readthedocs.io/en/latest/)** (>=0.4.0)
* **[dill](https://pypi.org/project/dill/)** (>=0.3.5)
* **[evalml](https://evalml.alteryx.com/en/stable/)** (>=0.62.0)
* **[gplearn](https://gplearn.readthedocs.io/en/stable/index.html)** (>=0.4.1)
* **[gradio](https://github.com/gradio-app/gradio)** (>=3.3.1)
* **[imbalanced-learn](https://imbalanced-learn.readthedocs.io/en/stable/api.html)** (>=0.9.0)
* **[featuretools](https://www.featuretools.com/)** (>=1.14.0)
* **[joblib](https://joblib.readthedocs.io/en/latest/)** (>=1.1.0, <1.2.0)
* **[matplotlib](https://matplotlib.org/)** (>=3.5.0, <3.6.0)
* **[mlflow](https://mlflow.org/)** (>=2.0.1)
* **[nltk](https://www.nltk.org/)** (>=3.7)
* **[numpy](https://numpy.org/)** (>=1.22)
* **[optuna](https://optuna.org/)** (>=3.0.0)
* **[pandas](https://pandas.pydata.org/)** (>=1.3.5)
* **[pandas-profiling](https://pandas-profiling.github.io/pandas-profiling/docs/)** (>=3.5.0)
* **[plotly](https://plotly.com/python/)** (>=5.10.0)
* **[shap](https://github.com/slundberg/shap/)** (>=0.41)
* **[schemdraw](https://schemdraw.readthedocs.io/en/latest/index.html)** (>=0.14)
* **[scikit-learn](https://scikit-learn.org/stable/)** (>=1.1.0)
* **[scikit-learn-intelex](https://github.com/intel/scikit-learn-intelex)** (>=2021.6.3)
* **[scipy](https://www.scipy.org/)** (>=1.8.1)
* **[typeguard](https://typeguard.readthedocs.io/en/latest/)** (>=2.13)
* **[wordcloud](http://amueller.github.io/word_cloud/)** (>=1.8.1)
* **[zoofs](https://jaswinder9051998.github.io/zoofs/)** (>=0.1.24)


### Optional

You can install some optional packages to be able to use some well-known
machine learning estimators that are not provided by sklearn but are
among ATOM's [predefined models][]. Install them using `pip install atom-ml[models]`.

* **[catboost](https://catboost.ai/docs/concepts/about.html)** (>=1.0.4)
* **[lightgbm](https://lightgbm.readthedocs.io/en/latest/)** (>=3.3.2)
* **[xgboost](https://xgboost.readthedocs.io/en/latest/)** (>=1.6.0)


### Development

The development dependencies are not installed with the package, and are
not required for any of its functionalities. These libraries are only
necessary to [contribute][contributing] to the project. Install them
using `pip install atom-ml[dev]`.

**Linting**

* **[isort](https://pycqa.github.io/isort/)** (>=5.10.1)
* **[flake8](https://github.com/pycqa/flake8)** (>=5.0.4)
* **[flake8-pyproject](https://github.com/john-hen/Flake8-pyproject)** (>=1.1.0)

**Testing**

* **[pytest](https://docs.pytest.org/en/latest/)** (>=7.1.0)
* **[pytest-cov](https://pytest-cov.readthedocs.io/en/latest/)** (>=3.0.0)

**Documentation**

* **[mike](https://github.com/jimporter/mike)** (>=1.1.2)
* **[mkdocs](https://www.mkdocs.org/)** (>=1.2.3)
* **[mkdocs-autorefs](https://mkdocstrings.github.io/autorefs/)** (>=0.4.1)
* **[mkdocs-jupyter](https://github.com/danielfrg/mkdocs-jupyter)** (>=0.22.0)
* **[mkdocs-material](https://squidfunk.github.io/mkdocs-material/)** (>=8.5.3)
* **[mkdocs-simple-hooks](https://github.com/aklajnert/mkdocs-simple-hooks)** (>=0.1.5)
* **[pyyaml](https://pyyaml.org/)** (>=6.0)
