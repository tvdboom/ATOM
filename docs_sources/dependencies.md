# Dependencies
--------------

## Python & OS

As of the moment, ATOM supports the following Python versions:

* [Python 3.7](https://www.python.org/downloads/release/python-370/)
* [Python 3.8](https://www.python.org/downloads/release/python-380/)
* [Python 3.9](https://www.python.org/downloads/release/python-390/)
* [Python 3.10](https://www.python.org/downloads/release/python-3100/)

And operating systems:

 * macOS
 * Unix-like (Ubuntu, Fedora, etc...)
 * Windows 8.1+

<br><br>


## Packages

### Required

ATOM is built on top of several existing Python libraries. These
packages are necessary for its correct functioning.

* **[numpy](https://numpy.org/)** (>=1.21.6)
* **[scipy](https://www.scipy.org/)** (>=1.8.1)
* **[pandas](https://pandas.pydata.org/)** (>=1.3.5)
* **[pandas-profiling](https://pandas-profiling.github.io/pandas-profiling/docs/)** (>=3.2.0)
* **[explainerdashboard](https://explainerdashboard.readthedocs.io/en/latest/)** (>=0.3.8.2)
* **[mlflow](https://mlflow.org/)** (>=1.26)
* **[dill](https://pypi.org/project/dill/)** (>=0.3.5)
* **[tqdm](https://tqdm.github.io/)** (>=4.64)
* **[joblib](https://joblib.readthedocs.io/en/latest/)** (>=1.1.0)
* **[typeguard](https://typeguard.readthedocs.io/en/latest/)** (>=2.13)
* **[scikit-learn](https://scikit-learn.org/stable/)** (>=1.0.1)
* **[scikit-optimize](https://scikit-optimize.github.io/stable/)** (>=0.9.0)
* **[nltk](https://www.nltk.org/)** (>=3.7)
* **[tpot](http://epistasislab.github.io/tpot/)** (>=0.11.7)
* **[category-encoders](https://contrib.scikit-learn.org/categorical-encoding/index.html)** (>=2.4.1)
* **[imbalanced-learn](https://imbalanced-learn.readthedocs.io/en/stable/api.html)** (>=0.9.0)
* **[featuretools](https://www.featuretools.com/)** (>=1.9.0)
* **[gplearn](https://gplearn.readthedocs.io/en/stable/index.html)** (>=0.4.1)
* **[zoofs](https://jaswinder9051998.github.io/zoofs/)** (>=0.1.24)
* **[matplotlib](https://matplotlib.org/)** (>=3.5.0)
* **[seaborn](https://seaborn.pydata.org/)** (>=0.11.0)
* **[shap](https://github.com/slundberg/shap/)** (>=0.40)
* **[wordcloud](http://amueller.github.io/word_cloud/)** (>=1.8.1)
* **[schemdraw](https://schemdraw.readthedocs.io/en/latest/index.html)** (>=0.14)

### Optional

You can install some optional packages to be able to use some well-known
machine learning estimators that are not provided by sklearn but are
among ATOM's [predefined models](../user_guide/models/#predefined-models).
Install them using `pip install atom-ml[models]`.

* **[xgboost](https://xgboost.readthedocs.io/en/latest/)** (>=0.90)
* **[lightgbm](https://lightgbm.readthedocs.io/en/latest/)** (>=3.3.2)
* **[catboost](https://catboost.ai/docs/concepts/about.html)** (>=1.0.4)


### Development

The development dependencies are not installed with the package, and are
not required for any of its functionalities. These libraries are only
necessary to [contribute](../contributing) to the project. Install them
using `pip install atom-ml[dev]`.

* **[isort](https://pycqa.github.io/isort/)** (>=5.10.1)
* **[flake8](https://github.com/pycqa/flake8)** (>=4.0.1)
* **[pytest](https://docs.pytest.org/en/latest/)** (>=7.1.0)
* **[pytest-cov](https://pytest-cov.readthedocs.io/en/latest/)** (>=3.0.0)
* **[tensorflow](https://www.tensorflow.org/)** (>=2.3.1)
* **[mkdocs](https://www.mkdocs.org/)** (>=1.2.3)
* **[mkdocs-material](https://squidfunk.github.io/mkdocs-material/)** (>=8.2.5)
* **[mkdocs-jupyter](https://github.com/danielfrg/mkdocs-jupyter)** (>=0.20.1)
