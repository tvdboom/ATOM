<div align="center">
<p align="center">
	<img src="https://github.com/tvdboom/ATOM/blob/master/images/logo.png?raw=true" alt="ATOM" title="ATOM" height="130" width="500"/>
</p>

# Automated Tool for Optimized Modelling
### A Python package for fast exploration of machine learning pipelines
</div>

<br><br>



📜 Overview 
-----------

<p align="center" style="font-size: 1.4em">
<a href="https://github.com/tvdboom" style="text-decoration: none" draggable="false"><img src="https://github.com/tvdboom/ATOM/blob/master/docs_sources/img/icons/avatar.png?raw=true" alt="Author" height=15 width=15 draggable="false" /> Mavs</a>
&nbsp;&nbsp;&nbsp;&nbsp;
<a href="mailto:m.524687@gmail.com" style="text-decoration: none" draggable="false"><img src="https://github.com/tvdboom/ATOM/blob/master/docs_sources/img/icons/email.png?raw=true" alt="Email" height=13 width=17 draggable="false" /> m.524687@gmail.com</a>
&nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://tvdboom.github.io/ATOM/" style="text-decoration: none" draggable="false"><img src="https://github.com/tvdboom/ATOM/blob/master/docs_sources/img/icons/documentation.png?raw=true" alt="Documentation" height=17 width=17 draggable="false" /> Documentation</a>
&nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://join.slack.com/t/atom-alm7229/shared_invite/zt-upd8uc0z-LL63MzBWxFf5tVWOGCBY5g" style="text-decoration: none" draggable="false"><img src="https://github.com/tvdboom/ATOM/blob/master/docs_sources/img/icons/slack.png?raw=true" alt="Slack" height=16 width=16 draggable="false"/> Slack</a>
</p>

<br>

**General Information** | |
--- | ---
**Repository** | [![Project Status: Active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active) [![Conda Recipe](https://img.shields.io/badge/recipe-atom--ml-green.svg)](https://anaconda.org/conda-forge/atom-ml) [![License: MIT](https://img.shields.io/github/license/tvdboom/ATOM)](https://opensource.org/licenses/MIT) [![Downloads](https://pepy.tech/badge/atom-ml)](https://pepy.tech/project/atom-ml)
**Release** | [![PyPI version](https://img.shields.io/pypi/v/atom-ml)](https://pypi.org/project/atom-ml/) [![Conda Version](https://img.shields.io/conda/vn/conda-forge/atom-ml.svg)](https://anaconda.org/conda-forge/atom-ml) [![DOI](https://zenodo.org/badge/195069958.svg)](https://zenodo.org/badge/latestdoi/195069958)
**Compatibility** | [![Python 3.8\|3.9\|3.10](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue?logo=python)](https://www.python.org) [![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/atom-ml.svg)](https://anaconda.org/conda-forge/atom-ml)
**Build status** | [![Build Status](https://github.com/tvdboom/ATOM/workflows/ATOM/badge.svg)](https://github.com/tvdboom/ATOM/actions) [![Azure Pipelines](https://dev.azure.com/conda-forge/feedstock-builds/_apis/build/status/atom-ml-feedstock?branchName=master)](https://dev.azure.com/conda-forge/feedstock-builds/_build/latest?definitionId=10822&branchName=master) [![codecov](https://codecov.io/gh/tvdboom/ATOM/branch/master/graph/badge.svg)](https://codecov.io/gh/tvdboom/ATOM)
**Code analysis** | [![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/) [![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/) [![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/tvdboom/ATOM.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/tvdboom/ATOM/context:python) [![Total alerts](https://img.shields.io/lgtm/alerts/g/tvdboom/ATOM.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/tvdboom/ATOM/alerts/)


<br><br>



💡 Introduction  
---------------

During the exploration phase of a machine learning project, a data
scientist tries to find the optimal pipeline for his specific use case.
This usually involves applying standard data cleaning steps, creating
or selecting useful features, trying out different models, etc. Testing
multiple pipelines requires many lines of code, and writing it all in
the same notebook often makes it long and cluttered. On the other hand,
using multiple notebooks makes it harder to compare the results and to
keep an overview. On top of that, refactoring the code for every test
can be quite time-consuming. How many times have you conducted the same
action to pre-process a raw dataset? How many times have you
copy-and-pasted code from an old repository to re-use it in a new use
case?

ATOM is here to help solve these common issues. The package acts as
a wrapper of the whole machine learning pipeline, helping the data
scientist to rapidly find a good model for his problem. Avoid
endless imports and documentation lookups. Avoid rewriting the same
code over and over again. With just a few lines of code, it's now
possible to perform basic data cleaning steps, select relevant
features and compare the performance of multiple models on a given
dataset, providing quick insights on which pipeline performs best
for the task at hand.

Example steps taken by ATOM's pipeline:

1. Data Cleaning
	* Handle missing values
	* Encode categorical features
    * Detect and remove outliers
	* Balance the training set
2. Feature engineering
    * Create new non-linear features
	* Remove multi-collinear features
	* Remove features with too low variance
	* Select the most promising features
3. Train and validate multiple models
	* Select hyperparameters using a Bayesian Optimization approach
	* Train and test the models on the provided data
	* Assess the robustness of the output using a bootstrap algorithm
4. Analyze the results
    * Get the model scores on various metrics
    * Make plots to compare the model performances


<br/><br/>

<img src="https://github.com/tvdboom/ATOM/blob/master/images/diagram_pipeline.png?raw=true" alt="diagram_pipeline" title="diagram_pipeline" width="900" height="300" />

<br><br>


🛠️ Installation
---------------

Install ATOM's newest release easily via `pip`:

    $ pip install -U atom-ml


or via `conda`:

    $ conda install -c conda-forge atom-ml

<br><br>


⚡ Usage
-------

[![Colab](https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/drive/1tgcn6qw_P0QLsrlQpSpMjjv_MV5GP17j#offline=true&sandboxMode=true)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tvdboom/ATOM/HEAD)

ATOM contains a variety of classes and functions to perform data cleaning,
feature engineering, model training, plotting and much more. The easiest
way to use everything ATOM has to offer is through one of the main classes:

* [ATOMClassifier](https://tvdboom.github.io/ATOM/latest//API/ATOM/atomclassifier) for binary or multiclass classification tasks.
* [ATOMRegressor](https://tvdboom.github.io/ATOM/latest//API/ATOM/atomregressor) for regression tasks.

Let's walk you through an example. Click on the Google Colab badge on top
of this section to run this example yourself.

Make the necessary imports and load the data.

```python
import pandas as pd
from atom import ATOMClassifier

# Load the Australian Weather dataset
X = pd.read_csv("https://raw.githubusercontent.com/tvdboom/ATOM/master/examples/datasets/weatherAUS.csv")
X.head()
```

Initialize the ATOMClassifier or ATOMRegressor class. These two classes
are convenient wrappers for the whole machine learning pipeline. Contrary
to sklearn's API, they are initialized providing the data you want to
manipulate.

```python
atom = ATOMClassifier(X, y="RainTomorrow", test_size=0.3, verbose=2)
```

Data transformations are applied through atom's methods. For example,
calling the [impute](https://tvdboom.github.io/ATOM/latest/API/ATOM/atomclassifier/#impute)
method will initialize an [Imputer](https://tvdboom.github.io/ATOM/latest/API/data_cleaning/imputer)
instance, fit it on the training set and transform the whole dataset.
The transformations are applied immediately after calling the method
(no fit and transform commands necessary).

```python
atom.impute(strat_num="median", strat_cat="most_frequent")  
atom.encode(strategy="LeaveOneOut", max_onehot=8)
```

Similarly, models are [trained and evaluated](https://tvdboom.github.io/ATOM/latest/user_guide/training)
using the [run](https://tvdboom.github.io/ATOM/latest/API/ATOM/atomclassifier/#run)
method. Here, we fit both a [Random Forest](https://tvdboom.github.io/ATOM/latest/API/models/rf)
and [AdaBoost](https://tvdboom.github.io/ATOM/latest/API/models/adab) model,
and apply [hyperparameter tuning](https://tvdboom.github.io/ATOM/latest/user_guide/training/#hyperparameter-tuning).

```python
atom.run(models=["RF", "AdaB"], metric="auc", n_calls=10, n_initial_points=4)
```

Lastly, visualize the result using the integrated [plots](https://tvdboom.github.io/ATOM/latest/user_guide/plots).

```python
atom.plot_roc()
atom.rf.plot_confusion_matrix(normalize=True)
```

<br><br>


<img src="https://github.com/tvdboom/ATOM/blob/master/docs_sources/img/icons/documentation.png?raw=true" alt="Documentation" height=28 width=28 draggable="false" /> Documentation
----------------
  
**Relevant links** | |
--- | ---
⭐ **[About](https://tvdboom.github.io/ATOM/latest/release_history/)** | Learn more about the package.
🚀 **[Getting started](https://tvdboom.github.io/ATOM/latest/getting_started/)** | New to ATOM? Here's how to get you started!
📢 **[Release history](https://tvdboom.github.io/ATOM/latest/release_history/)** | What are the new features of the latest release?
👨‍💻 **[User guide](https://tvdboom.github.io/ATOM/latest/user_guide/introduction/)** | How to use ATOM and its features.
🎛️ **[API Reference](https://tvdboom.github.io/ATOM/latest/API/ATOM/atomclassifier/)** | The detailed reference for ATOM's API.
📋 **[Examples](https://tvdboom.github.io/ATOM/latest/examples/binary_classification/)** | Example notebooks show you what can be done and how.
❔ **[FAQ](https://tvdboom.github.io/ATOM/latest/faq/)** | Get answers to frequently asked questions.
🔧 **[Contributing](https://tvdboom.github.io/ATOM/latest/contributing/)** | Do you wan to contribute to the project? Read this before creating a PR.
🌳 **[Dependencies](https://tvdboom.github.io/ATOM/latest/dependencies/)** | Which other packages does ATOM depend on?
📃 **[License](https://tvdboom.github.io/ATOM/latest/license/)** | Copyright and permissions under the MIT license.
