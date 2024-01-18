# Getting started
-----------------

## Installation

Install ATOM's newest release easily via `pip`:

    pip install -U atom-ml

or via `conda`:

    conda install -c conda-forge atom-ml

!!! note
    Since atom was already taken, download the package under the name `atom-ml`!

!!! warning
    ATOM makes use of many other ML libraries, making its [dependency list][packages]
    quite long. Because of that, the installation may take longer than you
    are accustomed to. Be patient!

<br style="display: block; margin-top: 2em; content: ' '">

**Optional dependencies**

Some specific models, utility methods or plots require the installation of
additional libraries. To install the [optional dependencies][optional], add
`[full]` after the package's name.

    pip install -U atom-ml[full]

<br style="display: block; margin-top: 2em; content: ' '">

**Latest source**

Sometimes, new features and bug fixes are already implemented in the
`development` branch, but waiting for the next release to be made
available. If you can't wait for that, it's possible to install the
package directly from git.

    pip install git+https://github.com/tvdboom/ATOM.git@development#egg=atom-ml

Don't forget to include `#egg=atom-ml` to explicitly name the project,
this way pip can track metadata for it without having to have run the
`setup.py` script.

<br style="display: block; margin-top: 2em; content: ' '">

**Contributing**

If you are planning to [contribute][contributing] to the project,
you'll need the [development dependencies][development]. Install them
adding `[dev]` after the package's name.

    pip install -U atom-ml[dev]

Click [here](https://pypi.org/simple/atom-ml/) for a complete list of
package files for all versions published on PyPI.

<br><br>


## Usage

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tvdboom/ATOM/blob/master/examples/getting_started.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tvdboom/ATOM/HEAD)

ATOM contains a variety of classes and functions to perform data cleaning,
feature engineering, model training, plotting and much more. The easiest
way to use everything ATOM has to offer is through one of the main classes:

* [ATOMClassifier][] for classification tasks.
* [ATOMForecaster][] for forecasting tasks.
* [ATOMRegressor][] for regression tasks.

Let's walk you through an example. Click on the SageMaker Studio Lab badge
on top of this section to run this example yourself.

Make the necessary imports and load the data.

```pycon
import pandas as pd
from atom import ATOMClassifier

# Load the Australian Weather dataset
X = pd.read_csv("./examples/datasets/weatherAUS.csv", nrows=100)
print(X.head())
```

Initialize the ATOMClassifier or ATOMRegressor class. These two classes
are convenient wrappers for the whole machine learning pipeline. Contrary
to sklearn's API, they are initialized providing the data you want to
manipulate.

```pycon
import pandas as pd  # hide
from atom import ATOMClassifier  # hide
X = pd.read_csv("./examples/datasets/weatherAUS.csv", nrows=100)  # hide

atom = ATOMClassifier(X, y="RainTomorrow", verbose=2)
```

Data transformations are applied through atom's methods. For example,
calling the [impute][atomclassifier-impute] method will initialize an
[Imputer][] instance, fit it on the training set and transform the whole
dataset. The transformations are applied immediately after calling the
method (no fit and transform commands necessary).

```pycon
import pandas as pd  # hide
from atom import ATOMClassifier  # hide
X = pd.read_csv("./examples/datasets/weatherAUS.csv", nrows=100)  # hide

atom = ATOMClassifier(X, y="RainTomorrow")  # hide
atom.verbose = 2  # hide

atom.impute(strat_num="median", strat_cat="most_frequent")  
atom.encode(strategy="Target", max_onehot=8)
```

Similarly, models are [trained and evaluated][training] using the
[run][atomclassifier-run] method. Here, we fit both a [LogisticRegression][]
and [LinearDiscriminantAnalysis][] model, and apply [hyperparameter tuning][].

```pycon
import pandas as pd  # hide
from atom import ATOMClassifier  # hide
X = pd.read_csv("./examples/datasets/weatherAUS.csv", nrows=100)  # hide

atom = ATOMClassifier(X, y="RainTomorrow")  # hide

atom.impute(strat_num="median", strat_cat="most_frequent")  # hide 
atom.encode(strategy="Target", max_onehot=8)  # hide
atom.verbose = 2  # hide

atom.run(models=["LR", "LDA"], metric="auc", n_trials=6)
```

And lastly, analyze the results.

```pycon
import pandas as pd  # hide
from atom import ATOMClassifier  # hide
X = pd.read_csv("./examples/datasets/weatherAUS.csv", nrows=100)  # hide

atom = ATOMClassifier(X, y="RainTomorrow")  # hide

atom.impute(strat_num="median", strat_cat="most_frequent")  # hide 
atom.encode(strategy="Target", max_onehot=8)  # hide

atom.run(models=["LR", "LDA"], metric="auc", n_trials=6)  # hide

print(atom.evaluate())

atom.plot_lift()
```
