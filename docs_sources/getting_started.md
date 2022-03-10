# Getting started
-----------------

## Installation

Install ATOM's newest release easily via `pip`:

    $ pip install -U atom-ml

or via `conda`:

    $ conda install -c conda-forge atom-ml

These commands will install ATOM and all [required dependencies](../dependencies/#required).
To install the [optional dependencies](../dependencies/#optional) as well, add [models]
after the package's name.

    $ pip install -U atom-ml[models]

!!! note
    Since atom was already taken, download the package under the name `atom-ml`!

<br>

Sometimes, new features and bug fixes are already implemented in the
`development` branch, but waiting for the next release to be made
available. If you can't wait for that, it's possible to install the
package directly from git.

    $ pip install git+https://github.com/tvdboom/ATOM.git@development#egg=atom-ml

Don't forget to include `#egg=atom-ml` to explicitly name the project,
this way pip can track metadata for it without having to have run the
`setup.py` script.

Click [here](https://pypi.org/simple/atom-ml/) for a complete list of
package files for all versions published on PyPI.

<br><br>


## Usage

[![Colab](https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/drive/1PnYfycwdmKw8dGyygwh7F0S3A4Rc47lI?usp=sharing)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tvdboom/ATOM/HEAD)

Make the necessary imports and load the data.

```python
import pandas as pd
from atom import ATOMClassifier

# Load the Australian Weather dataset
X = pd.read_csv("https://raw.githubusercontent.com/tvdboom/ATOM/master/examples/datasets/weatherAUS.csv")
X.head()
```

Initialize the [ATOMClassifier](../API/ATOM/atomclassifier) or [ATOMRegressor](../API/ATOM/atomregressor) class.

```python
atom = ATOMClassifier(X, y="RainTomorrow", n_rows=1e3, verbose=2)
```

Use the data cleaning methods to prepare the data for modelling.

```python
atom.impute(strat_num="median", strat_cat="most_frequent")  
atom.encode(strategy="LeaveOneOut", max_onehot=8)  
atom.feature_selection(strategy="PCA", n_features=12)
```

Train and evaluate the models you want to compare.

```python
atom.run(
    models=["LR", "RF", "LGB"],
    metric="auc",
    n_calls=10,
    n_initial_points=4,
)
```

Make plots to analyze the results.

```python
atom.plot_roc()
atom.rf.plot_confusion_matrix(normalize=True)
```