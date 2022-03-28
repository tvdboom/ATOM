# Getting started
-----------------

## Installation

**Standard installation**

Install ATOM's newest release easily via `pip`:

    $ pip install -U atom-ml

or via `conda`:

    $ conda install -c conda-forge atom-ml

!!! note
    Since atom was already taken, download the package under the name `atom-ml`!

<br style="display: block; margin-top: 2em; content: ' '">

**Optional dependencies**

To install the [optional dependencies](../dependencies/#optional), add
[models] after the package's name.

    $ pip install -U atom-ml[models]

<br style="display: block; margin-top: 2em; content: ' '">

**Latest source**

Sometimes, new features and bug fixes are already implemented in the
`development` branch, but waiting for the next release to be made
available. If you can't wait for that, it's possible to install the
package directly from git.

    $ pip install git+https://github.com/tvdboom/ATOM.git@development#egg=atom-ml

Don't forget to include `#egg=atom-ml` to explicitly name the project,
this way pip can track metadata for it without having to have run the
`setup.py` script.

<br style="display: block; margin-top: 2em; content: ' '">

**Contributing**

If you are planning to [contribute](../contributing) to the project,
you'll need the [development dependencies](../dependencies/#development).
Install them adding [dev] after the package's name.

    $ pip install -U atom-ml[dev]

Click [here](https://pypi.org/simple/atom-ml/) for a complete list of
package files for all versions published on PyPI.

<br><br>


## Usage

[![Colab](https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/drive/1PnYfycwdmKw8dGyygwh7F0S3A4Rc47lI?usp=sharing)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tvdboom/ATOM/HEAD)

ATOM contains a variety of classes and functions to perform data cleaning,
feature engineering, model training, plotting and much more. The easiest
way to use everything ATOM has to offer is through one of the main classes:

* [ATOMClassifier](../API/ATOM/atomclassifier) for binary or multiclass classification tasks.
* [ATOMRegressor](../API/ATOM/atomregressor) for regression tasks.

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
manipulate. This data is stored in the instance and can be accessed at
any moment through atom's [data attributes](../API/ATOM/atomclassifier/#data-attributes).
You can either let atom split the dataset into a train and test set or
provide the sets yourself.

```python
atom = ATOMClassifier(X, y="RainTomorrow", test_sixe=0.3, n_rows=1e3, verbose=2)
```

Data transformations are applied through atom's methods. For example,
calling the [impute](../API/ATOM/atomclassifier/#impute) method will
initialize an [Imputer](../API/data_cleaning/imputer) instance, fit it
on the training set and transform the whole dataset. The transformations
are applied immediately after calling the method (no fit and transform
commands necessary).

```python
atom.impute(strat_num="median", strat_cat="most_frequent")  
atom.encode(strategy="LeaveOneOut", max_onehot=8)
```

Similarly, models are [trained and evaluated](../user_guide/training) using the
[run](../API/ATOM/atomclassifier/#run) method. Here, we fit both a
[Random Forest](../API/models/rf) and [AdaBoost](../API/models/adab) model
while applying [hyperparameter tuning](../user_guide/training/#hyperparameter-tuning).

```python
atom.run(models=["RF", "AdaB"], metric="auc", n_calls=10, n_initial_points=4)
```

Lastly, visualize the result using the integrated [plots](../user_guide/plots).

```python
atom.plot_roc()
atom.rf.plot_confusion_matrix(normalize=True)
```