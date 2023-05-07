# Getting started
-----------------

## Installation

Install ATOM's newest release easily via `pip`:

    pip install -U atom-ml

or via `conda`:

    conda install -c conda-forge atom-ml

!!! note
    Since atom was already taken, download the package under the name `atom-ml`!

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

[![SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/tvdboom/ATOM/blob/master/examples/getting_started.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tvdboom/ATOM/HEAD)

ATOM contains a variety of classes and functions to perform data cleaning,
feature engineering, model training, plotting and much more. The easiest
way to use everything ATOM has to offer is through one of the main classes:

* [ATOMClassifier][] for binary or multiclass classification tasks.
* [ATOMRegressor][] for regression tasks.

Let's walk you through an example. Click on the SageMaker Studio Lab badge
on top of this section to run this example yourself.

Make the necessary imports and load the data.

```pycon
>>> import pandas as pd
>>> from atom import ATOMClassifier

>>> # Load the Australian Weather dataset
>>> X = pd.read_csv("https://raw.githubusercontent.com/tvdboom/ATOM/master/examples/datasets/weatherAUS.csv")
>>> print(X)

             Location  MinTemp  MaxTemp  ...  Temp9am  Temp3pm  RainToday
0    MelbourneAirport     18.0     26.9  ...     18.5     26.0        Yes
1            Adelaide     17.2     23.4  ...     17.7     21.9         No
2              Cairns     18.6     24.6  ...     20.8     24.1        Yes
3            Portland     13.6     16.8  ...     15.6     16.0        Yes
4             Walpole     16.4     19.9  ...     17.4     18.1         No
..                ...      ...      ...  ...      ...      ...        ...
995            Hobart     12.6     21.8  ...     18.0     18.7         No
996      PerthAirport      7.1     20.0  ...     14.3     19.7         No
997        WaggaWagga     10.0     25.9  ...     17.0     24.2         No
998            Albany     12.9     17.4  ...     15.6     16.8         No
999           Mildura     -1.0     14.9  ...      4.1     14.5         No

[1000 rows x 21 columns]

```

Initialize the ATOMClassifier or ATOMRegressor class. These two classes
are convenient wrappers for the whole machine learning pipeline. Contrary
to sklearn's API, they are initialized providing the data you want to
manipulate.

```pycon
>>> atom = ATOMClassifier(X, y="RainTomorrow", n_rows=1000, verbose=2)

<< ================== ATOM ================== >>
Algorithm task: binary classification.

Dataset stats ==================== >>
Shape: (1000, 22)
Memory: 434.38 kB
Scaled: False
Missing values: 2131 (9.7%)
Categorical features: 5 (23.8%)
-------------------------------------
Train set size: 800
Test set size: 200
-------------------------------------
|   |     dataset |       train |        test |
| - | ----------- | ----------- | ----------- |
| 0 |   775 (3.4) |   620 (3.4) |   155 (3.4) |
| 1 |   225 (1.0) |   180 (1.0) |    45 (1.0) |

```

Data transformations are applied through atom's methods. For example,
calling the [impute][atomclassifier-impute] method will initialize an
[Imputer][] instance, fit it on the training set and transform the whole
dataset. The transformations are applied immediately after calling the
method (no fit and transform commands necessary).

```pycon
>>> atom.impute(strat_num="median", strat_cat="most_frequent")  
>>> atom.encode(strategy="Target", max_onehot=8)

Fitting Imputer...
Imputing missing values...
 --> Imputing 1 missing values with median (12.0) in feature MinTemp.
 --> Imputing 2 missing values with median (22.8) in feature MaxTemp.
 --> Imputing 5 missing values with median (0.0) in feature Rainfall.
 --> Imputing 430 missing values with median (4.6) in feature Evaporation.
 --> Imputing 453 missing values with median (8.2) in feature Sunshine.
 --> Imputing 65 missing values with most_frequent (W) in feature WindGustDir.
 --> Imputing 65 missing values with median (39.0) in feature WindGustSpeed.
 --> Imputing 63 missing values with most_frequent (N) in feature WindDir9am.
 --> Imputing 24 missing values with most_frequent (WSW) in feature WindDir3pm.
 --> Imputing 8 missing values with median (13.0) in feature WindSpeed9am.
 --> Imputing 19 missing values with median (19.0) in feature WindSpeed3pm.
 --> Imputing 10 missing values with median (70.0) in feature Humidity9am.
 --> Imputing 19 missing values with median (53.0) in feature Humidity3pm.
 --> Imputing 94 missing values with median (1017.4) in feature Pressure9am.
 --> Imputing 96 missing values with median (1015.15) in feature Pressure3pm.
 --> Imputing 369 missing values with median (6.0) in feature Cloud9am.
 --> Imputing 386 missing values with median (5.0) in feature Cloud3pm.
 --> Imputing 3 missing values with median (16.8) in feature Temp9am.
 --> Imputing 14 missing values with median (21.2) in feature Temp3pm.
 --> Imputing 5 missing values with most_frequent (No) in feature RainToday.
Fitting Encoder...
Encoding categorical columns...
 --> Target-encoding feature Location. Contains 49 classes.
 --> Target-encoding feature WindGustDir. Contains 16 classes.
 --> Target-encoding feature WindDir9am. Contains 16 classes.
 --> Target-encoding feature WindDir3pm. Contains 16 classes.
 --> Ordinal-encoding feature RainToday. Contains 2 classes.

```

Similarly, models are [trained and evaluated][training] using the
[run][atomclassifier-run] method. Here, we fit both a [LinearDiscriminantAnalysis][]
and [AdaBoost][] model, and apply [hyperparameter tuning][].

```pycon
>>> atom.run(models=["LDA", "AdaB"], metric="auc", n_trials=10)

Training ========================= >>
Models: LDA, AdaB
Metric: roc_auc


Running hyperparameter tuning for LinearDiscriminantAnalysis...
| trial |  solver | shrinkage | roc_auc | best_roc_auc | time_trial | time_ht |    state |
| ----- | ------- | --------- | ------- | ------------ | ---------- | ------- | -------- |
| 0     |   eigen |      auto |  0.7888 |       0.7888 |     0.156s |  0.156s | COMPLETE |
| 1     |    lsqr |       0.9 |  0.7988 |       0.7988 |     0.141s |  0.297s | COMPLETE |
| 2     |    lsqr |       1.0 |  0.8125 |       0.8125 |     0.141s |  0.438s | COMPLETE |
| 3     |    lsqr |       0.6 |   0.858 |        0.858 |     0.125s |  0.563s | COMPLETE |
| 4     |   eigen |       1.0 |   0.782 |        0.858 |     0.125s |  0.688s | COMPLETE |
| 5     |    lsqr |       0.8 |  0.8396 |        0.858 |     0.141s |  0.828s | COMPLETE |
| 6     |     svd |       --- |  0.7968 |        0.858 |     0.141s |  0.969s | COMPLETE |
| 7     |    lsqr |       0.7 |  0.8208 |        0.858 |     0.125s |  1.094s | COMPLETE |
| 8     |   eigen |       0.9 |  0.8548 |        0.858 |     0.141s |  1.234s | COMPLETE |
| 9     |   eigen |       0.7 |  0.8401 |        0.858 |     0.125s |  1.359s | COMPLETE |
Hyperparameter tuning ---------------------------
Best trial --> 3
Best parameters:
 --> solver: lsqr
 --> shrinkage: 0.6
Best evaluation --> roc_auc: 0.858
Time elapsed: 1.359s
Fit ---------------------------------------------
Train evaluation --> roc_auc: 0.8321
Test evaluation --> roc_auc: 0.8668
Time elapsed: 0.016s
-------------------------------------------------
Total time: 1.375s


Running hyperparameter tuning for AdaBoost...
| trial | n_estimators | learning_rate | algorithm | roc_auc | best_roc_auc | time_trial | time_ht |    state |
| ----- | ------------ | ------------- | --------- | ------- | ------------ | ---------- | ------- | -------- |
| 0     |          480 |        1.6346 |     SAMME |   0.744 |        0.744 |     0.734s |  0.734s | COMPLETE |
| 1     |          410 |        6.7512 |     SAMME |  0.3454 |        0.744 |     0.125s |  0.859s | COMPLETE |
| 2     |          480 |        7.4271 |   SAMME.R |  0.3513 |        0.744 |     0.788s |  1.648s | COMPLETE |
| 3     |          460 |        0.0238 |   SAMME.R |  0.8365 |       0.8365 |     0.799s |  2.447s | COMPLETE |
| 4     |          400 |        1.8333 |   SAMME.R |  0.6328 |       0.8365 |     0.672s |  3.118s | COMPLETE |
| 5     |          420 |        6.7105 |   SAMME.R |  0.3772 |       0.8365 |     0.694s |  3.813s | COMPLETE |
| 6     |          240 |         0.358 |   SAMME.R |  0.7413 |       0.8365 |     0.469s |  4.282s | COMPLETE |
| 7     |          320 |        0.2011 |     SAMME |  0.8418 |       0.8418 |     0.531s |  4.813s | COMPLETE |
| 8     |          260 |        0.1667 |     SAMME |   0.808 |       0.8418 |     0.454s |  5.267s | COMPLETE |
| 9     |          450 |        0.0464 |     SAMME |  0.8001 |       0.8418 |     0.688s |  5.954s | COMPLETE |
Hyperparameter tuning ---------------------------
Best trial --> 7
Best parameters:
 --> n_estimators: 320
 --> learning_rate: 0.2011
 --> algorithm: SAMME
Best evaluation --> roc_auc: 0.8418
Time elapsed: 5.954s
Fit ---------------------------------------------
Train evaluation --> roc_auc: 0.9087
Test evaluation --> roc_auc: 0.7987
Time elapsed: 0.453s
-------------------------------------------------
Total time: 6.407s


Final results ==================== >>
Total time: 8.017s
-------------------------------------
LinearDiscriminantAnalysis --> roc_auc: 0.8668 !
AdaBoost                   --> roc_auc: 0.7987
```

And lastly, analyze the results.

```pycon
>>> atom.evaluate()

      accuracy  average_precision  ...  recall  roc_auc
LDA      0.850             0.6893  ...  0.5333   0.8668
AdaB     0.825             0.6344  ...  0.3556   0.7987

[2 rows x 9 columns]

```
