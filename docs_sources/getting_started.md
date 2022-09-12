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

To install the [optional dependencies][optional], add `[models]` after
the package's name.

    pip install -U atom-ml[models]

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

[![Colab](https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/drive/1tgcn6qw_P0QLsrlQpSpMjjv_MV5GP17j#offline=true&sandboxMode=true)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tvdboom/ATOM/HEAD)

ATOM contains a variety of classes and functions to perform data cleaning,
feature engineering, model training, plotting and much more. The easiest
way to use everything ATOM has to offer is through one of the main classes:

* [ATOMClassifier][] for binary or multiclass classification tasks.
* [ATOMRegressor][] for regression tasks.

Let's walk you through an example. Click on the Google Colab badge on top
of this section to run this example yourself.

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
manipulate. You can either let atom split the dataset into a train and
test set or provide the sets yourself.

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
>>> atom.encode(strategy="LeaveOneOut", max_onehot=8)

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
 --> LeaveOneOut-encoding feature Location. Contains 49 classes.
 --> LeaveOneOut-encoding feature WindGustDir. Contains 16 classes.
 --> LeaveOneOut-encoding feature WindDir9am. Contains 16 classes.
 --> LeaveOneOut-encoding feature WindDir3pm. Contains 16 classes.
 --> Ordinal-encoding feature RainToday. Contains 2 classes.

```

Similarly, models are [trained and evaluated][training] using the
[run][atomclassifier-run] method. Here, we fit both a [RandomForest][]
and [AdaBoost][] model, and apply [hyperparameter tuning][].

```pycon
>>> atom.run(models=["RF", "AdaB"], metric="auc", n_trials=10, n_initial_points=4)

Training ========================= >>
Models: RF, AdaB
Metric: roc_auc


Running BO for Random Forest...
| call             | n_estimators | criterion | max_depth | min_samples_split | min_samples_leaf | max_features | bootstrap | max_samples | ccp_alpha | roc_auc | best_roc_auc | time_trial | time_bo |
| ---------------- | ------------ | --------- | --------- | ----------------- | ---------------- | ------------ | --------- | ----------- | --------- | ------- | ------------ | ---------- | ------- |
| Initial point 1  |          499 |   entropy |         2 |                20 |                5 |          0.6 |      True |         0.7 |    0.0327 |  0.8013 |       0.8013 |     0.885s |  0.902s |
| Initial point 2  |          425 |      gini |         8 |                10 |                5 |          0.7 |     False |         --- |    0.0151 |  0.8405 |       0.8405 |     1.345s |  2.249s |
| Initial point 3  |          470 |   entropy |        11 |                16 |                3 |          0.7 |     False |         --- |     0.029 |  0.7664 |       0.8405 |     1.784s |  4.036s |
| Initial point 4  |          144 |      gini |        11 |                13 |               14 |          0.6 |      True |         0.5 |     0.005 |  0.9164 |       0.9164 |     0.329s |  4.368s |
| Iteration 5      |           84 |      gini |        15 |                13 |               16 |          0.6 |      True |         0.5 |    0.0028 |  0.8085 |       0.9164 |     0.250s |  4.967s |
| Iteration 6      |          454 |      gini |         1 |                 9 |               19 |         None |      True |         0.9 |     0.035 |  0.7037 |       0.9164 |     0.678s |  5.909s |
| Iteration 7      |          461 |   entropy |         1 |                15 |               13 |          0.9 |     False |         --- |    0.0286 |  0.6998 |       0.9164 |     0.666s |  6.836s |
| Iteration 8      |          111 |      gini |         2 |                 6 |               14 |         log2 |      True |         0.7 |    0.0106 |  0.8546 |       0.9164 |     0.245s |  7.370s |
| Iteration 9      |           33 |   entropy |         9 |                12 |                5 |         sqrt |     False |         --- |       0.0 |  0.8466 |       0.9164 |     0.181s |  7.832s |
| Iteration 10     |          301 |      gini |        15 |                11 |               14 |          0.7 |      True |        None |    0.0186 |  0.8342 |       0.9164 |     0.730s |  8.857s |
Bayesian Optimization ---------------------------
Best call --> Initial point 4
Best parameters --> {'n_estimators': 144, 'criterion': 'gini', 'max_depth': 11, 'min_samples_split': 13, 'min_samples_leaf': 14, 'max_features': 0.6, 'bootstrap': True, 'max_samples': 0.5, 'ccp_alpha': 0.005}
Best evaluation --> roc_auc: 0.9164
Time elapsed: 8.857s
Fit ---------------------------------------------
Train evaluation --> roc_auc: 0.904
Test evaluation --> roc_auc: 0.8496
Time elapsed: 0.255s
-------------------------------------------------
Total time: 9.463s


Running BO for AdaBoost...
| call             | n_estimators | learning_rate | algorithm | roc_auc | best_roc_auc | time_trial | time_bo |
| ---------------- | ------------ | ------------- | --------- | ------- | ------------ | ---------- | ------- |
| Initial point 1  |          499 |        6.2758 |   SAMME.R |  0.6546 |       0.6546 |     0.831s |  0.838s |
| Initial point 2  |          500 |        0.0511 |   SAMME.R |  0.8837 |       0.8837 |     0.846s |  1.686s |
| Initial point 3  |          225 |        1.0215 |     SAMME |  0.8136 |       0.8837 |     0.466s |  2.155s |
| Initial point 4  |          431 |        0.0871 |     SAMME |  0.8817 |       0.8837 |     0.700s |  2.857s |
| Iteration 5      |          150 |        0.0536 |     SAMME |  0.7296 |       0.8837 |     0.325s |  3.371s |
| Iteration 6      |          500 |        0.1425 |   SAMME.R |  0.7254 |       0.8837 |     0.824s |  4.366s |
| Iteration 7      |          446 |        0.0724 |     SAMME |  0.8069 |       0.8837 |     0.741s |  5.328s |
| Iteration 8      |          469 |        0.0959 |     SAMME |  0.7715 |       0.8837 |     0.739s |  6.271s |
| Iteration 9      |          425 |        0.0913 |     SAMME |  0.8237 |       0.8837 |     0.673s |  7.189s |
| Iteration 10     |          396 |        0.0844 |     SAMME |  0.7224 |       0.8837 |     0.636s |  8.166s |
Bayesian Optimization ---------------------------
Best call --> Initial point 2
Best parameters --> {'n_estimators': 500, 'learning_rate': 0.0511, 'algorithm': 'SAMME.R'}
Best evaluation --> roc_auc: 0.8837
Time elapsed: 8.166s
Fit ---------------------------------------------
Train evaluation --> roc_auc: 0.9371
Test evaluation --> roc_auc: 0.7811
Time elapsed: 0.868s
-------------------------------------------------
Total time: 9.351s


Final results ==================== >>
Total time: 18.813s
-------------------------------------
Random Forest --> roc_auc: 0.8496 !
AdaBoost      --> roc_auc: 0.7811

```

And lastly, analyze the results.

```pycon
>>> atom.evaluate()

      accuracy  average_precision  ...    recall   roc_auc
RF       0.840           0.681397  ...  0.355556  0.849606
AdaB     0.835           0.590445  ...  0.466667  0.781075

[2 rows x 9 columns]

```
