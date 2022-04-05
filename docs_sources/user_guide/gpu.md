# GPU
-----

Graphics Processing Units (GPUs) can significantly accelerate
calculations for preprocessing step or training machine learning
models. Training models involves compute-intensive matrix
multiplications and other operations that can take advantage of a
GPU's massively parallel architecture. Training on large datasets can
take hours to run on a single processor. However, if you offload those
tasks to a GPUs, you can reduce training time to minutes instead.

Training transformers and models in atom using a GPU is as easy as
initializing the instance with parameter `gpu=True`. The `gpu` parameter
accepts three options:

* False: Always use CPU implementation.
* True: Use GPU implementation. If this results in an error, use CPU instead.
  When this happens, a message is written to the logger.
* "force": Use GPU implementation. If this results in an error, raise it.

ATOM uses [cuML](https://docs.rapids.ai/api/cuml/stable/) for all estimators
except [XGB](../../API/models/xgb), [LGB](../../API/models/lgb) and
[CatB](../../API/models/catb), which come with their own GPU implementation.
Check which [prerequisites](#prerequisites) your machine needs for it
to work, and which [transformers](#transformers) and [models](#models)
are supported.

Be aware of the following:

* cuML estimators do not support sparse dataframes.
* cuML models sometimes use slightly different hyperparameters than
  their sklearn counterparts.
* cuML does not support multi-gpu training. If there is more than one
  GPU on the machine, the first one is used by default. Use `CUDA_VISIBLE_DEVICES`
  to use any of the other GPUs.

!!! example
    [![Colab](https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/drive/1b-piLK6_O99EDpLNmeKIsSRkQlk0WoTl#offline=true&sandboxMode=true)<br><br>
    Train a model on a GPU yourself using Google Colab. Just click on
    the badge above and follow the notebook. Note two things:

    * Make sure you've been allocated a Tesla T4, P4, or P100. If this is not
    the case (check it using `!nvidia-smi`), reset the runtime (Runtime -> 
    Factory reset runtime) until you get one.
    * Setting up the environment and installing the necessary libraries may
    take quite some time (usually up to 15min).


## Prerequisites

* Operating System:
    - Ubuntu 18.04/20.04 or CentOS 7/8 with gcc/++ 9.0+
    - Windows 8.1+ with WSL2 (see [here](https://developer.nvidia.com/blog/run-rapids-on-microsoft-windows-10-using-wsl-2-the-windows-subsystem-for-linux/) a tutorial)
* GPU: NVIDIA Pascal™ or better with [compute capability](https://developer.nvidia.com/cuda-gpus) 6.0+
* CUDA & NVIDIA Drivers: One of versions 11.0, 11.2, 11.4 or 11.5
* [cuML](https://docs.rapids.ai/api/cuml/stable/)>=0.15


## Transformers

* [Scaler](../../API/data_cleaning/scaler)
* [Imputer](../../API/data_cleaning/imputer) (not for strat_num="knn")
* [Discretizer](../../API/data_cleaning/discretizer) (not for strategy="custom")
* [FeatureSelector](../../API/feature_engineering/feature_selector) (only for strategy="pca")


## Models

* [Gaussian Naive Bayes](../../API/models/gnb)
* [Multinomial Naive Bayes](../../API/models/mnb)
* [Bernoulli Naive Bayes](../../API/models/bnb)
* [Categorical Naive Bayes](../../API/models/catnb)
* [Ordinary Least Squares](../../API/models/ols)
* [Ridge](../../API/models/ridge) (only for regression tasks)
* [Lasso](../../API/models/lasso)
* [ElasticNet](../../API/models/en)
* [Lars](../../API/models/lars)
* [Logistic Regression](../../API/models/lr)
* [K-Nearest Neighbors](../../API/models/knn)
* [Random Forest](../../API/models/rf)
* [XGBoost](../../API/models/xgb)
* [LightGBM](../../API/models/lgb) (requires [extra installations](https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html))
* [CatBoost](../../API/models/catb)
* [Linear SVM](../../API/models/lsvm)
* [Kernel SVM](../../API/models/ksvm)
