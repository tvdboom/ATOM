[![Colab](https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/drive/1PnYfycwdmKw8dGyygwh7F0S3A4Rc47lI?usp=sharing)

Graphics Processing Units (GPUs) can significantly accelerate the training
process for many machine learning models. Training models involves
compute-intensive matrix multiplications and other operations that can
take advantage of a GPU's massively parallel architecture. Training on
large datasets can take hours to run on a single processor. However, if
you offload those tasks to a GPUs, you can reduce training time to minutes
instead.

Training models in atom using a GPU is as easy as initializing the
instance with parameter `gpu=True`. Check which [prerequisites](#prerequisites)
your machine needs for it to work, and which [algorithms](#algorithms)
and [models](#models) are supported.

!!! warning
    Models trained on GPU do not support sparse dataframes!

!!! warning
    ATOM does not support multi-gpu training. If there is more than
    one GPU on the machine, the first one is used by default. Use
    `CUDA_VISIBLE_DEVICES` or model-specific parameters to use any of
    the other GPUs.

!!! warning
    Models can use different hyperparameters when trained with GPU than with CPU.


<a name="prerequisites"></a>
**Prerequisites**

* Operating System:
    - Ubuntu 18.04/20.04 or CentOS 7/8 with gcc/++ 9.0+
    - Windows 8.1+ with WSL2 (see [here](https://developer.nvidia.com/blog/run-rapids-on-microsoft-windows-10-using-wsl-2-the-windows-subsystem-for-linux/) a tutorial)
* GPU: NVIDIA Pascal™ or better with [compute capability](https://developer.nvidia.com/cuda-gpus) 6.0+
* CUDA & NVIDIA Drivers: One of versions 11.0, 11.2, 11.4 or 11.5

<a name="classes"></a>
**Classes**

* [ATOMClassifier](../../API/ATOM/atomclassifier) and [ATOMRegressor](../../API/ATOM/atomregressor) (for data splitting)
* [Scaler](../../API/data_cleaning/scaler)
* [Cleaner](../../API/data_cleaning/cleaner) (only to encode the target column)
* [Imputer](../../API/data_cleaning/imputer) (not for strat_num="knn")
* [Discretizer](../../API/data_cleaning/discretizer)
* [Vectorizer](../../API/nlp/vectorizer)
* [FeatureSelector](../../API/feature_engineering/feature_selector) (only for strategy="pca")

<a name="models"></a>
**Models**

* [Gaussian Naive Bayes](../../API/models/gnb)
* [Multinomial Naive Bayes](../../API/models/mnb)
* [Bernoulli Naive Bayes](../../API/models/bnb)
* [Categorical Naive Bayes](../../API/models/catnb)
* [Ordinary Least Squares](../../API/models/ols)
* [Ridge](../../API/models/ridge)
* [Lasso](../../API/models/lasso)
* [ElasticNet](../../API/models/en)
* [Lars](../../API/models/lars)
* [Logistic Regression](../../API/models/lr)
* [K-Nearest Neighbors](../../API/models/knn)
* [XGBoost](../../API/models/xgb)
* [LightGBM](../../API/models/lgb) (requires [extra installations](https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html))
* [CatBoost](../../API/models/catb)
