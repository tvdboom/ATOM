# Accelerating pipelines
------------------------

## CPU acceleration

ATOM uses [sklearnex](https://intel.github.io/scikit-learn-intelex/index.html)
to accelerate sklearn applications and still have full conformance
with its API. This tool can bring over 10-100X acceleration across a
variety of transformers and models.

### Prerequisites

* Operating System:
    - Linux (Ubuntu, Fedora, etc...)
    - Windows 8.1+
    - macOS
* CPU:
    - Processor must have x86 architecture.
    - Processor must support at least one of SSE2, AVX, AVX2, AVX512 instruction sets.
    - ARM* architecture is not supported.
* Libraries:
    - [sklearnex](https://intel.github.io/scikit-learn-intelex/index.html)>=2021.6.3 (automatically installed with atom)

!!! note
    Intel® processors provide better performance than other CPUs.

<br>

### Supported estimators

**Transformers**

* [Pruner][] (only for strategy="dbscan")
* [FeatureSelector][] (only for strategy="pca" and dense datasets)

**Models**

* [ElasticNet][]
* [K-Nearest Neighbors][]
* [Kernel SVM][]
* [Lasso][]
* [Logistic Regression][]
* [Ordinary Least Squares][]
* [Random Forest][]
* [Ridge][] (only for regression tasks)


<br><br>

## GPU acceleration

Graphics Processing Units (GPUs) can significantly accelerate
calculations for preprocessing step or training machine learning
models. Training models involves compute-intensive matrix
multiplications and other operations that can take advantage of a
GPU's massively parallel architecture. Training on large datasets can
take hours to run on a single processor. However, if you offload those
tasks to a GPU, you can reduce training time to minutes instead.

Training transformers and models in atom using a GPU is as easy as
initializing the instance with parameter `device="gpu"`. The [`device`]
[atomclassifier-device] parameter accepts any string that follows the
[SYCL_DEVICE_FILTER][] filter selector. Examples are:

* device="cpu" (use CPU)
* device="gpu" (use default GPU)
* device="gpu:1" (use second GPU)

Use the [`engine`][atomclassifier-engine] parameter to choose between the
[cuML][] and [sklearnex][] execution engines. The [XGB][xgboost],
[LGB][lightgbm] and [CatB][catboost] models come with their own GPU engine.
Setting device="gpu" is sufficient to accelerate them with GPU, regardless
of the engine parameter.

!!! warning
    * GPU accelerated estimators almost never support [sparse datasets][].
      Refer to their respective documentation to check which ones do.
    * GPU accelerated estimators use slightly different hyperparameters
      than their CPU counterparts.
    * ATOM does not support multi-GPU training. If there is more than one
      GPU on the machine and the `device` parameter does not specify which
      one to use, the first one is used by default.

!!! example
    [![Colab](https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/drive/1b-piLK6_O99EDpLNmeKIsSRkQlk0WoTl#offline=true&sandboxMode=true)<br><br>
    Train a model on a GPU yourself using Google Colab. Just click on
    the badge above and follow the notebook. Note two things:

    * Make sure you've been allocated a Tesla T4, P4, or P100. If this is not
    the case (check it using `!nvidia-smi`), reset the runtime (Runtime -> 
    Factory reset runtime) until you get one.
    * Setting up the environment and installing the necessary libraries may
    take quite some time (usually up to 15min).


### Prerequisites

* Operating System:
    - Ubuntu 18.04/20.04 or CentOS 7/8 with gcc/++ 9.0+
    - Windows 10+ with WSL2 (see [here](https://developer.nvidia.com/blog/run-rapids-on-microsoft-windows-10-using-wsl-2-the-windows-subsystem-for-linux/) a tutorial)
* GPU: 
    - For sklearnex: All Intel® integrated and discrete GPUs.
    - For cuML: NVIDIA Pascal™ or better with [compute capability](https://developer.nvidia.com/cuda-gpus) 6.0+
* Drivers:
    - For cuML: CUDA & NVIDIA Drivers of versions 11.0, 11.2, 11.4 or 11.5
    - For sklearnex: Intel® GPU drivers.
* Libraries:
    - [sklearnex](https://intel.github.io/scikit-learn-intelex/index.html)>=2021.6.3 (automatically installed with atom)
    - [cuML](https://docs.rapids.ai/api/cuml/stable/)>=22.08

### Supported estimators

**Transformers**

* [Cleaner][] (only for cuML with encode_target=True)
* [Discretizer][] (only for cuML with strategy!="custom")
* [Imputer][] (only for cuML with strat_num="knn")
* [Pruner][] (only for strategy="dbscan")
* [Scaler][] (only for cuML)
* [Vectorizer][] (only for cuML)
* [FeatureSelector][] (only for strategy="pca" and dense datasets)


**Models**

* [Bernoulli Naive Bayes][] (only for cuML)
* [CatBoost][]
* [Categorical Naive Bayes][] (only for cuML)
* [ElasticNet][] (only for cuML)
* [Gaussian Naive Bayes][] (only for cuML)
* [K-Nearest Neighbors][]
* [Kernel SVM][]
* [Lasso][] (only for cuML)
* [LightGBM][] (requires [extra installations](https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html))
* [Linear SVM][] (only for cuML)
* [Logistic Regression][]
* [Multinomial Naive Bayes][] (only for cuML)
* [Ordinary Least Squares][]
* [Random Forest][]
* [Ridge][] (only for regression tasks)
* [XGBoost][]
