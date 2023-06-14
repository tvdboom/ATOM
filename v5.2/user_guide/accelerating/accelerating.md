# Accelerating pipelines
------------------------

## CPU acceleration

With the [Intel® Extension for Scikit-learn](https://intel.github.io/scikit-learn-intelex/index.html)
package (or sklearnex, for brevity) you can accelerate your sklearn
models and transformers, keeping full conformance with sklearn's API.
Sklearnex is a free software AI accelerator that offers you a way to
make sklearn code 10–100 times faster. The software acceleration is
achieved through the use of vector instructions, IA hardware-specific
memory optimizations, threading, and optimizations for all upcoming
Intel platforms at launch time.

Select `#!python engine="sklearnex"` in atom's constructor to make use
of this feature. See [here][example-accelerating-pipelines] an example.

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
    - [sklearnex](https://intel.github.io/scikit-learn-intelex/index.html)>=2021.6.3 (automatically installed with atom when the processor has x86 architecture)

!!! tip
    * Intel® processors provide better performance than other CPUs.

<br>

### Supported estimators

**Transformers**

* [Pruner][] (only for strategy="dbscan")
* [FeatureSelector][] (only for strategy="pca" and dense datasets)

**Models**

* [ElasticNet][]
* [KNearestNeighbors][]
* [Lasso][]
* [LogisticRegression][]
* [OrdinaryLeastSquares][]
* [RandomForest][]
* [Ridge][] (only for regression tasks)
* [SupportVectorMachine][]


<br><br>

## GPU acceleration

Graphics Processing Units (GPUs) can significantly accelerate
calculations for preprocessing steps or training machine learning
models. Training models involves compute-intensive matrix
multiplications and other operations that can take advantage of a
GPU's massively parallel architecture. Training on large datasets can
take hours to run on a single processor. However, if you offload those
tasks to a GPU, you can reduce training time to minutes instead.

Training transformers and models in atom using a GPU is as easy as
initializing the instance with parameter `#!python device="gpu"`. The
[`device`][atomclassifier-device] parameter accepts any string that
follows the [SYCL_DEVICE_FILTER][] filter selector. Examples are:

* device="cpu" (use CPU)
* device="gpu" (use default GPU)
* device="gpu:0" (use first GPU)
* device="gpu:1" (use second GPU)

Use the [`engine`][atomclassifier-engine] parameter to choose between the
cuML and sklearnex execution engines. The [XGBoost][], [LightGBM][] and
[CatBoost][] models come with their own GPU engine. Setting device="gpu"
is sufficient to accelerate them with GPU, regardless of the engine parameter.

!!! warning
    * GPU accelerated estimators almost never support [sparse datasets][].
      Refer to their respective documentation to check which ones do.
    * GPU accelerated estimators often use slightly different hyperparameters
      than their CPU counterparts.
    * ATOM does not support multi-GPU training. If there is more than one
      GPU on the machine and the `device` parameter does not specify which
      one to use, the first one is used by default.

!!! example
    [![SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/tvdboom/ATOM/blob/master/examples/accelerating_cuml.ipynb)<br><br>
    Train a model on a GPU yourself using SageMaker Studio Lab. Just click on
    the badge above and run the notebook! Make sure to choose the GPU compute
    type.


### Prerequisites

* Operating System:
    - Ubuntu 18.04/20.04 or CentOS 7/8 with gcc/++ 9.0+
    - Windows 10+ with WSL2 (see [here](https://developer.nvidia.com/blog/run-rapids-on-microsoft-windows-10-using-wsl-2-the-windows-subsystem-for-linux/) a tutorial)
* GPU: 
    - For sklearnex: All Intel® integrated and discrete GPUs.
    - For cuML: NVIDIA Pascal™ or better with [compute capability](https://developer.nvidia.com/cuda-gpus) 6.0+
* Drivers:
    - For sklearnex: Intel® GPU drivers.
    - For cuML: CUDA & NVIDIA Drivers of versions 11.0, 11.2, 11.4 or 11.5
* Libraries:
    - [sklearnex](https://intel.github.io/scikit-learn-intelex/index.html)>=2021.6.3 (automatically installed with atom when the processor has x86 architecture)
    - [cuML](https://docs.rapids.ai/api/cuml/stable/)>=22.10

### Supported estimators

**Transformers**

* [Cleaner][] (only for cuML with encode_target=True)
* [Discretizer][] (only for cuML with strategy!="custom")
* [Imputer][] (only for cuML with strat_num!="knn")
* [Normalizer][] (only for cuML)
* [Pruner][] (only for strategy="dbscan")
* [Scaler][] (only for cuML)
* [Vectorizer][] (only for cuML)
* [FeatureSelector][] (only for strategy="pca" and dense datasets)


**Models**

* [BernoulliNB][] (only for cuML)
* [CatBoost][]
* [CategoricalNB][] (only for cuML)
* [ElasticNet][] (only for cuML)
* [GaussianNB][] (only for cuML)
* [KNearestNeighbors][]
* [Lasso][] (only for cuML)
* [LightGBM][] (requires [extra installations](https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html))
* [LinearSVM][] (only for cuML)
* [LogisticRegression][]
* [MultinomialNB][] (only for cuML)
* [OrdinaryLeastSquares][]
* [RandomForest][]
* [Ridge][] (only for regression tasks)
* [SupportVectorMachine][]
* [XGBoost][]


## Parallel execution

Another way to accelerate your pipelines is executing processes in parallel.
Use the [`backend`][atomclassifier-backend] parameter to select one of several
parallelization backends.

* **loky:** Used by default, can induce some communication and memory overhead
  when exchanging input and output data with the worker Python processes. On
  some rare systems (such as Pyiodide), the loky backend may not be available.
* **multiprocessing:** Previous process-based backend based on `multiprocessing.Pool`.
  Less robust than loky.
* **threading:** Very low-overhead backend but it suffers from the Python Global
  Interpreter Lock if the called function relies a lot on Python objects. It's 
  mostly useful when the execution bottleneck is a compiled extension that
  explicitly releases the GIL (for instance a Cython loop wrapped in a "with nogil"
  block or an expensive call to a library such as numpy).
* **ray:** [Ray](https://www.ray.io/) is an open-source unified compute framework
  that makes it easy to scale AI and Python workloads. Read more about Ray [here](https://docs.ray.io/en/latest/ray-core/walkthrough.html).
  Selecting the ray backend also parallelizes the data using [modin][], a
  multi-threading, drop-in replacement for pandas, that uses Ray as backend.
  See [here][example-ray-backend] an example use case.

!!! warning
    Using [modin][] can be considerably less performant than pandas for small
    datasets (<3M rows).

The parallelization backend is applied in the following cases:

* In every individual estimator that uses parallelization internally.
* To calculate cross-validated results during [hyperparameter tuning][].
* To train multiple models in parallel (when the trainer's `parallel` parameter is True).
* To calculate partial dependencies in [plot_partial_dependence][].

!!! note
    The [`njobs`][atomclassifier-n_jobs] parameter sets the number of cores
    for the individual models as well as for parallel training. You won't
    gain much training two models in parallel with 2 cores, when the models
    also parallelize computations internally. Instead, use parallel training
    for models that can't parallelize their training (their constructor doesn't
    have the `n_jobs` parameter).
