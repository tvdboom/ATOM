# Accelerating pipelines
------------------------

For very large datasets, ATOM offers various ways to accelerate its
pipeline:

 - [Run estimators on GPU][gpu-acceleration]
 - [Use a faster data engine][data-acceleration]
 - [Use a faster estimator engine][estimator-acceleration]
 - [Run processes in parallel][parallel-execution]

!!! warning
    Performance improvements are usually noticeable for datasets larger 
    than ~5M rows. For smaller datasets, using other values than the
    default can even harm performance!


## GPU acceleration

Graphics Processing Units (GPUs) can significantly accelerate
calculations for preprocessing steps or training machine learning
models. Training models involves compute-intensive matrix
multiplications and other operations that can take advantage of a
GPU's massively parallel architecture. Training on large datasets can
take hours to run on a single processor. However, if you offload those
tasks to a GPU, you can reduce training time to minutes instead.

Running transformers and models in atom using a GPU is as easy as
initializing the instance with parameter `#!python device="gpu"`. The
[`device`][atomclassifier-device] parameter accepts any string that
follows the [SYCL_DEVICE_FILTER][] filter selector. Examples are:

* device="cpu" (use CPU)
* device="gpu" (use default GPU)
* device="gpu:0" (use first GPU)
* device="gpu:1" (use second GPU)

Combine GPU acceleration with the [cuml][] and [sklearnex][] estimator engines.
The [XGBoost][], [LightGBM][] and [CatBoost][] models come with their own GPU
engine. Setting `#!python device="gpu"` is sufficient to accelerate them with GPU,
regardless of the engine parameter.

!!! warning
    ATOM does not support multi-GPU training. If there is more than one
    GPU on the machine and the `device` parameter does not specify which
    one to use, the first one is used by default.

!!! example
    [![SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/tvdboom/ATOM/blob/master/examples/accelerating_cuml.ipynb)<br><br>
    Train a model on a GPU yourself using SageMaker Studio Lab. Just click on
    the badge above and run the notebook! Make sure to choose the GPU compute
    type.


## Data acceleration

The data engine can be specified through the [`engine`][atomclassifier-engine]
parameter, which takes a dict with a key `data` that accepts three values:
[numpy][], [pyarrow][] and [modin][].


### numpy

ATOM uses [`pandas`](https://pandas.pydata.org/docs/index.html) as the
default library for data handling, which in turn, uses [`numpy`](https://numpy.org/)
for all data processing.


### pyarrow

[PyArrow](https://arrow.apache.org/docs/python/index.html) is a library
that provides a way to work with Apache Arrow memory structures. Apache
Arrow is a cross-language, platform-independent, in-memory data format
that provides an efficient and fast way to serialize and deserialize
data. Pandas offers [native integration](https://pandas.pydata.org/docs/user_guide/pyarrow.html)
with pyarrow, which atom uses when specifying the pyarrow data engine.

!!! warning
    - The pyarrow backend doesn't work for [sparse datasets][]. If the
      dataset has any sparse columns, an exception is raised.
    - The [LightGBM][] and [XGBoost][] models don't support pyarrow
      dtypes.


### modin

The [modin](https://modin.readthedocs.io/en/stable/) library is a multi-threading, drop-in replacement for
pandas, that uses [Ray](https://www.ray.io/) as backend.


## Estimator acceleration

The estimator engine can be specified through the [`engine`][atomclassifier-engine]
parameter, which takes a dict with a key `estimator` that accepts three
values: [sklearn][], [sklearnex][] and [cuml][]. Read [here][gpu-acceleration]
how to run the estimators on GPU instead of CPU.

!!! warning
    Estimators accelerated with sklearnex or cuML sometimes use slightly
    different hyperparameters than their sklearn counterparts.

### sklearn

This is the default option, which uses the standard estimators from
[sklearn](https://scikit-learn.org/stable/). Sklearn does not support
training on GPU.


### sklearnex

The [Intel® Extension for Scikit-learn](https://intel.github.io/scikit-learn-intelex/index.html) package (or sklearnex, for
brevity) accelerates sklearn models and transformers, keeping full
conformance with sklearn's API. Sklearnex is a free software AI
accelerator that offers a way to make sklearn code 10–100 times faster.
The software acceleration is achieved through the use of vector
instructions, IA hardware-specific memory optimizations, threading, and
optimizations for all upcoming Intel platforms at launch time. See
[here][example-accelerating-pipelines] an example using the sklearnex
engine.

!!! warning
    sklearnex estimators don't support [sparse datasets][] nor
    [multioutput tasks][].

!!! tip
    Intel® processors provide better performance than other CPUs.

#### Prerequisites

* Operating System:
    - Linux (Ubuntu, Fedora, etc...)
    - Windows 8.1+
    - macOS (no GPU support)
* CPU:
    - Processor must have x86 architecture.
    - Processor must support at least one of SSE2, AVX, AVX2, AVX512 instruction sets.
    - ARM* architecture is not supported.
* GPU:
    - All Intel® integrated and discrete GPUs.
    - Intel® GPU drivers.
* Libraries:
    - [sklearnex](https://intel.github.io/scikit-learn-intelex/index.html)>=2023.2.1 (automatically installed with atom when the processor has x86 architecture)
    - [dpcpp_cpp_rt](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html)>=2023.2  (only for GPU acceleration)

#### Supported estimators

* [Pruner][] (only for strategy="dbscan")
* [FeatureSelector][] (only for strategy="pca" and dense datasets)

* [ElasticNet][] (only for CPU acceleration)
* [KNearestNeighbors][]
* [Lasso][] (only for CPU acceleration)
* [LogisticRegression][]
* [OrdinaryLeastSquares][]
* [RandomForest][]
* [Ridge][] (only for regression tasks and CPU acceleration)
* [SupportVectorMachine][] (GPU acceleration only supports classification tasks)


### cuML

[cuML](https://github.com/rapidsai/cuml) is the machine learning library
of the [RAPIDS](https://rapids.ai/) project. cuML enables you to run
traditional tabular ML tasks on GPUs without going into the details of
CUDA programming. For large datasets, these GPU-based implementations can
complete 10-50x faster than their CPU equivalents.

!!! warning
    * cuML estimators don't support [multioutput tasks][] nor the [pyarrow][]
      data engine.
    * Install cuML using `pip install --extra-index-url=https://pypi.nvidia.com
      cuml-cu11` or `pip install --extra-index-url=https://pypi.nvidia.com
      cuml-cu12` depending on your CUDA version. Read more about RAPIDS'
      installation [here](https://docs.rapids.ai/install).

!!! tip
    Only transformers and predictors are converted to the requested engine.
    To use a metric from cuML, insert it directly in the [`run`][atomclassifier-run]
    method:

    ```python
    from atom import ATOMClassifier
    from cuml.metrics import accuracy_score
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=100, random_state=1)
    
    atom = ATOMClassifier(X, y, engine={"estimator": "cuml"}, verbose=2)
    atom.run("LR", metric=accuracy_score)
    ```

#### Prerequisites

* Operating System:
    - Ubuntu 18.04/20.04 or CentOS 7/8 with gcc/++ 9.0+
    - Windows 10+ with WSL2 (see [here](https://developer.nvidia.com/blog/run-rapids-on-microsoft-windows-10-using-wsl-2-the-windows-subsystem-for-linux/) a tutorial)
* GPU:
    - NVIDIA Pascal™ or better with [compute capability](https://developer.nvidia.com/cuda-gpus) 6.0+
* Drivers:
    - CUDA & NVIDIA Drivers of versions 11.0, 11.2, 11.4 or 11.5
* Libraries:
    - [cuML](https://docs.rapids.ai/api/cuml/stable/)>=23.08

#### Supported estimators

* [Cleaner][]
* [Discretizer][]
* [Imputer][] (only for strat_num!="knn")
* [Normalizer][]
* [Pruner][] (only for strategy="dbscan" and "hdbscan")
* [Scaler][]
* [Vectorizer][]
* [FeatureSelector][] (only for strategy="pca")

* [BernoulliNB][]
* [CategoricalNB][]
* [ElasticNet][]
* [GaussianNB][]
* [KNearestNeighbors][]
* [Lasso][]
* [LinearSVM][]
* [LogisticRegression][]
* [MultinomialNB][]
* [OrdinaryLeastSquares][]
* [RandomForest][]
* [Ridge][] (only for regression tasks)
* [SupportVectorMachine][]


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
  See [here][example-ray-backend] an example use case.


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
