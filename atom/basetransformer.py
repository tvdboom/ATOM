"""Automated Tool for Optimized Modeling (ATOM).

Author: Mavs
Description: Module containing the BaseTransformer class.

"""

from __future__ import annotations

import os
import random
import re
import tempfile
import warnings
from collections.abc import Hashable
from copy import deepcopy
from datetime import datetime as dt
from importlib import import_module
from importlib.util import find_spec
from logging import DEBUG, FileHandler, Formatter, Logger, getLogger
from multiprocessing import cpu_count
from pathlib import Path
from typing import Literal, TypeVar, overload

import dagshub
import mlflow
import numpy as np
import ray
import requests
from beartype import beartype
from dagshub.auth.token_auth import HTTPBearerAuth
from joblib.memory import Memory
from pandas._typing import Axes
from ray.util.joblib import register_ray
from sklearn.utils.validation import check_memory

from atom.utils.types import (
    Backend, Bool, DataFrame, Engine, EngineDataOptions,
    EngineEstimatorOptions, EngineTuple, Estimator, Int, IntLargerEqualZero,
    Pandas, Sequence, Severity, Verbose, Warnings, XSelector, YSelector,
    bool_t, dataframe_t, int_t, sequence_t,
)
from atom.utils.utils import (
    crash, flt, lst, make_sklearn, n_cols, to_df, to_tabular,
)


T_Estimator = TypeVar("T_Estimator", bound=Estimator)


class BaseTransformer:
    """Base class for transformers in the package.

    Note that this includes atom and runners. Contains shared
    properties, data preparation methods, and utility methods.

    Parameters
    ----------
    **kwargs
        Standard keyword arguments for the classes. Can include:

        - n_jobs: Number of cores to use for parallel processing.
        - device: Device on which to run the estimators.
        - engine: Execution engine to use for data and estimators.
        - backend: Parallelization backend.
        - verbose: Verbosity level of the output.
        - warnings: Whether to show or suppress encountered warnings.
        - logger: Name of the log file, Logger object or None.
        - experiment: Name of the mlflow experiment used for tracking.
        - random_state: Seed used by the random number generator.

    """

    attrs = (
        "n_jobs",
        "device",
        "engine",
        "backend",
        "memory",
        "verbose",
        "warnings",
        "logger",
        "experiment",
        "random_state",
    )

    def __init__(self, **kwargs):
        """Update the properties with the provided kwargs."""
        for key, value in kwargs.items():
            setattr(self, key, value)

    # Properties =================================================== >>

    @property
    def n_jobs(self) -> int:
        """Number of cores to use for parallel processing."""
        return self._n_jobs

    @n_jobs.setter
    @beartype
    def n_jobs(self, value: Int):
        # Check the number of cores for multiprocessing
        if value > (n_cores := cpu_count()):
            self._n_jobs = n_cores
        else:
            self._n_jobs = int(n_cores + 1 + value if value < 0 else value)

    @property
    def device(self) -> str:
        """Device on which to run the estimators."""
        return self._device

    @device.setter
    @beartype
    def device(self, value: str):
        self._device = value
        if "gpu" in value.lower():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self._device_id)

    @property
    def engine(self) -> EngineTuple:
        """Execution engine for data and estimators."""
        return self._engine

    @engine.setter
    @beartype
    def engine(self, value: Engine):
        if value is None:
            engine = EngineTuple()
        elif value in EngineDataOptions.__args__:
            engine = EngineTuple(data=value)  # type: ignore[arg-type]
        elif value in EngineEstimatorOptions.__args__:
            engine = EngineTuple(estimator=value)  # type: ignore[arg-type]
        elif isinstance(value, dict):
            engine = EngineTuple(
                data=value.get("data", EngineTuple().data),
                estimator=value.get("estimator", EngineTuple().estimator),
            )
        else:
            engine = value  # type: ignore[assignment]

        if engine.data == "modin" and not ray.is_initialized():
            ray.init(
                runtime_env={"env_vars": {"__MODIN_AUTOIMPORT_Pandas__": "1"}},
                log_to_driver=False,
            )

        # Update env variable to use for PandasModin in utils.py
        os.environ["ATOM_DATA_ENGINE"] = engine.data

        if engine.estimator == "sklearnex":
            if not find_spec("sklearnex"):
                raise ModuleNotFoundError(
                    "Failed to import scikit-learn-intelex. The library is "
                    "not installed. Note that the library only supports CPUs "
                    "with a x86 architecture."
                )
            else:
                import sklearnex

                sklearnex.set_config(self.device.lower() if self._gpu else "auto")
        elif engine.estimator == "cuml":
            if not find_spec("cuml"):
                raise ModuleNotFoundError(
                    "Failed to import cuml. Package is not installed. "
                    "Refer to: https://rapids.ai/start.html#install."
                )
            else:
                from cuml.common.device_selection import set_global_device_type

                set_global_device_type("gpu" if self._gpu else "cpu")

                # See https://github.com/rapidsai/cuml/issues/5564
                from cuml.internals.memory_utils import set_global_output_type

                set_global_output_type("numpy")

        self._engine = engine

    @property
    def backend(self) -> Backend:
        """Parallelization backend."""
        return self._backend

    @backend.setter
    @beartype
    def backend(self, value: Backend):
        if value == "ray":
            register_ray()  # Register ray as joblib backend
            if not ray.is_initialized():
                ray.init(log_to_driver=False)

        self._backend = value

    @property
    def memory(self) -> Memory:
        """Get the internal memory object."""
        return self._memory

    @memory.setter
    @beartype
    def memory(self, value: Bool | str | Path | Memory):
        """Create a new internal memory object."""
        if value is False:
            value = None
        elif value is True:
            value = tempfile.gettempdir()
        elif isinstance(value, Path):
            value = str(value)

        self._memory = check_memory(value)

    @property
    def verbose(self) -> Verbose:
        """Verbosity level of the output."""
        return self._verbose

    @verbose.setter
    @beartype
    def verbose(self, value: Verbose):
        self._verbose = value

    @property
    def warnings(self) -> Warnings:
        """Whether to show or suppress encountered warnings."""
        return self._warnings

    @warnings.setter
    @beartype
    def warnings(self, value: Bool | Warnings):
        if isinstance(value, bool_t):
            self._warnings: Warnings = "once" if value else "ignore"
        else:
            self._warnings = value

        warnings.filterwarnings(self._warnings)  # Change the filter in this process
        warnings.filterwarnings("ignore", category=FutureWarning, module=".*pandas.*")
        warnings.filterwarnings("ignore", category=FutureWarning, module=".*imblearn.*")
        warnings.filterwarnings("ignore", category=UserWarning, module=".*sktime.*")
        warnings.filterwarnings("ignore", category=ResourceWarning, module=".*ray.*")
        warnings.filterwarnings("ignore", category=UserWarning, module=".*modin.*")
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=".*shap.*")
        os.environ["PYTHONWARNINGS"] = self._warnings  # Affects subprocesses (joblib)

    @property
    def logger(self) -> Logger | None:
        """Logger for this instance."""
        return self._logger

    @logger.setter
    @beartype
    def logger(self, value: str | Path | Logger | None):
        external_loggers = ["dagshub", "mlflow", "optuna", "ray", "modin", "featuretools"]

        # Clear existing handlers for external loggers
        for name in external_loggers:
            for handler in (log := getLogger(name)).handlers:
                handler.close()
            log.handlers.clear()

        if not value:
            logger = None
        else:
            if isinstance(value, Logger):
                logger = value
            else:
                logger = getLogger(self.__class__.__name__)
                logger.setLevel(DEBUG)

                # Clear existing handlers for current logger
                for handler in logger.handlers:
                    handler.close()
                logger.handlers.clear()

                # Prepare the FileHandler
                if (path := Path(value)).suffix != ".log":
                    path = path.with_suffix(".log")
                if path.name == "auto.log":
                    now = dt.now().strftime("%d%b%y_%Hh%Mm%Ss")
                    path = path.with_name(f"{self.__class__.__name__}_{now}.log")

                fh = FileHandler(path)
                fh.setFormatter(Formatter("%(asctime)s - %(levelname)s: %(message)s"))

                # Redirect loggers to file handler
                for name in [logger.name, *external_loggers]:
                    getLogger(name).addHandler(fh)

        self._logger = logger

    @property
    def experiment(self) -> str | None:
        """Name of the mlflow experiment used for tracking."""
        return self._experiment

    @experiment.setter
    @beartype
    def experiment(self, value: str | None):
        self._experiment = value
        if value:
            if value.lower().startswith("dagshub:"):
                value = value[8:]  # Drop dagshub:

                token = dagshub.auth.get_token()
                os.environ["MLFLOW_TRACKING_USERNAME"] = token
                os.environ["MLFLOW_TRACKING_PASSWORD"] = token

                # Fetch username from dagshub api
                username = requests.get(
                    url="https://dagshub.com/api/v1/user",
                    auth=HTTPBearerAuth(token),
                    timeout=5,
                ).json()["username"]

                if f"{username}/{value}" not in os.getenv("MLFLOW_TRACKING_URI", ""):
                    dagshub.init(repo_name=value, repo_owner=username)
                    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

            elif "dagshub" in mlflow.get_tracking_uri():
                mlflow.set_tracking_uri("")  # Reset URI to ./mlruns

            mlflow.sklearn.autolog(disable=True)
            mlflow.set_experiment(value)

    @property
    def random_state(self) -> int | None:
        """Seed used by the random number generator."""
        return self._random_state

    @random_state.setter
    @beartype
    def random_state(self, value: IntLargerEqualZero | None):
        if value is not None:
            value = int(value)

        random.seed(value)
        np.random.seed(value)  # noqa: NPY002
        self._random_state = value

    @property
    def _gpu(self) -> bool:
        """Return whether the instance uses a GPU implementation."""
        return "gpu" in self.device.lower()

    @property
    def _device_id(self) -> int:
        """Which GPU device to use."""
        if len(value := self.device.split(":")) == 1:
            return 0  # Default value
        else:
            try:
                return int(value[-1])
            except (TypeError, ValueError):
                raise ValueError(
                    f"Invalid value for the device parameter. GPU device {value[-1]} "
                    "isn't understood. Use a single integer to denote a specific "
                    "device. Note that ATOM doesn't support multi-GPU training."
                ) from None

    # Methods ====================================================== >>

    def _inherit(self, obj: T_Estimator, fixed: tuple[str, ...] = ()) -> T_Estimator:
        """Inherit parameters from parent.

        Utility method to set the sp (seasonal period), n_jobs and
        random_state parameters of an estimator (if available) equal
        to that of this instance. If `obj` is a meta-estimator, it
        also adjusts the parameters of the base estimator.

        Parameters
        ----------
        obj: Estimator
            Instance for which to change the parameters.

        fixed: tuple of str, default=()
            Fixed parameters that should not be overriden.

        Returns
        -------
        Estimator
            Same object with changed parameters.

        """
        for p in obj.get_params():
            if p in fixed:
                continue
            elif match := re.search("^(n_jobs|random_state)$|__\1$", p):
                obj.set_params(**{p: getattr(self, match.group())})
            elif re.search(r"^sp$|__sp$", p) and hasattr(self, "_config") and self._config.sp:
                if self.multiple_seasonality:
                    obj.set_params(**{p: self._config.sp.sp})
                else:
                    obj.set_params(**{p: lst(self._config.sp.sp)[0]})

        return obj

    def _get_est_class(self, name: str, module: str) -> type[Estimator]:
        """Import a class from a module.

        When the import fails, for example, if atom uses sklearnex and
        that's passed to a transformer, use sklearn's (default engine).

        Parameters
        ----------
        name: str
            Name of the class to get.

        module: str
            Module from which to get the class.

        Returns
        -------
        Estimator
            Class of the estimator.

        """
        try:
            mod = import_module(f"{self.engine.estimator}.{module}")
        except (ModuleNotFoundError, AttributeError):
            mod = import_module(f"sklearn.{module}")

        return make_sklearn(getattr(mod, name))

    @crash
    def _log(self, msg: str, level: Int = 0, severity: Severity = "info"):
        """Print message and save to log file.

        Parameters
        ----------
        msg: str
            Message to save to the logger and print to stdout.

        level: int, default=0
            Minimum verbosity level to print the message.

        severity: str, default="info"
            Severity level of the message. Choose from: debug, info,
            warning, error, critical.

        """
        if severity in ("error", "critical"):
            raise UserWarning(msg)
        elif severity == "warning":
            warnings.warn(msg, category=UserWarning, stacklevel=2)
        elif severity == "info" and self.verbose >= level:
            print(msg)  # noqa: T201

        if getattr(self, "logger", None):
            for text in str(msg).split("\n"):
                getattr(self.logger, severity)(str(text))
