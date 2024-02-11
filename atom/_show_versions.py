"""Automated Tool for Optimized Modeling (ATOM).

Author: Mavs
Description: Utility methods to print system info for debugging.
             Adapted from sklearn.show_versions.

"""

__all__ = ["show_versions"]


import importlib
import platform
import sys


# Dependencies to print versions of
DEFAULT_DEPS = [
    "pip",
    "atom",
    "beartype",
    "category_encoders",
    "dagshub",
    "dill",
    "gplearn",
    "imblearn",
    "ipywidgets",
    "featuretools",
    "joblib",
    "matplotlib",
    "mlflow",
    "modin",
    "nltk",
    "numpy",
    "optuna",
    "pandas",
    "plotly",
    "polars",
    "pyarrow",
    "ray",
    "requests",
    "sklearn",
    "sklearnex",  # Has no __version__ attribute
    "scipy",
    "shap",
    "sktime",
    "statsmodels",
    "zoofs",  # Has no __version__ attribute
]


def _get_sys_info():
    """Get the system and Python version.

    Returns
    -------
    dict
        Information.

    """
    return {
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "machine": platform.platform(),
    }


def _get_deps_info(deps: list[str]) -> dict[str, str | None]:
    """Overview of the installed version of main dependencies.

    Parameters
    ----------
    deps: list of str
        Dependencies to get the version from.

    Returns
    -------
    dict
        Version information on libraries in `deps`, where the keys are
        import names and the values are PEP 440 version strings of the
        import as present in the current python environment.

    """
    deps_info = {}
    for modname in deps:
        try:
            if modname in sys.modules:
                mod = sys.modules[modname]
            else:
                mod = importlib.import_module(modname)
            deps_info[modname] = mod.__version__
        except (ImportError, AttributeError):  # noqa: PERF203
            deps_info[modname] = None

    return deps_info


def show_versions():
    """Print system and package information.

    The following information is displayed:

    - Python version of environment.
    - Python executable location.
    - OS version.
    - Import name and version number for selected python dependencies.

    """
    sys_info = _get_sys_info()
    deps_info = _get_deps_info(deps=DEFAULT_DEPS)

    print("\nSystem:")  # noqa: T201
    for k, stat in sys_info.items():
        print(f"{k:>10}: {stat}")  # noqa: T201

    print("\nPython dependencies:")  # noqa: T201
    for k, stat in deps_info.items():
        print(f"{k:>13}: {stat}")  # noqa: T201
