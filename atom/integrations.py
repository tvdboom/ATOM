"""Automated Tool for Optimized Modeling (ATOM).

Author: Mavs
Description: Module containing the experiment integration classes.

"""

from __future__ import annotations

import os
from abc import ABCMeta, abstractmethod

import mlflow

from atom.utils.utils import check_dependency


class Integrator(metaclass=ABCMeta):
    """Base class for experiment integrations."""

    @abstractmethod
    def __init__(self, project_name: str): ...


class DAGsHubIntegrator(Integrator):
    """DAGsHub integration class.

    Read more in the [user guide][dagshub-integration].

    Parameters
    ----------
    project_name: str
        The name of the project on DAGsHub. If the project does not exist,
        a new one is created.

    Notes
    -----
    https://dagshub.com/

    """

    def __init__(self, project_name: str):
        check_dependency("dagshub")
        check_dependency("requests")
        import dagshub
        import requests
        from dagshub.auth.token_auth import HTTPBearerAuth

        token = dagshub.auth.get_token()

        # Fetch username from dagshub api
        username = requests.get(
            url="https://dagshub.com/api/v1/user",
            auth=HTTPBearerAuth(token),
            timeout=5,
        ).json()["username"]

        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token

        if f"{username}/{project_name}" not in mlflow.get_tracking_uri():
            dagshub.init(repo_name=project_name, repo_owner=username, mlflow=True)
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))


INTEGRATIONS = {"dagshub": DAGsHubIntegrator}
