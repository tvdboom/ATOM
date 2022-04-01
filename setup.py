# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Package's setup code.

"""

import re
from setuptools import setup


with open("atom/__init__.py", encoding="utf8") as f:
    version = re.search(r"^__version__ = \"([\d.]*)\"", f.read(), re.M).group(1)

with open("README.md", encoding="utf8") as f:
    long_description = f.read()

with open("requirements.txt", encoding="utf8") as f:
    requirements = f.read().splitlines()

with open("requirements-optional.txt", encoding="utf8") as f:
    optional_requirements = f.read().splitlines()

with open("requirements-dev.txt", encoding="utf8") as f:
    dev_requirements = f.read().splitlines()

setup(
    name="atom-ml",
    version=version,
    license="MIT",
    description="A Python package for fast exploration of machine learning pipelines",
    download_url=f"https://github.com/tvdboom/ATOM/archive/v{version}.tar.gz",
    url="https://github.com/tvdboom/ATOM",
    author="tvdboom",
    author_email="m.524687@gmail.com",
    keywords=["Python package", "Machine Learning", "Modelling", "Data Pipeline"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["atom"],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements,
    extras_require={
        "models": optional_requirements,
        "dev": optional_requirements + dev_requirements,
    },
    tests_require=dev_requirements,
    python_requires=">=3.7"
)
