# -*- coding: utf-8 -*-

'''
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom

'''

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
      name='atom-ml',
      version='v3.0.1',
      license='MIT',
      description='A Python AutoML tool for fast exploration and experimentation of supervised machine learning pipelines.',
      download_url='https://github.com/tvdboom/ATOM/archive/v3.0.1.tar.gz',
      url='http://github.com/tvdboom/ATOM',
      author='tvdboom',
      author_email='m.524687@gmail.com',
      keywords=['AutoML', 'Machine Learning'],
      long_description=long_description,
      long_description_content_type="text/markdown",
      packages=['atom'],
      classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
      ],
      install_requires=[
          'numpy>=1.17.2',
          'scipy>=1.4.1',
          'pandas>=1.0.1',
          'scikit-learn>=0.22',
          'tabulate>=0.8.6',
          'tqdm>=4.35.0',
          'typeguard>=2.7.1',
          'gpyopt>=1.2.5',
          'matplotlib>=3.1.0',
          'seaborn>=0.10.0',
      ],
      extras_require={
          'pandas-profiling': ['pandas-profiling>=2.3.0'],
          'imbalanced-learn': ['imbalanced-learn>=0.5.0'],
          'gplearn': ['gplearn>=0.4.1'],
          'xgboost': ['xgboost>=0.90'],
          'lightgbm': ['lightgbm>=2.3.0'],
          'catboost': ['catboost>=0.19.1'],
      },
      python_requires='>=3.6')
