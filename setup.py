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
      version='v1.1.4',
      license='MIT',
      description='AutoML package for model comparison tasks',
      #download_url='https://github.com/tvdboom/ATOM/archive/v1.1.3.tar.gz',
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
          'pandas>=0.25.1',
          'scikit-learn>=0.21.3',
          'tqdm>=4.35.0',
          'GpyOpt>=1.2.5',
          'matplotlib>=3.1.1',
          'seaborn>=0.9.0',
          'xgboost>=0.90'
      ],
      python_requires='>=3.6')
