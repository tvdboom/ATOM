# Contributing
--------------

Are you interested in contributing to ATOM? Do you want to report a bug?
Do you have a question? Before you do, please read the following guidelines.

<br>


## Submission context

### Question or problem?

For quick questions, there's no need to open an issue. Check first if the
question isn't already answered in the [FAQ][] section. If not, reach us
through the [discussions](https://github.com/tvdboom/ATOM/discussions) page or on the [slack](https://join.slack.com/t/atom-alm7229/shared_invite/zt-upd8uc0z-LL63MzBWxFf5tVWOGCBY5g) channel.


### Report a bug?

If you found a bug in the source code, you can help by submitting an issue
to the [issue tracker](https://github.com/tvdboom/ATOM/issues) in the GitHub
repository. Even better, you can submit a Pull Request with a fix. However,
before doing so, please read the [submission guidelines](#submission-guidelines).


### Missing a feature?

You can request a new feature by submitting an [issue](https://github.com/tvdboom/ATOM/issues)
to the GitHub Repository. If you would like to implement a new feature,
please submit an issue with a proposal for your work first. Please consider
what kind of change it is:

* For a **major feature**, first open an issue and outline your proposal so
  that it can be discussed. This will also allow us to better coordinate our
  efforts, prevent duplication of work, and help you to craft the change so
  that it is successfully accepted into the project.

* **Small features and bugs** can be crafted and directly submitted as a Pull
  Request. However, there is no guarantee that your feature will make it into
  `master`, as it's always a matter of opinion whether if benefits the
  overall functionality of the project.


### Project layout

The latest stable release of ATOM is on the `master` branch, whereas the
latest version of ATOM in development is on the `development` branch. Make
sure you are looking at and working on the correct branch if you're looking
to contribute code.

In terms of directory structure:

* All of ATOM's code sources are in the `atom` directory.
* The documentation sources are in the `docs_sources` directory.
* Images in the documentation are in the `docs_sources/img` directory.
* Tutorial notebooks are in the `examples` directory. If you want to
  include the example to the documentation as well, add the `.ipynb` file
  to `docs_sources/examples` and update the `mkdocs.yml` file accordingly.
* Unit tests are in the `tests` directory. Make sure to add the tests to the
  file corresponding to the module in the `atom` directory with the code that
  is being tested.

Make sure to familiarize yourself with the project layout before making any
major contributions, and especially make sure to send all code changes to the
`development` branch.

<br><br>


## Submission guidelines

### Submitting an issue

Before you submit an issue, please search the [issue tracker](https://github.com/tvdboom/ATOM/issues),
maybe an issue for your problem already exists and the discussion
might inform you of workarounds readily available.

We want to fix all the issues as soon as possible, but before fixing a
bug we need to reproduce and confirm it. In order to reproduce bugs we
will systematically ask you to provide a minimal reproduction scenario
using the custom issue template.


### Submitting a pull request

Before you submit a pull request, please work through this checklist to
make sure that you have done everything necessary so we can efficiently
review and accept your changes.

* Update the documentation so all of your changes are reflected there.
* Adhere to [PEP 8](https://peps.python.org/pep-0008/) standards.
* Use a maximum of 90 characters per line. Try to keep docstrings below
  74 characters.
* Update the project unit tests to test your code changes as thoroughly
  as possible.
* Make sure that your code is properly commented with docstrings and
  comments explaining your rationale behind non-obvious coding practices.
* Run [isort](https://pycqa.github.io/isort/): `isort atom tests`.
* Run [flake8](https://github.com/pycqa/flake8): `flake8 --show-source --statistics atom tests`.
* Run [mypy](https://www.mypy-lang.org/): `mypy atom tests`.

If your contribution requires a new library dependency:

* Double-check that the new dependency is easy to install via pip and Anaconda.
* The library should support Python 3.9 and higher.
* Make sure the code works with the latest version of the library.
* Update the dependencies in the documentation.
* Add the library with the minimum required version to `pyproject.toml`.

After submitting your pull request, GitHub will automatically run the tests
on your changes and make sure that the updated code builds successfully.
The checks run on Python 3.9, 3.10 and 3.11, on Ubuntu and Windows. We also
use services that automatically check code style and test coverage.
