# Contributing
--------------

## Project layout

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

## Submitting a pull request

Before you submit a pull request, please work through this checklist to
make sure that you have done everything necessary so we can efficiently
review and accept your changes.

If your contribution changes ATOM in any way:

* Update the documentation so all of your changes are reflected there.
* Update the README if anything there has changed.

If your contribution involves any code changes:

* Adhere to [PEP 8](https://www.python.org/dev/peps/pep-0008/) standards.
* Use a maximum of 88 characters per line. Try to keep comments and
  docstrings below 72 characters.
* Update the project unit tests to test your code changes as thoroughly
  as possible.
* Make sure that your code is properly commented with docstrings and
  comments explaining your rationale behind non-obvious coding practices.

If your contribution requires a new library dependency:

* Double-check that the new dependency is easy to install via pip and Anaconda.
* The library should support Python 3.6 and higher.
* Make sure the code works with the latest version of the library.
* Update the dependencies in the documentation.
* Add the library with the minimum required version to `requirements.txt`
  and `setup.py`.

After submitting your pull request, GitHub will automatically run the tests
on your changes and make sure that the updated code builds successfully.
The tests are checked on Python 3.7, 3.8, 3.9 and 3.10, on the latest Ubuntu
and Windows builds. We also use services that automatically run test coverage.
