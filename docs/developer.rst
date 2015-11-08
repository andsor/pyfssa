Developer Guide
===============

* **Repository**: `github.com/andsor/pyfssa <http://github.com/andsor/pyfssa>`_
* **Bibliography**: `www.citeulike.org/group/19073 <http://www.citeulike.org/group/19073>`_

Development environment
-----------------------

Use `tox`_ to `prepare virtual environments for development`_.

.. _prepare virtual environments for development: http://testrun.org/tox/latest/example/devenv.html>

.. _tox: http://tox.testrun.org

To set up a **Python 2.7** environment in ``.devenv27``, run::

    $ tox -e devenv27

To set up a **Python 3.4** environment in ``.devenv34``, run::

    $ tox -e devenv34

To set up a **Python 3.5** environment in ``.devenv35``, run::

    $ tox -e devenv35

Add `requirements`_ for the development environments to the
`requirements-dev.txt <requirements-dev.txt>`_ file.

.. _requirements: http://pip.readthedocs.org/en/latest/user_guide.html#requirements-files


Packaging
---------

This package uses `setuptools`_.

.. _setuptools: http://pythonhosted.org/setuptools

Run ::

    $ python setup.py sdist
   
or ::

    $ python setup.py bdist
   
or ::

    $ python setup.py bdist_wheel
    
to build a source, binary or wheel distribution.


Complete Git Integration
------------------------

The package is maintained in a git repository.
The setuptools script ``setup.py`` uses the information of tags to infer the
version of your project with the help of `setuptools_scm
<https://pypi.python.org/pypi/setuptools_scm/>`_.
To use this feature you need to tag with the format ``MAJOR.MINOR[.PATCH]``
, e.g. ``0.0.1`` or ``0.1``.

Run ::
        
    $ python setup.py --version
    
to retrieve the current `PEP440`_-compliant version.
This version will be used when building a package and is also accessible
through ``fssa.__version__``.

.. _PEP440: http://www.python.org/dev/peps/pep-0440

Unleash the power of Git by using its `pre-commit hooks
<http://pre-commit.com/>`_.
Make sure pre-commit is installed, e.g. ``pip install pre-commit``, then just
run ``pre-commit install``.


Sphinx Documentation
--------------------

This project follows the `NumPy documentation style
<https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_.

Build the documentation with::
        
    $ tox -e docs

Add further options separated from tox options by a double dash ``--``::

    $ tox -e docs -- --help

Add `requirements`_ for building the documentation to the
`requirements-doc.txt <requirements-doc.txt>`_ file.

.. _requirements: http://pip.readthedocs.org/en/latest/user_guide.html#requirements-files


Continuous documentation building
---------------------------------

For continuously building the documentation during development, run::
        
    $ tox -e cdocs

Unittest & Coverage
-------------------

Run ::

    $ tox -e py27
    
or::

    $ tox -e py34

or::

    $ tox -e py35

to run all unittests defined in the subfolder ``test`` with the help of `tox`_
and `py.test`_.

.. _py.test: http://pytest.org

The py.test plugin `pytest-cov`_ is used to automatically generate a coverage
report. 

.. _pytest-cov: http://github.com/schlamar/pytest-cov

Continuous testing
------------------

For continuous testing in a **Python 2.7** environment, run::
       
    $ tox -e c27

For continuous testing in a **Python 3.4** environment, run::
       
    $ tox -e c34

Requirements Management
-----------------------

Add `requirements`_ to the `requirements.txt <requirements.txt>`_ file which
will be automatically used by ``setup.py``.

Bibliography
------------

A `CiteULike group`_ manages the bibliography.

.. _CiteULike group: http://www.citeulike.org/group/19073

To download the bibliography, run ::

    $ doit download_bib

Continuous Integration
----------------------

pyfssa uses `Travis <https://travis-ci.org/andsor/pyfssa>`_ to run the tests on each commit.
Travis also reports the test coverage to `Coveralls <https://coveralls.io/github/andsor/pyfssa>`_.
If further deploys each tagged commit as a release to the Python Package Index (PyPI).

`ReadTheDocs <https://readthedocs.org/projects/pyfssa/>`_ builds and hosts this documentation.
