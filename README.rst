pyfss
=====

pyfss is a scientific Python package for finite-size scaling analysis at phase
transitions.

Usage
-----

The **fss** package expects finite-size data in the following setting.

.. math::

   A_L(\varrho) = L^{\zeta/\nu} \tilde{f}\left(L^{1/\nu} (\varrho -
   \varrho_c)\right), \qquad (L \to \infty, \varrho \to \varrho_c),

`l` is like a 1-D numpy array which contains the finite system sizes :math:`L`.
`rho` is like a 1-D numpy array which contains the parameter values
:math:`\varrho`.
`a` is like a 2-D numpy array which contains the observations (the data)
:math:`A_L(\varrho)`, where `a[i, j]` is the data at the `i`-th system size and
the `j`-th parameter value.
`da` is like a 2-D numpy array which contains the standard errors in the
observations.

The **fss.autoscale** function attempts to determine the critical parameter and
exponents which entail an optimal data collapse. The initial guesses for
:math:`\varrho_c, \nu, \zeta` are `rho_c0`, `nu0`, and `zeta0`.

>>> import fss
>>> fss.autoscale(l, rho, a, da, rho_c0, nu0, zeta0)

Package maintainer
------------------

The author and maintainer of the **pyfss** package is Andreas Sorge <as@asorge.de>.

Python 2/3 compatibility
------------------------

This project uses `python-future`_.
It is written in standard Python 3 code, with `python-future`_ providing
support for running the code on Python 2.7 `mostly unchanged
<http://python-future.org/compatible_idioms.html>`_.

.. _python-future: http://python-future.org

Building the documentation
--------------------------

This project uses `sphinx`_, `sphinx_rtd_theme`_, `sphinxcontrib-bibtex`_ and
`numpydoc`_.
A `CiteULike group`_ manages the bibliography.
We configure a `custom style`_ in `docs/conf.py <docs/conf.py>`_ which
suppresses URLs in the bibliography output.
We employ the `numpy docstring conventions`_.

To update the local bibliography, run

.. code:: bash

   cd docs; make bib

in the working directory of the repository.

To build the documentation, run

.. code:: bash

   cd docs; make html

in the working directory of the repository, or run

.. code:: bash
   
   cd docs; make

to list other output formats.

To automatically build the documentation, run

.. code:: bash

   cd docs; ./automake.sh

in the working directory of the repository.


.. _sphinx: http://sphinx-doc.org
.. _sphinx_rtd_theme: http://pypi.python.org/pypi/sphinx_rtd_theme
.. _sphinxcontrib-bibtex: http://pypi.python.org/pypi/sphinxcontrib-bibtex/
.. _CiteULike group: http://www.citeulike.org/group/19073
.. _custom style: http://sphinxcontrib-bibtex.readthedocs.org/en/latest/usage.html#custom-formatting-sorting-and-labelling
.. _numpydoc: http://pypi.python.org/pypi/numpydoc
.. _numpy docstring conventions: http://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt

Developing
----------

Deployment and Packaging
~~~~~~~~~~~~~~~~~~~~~~~~

This project uses `setuptools`_.
The `Development Mode`_ deploys the project locally without copying any files.
Run

.. code:: bash

           python setup.py develop

in the working directory root of the repository.

.. _setuptools: https://pypi.python.org/pypi/setuptools/

.. _Development Mode: http://pythonhosted.org//setuptools/setuptools.html#development-mode


Versioning
~~~~~~~~~~

This project `calculates the current package version number based on git tags <https://gist.github.com/ryanvolz/9e095624d46756ca0045>`_.

Testing
~~~~~~~

This project uses `unittest`_.

.. _unittest: http://docs.python.org/3/library/unittest.html

Run

.. code:: bash

   python setup.py test

to `build the package and run the tests
<http://pythonhosted.org/setuptools/setuptools.html#test-build-package-and-run-a-unittest-suite>`_.

Run

.. code:: bash
   
   python -m unittest discover

from the working directory root of the repository to `discover and run the
tests <http://docs.python.org/3.4/library/unittest.html#test-discovery>`_.

For automatic test runs upon file changes run

.. code:: bash

   ./autotest.sh

from the working directory of the repository.

.. license-before-anchor

License
-------

.. license-after-anchor

Copyright 2014 Max Planck Society, Andreas Sorge

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
