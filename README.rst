pyfss
=====

pyfss is a scientific Python package for finite-size scaling analysis at phase
transitions.

Python 2/3 compatibility
------------------------

This project uses `python-future`_.
It is written in standard Python 3 code, with `python-future`_ providing
support for running the code on Python 2.7 `mostly unchanged
<http://python-future.org/compatible_idioms.html>`_.

.. _python-future: http://python-future.org

Building the documentation
--------------------------

This project uses `sphinx`_, `sphinx_rtd_theme`_, and `sphinxcontrib-bibtex`_.
A `CiteULike group`_ manages the bibliography.
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

.. image:: http://gnu.org/graphics/gplv3-88x31.png
   :target: http://gnu.org/licenses/gpl.html

This program is free software: you can redistribute it and/or modify it under
the terms of the `GNU General Public License`_ as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the `GNU General Public License`_ for more details.

You should have received a copy of the `GNU General Public License`_ along with
this program.  If not, see http://www.gnu.org/licenses/.

.. _GNU General Public License: http://gnu.org/licenses/gpl.html

