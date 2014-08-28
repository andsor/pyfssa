pyfss
=====

pyfss is a scientific Python package for finite-size scaling analysis at phase
transitions.

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

Building the documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~

This project uses `sphinx`_ and `sphinx_rtd_theme`_.
To build the documentation, run

.. code:: bash

   cd docs; make html

in the working directory of the repository, or run

.. code:: bash
   
   cd docs; make

to list other output formats.


.. _sphinx: http://sphinx-doc.org
.. _sphinx_rtd_theme: http://pypi.python.org/pypi/sphinx_rtd_theme

License
-------

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

