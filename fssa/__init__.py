#!/usr/bin/env python
# encoding: utf-8

r"""
Implements algorithmic finite-size scaling analysis at phase transitions

This module implements the algorithmic finite-size scaling analysis at phase
transitions as demonstrated by Oliver Melchert and his superb autoscale.py
script.

The :mod:`fssa` module provides these high-level functions from the
:mod:`fssa.fssa` module:

.. autosummary::

   fssa.scaledata
   fssa.quality
   fssa.autoscale

See Also
--------

fssa.fssa : low-level functions

Notes
-----

The :func:`fssa.scaledata` function scales finite-size data in order for the
data to hopefully collapse onto a single universal scaling function, also
known as master curve.
The :func:`fssa.quality` function assesses the quality of this very data
collapse onto a single curve.
Finally, the :func:`fssa.autoscale` function frames the data collapse as an
optimization problem and searches for the critical values that minimize the
quality function.

The **fssa** package expects finite-size data in the following setting.

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
This implementation uses the quality function by Houdayer & Hartmann [1]_
which measures the quality of the data collapse, see the sections
:ref:`data-collapse-method` and :ref:`quality-function` in the manual.

This function and the whole fssa package have been inspired by Oliver
Melchert and his superb **autoScale** package [2]_.

The critical point and exponents, including its standard errors and
(co)variances, are fitted by the Nelder--Mead algorithm, see the section
:ref:`neldermead` in the manual.

Currently, the module only implements homogeneous data arrays:
Data must be available for all finite system sizes and parameter values.

References
----------
.. [1] J. Houdayer and A. Hartmann, Physical Review B 70, 014418+ (2004)
    `doi:10.1103/physrevb.70.014418
    <http://dx.doi.org/doi:10.1103/physrevb.70.014418>`_

.. [2] O. Melchert, `arXiv:0910.5403 <http://arxiv.org/abs/0910.5403>`_
    (2009)

.. todo::

   `Implement heterogeneous finite-size data handling`__

__ https://github.com/andsor/pyfssa/issues/2
"""
from __future__ import absolute_import

import pkg_resources
from .fssa import scaledata, quality, autoscale

__version__ = pkg_resources.get_distribution(__name__).version
