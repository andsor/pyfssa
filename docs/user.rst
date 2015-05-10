User Guide
==========

.. toctree::

   fss-theory
   quality
   nelder-mead

Usage
-----

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

The **fssa.autoscale** function attempts to determine the critical parameter and
exponents which entail an optimal data collapse. The initial guesses for
:math:`\varrho_c, \nu, \zeta` are `rho_c0`, `nu0`, and `zeta0`.

>>> import fssa
>>> fssa.autoscale(l, rho, a, da, rho_c0, nu0, zeta0)
