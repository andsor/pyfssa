Finite-size scaling analysis
============================

The finite-size scaling ansatz
------------------------------

Consider a system with some parameter :math:`\varrho`, which undergoes a phase
transition at a critical value :math:`\varrho_c`.
Divergences in the correlation length :math:`\xi` and of fluctuations (e.g. the
susceptibility :math:`\chi`) characterize the critical point in infinite
systems.
Typically, in the critical region a diverging quantity
:math:`A_\infty(\varrho)` scales as :math:`|\varrho - \varrho_c|^{-\zeta}` with
some critical exponent :math:`\zeta`.
In fact, this critical behavior should also hold in systems of finite size
:math:`L` at scales much larger than the characteristic length scale
:math:`\xi`.
Here and in the following, the characteristic length scale :math:`\xi` is the
correlation length in the infinite system (:math:`L \to \infty`).
As the correlation length diverges as :math:`\xi \sim |\varrho -
\varrho_c|^{-\nu}` for :math:`\varrho \to \varrho_c`, we have

.. math::

   A_L(\varrho) \sim |\varrho - \varrho_c|^{-\zeta} \sim \xi^{\zeta / \nu},
   \qquad (L \gg \xi, \varrho \to \varrho_c).

For :math:`L \ll \xi`, the system size :math:`L` takes over the role as the
cutoff, such that we expect

.. math::

   A_L(\varrho) \sim L^{\zeta/\nu}, \qquad (L \ll \xi, \varrho \to \varrho_c).

These considerations constitute the *finite-size scaling ansatz*
:cite:`Newman1999Monte,Binder2010Monte,Fisher1972Scaling`

.. math::

   A_L(\varrho) = \xi^{\zeta/\nu} f(L / \xi), \qquad (L \to \infty, \varrho \to
   \varrho_c),

with

.. math::

   f(x) \begin{cases}
   = \text{const.} & \text{for } |x| \gg 1, \\
   \sim x^{\zeta/\nu} & \text{for } x \to 0.
   \end{cases}


The *scaling function* :math:`f(x)` is a dimensionless function of the
dimensionless ratio :math:`L/\xi` of the finite system size and the
infinite-system correlation length.
This ratio controls the finite-size effects.
The conventional scaling function is :math:`\tilde{f}(x) = x^{-\zeta} f(x^\nu)`
:cite:`Newman1999Monte,Binder2010Monte` such that

.. math::

   A_L(\varrho) = L^{\zeta/\nu} \tilde{f}\left(L^{1/\nu} (\varrho -
   \varrho_c)\right), \qquad (L \to \infty, \varrho \to \varrho_c),

with

.. math::

   \tilde{f}(x) \begin{cases}
   = \text{const.} & \text{for } x \to 0 \quad (L \ll \xi), \\
   \sim L^{-\zeta/\nu} (\varrho - \varrho_c)^{-\zeta} & \text{for } |x| \gg 1
   \quad (L \gg \xi).
   \end{cases}

.. _data-collapse-method:

The data collapse method
------------------------

A simulation experiment yields the quantity :math:`a_{L, \varrho}` at system
size :math:`L` and parameter :math:`\varrho`, with standard error :math:`da_{L,
\varrho}`.
The scaling function is

.. math::

   \tilde{f}\left(L^{1/\nu} (\varrho - \varrho_c) \right) = L^{-\zeta/\nu}
   A_L(\varrho).

Thus, plotting :math:`L^{-\zeta/\nu} a_{L, \varrho}` against
:math:`L^{1/\nu}(\varrho-\varrho_c)` should let the experimental data
collapse onto the single curve :math:`\tilde{f}(x)`.
For this to happen, the critical parameter value :math:`\varrho_c` and the
critical exponents :math:`\zeta, \nu` need to be correct.
These assumptions hold for :math:`L \to \infty`, with systematic errors at
finite sizes :cite:`Newman1999Monte,Binder2010Monte`.

The intersection method
-----------------------

At a first-order transition where the order parameter :math:`P` remains
constant on both sides of the transition, its critical exponent is :math:`\beta
= 0`.

So we have

.. math::

   P_L(\varrho) = \tilde{P}\left(L^{1/\nu}(\varrho-\varrho_c)\right),

and hence, :math:`P_L(\varrho_c) = \tilde{P}(0)` independent of the system size
:math:`L`.
Thus, the common intersection point of the measured curves :math:`p_{L,
\varrho}` yields an estimate of the threshold :math:`\varrho_c`.
This estimate is unbiased with regards to the critical exponents, and "should
be free" from systematic errors due to finite system size
:cite:`Binder2010Monte`.

Implementation in the fss package
---------------------------------

Routines
~~~~~~~~

.. autosummary::
   :nosignatures:

   fss.fss.scaledata

Classes
~~~~~~~

.. autosummary::

   fss.fss.ScaledData

