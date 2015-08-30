The quality of data collapse
============================

.. _quality-function:

The quality function
--------------------

In the following, we present a measure by Houdayer & Hartmann
:cite:`Houdayer2004Lowtemperature` for the quality of the :ref:`data collapse
<data-collapse-method>`.
Melchert :cite:`Melchert2009AutoScalepy` refers to some alternative measures,
for example :cite:`Bhattacharjee2001Measure`, :cite:`Wenzel2008Percolation`,
and to some applications of these measures in the literature.

Houdayer & Hartmann :cite:`Houdayer2004Lowtemperature` refine a method proposed
by Kawashima & Ito :cite:`Kawashima1993Critical`.
They define the quality as the reduced :math:`\chi^2` statistic

.. math::

   S = \frac{1}{\mathcal{N}} \sum_{i,j} \frac{(y_{ij} -
   Y_{ij})^2}{dy_{ij}^2+dY_{ij}^2},

where the values :math:`y_{ij}, dy_{ij}` are the scaled observations and its
standard errors at :math:`x_{ij}`, and the values :math:`Y_{ij}, dY_{ij}` are
the estimated value of the master curve and its standard error at
:math:`x_{ij}`.

The quality :math:`S` is the mean square of the weighted deviations from the
master curve.
As we expect the individual deviations :math:`y_{ij} - Y_{ij}` to be of the
order of the individual error :math:`\sqrt{dy_{ij}^2 + dY_{ij}^2}` for an
optimal fit, the quality :math:`S` should attain its minimum :math:`S_{\min}`
at around :math:`1` and be much larger otherwise :cite:`Bevington2003Data`.

Let :math:`i` enumerate the system sizes :math:`L_i`, :math:`i = 1, \ldots, k`
and let :math:`j` enumerate the parameters :math:`\varrho_j`, :math:`j = 1,
\ldots, n` with :math:`\varrho_1 < \varrho_2 < \ldots < \varrho_n`.
The scaled data are

.. math::

   y_{ij} & := L_i^{-\zeta/\nu} a_{L_i, \varrho_j} \\
   dy_{ij} & := L_i^{-\zeta/\nu} da_{L_i, \varrho_j} \\
   x_{ij}  & := L_i^{1/\nu}(\varrho_j - \varrho_c).

The sum in the quality function :math:`S` only involves terms for which the
estimated value :math:`Y_{ij}` of the master curve at :math:`x_{ij}` is
defined. The number of such terms is :math:`\mathcal{N}`.

The master curve itself depends on the scaled data.
For a given :math:`i`, :math:`L_i`, we estimate the master curve at
:math:`x_{ij}` by the two respective data from all the other system sizes which
respectively enclose :math:`x_{ij}`:
for each :math:`i \neq i`, let :math:`j'` be such that :math:`x_{i'j'} \leq
x_{ij} \leq x_{i'(j'+1)}`, and select the points :math:`(x_{i'j'}, y_{i'j'},
dy_{i'j'}), (x_{i'(j'+1)}, y_{i'(j'+1)}, dy_{i'(j'+1)})`.
Do not select points for some :math:`i'`, if there is no such :math:`j'`. If
there is no such :math:`j'` for all :math:`i'`, the master curve remains
undefined at :math:`x_{ij}`.

Given the selected points :math:`(x_l, y_l, dy_l)`, the local approximation of
the master curve is the linear fit

.. math::

   y = mx + b

with weighted least squares :cite:`Strutz2011Data`.
The weights :math:`w_l` are the reciprocal variances, :math:`w_l :=
1/dy_{ij}^2`.
The estimates and (co)variances of the slope :math:`m` and intercept :math:`b`
are

.. math::

   \hat{b} &= \frac{1}{\Delta} (K_{xx}K_y - K_xK_{xy}) \\
   \hat{m} &= \frac{1}{\Delta} (K K_{xy} - K_x K_y)

   \hat{\sigma}_b^2 = \frac{K_{xx}}{\Delta} , \hat{\sigma}_m^2 = \frac{K}{\Delta},
   \hat{\sigma}_{bm} = - \frac{K_x}{\Delta}

with :math:`K_{nm} := \sum w_l x_l^n y_l^m`, :math:`K := K_{00}`, :math:`K_x :=
K_{10}`, :math:`K_y := K_{01}`, :math:`K_{xx} := K_{20}`, :math:`K_{xy} :=
K_{11}`, :math:`\Delta := KK_{xx} - K_x^2`.

Hence, the estimated value of the master curve at :math:`x_{ij}` is

.. math::

   Y_{ij} = \hat{m} x_{ij} + \hat{b}

with error propagation

.. math::

   dY_{ij}^2 = \hat{\sigma}^2 x_{ij}^2 + 2 \hat{\sigma}_{bm} x_{ij} +
   \hat{\sigma}_b^2.


 
Refinement of the quality function
----------------------------------

The fssa package further refines the quality function.
The original sum involves only terms for which the master curve is defined.
As the number of missing terms in general differs from system size to system
size, the sum implicitly weights system sizes differently.
This is unintended behavior, especially when it comes to scalings with less
dense coverage of the critical region at large system sizes.

To alleviate this, we modify the sum as follows:

.. math::

   S' = \frac{1}{k} \sum_i \frac{1}{\mathcal{N}_i} \sum_{j} \frac{(y_{ij} -
   Y_{ij})^2}{dy_{ij}^2+dY_{ij}^2},

where the number of system sizes is :math:`k` (as before), and
:math:`\mathcal{N}_i` is the number of terms for the :math:`i`-th system size.
By separately averaging over all available terms for each system size, and then
averaging over all system sizes, the contributions of each system size have
equal weight in the final sum.

Implementation in the fssa package
----------------------------------

Routines
~~~~~~~~

.. autosummary::
   :nosignatures:

   fssa.fssa.quality

