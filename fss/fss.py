#!/usr/bin/env python
# encoding: utf-8

"""
Routines for finite-size scaling analyses

The **fss** package provides routines to perform finite-size scaling analyses on
experimental data [1]_ [2]_.

Routines
--------

.. autosummary::
   :nosignatures:

   scaledata

Classes
-------

.. autosummary::

   ScaledData

References
----------

.. [1] M. E. J. Newman and G. T. Barkema, Monte Carlo Methods in Statistical
   Physics* (Oxford University Press, 1999)

.. [2] K. Binder and D. W. Heermann, `Monte Carlo Simulation in Statistical
   Physics <http://dx.doi.org/10.1007/978-3-642-03163-2>`_ (Springer, Berlin,
   Heidelberg, 2010)

"""

# Python 2/3 compatibility
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import *

import numpy as np
from collections import namedtuple


class ScaledData(namedtuple('ScaledData', ['x', 'y', 'dy'])):
    """
    A :py:func:`namedtuple <collections.namedtuple>` for :py:func:`scaledata` output
    """

    # set this to keep memory requirements low, according to
    # http://docs.python.org/3/library/collections.html#namedtuple-factory-function-for-tuples-with-named-fields
    __slots__ = ()


def scaledata(l, rho, a, da, rho_c, nu, zeta):
    r'''
    Scale experimental data according to critical exponents

    Parameters
    ----------
    l, rho : 1-D array_like
       finite system sizes `l` and parameter values `rho`

    a, da : 2-D array_like of shape (`l`.size, `rho`.size)
       experimental data `a` with standard errors `da` obtained at finite system
       sizes `l` and parameter values `rho`, with
       ``a.shape == da.shape == (l.size, rho.size)``

    rho_c : float in range [rho.min(), rho.max()]
       (assumed) critical parameter value with ``rho_c >= rho.min() and rho_c <=
       rho.max()``

    nu, zeta : float
       (assumed) critical exponents

    Returns
    -------
    :py:class:`ScaledData`
       scaled data `x`, `y` with standard errors `dy`

    x, y, dy : ndarray
       two-dimensional arrays of shape ``(l.size, rho.size)``

    Notes
    -----
    Scale data points :math:`(\varrho_j, a_{ij}, da_{ij})` observed at finite
    system sizes :math:`L_i` and parameter values :math:`\varrho_i` according to
    the finite-size scaling ansatz

    .. math::

       L^{-\zeta/\nu} a_{ij} = \tilde{f}\left( L^{1/\nu} (\varrho_j - \varrho_c)
       \right).

    The output is the scaled data points :math:`(x_{ij}, y_{ij}, dy_{ij})` with

    .. math::

       x_{ij} & = L_i^{1/\nu} (\varrho_j - \varrho_c) \\
       y_{ij} & = L_i^{-\zeta/\nu} a_{ij} \\
       dy_{ij} & = L_i^{-\zeta/\nu} da_{ij}

    such that all data points :ref:`collapse <data-collapse-method>` onto the
    single curve :math:`\tilde{f}(x)` with the right choice of :math:`\varrho_c,
    \nu, \zeta` [1]_ [2]_.

    Raises
    ------
    ValueError
       If `l` or `rho` is not 1-D array_like, if `a` or `da` is not 2-D
       array_like, if the shape of `a` or `da` differs from ``(l.size,
       rho.size)``, if `da` has non-positive entries, or if `rho_c` is out of
       range

    References
    ----------

    .. [1] M. E. J. Newman and G. T. Barkema, Monte Carlo Methods in Statistical
       Physics* (Oxford University Press, 1999)

    .. [2] K. Binder and D. W. Heermann, `Monte Carlo Simulation in Statistical
       Physics <http://dx.doi.org/10.1007/978-3-642-03163-2>`_ (Springer,
       Berlin, Heidelberg, 2010)
    '''

    # l should be 1-D array_like
    l = np.asanyarray(l)
    if l.ndim != 1:
        raise ValueError("l should be 1-D array_like")

    # rho should be 1-D array_like
    rho = np.asanyarray(rho)
    if rho.ndim != 1:
        raise ValueError("rho should be 1-D array_like")

    # a should be 2-D array_like
    a = np.asanyarray(a)
    if a.ndim != 2:
        raise ValueError("a should be 2-D array_like")

    # a should have shape (l.size, rho.size)
    if a.shape != (l.size, rho.size):
        raise ValueError("a should have shape (l.size, rho.size)")

    # da should be 2-D array_like
    da = np.asanyarray(da)
    if da.ndim != 2:
        raise ValueError("da should be 2-D array_like")

    # da should have shape (l.size, rho.size)
    if da.shape != (l.size, rho.size):
        raise ValueError("da should have shape (l.size, rho.size)")

    # da should have only positive entries
    if not np.all(da > 0.0):
        raise ValueError("da should have only positive values")

    # rho_c should be float
    rho_c = float(rho_c)

    # rho_c should be in range
    if rho_c > rho.max() or rho_c < rho.min():
        raise ValueError("rho_c is out of range")

    # nu should be float
    nu = float(nu)

    # zeta should be float
    zeta = float(zeta)

    l_mesh, rho_mesh = np.meshgrid(l, rho, indexing='ij')

    x = np.power(l_mesh, 1. / nu) * (rho_mesh - rho_c)
    y = np.power(l_mesh, - zeta / nu) * a
    dy = np.power(l_mesh, - zeta / nu) * da

    return ScaledData(x, y, dy)


def _wls_linearfit_predict(x, w, wx, wy, wxx, wxy, select):
    """
    Predict a point according to a weighted least squares linear fit of the data

    This function is a helper function for :py:func:`quality`. It is not
    supposed to be called directly.

    Parameters
    ----------
    x : float
        The position for which to predict the function value

    w : ndarray
        The pre-calculated weights :math:`w_l`

    wx : ndarray
        The pre-calculated weighted `x` data :math:`w_l x_l`

    wy : ndarray
        The pre-calculated weighted `y` data :math:`w_l y_l`

    wxx : ndarray
    The pre-calculated weighted :math:`x^2` data :math:`w_l x_l^2`

    wxy : ndarray
        The pre-calculated weighted `x y` data :math:`w_l x_l y_l`

    select : indexing array
        To select the subset from the `w`, `wx`, `wy`, `wxx`, `wxy` data

    Returns
    -------
    float, float
        The estimated value of the master curve for the selected subset and the
        squared standard error
    """

    # linear fit
    k = w[select].sum()
    kx = wx[select].sum()
    ky = wy[select].sum()
    kxx = wxx[select].sum()
    kxy = wxy[select].sum()
    delta = k * kxx - kx ** 2
    m = 1. / delta * (k * kxy - kx * ky)
    b = 1. / delta * (kxx * ky - kx * kxy)
    b_var = kxx / delta
    m_var = k / delta
    bm_covar = - kx / delta

    # estimation
    y = b + m * x
    dy2 = b_var + 2 * bm_covar * x + m_var * x**2

    return y, dy2

def quality(x, y, dy):

    # arguments should be 2-D array_like
    x = np.asanyarray(x)
    y = np.asanyarray(y)
    dy = np.asanyarray(dy)

    args = {"x": x, "y": y, "dy": dy}
    for arg_name, arg in args.items():
        if arg.ndim != 2:
            raise ValueError("{} should be 2-D array_like".format(arg_name))

    # arguments should have all the same shape
    if not x.shape == y.shape == dy.shape:
        raise ValueError("arguments should be of same shape")

    # x should be sorted for all system sizes l
    if not np.array_equal(x, np.sort(x, axis=1)):
        raise ValueError("x should be sorted for each system size")

    # first dimension: system sizes l
    # second dimension: parameter values rho
    k, n = x.shape

    # pre-calculate weights and other matrices
    w = dy ** (-2)
    wx = w * x
    wy = w * y
    wxx = w * x * x
    wxy = w * x * y

    # calculate master curve estimates
    master_y = np.zeros_like(y)
    master_y[:] = np.nan
    master_dy2 = np.zeros_like(dy)
    master_dy2[:] = np.nan

    for i in range(k):

        # initialize j' to -1 (which is invalid value)
        j_primes = - np.ones_like(x, dtype=int)

        for i_prime in range(k):
            if i_prime == i:
                continue

            # for each j determine the pairs x_i'j', x_i'(j'+1) that enclose
            # the value x_ij
            j_primes[i_prime, :] = (
                np.searchsorted(
                    x[i_prime, :], x[i, :], side='right'
                ) - 1
            )

        # boolean mask for valid values of j'
        j_primes_mask = np.logical_and(j_primes >= 0, j_primes < n - 1)

        for j in range(n):

            # get the mask for the j' for this specific j
            j_primes_mask_j = j_primes_mask[:, j]

            if not j_primes_mask_j.any():
                # no data to select
                # master curve estimate Y_ij remains undefined
                continue

            # boolean mask for selected data x_l, y_l, dy_l
            select = np.zeros_like(x, dtype=bool)

            # select all x_i'j' for x_ij
            select[
                j_primes_mask[:, j].nonzero(),  # the i' indices
                j_primes[:, j][                 # the j' indices
                    j_primes_mask[:, j]
                ]
            ] = True

            # select all x_i'(j'+1) for x_ij
            select[
                j_primes_mask[:, j].nonzero(),  # the i' indices
                j_primes[:, j][                 # the j' + 1 indices
                    j_primes_mask[:, j]
                ] + 1
            ] = True

            # master curve estimate
            master_y[i, j], master_dy2[i, j] = _wls_linearfit_predict(
                x=x[i, j], w=w, wx=wx, wy=wy, wxx=wxx, wxy=wxy, select=select
            )

    return np.nanmean((y - master_y) ** 2 / (dy ** 2 + master_dy2))
