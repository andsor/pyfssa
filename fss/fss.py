#!/usr/bin/env python
# encoding: utf-8

"""
Routines for finite-size scaling analyses

The **fss** package provides routines to perform finite-size scaling analyses on
experimental data [1]_ [2]_.

It has been inspired by Oliver Melchert and his superb **autoScale** package
[3]_.

Routines
--------

.. autosummary::
   :nosignatures:

   quality
   scaledata
   autoscale

Classes
-------

.. autosummary::

   ScaledData

References
----------

.. [1] M. E. J. Newman and G. T. Barkema, Monte Carlo Methods in Statistical
   Physics (Oxford University Press, 1999)

.. [2] K. Binder and D. W. Heermann, `Monte Carlo Simulation in Statistical
   Physics <http://dx.doi.org/10.1007/978-3-642-03163-2>`_ (Springer, Berlin,
   Heidelberg, 2010)

.. [3] O. Melchert, `arXiv:0910.5403 <http://arxiv.org/abs/0910.5403>`_
   (2009)

"""

# Python 2/3 compatibility
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import *

import numpy as np
from collections import namedtuple

import scipy.optimize
from scipy.optimize.optimize import (
    wrap_function, _status_message, OptimizeResult
)


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
       Physics (Oxford University Press, 1999)

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


def _jprimes(x, i):
    """
    Helper function to return the j' indices for the master curve fit

    This function is a helper function for :py:func:`quality`. It is not
    supposed to be called directly.

    Parameters
    ----------
    x : 2-D ndarray
        The x values

    i : int
        The column index (finite size index)

    Returns
    -------
    2-D ndarray of floats
        Has the same shape as `x`. Its element with index (i', j) is the j'
        such that :math:`x_{i'j'} \leq x_{ij} < x_{i'(j'+1)}`. If no such j'
        exists, the element is np.nan. Convert the element to int to use as
        an index.
    """
    ret = np.zeros_like(x)
    ret[:] = np.nan
    j_primes = - np.ones_like(x)

    k, n = x.shape
    for i_prime in range(k):
        if i_prime == i:
            continue

        j_primes[i_prime, :] = (
            np.searchsorted(
                x[i_prime, :], x[i, :], side='right'
            ) - 1
        )

    # boolean mask for valid values of j'
    j_primes_mask = np.logical_and(j_primes >= 0, j_primes < n - 1)

    ret[j_primes_mask] = j_primes[j_primes_mask]

    return ret


def _select_mask(j, j_primes):
    """
    Return a boolean mask for selecting the data subset according to the j'

    Parameters
    ----------
    j : int
        current j index

    j_primes : ndarray
        result from _jprimes call
    """

    ret = np.zeros_like(j_primes, dtype=bool)
    my_iprimes = np.invert(np.isnan(j_primes[:, j])).nonzero()[0]
    my_jprimes = j_primes[my_iprimes, j]
    my_jprimes = my_jprimes.astype(np.int)
    ret[my_iprimes, my_jprimes] = True
    ret[my_iprimes, my_jprimes + 1] = True

    return ret


def quality(x, y, dy):
    r'''
    Quality of data collapse onto a master curve defined by the data

    This is the reduced chi-square statistic for a data fit except that the
    master curve is fitted from the data itself.

    Parameters
    ----------
    x, y, dy : 2-D array_like
        output from :py:func:`scaledata`, scaled data `x`, `y` with standard
        errors `dy`

    Returns
    -------
    float
        the quality of the data collapse

    Raises
    ------
    ValueError
        if not all arrays `x`, `y`, `dy` have dimension 2, or if not all arrays
        are of the same shape, or if `x` is not sorted along rows (``axis=1``)

    Notes
    -----
    This is the implementation of the reduced :math:`\chi^2` quality function
    :math:`S` by Houdayer & Hartmann [1]_.
    It should attain a minimum of around :math:`1` for an optimal fit, and be
    much larger otherwise.

    For further information, see the :ref:`quality-function` section in the
    manual.

    References
    ----------
    .. [1] J. Houdayer and A. Hartmann, Physical Review B 70, 014418+ (2004)
        `doi:10.1103/physrevb.70.014418
        <http://dx.doi.org/doi:10.1103/physrevb.70.014418>`_

    '''

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

        j_primes = _jprimes(x=x, i=i)

        for j in range(n):

            # boolean mask for selected data x_l, y_l, dy_l
            select = _select_mask(j=j, j_primes=j_primes)

            if not select.any():
                # no data to select
                # master curve estimate Y_ij remains undefined
                continue

            # master curve estimate
            master_y[i, j], master_dy2[i, j] = _wls_linearfit_predict(
                x=x[i, j], w=w, wx=wx, wy=wy, wxx=wxx, wxy=wxy, select=select
            )

    return np.nanmean((y - master_y) ** 2 / (dy ** 2 + master_dy2))


def _neldermead_errors(sim, fsim, fun):
    """
    Estimate the errors from the final simplex of the Nelder--Mead algorithm

    This is a helper function and not supposed to be called directly.

    Parameters
    ----------
    sim : ndarray
        the final simplex

    fsim : ndarray
        the function values at the vertices of the final simplex

    fun : callable
        the goal function to minimize
    """

    # fit quadratic coefficients
    n = len(sim) - 1

    ymin = fsim[0]

    sim = np.copy(sim)
    fsim = np.copy(fsim)

    centroid = np.mean(sim, axis=0)
    fcentroid = fun(centroid)

    # enlarge distance of simplex vertices from centroid until all have at least
    # an absolute function value distance of 0.1
    for i in range(n + 1):
        while np.abs(fsim[i] - fcentroid) < 0.01:
            sim[i] += sim[i] - centroid
            fsim[i] = fun(sim[i])

    # the vertices and the midpoints x_ij
    x = 0.5 * (
        sim[np.mgrid[0:n+1, 0:n+1]][1]
        + sim[np.mgrid[0:n+1, 0:n+1]][0]
    )

    #for i in range(n + 1):
    #    assert(np.array_equal(x[i,i], sim[i]))
    #for j in range(n + 1):
    #        assert(np.array_equal(x[i,j], 0.5 * (sim[i] + sim[j])))

    y = np.nan * np.ones(shape=(n + 1, n + 1))
    for i in range(n + 1):
        y[i, i] = fsim[i]
        for j in range(i + 1, n + 1):
            y[i, j] = y[j, i] = fun(x[i, j])

    y0i = y[np.mgrid[0:n+1, 0:n+1]][0][1:, 1:, 0]
    #for i in range(n):
    #    for j in range(n):
    #        assert y0i[i, j] == y[0, i + 1], (i, j)

    y0j = y[np.mgrid[0:n+1, 0:n+1]][0][0, 1:, 1:]
    #for i in range(n):
    #    for j in range(n):
    #        assert y0j[i, j] == y[0, j + 1], (i, j)

    b = 2 * (y[1:, 1:] + y[0, 0] - y0i - y0j)
    #for i in range(n):
    #    assert abs(b[i, i] - 2 * (fsim[i + 1] + fsim[0] - 2 * y[0, i + 1])) < 1e-12
    #    for j in range(n):
    #        if i == j:
    #            continue
    #        assert abs(b[i, j] - 2 * (y[i + 1, j + 1] + fsim[0] - y[0, i + 1] -
    #            y[0, j + 1])) < 1e-12

    q = (sim - sim[0])[1:].T
    #for i in range(n):
    #    assert np.array_equal(q[:, i], sim[i + 1] - sim[0])

    varco = ymin * np.dot(q, np.dot(np.linalg.inv(b), q.T))
    return np.sqrt(np.diag(varco)), varco


def _minimize_neldermead_witherrors(
    fun, x0, args=(), callback=None,
    xtol=1e-4, ftol=1e-4, maxiter=None, maxfev=None,
    disp=False, return_all=False, with_errors=True,
    **unknown_options
):
    """
    Minimization of scalar function of one or more variables using the
    Nelder-Mead algorithm.

    Options for the Nelder-Mead algorithm are:
        disp : bool
            Set to True to print convergence messages.
        xtol : float
            Relative error in solution `xopt` acceptable for convergence.
        ftol : float
            Relative error in ``fun(xopt)`` acceptable for convergence.
        maxiter : int
            Maximum number of iterations to perform.
        maxfev : int
            Maximum number of function evaluations to make.

    This function is called by the `minimize` function with
    `method=minimize_neldermead_with_errors`. It is not supposed to be called
    directly.

    Notes
    -----
    Adapted from the original
    :py:func:`scipy.optimize.optimize._minimize_neldermead` function to include
    errors according to [1]_
    Note that the errors are calculated for minimizing the reduced chi-square
    statistic!

    ******NOTICE***************
    optimize.py module by Travis E. Oliphant

    You may copy and use this module as you see fit with no
    guarantee implied provided you keep this notice in all copies.
    *****END NOTICE************

    References
    ----------
    .. [1] J. A. Nelder and R. Mead, The Computer Journal 7, 308 (1965),
       `doi:10.1093/comjnl/7.4.308 <http://dx.doi.org/10.1093/comjnl/7.4.308>`_

    """

    maxfun = maxfev
    retall = return_all

    fcalls, func = wrap_function(fun, args)
    x0 = np.asfarray(x0).flatten()
    N = len(x0)
    rank = len(x0.shape)
    if not -1 < rank < 2:
        raise ValueError("Initial guess must be a scalar or rank-1 sequence.")
    if maxiter is None:
        maxiter = N * 200
    if maxfun is None:
        maxfun = N * 200

    rho = 1
    chi = 2
    psi = 0.5
    sigma = 0.5
    one2np1 = list(range(1, N + 1))

    if rank == 0:
        sim = np.zeros((N + 1,), dtype=x0.dtype)
    else:
        sim = np.zeros((N + 1, N), dtype=x0.dtype)
    fsim = np.zeros((N + 1,), float)
    sim[0] = x0
    if retall:
        allvecs = [sim[0]]
    fsim[0] = func(x0)
    nonzdelt = 0.05
    zdelt = 0.00025
    for k in range(0, N):
        y = np.array(x0, copy=True)
        if y[k] != 0:
            y[k] = (1 + nonzdelt)*y[k]
        else:
            y[k] = zdelt

        sim[k + 1] = y
        f = func(y)
        fsim[k + 1] = f

    ind = np.argsort(fsim)
    fsim = np.take(fsim, ind, 0)
    # sort so sim[0,:] has the lowest function value
    sim = np.take(sim, ind, 0)

    iterations = 1

    while (fcalls[0] < maxfun and iterations < maxiter):
        if (np.max(np.ravel(np.abs(sim[1:] - sim[0]))) <= xtol and
                np.max(np.abs(fsim[0] - fsim[1:])) <= ftol):
            break

        xbar = np.add.reduce(sim[:-1], 0) / N
        xr = (1 + rho) * xbar - rho * sim[-1]
        fxr = func(xr)
        doshrink = 0

        if fxr < fsim[0]:
            xe = (1 + rho * chi) * xbar - rho * chi * sim[-1]
            fxe = func(xe)

            if fxe < fxr:
                sim[-1] = xe
                fsim[-1] = fxe
            else:
                sim[-1] = xr
                fsim[-1] = fxr
        else:  # fsim[0] <= fxr
            if fxr < fsim[-2]:
                sim[-1] = xr
                fsim[-1] = fxr
            else:  # fxr >= fsim[-2]
                # Perform contraction
                if fxr < fsim[-1]:
                    xc = (1 + psi * rho) * xbar - psi * rho * sim[-1]
                    fxc = func(xc)

                    if fxc <= fxr:
                        sim[-1] = xc
                        fsim[-1] = fxc
                    else:
                        doshrink = 1
                else:
                    # Perform an inside contraction
                    xcc = (1 - psi) * xbar + psi * sim[-1]
                    fxcc = func(xcc)

                    if fxcc < fsim[-1]:
                        sim[-1] = xcc
                        fsim[-1] = fxcc
                    else:
                        doshrink = 1

                if doshrink:
                    for j in one2np1:
                        sim[j] = sim[0] + sigma * (sim[j] - sim[0])
                        fsim[j] = func(sim[j])

        ind = np.argsort(fsim)
        sim = np.take(sim, ind, 0)
        fsim = np.take(fsim, ind, 0)
        if callback is not None:
            callback(sim[0])
        iterations += 1
        if retall:
            allvecs.append(sim[0])

    x = sim[0]
    fval = np.min(fsim)
    warnflag = 0
    errors = None

    if fcalls[0] >= maxfun:
        warnflag = 1
        msg = _status_message['maxfev']
        if disp:
            print('Warning: ' + msg)
    elif iterations >= maxiter:
        warnflag = 2
        msg = _status_message['maxiter']
        if disp:
            print('Warning: ' + msg)
    else:
        msg = _status_message['success']
        errors, varco = _neldermead_errors(sim, fsim, func)
        if disp:
            print(msg)
            print("         Current function value: %f" % fval)
            print("         Iterations: %d" % iterations)
            print("         Function evaluations: %d" % fcalls[0])

    result = OptimizeResult(fun=fval, nit=iterations, nfev=fcalls[0],
                            status=warnflag, success=(warnflag == 0),
                            message=msg, x=x, errors=errors, sim=sim,
                            fsim=fsim, varco=varco)

    if retall:
        result['allvecs'] = allvecs
    return result


def autoscale(l, rho, a, da, rho_c0, nu0, zeta0, **kwargs):
    """
    Automatically scale finite-size data and fit critical point and exponents

    Parameters
    ----------
    l, rho, a, da : array_like
        input for the :py:func:`scaledata` function

    rho_c0, nu0, zeta0 : float
        initial guesses for the critical point and exponents

    Returns
    -------
    res : OptimizeResult

    res['success'] : bool
        Indicates whether the optimization algorithm has terminated
        successfully.

    res['x'] : ndarray

    res['rho'], res['nu'], res['zeta'] : float
        The fitted critical point and exponents, ``res['x'] == [res['rho'],
        res['nu'], res['zeta']]``

    res['drho'], res['dnu'], res['dzeta'] : float
        The respective standard errors derived from fitting the curvature at the
        minimum, ``res['errors'] == [res['drho'], res['dnu'], res['dzeta']]``.

    res['errors'], res['varco'] : ndarray
        The standard errors as a vector, and the full variance--covariance
        matrix (the diagonal entries of which are the squared standard errors),
        ``np.sqrt(np.diag(res['varco'])) == res['errors']``

    See also
    --------
    scaledata
        For the `l`, `rho`, `a`, `da` input parameters

    quality
        The goal function of the optimization

    scipy.optimize.minimize
        The optimization wrapper routine

    scipy.optimize.OptimizeResult
        The return type

    Notes
    -----
    This implementation uses the quality function by Houdayer & Hartmann [1]_
    which measures the quality of the data collapse, see the sections
    :ref:`data-collapse-method` and :ref:`quality-function` in the manual.

    This function and the whole fss package have been inspired by Oliver
    Melchert and his superb **autoScale** package [2]_.

    The critical point and exponents, including its standard errors and
    (co)variances, are fitted by the Nelder--Mead algorithm, see the section
    :ref:`neldermead` in the manual.

    References
    ----------
    .. [1] J. Houdayer and A. Hartmann, Physical Review B 70, 014418+ (2004)
        `doi:10.1103/physrevb.70.014418
        <http://dx.doi.org/doi:10.1103/physrevb.70.014418>`_

    .. [2] O. Melchert, `arXiv:0910.5403 <http://arxiv.org/abs/0910.5403>`_
       (2009)

    Examples
    --------
    >>> # generate artificial scaling data from master curve
    >>> # with rho_c == 1.0, nu == 2.0, zeta == 0.0
    >>> import fss
    >>> l = [ 10, 100, 1000 ]
    >>> rho = np.linspace(0.9, 1.1)
    >>> l_mesh, rho_mesh = np.meshgrid(l, rho, indexing='ij')
    >>> master_curve = lambda x: 1. / (1. + np.exp( - x))
    >>> x = np.power(l_mesh, 0.5) * (rho_mesh - 1.)
    >>> y = master_curve(x)
    >>> dy = y / 100.
    >>> y += np.random.randn(*y.shape) * dy
    >>> a = y
    >>> da = dy
    >>>
    >>> # run autoscale
    >>> fss.autoscale(l=l, rho=rho, a=a, da=da, rho_c0=0.9, nu0=2.0, zeta0=0.0)
    varco: array([[  3.48404128e-06,   5.72603874e-05,  -2.26338928e-06],
        [  5.72603874e-05,   2.10915133e-02,  -1.28212500e-04],
        [ -2.26338928e-06,  -1.28212500e-04,   1.88163502e-06]])
    dzeta: 0.0013717270192608684
    fun: 2.2095309418677989
    zeta: -0.00035713114892812982
    drho: 0.0018665586740112224
    fsim: array([ 2.20953094,  2.20953226,  2.20953272,  2.20953504])
    dnu: 0.14522917505380872
    rho: 0.9999052186821713
    nu: 2.0366578589015862
    status: 0
    nit: 69
    success: True
    sim: array([[  9.99905219e-01,   2.03665786e+00,  -3.57131149e-04],
        [  9.99905097e-01,   2.03658480e+00,  -3.56730933e-04],
        [  9.99905351e-01,   2.03671238e+00,  -3.57425026e-04],
        [  9.99905303e-01,   2.03665797e+00,  -3.57164729e-04]])
    x: array([  9.99905219e-01,   2.03665786e+00,  -3.57131149e-04])
    message: 'Optimization terminated successfully.'
    errors: array([ 0.00186656,  0.14522918,  0.00137173])
    nfev: 153

    """

    def goal_function(x):
        return quality(*scaledata(
            rho=rho, l=l, a=a, da=da, nu=x[1], zeta=x[2], rho_c=x[0]
        ))

    ret = scipy.optimize.minimize(
        goal_function,
        [rho_c0, nu0, zeta0],
        method=_minimize_neldermead_witherrors,
        options={
            'xtol': 1e-2,
            'ftol': 1e-2
        }
    )

    ret['rho'], ret['nu'], ret['zeta'] = ret['x']
    ret['drho'], ret['dnu'], ret['dzeta'] = ret['errors']

    return ret
