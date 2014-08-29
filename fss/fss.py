#!/usr/bin/env python
# encoding: utf-8

# Python 2/3 compatibility
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import *

import numpy as np

from collections import namedtuple

ScaledData = namedtuple('ScaledData', ['x', 'y', 'dy'])

def scaledata(l, rho, a, da, rho_c, nu, zeta):
    """TODO: Docstring for scaledata.

    Parameters
    ----------
    :l: 1-D array_like

    :rho: 1-D array_like

    :a: 2-D array_like of shape (l.size, rho.size)

    :da: 2-D array_like of shape (l.size, rho.size)

    :rho_c: float in range [rho.min(), rho.max()]

    :nu: float

    :zeta: float

    :returns: TODO

    """

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
    dy = np.power(l_mesh, - zeta / nu) * a

    return ScaledData(x, y, dy)
