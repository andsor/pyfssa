"""
A scientific Python package for finite-size scaling analysis
"""
from __future__ import absolute_import
from ._version import get_versions

from .fss import scaledata, quality, autoscale
__version__ = get_versions()['version']
del get_versions
