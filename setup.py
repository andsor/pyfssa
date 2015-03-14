#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os
import inspect
import os.path

import version

__location__ = os.path.join(os.getcwd(), os.path.dirname(
    inspect.getfile(inspect.currentframe())))
here = __location__


def get_version(*file_paths):
    try:
        # read version from git tags
        ver = version.read_version_git()
    except:
        # read version from file
        ver = version.read_version_file(here, *file_paths)
    else:
        # write version to file if we got it successfully from git
        version.write_version_file(ver, here, *file_paths)
    return ver


def get_install_requirements(path):
    content = open(os.path.join(__location__, path)).read()
    return [req for req in content.split("\\n") if req != '']


def setup_package():
    # Get the long description from the relevant file
    with open(os.path.join(here, 'README.rst')) as f:
        long_description = f.read()

    setup(
        name='fssa',
        version=get_version('fssa', '_version.py'),
        description=('Finite-size scaling analysis at phase transitions'),
        long_description=long_description,
        url='http://github.com/andsor/pyfss',
        author='Andreas Sorge',
        author_email='as@asorge.de',
        license='ASL',
        # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3.4',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Scientific/Engineering :: Physics',
        ],
        keywords='',
        packages=find_packages(exclude=['docs', 'tests*']),
        install_requires=get_install_requirements('requirements.txt') + [
            'setuptools_git'
        ],
        include_package_data=True,  # include everything in source control

        # but exclude these files
        exclude_package_data={'': [
            '.gitignore',
        ]},
        package_data={
            'fssa': [],
        },

        setup_requires=['setuptools_git >= 1.1', ],
        extras_require={
            'doc': get_install_requirements('requirements-doc.txt'),
        },
        test_suite="tests"
    )


if __name__ == "__main__":
        setup_package()
