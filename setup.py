import ez_setup
ez_setup.use_setuptools()

from setuptools import setup, find_packages
from codecs import open  # To use a consistent encoding
import os
import os.path

import version

here = os.path.abspath(os.path.dirname(__file__))


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

# Get the long description from the relevant file
with open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='fss',
    version=get_version('fss', '_version.py'),
    description=('Finite-size scaling analysis at phase transitions'),
    long_description=long_description,
    url='',
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
    install_requires=[
        'numpy>=1.8',
        'future>=0.13'
    ],
    include_package_data=True,  # include everything in source control

    # but exclude these files
    exclude_package_data={'': [
        '.gitignore',
    ]},
    package_data={
        'fss': [],
    },

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages.
    # see http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    data_files=[],

    setup_requires=['setuptools_git >= 1.1', ],
    extras_require={
        'doc': [
            "sphinx>=1.2.2",
            "sphinx_rtd_theme>=0.1.6",
            "sphinxcontrib-bibtex>=0.3.1",
            "numpydoc>=0.5",
        ],
    },
    test_suite="tests"
)
