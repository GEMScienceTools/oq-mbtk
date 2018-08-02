# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2018 GEM Foundation
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.

from setuptools import setup, find_packages
import re
import sys

url = "https://github.com/GEMScienceTools/oq-mbtk"

README = """
Python and OpenQuake-based Toolkit for the construction of Seismic Source
Models
Copyright (C) 2017-2018 GEM Foundation
"""


def get_version():
    version_re = r"^__version__\s+=\s+['\"]([^'\"]*)['\"]"
    version = None

    package_init = 'openquake/mbt/__init__.py'
    for line in open(package_init, 'r'):
        version_match = re.search(version_re, line, re.M)
        if version_match:
            version = version_match.group(1)
            break
    else:
        sys.exit('__version__ variable not found in %s' % package_init)

    return version

version = get_version()

setup(
    name='openquake.mbt',
    version=version,
    description=README,
    url=url,
    packages=find_packages(exclude=['tests', 'tests.*']),
    # Minimal requirements, for a complete list see requirements-*.txt
    # matplotlib is brought by the openquake engine
    install_requires=[
        'openquake.engine',
        'nbstripout >= 0.3.0',
        'jupyter_client >= 5.0.0',
        'ipykernel >= 4.8.0',
        'ipython >= 6.2.0',
        'notebook >= 5.0.0',
        'nbformat',
        'GDAL',
        'pyproj',
        'rtree',
        'nbconvert',
        'prettytable',
        'geopandas'
    ],
    python_requires='>=3.5',
    author='GEM Foundation',
    author_email='hazard@globalquakemodel.org',
    maintainer='GEM Foundation',
    maintainer_email='hazard@globalquakemodel.org',
    classifiers=(
        'Development Status :: 4 - Beta',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Affero General Public License v3',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
    ),
    namespace_packages=['openquake'],
    keywords="seismic hazard",
    license="AGPL3",
    platforms=["any"],
    package_data={"openquake.mbt": [
        "README.md", "LICENSE"]},
    include_package_data=True,
    zip_safe=False,
)
