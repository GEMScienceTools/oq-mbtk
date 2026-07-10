# ------------------- The OpenQuake Model Building Toolkit --------------------
# Copyright (C) 2026 GEM Foundation and Électricité de France
#           _______  _______        __   __  _______  _______  ___   _
#          |       ||       |      |  |_|  ||  _    ||       ||   | | |
#          |   _   ||   _   | ____ |       || |_|   ||_     _||   |_| |
#          |  | |  ||  | |  ||____||       ||       |  |   |  |      _|
#          |  |_|  ||  |_|  |      |       ||  _   |   |   |  |     |_
#          |       ||      |       | ||_|| || |_|   |  |   |  |    _  |
#          |_______||____||_|      |_|   |_||_______|  |___|  |___| |_|
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# This script is produced within the scope of Work Package 5, named Simulation 
# platform, under SIGMA3 project. For more detailed information about 
# the project, please visit to https://sigma-programs.com/.
# -----------------------------------------------------------------------------
# vim: tabstop=4 shiftwidth=4 softtabstop=4
# coding: utf-8

"""
module :mod:`openquake.man.ses_utils.ses_generator` provides a function to 
generate 1000 years Stochastic Event Sets (SES) by sampling earthquake ruptures 
from configured seismic sources using Monte Carlo sampling.
"""

from openquake.man.ses_utils.ses_source import get_area_source


def ses_from_area_source(fname: str, mfd, hdd=None):
    """
    Generates a stochastic event set from an Area Source

    :param fname:
        A string with the name of the file with the geometry of
        the area source. It can be of any format (e.g. shapefile, 
        geojson, geopackage).
    :param mfd:
        An instance of a concrete class of 
        :class:`openquake.hazardlib.mfd.base.BaseMFD`
    :param hdd:
        An instance of :class:`openquake.hazardlib.pmf.PMF` instance 
        describing the hypocentral depth distribution. For example:
        pmf = PMF([(0.3, 5.0), (0.7, 10.0)])
    """
    src = get_area_source(mfd, polygon_fname=fname, hdd=hdd)
    src.smweight = 1.0
    rups = []
    result = src.sample_ruptures(1000, 0)
    if result is not None:
        rups.extend(result)

    return rups
