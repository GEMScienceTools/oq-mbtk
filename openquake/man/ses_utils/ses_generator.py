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
simulate 1000 Stochastic Event Set (SES) by sampling earthquake ruptures 
from configured seismic sources using Monte Carlo sampling.
"""

import numpy as np

from openquake.man.ses_utils.ses_source import make_area_source


def prepare_source_for_sampling(src, samples=1):
    """
    Prepares a source for rupture sampling.

    It assigns the minimum metadata normally initialized by the
    OpenQuake Engine so that ``sample_ruptures()`` can be called directly
    on a source created through hazardlib.

    :param src:
        An OpenQuake seismic source.
    :param samples:
        Number of source-model samples associated with the source. Default is 1.
    :returns:
        The prepared source with the required sampling metadata.
    """
    src.id = 0
    src.offset = 0
    src.smweight = 1.0
    src.sampling = {
        'samples': np.array([samples], dtype=np.uint32),
        'trt_smr': np.array([0], dtype=np.uint32),
    }
    return src

def ses_from_area_source(fname: str, mfd, hdd=None, num_ses=1000, ses_seed=0, samples=1):
    """
    Generates a stochastic event set from an Area Source.

    :param fname:
        Path to the GeoJSON file containing the area-source geometry.
    :param mfd:
        An OpenQuake magnitude-frequency distribution instance.
    :param hdd:
        Optional hypocentral-depth probability mass function.
    :param num_ses:
        Number of stochastic event sets to simulate.
    :param ses_seed:
        Seed used for stochastic rupture sampling.
    :param samples:
        Number of source-model samples associated with the source.
    :returns:
        A list of :class:`~openquake.hazardlib.source.rupture.EBRupture`
        objects representing the simulated stochastic event set.
    """
    src = make_area_source(mfd, polygon_fname=fname, hdd=hdd)
    prepare_source_for_sampling(src, samples=samples)
    return src.sample_ruptures(num_ses=num_ses, ses_seed=ses_seed)
