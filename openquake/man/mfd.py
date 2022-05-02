# ------------------- The OpenQuake Model Building Toolkit --------------------
# Copyright (C) 2022 GEM Foundation
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
# -----------------------------------------------------------------------------
# vim: tabstop=4 shiftwidth=4 softtabstop=4
# coding: utf-8

import numpy

from openquake.mbt.tools.mfd import mag_to_mo
from openquake.hazardlib.mfd import TruncatedGRMFD, EvenlyDiscretizedMFD


def get_rates_within_m_range(mfd, mmint=0.0, mmaxt=11.0):
    """
    :parameter mfd:
    :parameter mmint:
    :parameter mmaxt:
    """
    rtes = numpy.array(mfd.get_annual_occurrence_rates())
    idx = numpy.nonzero((rtes[:, 0] > mmint) & (rtes[:, 0] < mmaxt))
    return sum(rtes[idx[0], 1])


def get_moment_from_mfd(mfd):
    """
    This computed the total scalar seismic moment released per year by a
    source

    :parameter mfd:
        An instance of openquake.hazardlib.mfd
    :returns:
        A float corresponding to the rate of scalar moment released
    """
    if isinstance(mfd, TruncatedGRMFD):
        return mfd._get_total_moment_rate()
    elif isinstance(mfd, EvenlyDiscretizedMFD):
        occ_list = mfd.get_annual_occurrence_rates()
        mo_tot = 0.0
        for occ in occ_list:
            mo_tot += occ[1] * mag_to_mo(occ[0])
    return mo_tot
