# ------------------- The OpenQuake Model Building Toolkit --------------------
# Copyright (C) 2024 GEM Foundation
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

"""
:module:`openquake.sub.misc.utils`
"""

import numpy as np

# _read_profile imported but never used
from openquake.sub.misc.profile import _read_profile
from openquake.sub.misc.edge import create_faults


def get_centroids(lons, lats, deps):
    """
    Computes the centroids of all the ruptures admitted by the fault described
    by the coordinates in the three input arrays.
    """
    cen = np.zeros((lons.shape[0] - 1, lons.shape[1] - 1, 3))
    cen[:, :, :] = np.nan
    for j in range(0, lons.shape[1] - 1):
        for i in range(0, lons.shape[0] - 1):
            if np.all(np.isfinite(lons[i:i + 1, j:j + 1])):
                dst1 = ((lons[i, j] - lons[i + 1, j + 1])**2 +
                        (lats[i, j] - lats[i + 1, j + 1])**2 +
                        (deps[i, j] - deps[i + 1, j + 1])**2)**.5
                dst2 = ((lons[i + 1, j] - lons[i, j + 1])**2 +
                        (lats[i + 1, j] - lats[i, j + 1])**2 +
                        (deps[i + 1, j] - deps[i, j + 1])**2)**.5
                if dst1 > dst2:
                    x = (lons[i, j] + lons[i + 1, j + 1]) / 2
                    y = (lats[i, j] + lats[i + 1, j + 1]) / 2
                    z = (deps[i, j] + deps[i + 1, j + 1]) / 2
                else:
                    x = (lons[i + 1, j] + lons[i, j + 1]) / 2
                    y = (lats[i + 1, j] + lats[i, j + 1]) / 2
                    z = (deps[i + 1, j] + deps[i, j + 1]) / 2

                # Save the centroid
                cen[i, j, 0] = x
                cen[i, j, 1] = y
                cen[i, j, 2] = z
    return cen


def create_inslab_meshes(msh, dips, slab_thickness, sampling):
    """
    :param msh:
    :param dips:
    :param slab_thickness:
    :param sampling:
    """
    oms = {}
    for dip in dips:
        for i in range(0, msh.shape[0]):
            out = create_faults(msh, i, slab_thickness, dip, sampling)
            for subfault in out:
                if dip not in oms:
                    oms[dip] = [subfault]
                else:
                    oms[dip].append(subfault)
    return oms


def get_min_max(msh, lmsh):
    """
    :param msh:
    :param lmsh:
    """
    #
    # Creating the 3d mesh filling the slab volume
    mx = msh[:, :, 0]
    xx = np.isfinite(mx)
    my = msh[:, :, 1]
    yy = np.isfinite(my)
    mz = msh[:, :, 2]
    zz = np.isfinite(mz)

    lmx = lmsh[:, :, 0]
    lmy = lmsh[:, :, 1]
    lmz = lmsh[:, :, 2]
    lzz = np.isfinite(lmz)

    np.testing.assert_equal(xx, zz)
    np.testing.assert_equal(yy, zz)
    np.testing.assert_equal(zz, lzz)

    milo = np.amin((np.amin(mx[zz]), np.amin(lmx[zz]))) - 0.1
    mila = np.amin((np.amin(my[zz]), np.amin(lmy[zz]))) - 0.1
    mide = np.amin((np.amin(mz[zz]), np.amin(lmz[zz])))

    malo = np.amax((np.amax(mx[zz]), np.amax(lmx[zz]))) + 0.1
    mala = np.amax((np.amax(my[zz]), np.amax(lmy[zz]))) + 0.1
    made = np.amax((np.amax(mz[zz]), np.amax(lmz[zz])))

    return milo, mila, mide, malo, mala, made
