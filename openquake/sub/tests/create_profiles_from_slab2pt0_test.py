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

import pathlib
import tempfile
import unittest
import numpy as np
from openquake.sub.get_profiles_from_slab2pt0 import (get_profiles,
                                                      get_bounding_box,
                                                      get_initial_traces)

HERE = pathlib.Path(__file__).parent.resolve()


class GetBBox(unittest.TestCase):

    def test_points_across_idl(self):
        """ test calculation of bounding box for a set crossing IDL """
        lons = np.array([-179.5, -179.9, 179.4, 179.2])
        lats = np.ones_like(lons) * 85
        bb, idl = get_bounding_box(lons, lats)
        expected = np.array([179.2, 180.5, 85.0, 85.0])
        np.testing.assert_almost_equal(bb, expected)

    def test_points_west_of_idl(self):
        """ test calculation of bounding box for a set crossing IDL """
        lons = np.array([-170.5, -173.9, -179.2, -179.4])
        lats = np.ones_like(lons) * 85
        bb, idl = get_bounding_box(lons, lats)
        expected = np.array([180.6, 189.5, 85.0, 85.0])
        np.testing.assert_almost_equal(bb, expected)


class GetInitialProfiles(unittest.TestCase):

    def test_bb_across_idl(self):

        bb = [164.45, 223.15, 49.4, 65.75]

        dip_dir = 330
        spacing = 50.
        profiles, _ = get_initial_traces(bb, dip_dir, spacing)

        if True:
            import pygmt
            import netCDF4

            grid = False

            # Reading file with strike values
            if grid:
                fname = HERE / 'data' / 'slab2pt0' / 'alu_slab2_dep_02.23.18.grd'
                dat = netCDF4.Dataset(fname)
                xx = np.array(dat.variables['x'])
                yy = np.array(dat.variables['y'])
                z = np.array(dat.variables['z'])
                x, y = np.meshgrid(xx, yy)
                mask = np.where(np.isfinite(z))
                z = z[mask]
                x = x[mask]
                y = y[mask]
                tmp = zip(x.flatten(), y.flatten(), y.flatten())
                dd = np.array([[x, y, z] for x, y, z in tmp])
                xx, yy = np.meshgrid(x, y)

            fig = pygmt.Figure()
            buf = 5
            reg = [bb[0]-buf, bb[1]+buf, bb[2]-buf, bb[3]+buf]
            loc = np.mean([bb[0], bb[1]])
            lac = np.mean([bb[2], bb[3]])
            prjstr = f"D{loc}/{lac}/{bb[2]}/{bb[3]}/15c"
            fig.basemap(region=reg, projection=prjstr, frame=True)

            if grid:
                idx = np.random.choice(np.arange(0, len(dd)), 1000)
                fig.plot(x=dd[idx, 0], y=dd[idx, 1], style="c0.01c", pen="grey")

            for pro in profiles:
                tmpx = pro[:, 0]
                tmpx[tmpx < 0] = tmpx[tmpx < 0] + 360
                fig.plot(x=tmpx, y=pro[:, 1], pen="red")

            # This is the bounding box
            fig.plot(x=[bb[0], bb[1], bb[1], bb[0], bb[0]],
                     y=[bb[2], bb[2], bb[3], bb[3], bb[2]], pen="blue")

            fig.show()


class CreateProfilesFromSlab2pt0(unittest.TestCase):

    def test_aleutian(self):
        strike_fname = HERE / 'data' / 'slab2pt0' / 'alu_slab2_str_02.23.18.grd'
        depth_fname = HERE / 'data' / 'slab2pt0' / 'alu_slab2_dep_02.23.18.grd'
        spacing = 50.
        tmp_dir = tempfile.mkdtemp()
        fname_fig = pathlib.Path(tmp_dir) / 'figure.png'
        slb = get_profiles(strike_fname, depth_fname, spacing, fname_fig)

    def test_central_america(self):
        strike_fname = HERE / 'data' / 'slab2pt0' / 'cam_slab2_str_02.24.18.grd'
        depth_fname = HERE / 'data' / 'slab2pt0' / 'cam_slab2_dep_02.24.18.grd'
        spacing = 50.
        tmp_dir = tempfile.mkdtemp()
        fname_fig = pathlib.Path(tmp_dir) / 'figure.png'
        slb = get_profiles(strike_fname, depth_fname, spacing, fname_fig)
