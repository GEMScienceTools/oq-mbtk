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
                                                      get_initial_traces, 
                                                      rotate)
from openquake.sub.cross_sections import CrossSection, Slab2pt0


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
        
        x = [164.45, 165.45, 167.45, 220.25, 223.15]
        y = [49.4, 50.1, 53.3, 60.15, 65.75]
        xx, yy = np.meshgrid(x, y)
        
        depths = [50, 80, 130, 200, 350]
        tmp = zip(xx.flatten(), yy.flatten(), depths)
        
        depths = [[x, y, z] for x, y, z in tmp]
        depths = np.array(depths)
        
        #box = [164.45, 223.15]
        #boy = [49.4, 65.75]
        
        cx = np.mean([bb[0:2]])
        cy = np.mean([bb[2:4]])

        # Rotate the grid with the fault information and get the bounding box
        rx, ry = rotate(xx.flatten(), yy.flatten(), cx, cy, -dip_dir)
        #bb = tmp_bb
        r_bb, _ = get_bounding_box(rx, ry, delta=1.)

        # Compute the rotated and buffered bounding box
        dlt = 3.0
        coox = [r_bb[0]-dlt, r_bb[1]+dlt, r_bb[1]+dlt, r_bb[0]-dlt]
        cooy = [r_bb[2]-dlt, r_bb[2]-dlt, r_bb[3]+dlt, r_bb[3]+dlt]
        nbbx, nbby = rotate(coox, cooy, cx, cy, dip_dir)
        
        
        traces, plen = get_initial_traces(nbbx, nbby, dip_dir, spacing)

        css = []
        for pro in traces:
            xlo = pro[0, 0]
            xla = pro[0, 1]
            xlo = xlo if xlo < 180 else xlo - 360
            cs = CrossSection(xlo, xla, plen, dip_dir)
            css.append(cs)
        
        slb = Slab2pt0(depths, css)
        slb.compute_profiles(spacing/2)
        
        if True:
            import pygmt
            import netCDF4

            grid = True

            # Reading file with strike values
            #if grid:
            #    fname = HERE / 'data' / 'slab2pt0' / 'alu_slab2_dep_02.23.18.grd'
            #    dat = netCDF4.Dataset(fname)
            #    xx = np.array(dat.variables['x'])
            #    yy = np.array(dat.variables['y'])
            #    z = np.array(dat.variables['z'])
            #    x, y = np.meshgrid(xx, yy)
            #    mask = np.where(np.isfinite(z))
            #    z = z[mask]
            #    x = x[mask]
            #    y = y[mask]
            #    tmp = zip(x.flatten(), y.flatten(), z.flatten())
            #    dd = np.array([[x, y, z] for x, y, z in tmp])
            #    xx, yy = np.meshgrid(x, y)

            fig = pygmt.Figure()
            buf = 5
            reg = [bb[0]-buf, bb[1]+buf, bb[2]-buf, bb[3]+buf]
            loc = np.mean([bb[0], bb[1]])
            lac = np.mean([bb[2], bb[3]])
            prjstr = f"D{loc}/{lac}/{bb[2]}/{bb[3]}/15c"
            fig.basemap(region=reg, projection=prjstr, frame=True)

            if grid:
                idx = np.random.choice(np.arange(0, len(depths)), 1000)
                fig.plot(x=depths[idx, 0], y=depths[idx, 1], style="c0.01c", pen="grey")

            for pro in traces:
                tmpx = pro[:, 0]
                tmpx[tmpx < 0] = tmpx[tmpx < 0] + 360
                fig.plot(x=tmpx, y=pro[:, 1], pen="red")
            
            for key in slb.profiles:
                pro = slb.profiles[key]
                if pro.shape[0] > 0:
                    fig.plot(x=pro[:, 0],
                         y=pro[:, 1],
                         color=pro[:, 2],
                         cmap=True,
                         style="h0.025c",
                         pen='black')

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