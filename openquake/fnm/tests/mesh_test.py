# ------------------- The OpenQuake Model Building Toolkit --------------------
# ------------------- FERMI: Fault nEtwoRks ModellIng -------------------------
# Copyright (C) 2023 GEM Foundation
#         .-.
#        /    \                                        .-.
#        | .`. ;    .--.    ___ .-.     ___ .-. .-.   ( __)
#        | |(___)  /    \  (   )   \   (   )   '   \  (''")
#        | |_     |  .-. ;  | ' .-. ;   |  .-.  .-. ;  | |
#       (   __)   |  | | |  |  / (___)  | |  | |  | |  | |
#        | |      |  |/  |  | |         | |  | |  | |  | |
#        | |      |  ' _.'  | |         | |  | |  | |  | |
#        | |      |  .'.-.  | |         | |  | |  | |  | |
#        | |      '  `-' /  | |         | |  | |  | |  | |
#       (___)      `.__.'  (___)       (___)(___)(___)(___)
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

import unittest
import numpy as np

from openquake.fnm.mesh import get_mesh_polygon, get_mesh_bb
from openquake.hazardlib.geo import Line, Point
from openquake.hazardlib.geo.surface.kite_fault import (
    KiteSurface, get_profiles_from_simple_fault_data)

PLOTTING = False


class TestGeom(unittest.TestCase):

    def setUp(self):
        usd = 0
        lsd = 12.0
        dip = 80.0
        mesh_spacing = 2.5
        profile_sd = 2.5
        edge_sd = 5.0

        # Create the Kite Fault Surface
        fault_trace = Line([Point(10.0, 45.0), Point(10.3, 45.0)])
        profiles = get_profiles_from_simple_fault_data(
            fault_trace, usd, lsd, dip, mesh_spacing)
        self.surf = KiteSurface.from_profiles(profiles, profile_sd, edge_sd)

    def test_get_polygon(self):
        """ test the calculation of the mesh polygon """
        # Get polygon
        poly = get_mesh_polygon(self.surf.mesh)

        expected = np.array([[10.253752, 45.000051,  0.],
                             [10.253762, 44.996147,  2.462019]])
        np.testing.assert_allclose(poly[4:6], expected, rtol=1e-5)

        if PLOTTING:
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, 1)
            plt.plot(self.surf.mesh.lons, self.surf.mesh.lats, '.')
            plt.plot(poly[:, 0], poly[:, 1], '-')
            plt.show()

    def test_get_bounding_box(self):
        """ test the calculation of the bounding box """
        # Get polygon
        bbox = get_mesh_bb(self.surf.mesh)

        expected = np.array([10.0, 10.253803, 44.980479, 45.000096])
        np.testing.assert_allclose(bbox, expected, atol=1e-2)

        if PLOTTING:
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, 1)
            plt.plot(self.surf.mesh.lons, self.surf.mesh.lats, '.')
            xlim = axs.get_xlim()
            ylim = axs.get_ylim()
            plt.hlines([bbox[2], bbox[3]], xlim[0], xlim[1], )
            plt.vlines([bbox[0], bbox[1]], ylim[0], ylim[1], )
            plt.show()


def plot_mesh(mesh):
    import pyvista as pv
    pl = pv.Plotter()
    scl = 1. / 100
    grd = np.zeros((mesh.lons.size, 3))
    grd[:, 0] = mesh.lons.flatten()
    grd[:, 1] = mesh.lats.flatten()
    grd[:, 2] = mesh.depths.flatten() * scl
    mesh = pv.PolyData(grd)
    pl.add_points(mesh.points, color='red', point_size=20)
    pl.view_isometric()
    pl.set_viewup((0, 0, 1))
    pl.show_grid()
    pl.show(interactive=True)
