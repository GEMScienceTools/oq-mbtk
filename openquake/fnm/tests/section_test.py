#!/usr/bin/env python
# coding: utf-8

import unittest
import pathlib
import numpy as np

from openquake.fnm.section import split_into_subsections
from openquake.hazardlib.geo import Line, Point
from openquake.hazardlib.geo.surface.kite_fault import (
    KiteSurface, get_profiles_from_simple_fault_data)
from openquake.fnm.importer import kite_surfaces_from_geojson
from openquake.fnm.plot import plot

HERE = pathlib.Path(__file__).parent
PLOTTING = False

# TODO:
# - Add a test for section 5 in Kunlun using [12, -1] as parameters


class TestCreateSubSections(unittest.TestCase):

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

    def test_create_subsections_from_one_section(self):
        """ test the creation of subsections using a simple surface """
        # Create subsections given a number of cells along the strike and
        # dip
        nc_strike = -1
        nc_dip = 2
        mesh = self.surf.mesh
        tmp_ul = split_into_subsections(mesh, nc_strike, nc_dip)

        # The mesh has five columns and six rows (this means 4 cells along
        # strike and 5 cells along dip). Since each subsection must have 2
        # cells along strike and two along the dip, the expected number
        # corresponds to 6 (i.e. 2.5 rows times 2 cols, the former
        # considered as three)
        msg = 'The number of subsections does not match the expected number'
        self.assertEqual(tmp_ul.shape[0]*tmp_ul.shape[1], 6, msg)

        if PLOTTING:
            _plot_mesh(mesh)

    def test_create_subsections_one_section_full_width(self):
        """ test the creation of subsections with width = section width """
        # Create subsections given a number of cells along the strike and
        # dip
        nc_strike = 1
        nc_dip = -1
        mesh = self.surf.mesh
        tmp_ul = split_into_subsections(mesh, nc_strike, nc_dip)

        # Test the mesh
        np.testing.assert_array_equal([6, 5], mesh.lons.shape)

        # These are full width subsections
        msg = 'The number of subsections does not match the expected number'
        self.assertEqual(tmp_ul.shape[0]*tmp_ul.shape[1], 4, msg)

        # Test number of cells
        msg = 'The number of cells along the strike is wrong'
        self.assertEqual(tmp_ul[0, 0, 2], 1, msg)
        msg = 'The number of cells along the dip is wrong'
        self.assertEqual(tmp_ul[0, 0, 3], 5, msg)

        aae = np.testing.assert_almost_equal
        expected = np.ones((6)) * 10.0
        aae(mesh.lons[:, 0], expected, decimal=4)

    @unittest.skip('to review')
    def test_create_subsections_kunlun_11(self):
        size = [-.5, -1]
        fname = HERE / 'data' / 'kunlun_faults.geojson'
        surfs = kite_surfaces_from_geojson(fname, 2)
        surf = surfs[11]
        mesh = surf.mesh
        tmp_ul = split_into_subsections(mesh, nc_stk=size[0], nc_dip=size[1])
        if PLOTTING:
            plot([mesh], subsections=tmp_ul)
        msg = 'The number of subsections does not match the expected number'
        self.assertEqual(tmp_ul.shape[0]*tmp_ul.shape[1], 3, msg)

    def test_create_subsections_kunlun_0(self):
        size = [-.5, -1]
        fname = HERE / 'data' / 'kunlun_faults.geojson'
        surfs = kite_surfaces_from_geojson(fname, 2)
        surf = surfs[0]
        mesh = surf.mesh
        tmp_ul = split_into_subsections(mesh, nc_stk=size[0], nc_dip=size[1])
        if PLOTTING:
            plot([mesh], subsections=tmp_ul)
        msg = 'The number of subsections does not match the expected number'
        self.assertEqual(tmp_ul.shape[0]*tmp_ul.shape[1], 31, msg)


def _plot_mesh(mesh):
    import pyvista as pv
    pl = pv.Plotter()
    scl = 1./100
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
