# coding: utf-8

import pathlib
import unittest
import numpy as np

from openquake.hazardlib.geo import Line, Point
from openquake.hazardlib.geo.surface.kite_fault import (
    KiteSurface, get_profiles_from_simple_fault_data)

from openquake.fnm.section import split_into_subsections
from openquake.fnm.rupture import (
    get_ruptures_section, _get_ruptures_first_level, get_ruptures_area,
    _get_rupture_area)
from openquake.fnm.importer import kite_surfaces_from_geojson
from openquake.fnm.fault_system import get_fault_system
from openquake.fnm.tests.connection_test import _get_surfs

HERE = pathlib.Path(__file__).parent
PLOTTING = False


class TestGetAreaSubsection(unittest.TestCase):

    def test_get_area_rupture_one_section(self):
        """ compute the area for a rupture on one section """

        # The mesh spacing is 1 km so each cell should have an area of about
        # one square kilometer
        surfs = _get_surfs()

        nc_strike = 12
        nc_dip = -1

        mesh = surfs[0].mesh
        tmp_ul = split_into_subsections(mesh, nc_strike, nc_dip)
        rups = get_ruptures_section(np.array(tmp_ul))

        # The mesh has 12x24 cells. Since each cell has an area of about 1km
        # the total area if this section is about 288 km2
        areas = []
        for i, rup in enumerate(rups):
            area = get_ruptures_area(surfs, rups[rups[:, 4] == i, :])
            areas.append(area)
        tmp = areas[0] + areas[2]
        np.testing.assert_array_almost_equal(tmp, areas[1], decimal=0)

        # tmp_areas = np.take(areas, (0, 4, 7, 9))
        # np.testing.assert_almost_equal(areas[3], np.sum(tmp_areas))
        # TODO compute the length of the trace and compute the area

    def test_get_area_rupture_multiple_sections(self):
        """ compute the area of a rupture on multiple sections """

        # The mesh spacing is 1 km so each cell should have an area of about
        # one square kilometer
        surfs = _get_surfs()

        # Get ruptures for one section
        rups = np.array([[0., 12., 12., 12.,  2.,  1.,  0.],
                         [0.,  0., 24., 12.,  2.,  1.,  1.]])

        # Get the area for one subsection
        area2 = _get_rupture_area(surfs, rups[rups[:, 4] == 2, :])

        expected = 143.8 + 287.30
        np.testing.assert_almost_equal(expected, area2, decimal=1)


class TestGetAreaRuptures(unittest.TestCase):

    def test_get_areas_ss_rups(self):
        aratios = np.array([0, 100], dtype=int)
        subs_size = [-0.5, -1]
        surfs = _get_surfs()
        fault_system = get_fault_system(surfs, subs_size)
        rups = _get_ruptures_first_level(fault_system, aratios)
        areas = get_ruptures_area(surfs, rups)


class TestCreateRupturesSection(unittest.TestCase):

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

    def test_generate_ruptures_one_section_1row(self):
        """ Test the construction of all the ruptures for one section """

        # Create subsections given a number of cells along the strike and
        # dip. The mesh contains 6 rows and 5 columns.
        nc_strike = 1
        nc_dip = -1
        mesh = self.surf.mesh
        # nc_stk_dip is [1, 5], that is one cell along columns and 5 along rows
        tmp_ul = split_into_subsections(mesh, nc_strike, nc_dip)

        if 0:
            from openquake.fnm.tests.mesh_test import plot_mesh
            plot_mesh(mesh)

        # Create the ruptures. Each rupture is represented by four values:
        # The first one defines the index of the row of the starting
        # subsection, the second is the index of the column of the starting
        # subsection, the third is the lenght of the rupture (number of
        # subsections) and the last one is the width of the rupture (number
        # of subsections
        rups = get_ruptures_section(tmp_ul)

        # The number of ruptures admitted by this section is 10
        np.testing.assert_equal(len(rups), 10)

        # We have only one row of ruptures (first index always 0). All the
        # ruptures have the same width (equal to 5).
        expected = np.array([[0, 0, 1, 5, 0, 1, 0, 0],
                             [0, 0, 2, 5, 1, 1, 0, 1],
                             [0, 0, 3, 5, 2, 1, 0, 2],
                             [0, 0, 4, 5, 3, 1, 0, 3],
                             [0, 1, 1, 5, 4, 1, 0, 4],
                             [0, 1, 2, 5, 5, 1, 0, 5],
                             [0, 1, 3, 5, 6, 1, 0, 6],
                             [0, 2, 1, 5, 7, 1, 0, 7],
                             [0, 2, 2, 5, 8, 1, 0, 8],
                             [0, 3, 1, 5, 9, 1, 0, 9]]
                            )
        np.testing.assert_equal(rups, expected)

    def test_generate_ruptures_one_section_3rows(self):

        # Create subsections given a number of cells along the strike and
        # dip. The mesh has shape 6 x 5 which means we have ruptures
        # distributed on three rows and 5 columns.
        nc_strike = 1
        nc_dip = 2
        mesh = self.surf.mesh

        # Computes subsections
        tmp_ul = split_into_subsections(mesh, nc_strike, nc_dip)

        # Create the ruptures. Each rupture is represented by four values:
        # The first one defines the index of the column of the starting
        # subsection, the second is the index of the row of the starting
        # subsection, the third is the lenght of the rupture (number of
        # subsections) and the last one is the width of the rupture (number
        # of subsections
        rups = get_ruptures_section(tmp_ul)

        # Check the number of ruptures admitted by this sections.
        np.testing.assert_equal(len(rups), 60)

        #  Checking some of them
        expected = np.array([[0, 0, 1, 2, 0, 1, 0, 0],
                             [0, 0, 1, 4, 1, 1, 0, 1],
                             [0, 0, 1, 5, 2, 1, 0, 2],
                             [0, 0, 2, 2, 3, 1, 0, 3],
                             [0, 0, 2, 4, 4, 1, 0, 4],
                             [0, 0, 2, 5, 5, 1, 0, 5],
                             [0, 0, 3, 2, 6, 1, 0, 6],
                             [0, 0, 3, 4, 7, 1, 0, 7],
                             [0, 0, 3, 5, 8, 1, 0, 8],
                             [0, 0, 4, 2, 9, 1, 0, 9]])
        np.testing.assert_equal(rups[:10], expected)

    def test_generate_ruptures_one_section_aratio_3rows(self):

        # Create subsections given a number of cells along the strike and dip
        nc_strike = 1
        nc_dip = 2
        mesh = self.surf.mesh
        tmp_ul = split_into_subsections(mesh, nc_strike, nc_dip)

        # Get the ruptures
        aspect_ratios = np.array([1.0, 10.0])
        rups = get_ruptures_section(tmp_ul, aspect_ratios=aspect_ratios)

        # Check that the length of each rupture is equal or larger than the
        # width
        np.testing.assert_equal(np.all(rups[:, 2] >= rups[:, 3]), True)

        #  Checking indexes of ruptures
        expected = np.array([[0, 0, 2, 2,  0, 1, 0, 0],
                             [0, 0, 3, 2,  1, 1, 0, 1],
                             [0, 0, 4, 2,  2, 1, 0, 2],
                             [0, 0, 4, 4,  3, 1, 0, 3],
                             [0, 1, 2, 2,  4, 1, 0, 4],
                             [0, 1, 3, 2,  5, 1, 0, 5],
                             [0, 2, 2, 2,  6, 1, 0, 6],
                             [2, 0, 2, 2,  7, 1, 0, 7],
                             [2, 0, 3, 2,  8, 1, 0, 8],
                             [2, 0, 3, 3,  9, 1, 0, 9],
                             [2, 0, 4, 2, 10, 1, 0, 10],
                             [2, 0, 4, 3, 11, 1, 0, 11]])

        np.testing.assert_equal(rups[:12], expected)

    def test_generate_ruptures_kunlun(self):

        nc_strike = 10
        nc_dip = -1

        fname = HERE / 'data' / 'kunlun_faults.geojson'
        surfs = kite_surfaces_from_geojson(fname, 2)

        mesh = surfs[0].mesh
        tmp_ul = split_into_subsections(mesh, nc_strike, nc_dip)

        if PLOTTING:
            from openquake.fnm.tests.mesh_test import plot_mesh
            plot_mesh(mesh)
