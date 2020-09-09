import pathlib
import unittest
import numpy
from openquake.hazardlib.geo.point import Point
from openquake.hazardlib.geo.polygon import Polygon
from openquake.hmtk.parsers.catalogue import CsvCatalogueParser
from openquake.mbt.tools.smooth import Smoothing
from openquake.mbt.tests import __file__ as tests__init__

TESTDIR = pathlib.Path(tests__init__).parent


def setUpMesh(lons, lats):
    # make the mesh
    points = []
    for lon, lat in zip(lons, lats):
        points.append(Point(lon, lat))

    newpoly = Polygon(points)
    # must use a fine mesh spacing to preserve symmetry (because
    # of auto-generated mesh coordinates)
    new_polygon_mesh = newpoly.discretize(2)

    return new_polygon_mesh


def check_symmetry(mesh, values):
    # take the mesh row with the median longitude
    row = numpy.median(mesh.lats)
    idx = numpy.where(mesh.lats == row)

    # smooth values from that row
    vals_row = values[idx]

    # find the center: the point with the highest smoothing value
    center = int(numpy.where(vals_row == max(vals_row))[0])

    # take the smoothed values on either side of the center
    center_left = center - int(numpy.floor(0.45*len(vals_row)))
    center_right = center + int(numpy.floor(0.45*len(vals_row)))
    vals_west = vals_row[center_left:center+1]
    vals_east = numpy.flip(vals_row[center:center_right+1])

    # check that the smoothing values are approximately symmetrical
    max_vals = max(vals_row)
    diff = vals_east-vals_west
    max_diff = max(abs(diff))

    # compute the largest percentage difference
    p_diff = max_diff/max_vals*100

    return p_diff


class SmoothTestCase(unittest.TestCase):
    """
    Tests that smoothing values sum to N earthquakes and that
    smoothing distribution is symmetric. catalogues use a
    single earthquake with latitude assigned the median latitude
    of the mesh
    """
    def test_01(self):
        # create the mesh
        # set bounds crossing the IDL
        lons = [150, 151, 151, 150, 150]
        lats = [-15.5, -15.5, -16.0, -16.0, -15.5]
        mesh = setUpMesh(lons, lats)

        # read in the test catalogue
        cat_filename = TESTDIR / 'data/tools/test_catalogue.csv'
        catalogue_parser = CsvCatalogueParser(cat_filename)
        cat = catalogue_parser.read_file()

        # smooth the catalogue onto the mesh grid
        smooth = Smoothing(cat, mesh, 20)
        values = smooth.gaussian(50, 20)

        # check that smoothed values sum to 1.0
        self.assertAlmostEqual(sum(values), len(cat.data['depth']), 5)

        # check that the Gaussian distribution works across IDL:
        # assert that max %-difference is < 1
        self.assertLess(check_symmetry(mesh, values), 1)

    def test_IDL_02(self):
        # tests that the smoothing works accross the IDL
        # set bounds crossing the IDL
        lons = [-179.5, 179.5, 179.5, -179.5, -179.5]
        lats = [-15.5, -15.5, -16.0, -16.0, -15.5]
        mesh = setUpMesh(lons, lats)

        # read in the test catalogue
        cat_filename = TESTDIR / 'data/tools/idl_test_catalogue.csv'
        catalogue_parser = CsvCatalogueParser(cat_filename)
        cat = catalogue_parser.read_file()

        # smooth the catalogue onto the mesh grid
        smooth = Smoothing(cat, mesh, 20)
        values = smooth.gaussian(50, 20)

        # check that smoothed values sum to 1.0
        self.assertAlmostEqual(sum(values), len(cat.data['depth']), 5)

        # check that the Gaussian distribution works across IDL:
        # assert that max %-difference is < 1
        self.assertLess(check_symmetry(mesh, values), 1)
