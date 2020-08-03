import numpy 
import unittest

from pyproj import Proj
from openquake.hazardlib.geo.point import Point
from openquake.hazardlib.geo.polygon import Polygon
from openquake.mbt.tools.smooth import Smoothing
from openquake.hazardlib.geo.mesh import Mesh
from openquake.hmtk.parsers.catalogue import CsvCatalogueParser

def setUpMesh():
    #
    # set bounds crossing the IDL
    lons = [-179.5, 179.5, 179.5, -179.5, -179.5]
    lats = [-15.5, -15.5, -16.0, -16.0, -15.5]
    #
    # make the mesh
    points = []
    for lon,lat in zip(lons,lats):
        points.append(Point(lon,lat))

    newpoly = Polygon(points)
    # must use a fine mesh spacing to preserve symmetry (because
    # of auto-generated mesh coordinates)
    new_polygon_mesh = newpoly.discretize(2)

    return new_polygon_mesh


class SmoothTestCaseIDL(unittest.TestCase):
    """
    """

    def testIDL(self):
        """
        """
        # create the mesh 
        mesh = setUpMesh()

        # read in the test catalogue
        cat_filename = '../data/tools/idl_test_catalogue.csv' 
        catalogue_parser = CsvCatalogueParser(cat_filename)
        cat = catalogue_parser.read_file()

        # smooth the catalogue onto the mesh grid
        smooth = Smoothing(cat, mesh, 20)
        values = smooth.gaussian(50, 20)

        # check that smoothed values sum to 1.0
        self.assertAlmostEqual(sum(values),len(cat.data['depth']),5)

        # check that the Gaussian distribution works across IDL

        # take the mesh row with the median longitude  
        row = numpy.median(mesh.lats)
        idx = numpy.where(mesh.lats==row)

        # take the lats, lons, and smooth values from that row
        lats_row = mesh.lats[idx]
        lons_row = mesh.lons[idx]
        vals_row = values[idx]

        # find the center: the point with the highest smoothing value
        center = int(numpy.where(vals_row==max(vals_row))[0])

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

        # assert that max %-difference is < 1
        self.assertLess(p_diff,1)

