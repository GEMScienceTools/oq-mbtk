import numpy as np
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
    lons = [-176.0, 176.0, 176.0, -176.0, -176.0]
    lats = [-15.5, -15.5, -16.5, -16.5, -15.5]
    #
    # make the mesh
    points = []
    for lon,lat in zip(lons,lats):
        points.append(Point(lon,lat))

    newpoly = Polygon(points)
    new_polygon_mesh = newpoly.discretize(5)
    import pdb; pdb.set_trace()

    return new_polygon_mesh


class SmoothTestCaseIDL(unittest.TestCase):
    """
    """

    def testcase01(self):
        """
        """
        mesh = setUpMesh()
        cat_filename = '../data/tools/idl_test_catalogue.csv' 
        catalogue_parser = CsvCatalogueParser(cat_filename)
        cat = catalogue_parser.read_file()

        smooth = Smoothing(cat, mesh, 20)
        values = smooth.gaussian(50, 20)
        print('sum:', sum(values))

