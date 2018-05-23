import os
import numpy
import unittest

from pathlib import Path

from openquake.mbt.tools.tr.catalogue import get_catalogue
from openquake.mbt.tools.area import create_catalogue, load_geometry_from_shapefile
from openquake.mbt.oqt_project import OQtModel, OQtSource

BASE_DATA_PATH = os.path.dirname(__file__)


class SelectEarthquakesWithinAreaTestCase(unittest.TestCase):
    """
    This class tests the selection of earthquakes within a polygon
    """

    def setUp(self):
        self.catalogue_fname = None

    def testcase01(self):
        """
        Simple area source
        """
        datafold = '../data/tools/area/case01/'
        datafold = os.path.join(BASE_DATA_PATH, datafold)
        #
        # create the source and set the geometry
        model = OQtModel('0', 'test')
        #
        # read geometries
        shapefile = os.path.join(datafold, 'polygon.shp')
        srcs = load_geometry_from_shapefile(shapefile)
        model.sources = srcs
        #
        # read catalogue
        self.catalogue_fname = os.path.join(datafold, 'catalogue.csv')
        cat = get_catalogue(self.catalogue_fname)
        #
        # select earthquakes within the polygon
        scat = create_catalogue(model, cat, ['1'])
        #
        # cleaning
        os.remove(os.path.join(datafold, 'catalogue.pkl'))
        #
        # check
        self.assertEqual(len(scat.data['longitude']), 5)


    def testcase02(self):
        """
        Area source straddling the IDL
        """
        datafold = '../data/tools/area/case02/'
        datafold = os.path.join(BASE_DATA_PATH, datafold)
        #
        # create the source and set the geometry
        model = OQtModel('0', 'test')
        #
        # read geometries
        shapefile = os.path.join(datafold, 'area_16.shp')
        srcs = load_geometry_from_shapefile(shapefile)
        model.sources = srcs
        #
        # read catalogue
        self.catalogue_fname = os.path.join(datafold, 'catalogue.csv')
        cat = get_catalogue(self.catalogue_fname)
        #
        # select earthquakes within the polygon
        scat = create_catalogue(model, cat, ['16'])
        #
        # cleaning
        os.remove(os.path.join(datafold, 'catalogue.pkl'))
        #
        # check
        self.assertEqual(len(scat.data['longitude']), 4)
