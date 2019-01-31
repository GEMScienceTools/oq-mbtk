"""
Module create_map_functions_test
"""

import os
import unittest
import geopandas as gpd

from openquake.ghm.create_homogenised_curves import get_hcurves_geodataframe

DATA = os.path.join(os.path.dirname(__file__), 'data', 'hazardcurve')


class ReadHazardCurveTestCase(unittest.TestCase):
    """ Check information read from hazard curve file """

    def test_read_file_search(self):
        """ Testing homogenisation between models"""
        fname = os.path.join(DATA, 'hazard_curve-mean-PGA_18150.csv')
        gdf, header = get_hcurves_geodataframe(fname)
        self.assertTrue(header[0] == 'mean', 'Header not matching')
        self.assertTrue(header[1] == 1.0, 'Header not matching')
        self.assertTrue(header[2] == 'PGA', 'Header not matching')
        msg = 'The output type is not a GeoDataFrame'
        self.assertTrue(isinstance(gdf, gpd.GeoDataFrame), msg)
