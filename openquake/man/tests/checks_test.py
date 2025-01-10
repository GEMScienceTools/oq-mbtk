import os
import numpy as np
import geopandas as gpd
import unittest

from openquake.man.checks.plotting import get_fault_geojsons

BASE_DATA_PATH = os.path.dirname(__file__)

exp_polys_path = os.path.join(
    BASE_DATA_PATH, 'data', 'checks', 'expected', 'fault_sections.geojson')


class TestReadModel(unittest.TestCase):

    def test_get_fault_geoJSONs(self):
        """
        Check execution of the functions which convert the fault sources
        within a hazard model into geoJSONs.

        Uses a source XML with a single SimpleFaultSource.
        """
        fname = os.path.join(BASE_DATA_PATH, 'data', 'checks')
        results = get_fault_geojsons(fname)

        # Check the geoJSON contents
        obs_polys = results[0]
        exp_polys = gpd.read_file(exp_polys_path)
        obs_p_coo = obs_polys.geometry[0].exterior.coords._coords
        exp_p_coo = exp_polys.geometry[0].exterior.coords._coords
        np.testing.assert_allclose(obs_p_coo, exp_p_coo)

        # Check the geojsons exist
        assert os.path.exists(results[2]) # polys geojson
        assert os.path.exists(results[3]) # traces geojson
        
        # Then remove them
        os.remove(results[2])
        os.remove(results[3])
