import os
import unittest

import pandas as pd
import xarray as xr

from openquake.sep.io import make_dataset_from_oq_gmfs, make_pga_xr

BASE_DATA_PATH = os.path.join(os.path.dirname(__file__), "data")
gmf_file = os.path.join(BASE_DATA_PATH, "test_gmfs.csv")
event_file = os.path.join(BASE_DATA_PATH, "test_events.csv")
rupture_file = os.path.join(BASE_DATA_PATH, "test_ruptures.csv")
site_file = os.path.join(BASE_DATA_PATH, "test_sites.csv")

class test_make_pga_dataset(unittest.TestCase):

    def setUp(self):
        return

    def test_make_dataset_from_oq_gmfs(self):
        ds = make_dataset_from_oq_gmfs(gmf_file=gmf_file, 
                                       sitemesh_file=site_file,
                                       ruptures_file=rupture_file, 
                                       events_file=event_file,
                                       ds_name='test')

        ds.to_netcdf("./data/test_pga.nc")

        


