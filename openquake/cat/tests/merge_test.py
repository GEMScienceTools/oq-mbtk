import os
import pandas as pd
import unittest
import tempfile
import toml

from openquake.cat.hmg import merge

BASE_PATH = os.path.dirname(__file__)

SETTINGS = """

[general]
region_buffer = 5.0
output_path = "{:s}"
output_prefix = "test_"
region_shp = "{:s}"
# region_buffer = 1.0
log_file = "{:s}"

# Catalogues

[[catalogues]]
code = "ISC"
name = "ISC Bulletin"
filename = "{:s}"
type = "isf"
select_region = false

[[catalogues]]
code = "oGCMT"
name = "Original GCMT"
filename = "{:s}"
type = "csv"
delta_ll = 0.50
delta_t =  40.0
timezone = 0
buff_ll = 0.0
buff_t = 5.0
use_ids = false
"""


class MergeGCMTTest(unittest.TestCase):

    def setUp(self):

        data_path = os.path.join(BASE_PATH, 'data', 'test_merge')

        # Create the temporary folder
        self.tmpd = tempfile.mkdtemp()

        # Update settings
        # Use toml.load and toml dump to ensure that Windows paths
        # are escaped correctly and the resulting TOML file is valid
        td = toml.loads(SETTINGS)
        td["general"]["output_path"] = self.tmpd
        td["general"]["log_file"] = os.path.join(self.tmpd, "log.txt")
        td["general"]["region_shp"] = \
            os.path.join(data_path, "shp", "test_area.shp")
        td["catalogues"][0]["filename"] = \
            os.path.join(data_path, "test_isc_bulletin.isf")
        td["catalogues"][1]["filename"] = \
            os.path.join(data_path, "test_gcmt.csv")

        # Create settings file
        self.settings = os.path.join(self.tmpd, "settings.toml")
        with open(self.settings, "w") as fou:
            toml.dump(td, fou)

    def test_case01(self):
        """Merging GCMT catalogue"""

        # Read the ISF formatted file
        print(self.settings)

        # Merge
        merge.process_catalogues(self.settings)

        # Reading catalogue
        fname = os.path.join(self.tmpd, "test_otab.h5")
        odf = pd.read_hdf(fname)
        self.assertEqual(len(odf[odf["prime"] == 1]), 635)
