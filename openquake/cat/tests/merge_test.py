# ------------------- The OpenQuake Model Building Toolkit --------------------
# Copyright (C) 2022 GEM Foundation
#           _______  _______        __   __  _______  _______  ___   _
#          |       ||       |      |  |_|  ||  _    ||       ||   | | |
#          |   _   ||   _   | ____ |       || |_|   ||_     _||   |_| |
#          |  | |  ||  | |  ||____||       ||       |  |   |  |      _|
#          |  |_|  ||  |_|  |      |       ||  _   |   |   |  |     |_
#          |       ||      |       | ||_|| || |_|   |  |   |  |    _  |
#          |_______||____||_|      |_|   |_||_______|  |___|  |___| |_|
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------
# vim: tabstop=4 shiftwidth=4 softtabstop=4
# coding: utf-8

import os
import unittest
import tempfile
import numpy as np
import pandas as pd
import toml

from openquake.cat.hmg import merge

aeq = np.testing.assert_equal


BASE_PATH = os.path.dirname(__file__)

SETTINGS = """

[general]
region_buffer = 5.0
output_path = "{:s}"
output_prefix = "test_"
region_shp = "{:s}"
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

SETTINGS_GCMT = """

[general]
region_buffer = 5.0
output_path = "{:s}"
output_prefix = "test_"
region_shp = "{:s}"
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
delta_ll = [['1800', '0.50 * m / m']]
delta_t =  [['1800', '40.0']]
timezone = 0
buff_ll = 0.0
buff_t = 5.0
use_ids = false
"""

SETTINGS_GCMT2 = """

[general]
region_buffer = 5.0
output_path = "{:s}"
output_prefix = "test_"
region_shp = "{:s}"
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
delta_ll = [['1800', '0.50 * m / m']]
delta_t =  [['1800', '40.0 * m / m']]
timezone = 0
buff_ll = 0.0
buff_t = 5.0
use_ids = false
"""

SETTINGS_COMCAT = """

[general]
log_file = "{:s}"

[[catalogues]]
code = "ISCGEM"
name = "ISC GEM Test"
type = "csv"
select_region = false

[[catalogues]]
code = "COMCAT"
name = "USGS ComCat"
type = "csv"
delta_ll = 0.30
delta_t =  40.0
timezone = 0
buff_ll = 0.0
buff_t = 5.0
use_ids = false
"""

SETTINGS_GLOBAL = """

[general]
output_path = "{:s}"
output_prefix = "global_"
log_file = "{:s}"

# Catalogues

[[catalogues]]
code = "ISC"
name = "ISC Bulletin"
filename = "{:s}"
type = "csv"
select_region = false

[[catalogues]]
code = "comcat"
name = "comcat"
#filename = "./comcat_concat.csv"
filename = "{:s}"
type = "csv"
delta_ll = 20
delta_t =  5
timezone = 0
buff_ll = 0.0
buff_t = 5.0
use_kms = true

[[catalogues]]
code = "oGCMT"
name = "Original GCMT"
filename = "{:s}"
type = "csv"
delta_ll = [['1900', '12*m']]
delta_t =  [['1900', '8*m']]
timezone = 0
buff_ll = 0.0
buff_t = 5.0
use_kms = true
"""

class MergeGCMTTestCase(unittest.TestCase):

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

        #td["catalogues"][0]["log_file"] = os.path.join(data_path, "log_isc.txt")
        #td["catalogues"][1]["log_file"] = os.path.join(data_path, "log_gcmt.txt")
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


class MergeGCMTWithTimeVaryingParametersTestCase(unittest.TestCase):

    def setUp(self):

        data_path = os.path.join(BASE_PATH, 'data', 'test_merge')

        # Create the temporary folder
        self.tmpd = tempfile.mkdtemp()

        # Update settings
        # Use toml.load and toml dump to ensure that Windows paths
        # are escaped correctly and the resulting TOML file is valid
        td = toml.loads(SETTINGS_GCMT)
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

    def test_case_tv_01(self):
        """Merging GCMT catalogue"""

        # Read the ISF formatted file
        print(self.settings)

        # Merge
        merge.process_catalogues(self.settings)

        # Reading catalogue
        fname = os.path.join(self.tmpd, "test_otab.h5")
        odf = pd.read_hdf(fname)
        self.assertEqual(len(odf[odf["prime"] == 1]), 635)
        
class MergeGCMTWithTimeVaryingParameterFunctionsTestCase(unittest.TestCase):

    def setUp(self):

        data_path = os.path.join(BASE_PATH, 'data', 'test_merge')

        # Create the temporary folder
        self.tmpd = tempfile.mkdtemp()

        # Update settings
        # Use toml.load and toml dump to ensure that Windows paths
        # are escaped correctly and the resulting TOML file is valid
        td = toml.loads(SETTINGS_GCMT2)
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

    def test_case_tv_01(self):
        """Merging GCMT catalogue"""

        # Read the ISF formatted file
        print(self.settings)

        # Merge
        merge.process_catalogues(self.settings)

        # Reading catalogue
        fname = os.path.join(self.tmpd, "test_otab.h5")
        odf = pd.read_hdf(fname)
        self.assertEqual(len(odf[odf["prime"] == 1]), 635)        


class MergeComCatTestCase(unittest.TestCase):

    def setUp(self):

        data_path = os.path.join(BASE_PATH, 'data', 'test_merge_comcat')

        # Create the temporary folder
        self.tmpd = tempfile.mkdtemp()

        # Update settings
        # Use toml.load and toml dump to ensure that Windows paths
        # are escaped correctly and the resulting TOML file is valid
        td = toml.loads(SETTINGS_COMCAT)
        td["general"]["output_path"] = self.tmpd
        td["general"]["log_file"] = os.path.join(self.tmpd, "log.txt")
        td["catalogues"][0]["filename"] = \
            os.path.join(data_path, "isc_sample.csv")
        td["catalogues"][1]["filename"] = \
            os.path.join(data_path, "comcat_sample.csv")

        # Create settings file
        self.settings = os.path.join(self.tmpd, "settings.toml")
        with open(self.settings, "w") as fou:
            toml.dump(td, fou)

    def test_merge_comcat_case01(self):
        """Merging COMCAT catalogue"""

        print(self.settings)

        # Merge
        merge.process_catalogues(self.settings)

        # Reading output catalogue
        fname = os.path.join(self.tmpd, "otab.h5")
        odf = pd.read_hdf(fname)

        # Checking prime events
        expected = [1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
        aeq(odf.prime.to_numpy(), expected)
        
class MergeGlobalTestCase(unittest.TestCase):
    # Merge ISC-GEM, comcat and GCMT sample files using km distances for delta_ll
    
    def setUp(self):

        data_path = os.path.join(BASE_PATH, 'data', 'test_merge_comcat')

        # Create the temporary folder
        self.tmpd = tempfile.mkdtemp()

        # Update settings
        # Use toml.load and toml dump to ensure that Windows paths
        # are escaped correctly and the resulting TOML file is valid
        td = toml.loads(SETTINGS_GLOBAL)
        td["general"]["output_path"] = self.tmpd
        td["general"]["log_file"] = os.path.join(self.tmpd, "log.txt")
        td["general"]["region_shp"] = \
            os.path.join(data_path, "shp", "test_area.shp")
        td["catalogues"][0]["filename"] = \
            os.path.join(data_path, "isc_sample.csv")
        td["catalogues"][1]["filename"] = \
            os.path.join(data_path, "comcat_sample.csv")
        td["catalogues"][2]["filename"] = \
            os.path.join(data_path, "gcmt_sample.csv")

        # Create settings file
        self.settings = os.path.join(self.tmpd, "settings.toml")
        with open(self.settings, "w") as fou:
            toml.dump(td, fou)

    def test_case_global_01(self):
        """Merging GCMT catalogue"""

        # Read the ISF formatted file
        print(self.settings)

        # Merge
        merge.process_catalogues(self.settings)

        # Reading catalogue
        fname = os.path.join(self.tmpd, "global_otab.h5")
        odf = pd.read_hdf(fname)
        # Check prime events
        expected = [1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1]
        aeq(odf.prime.to_numpy(), expected)
       

