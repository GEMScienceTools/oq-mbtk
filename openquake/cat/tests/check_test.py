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
import toml

from openquake.cat.hmg import check

BASE_PATH = os.path.dirname(__file__)

SETTINGS_DICT = {
    "general": {
        "delta_ll": 0.3,
        "delta_t": 10.0
    }
}

SETTINGS_GLOBAL = """

[general]
output_path = "{:s}"
delta_ll = [['2000', '10*m'], ['2003', '5+m']]
delta_t = [['2000', '10*m'], ['2003', '2*m']]
use_kms = true
"""

class CheckHomogenisedCatalogue(unittest.TestCase):
    """ Test checking the check function in the catalogue toolkit """

    def setUp(self):

        data_path = os.path.join(BASE_PATH, 'data', 'test_check')
        self.catalogue_fname = os.path.join(data_path, "test_check.csv")

        # Create the temporary folder
        self.tmpd = tempfile.mkdtemp()

        # Update settings
        SETTINGS_DICT["general"]["output_path"] = self.tmpd

        # Create settings file
        self.settings = os.path.join(self.tmpd, "settings.toml")
        with open(self.settings, "w") as fou:
            toml.dump(SETTINGS_DICT, fou)

    def test_case01(self):
        """Searching for the duplicate"""

        # In this test we created a small catalogue with 5 earthquakes. The
        # last one is a duplication of the first one with slightly modified
        # longitude (+0.1) and seconds (+2.0). We expect that the code that
        # checks the catalogue will find this duplicate (given the settings
        # above)

        # Run the check
        cnt = check.check_catalogue(self.catalogue_fname, self.settings)

        # Testing
        self.assertEqual(cnt, 1, "Found a wrong number of checks")
        
       
class CheckHomogenisedCatalogueFunctions(unittest.TestCase):
    """ Test checking the check function works with tuples describing different behaviour in different years """
    def setUp(self):

        data_path = os.path.join(BASE_PATH, 'data', 'test_check')
        self.catalogue_fname = os.path.join(data_path, "test_check2.csv")

        # Create the temporary folder
        self.tmpd = tempfile.mkdtemp()

        # Update settings
        # Use toml.load and toml dump to ensure that Windows paths
        # are escaped correctly and the resulting TOML file is valid
        td = toml.loads(SETTINGS_GLOBAL)
        td["general"]["output_path"] = self.tmpd
        # Create settings file
        self.settings = os.path.join(self.tmpd, "settings.toml")
        with open(self.settings, "w") as fou:
            toml.dump(td, fou)

    def test_case01(self):
        """Searching for events to check"""

        # In this test we created a small catalogue with 8 earthquakes between 2002 and 2004
        # Check should count:
        # - Duplicate event in 2002 at slightly different location (12kms sep)
        # - Event in 2003 with 2 second seperation
        # Check should *not* check 
        # - events in 2002 at same location one hour apart
        # - events in 2004 that are 10s apart (greater than the time threshold in later year) 
        

        # Run the check
        cnt = check.check_catalogue(self.catalogue_fname, self.settings)
        print("number of checks = ", cnt)

        # Testing
        self.assertEqual(cnt, 2, "Found a wrong number of checks")
