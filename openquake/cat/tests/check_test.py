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
import pandas as pd
import unittest
import tempfile

from openquake.cat.hmg import check

BASE_PATH = os.path.dirname(__file__)

SETTINGS = """

[general]
delta_ll = 0.3
delta_t = 10.0
output_path = "{:s}"
"""


class CheckHomogenisedCatalogue(unittest.TestCase):
    """ Test checking the check function in the catalogue toolkit """

    def setUp(self):

        data_path = os.path.join(BASE_PATH, 'data', 'test_check')
        self.catalogue_fname = os.path.join(data_path, "test_check.csv")

        # Create the temporary folder
        self.tmpd = tempfile.mkdtemp()

        # Update settings
        settings = SETTINGS.format(self.tmpd)

        # Create settings file
        self.settings = os.path.join(self.tmpd, "settings.toml")
        fou = open(self.settings, "w")
        fou.write(settings)
        fou.close()

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
