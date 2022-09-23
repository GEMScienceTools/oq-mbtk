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

import re
import os
import unittest
import tempfile
from openquake.cat.hmg import merge
from openquake.cat.parsers.generic_catalogue import GeneralCsvCatalogue
from openquake.cat.parsers.converters import GenericCataloguetoISFParser

BASE_PATH = os.path.dirname(__file__)


class GenericCatalogueTestCase(unittest.TestCase):

    def setUp(self):

        fname = os.path.join(BASE_PATH, 'data', 'comcat_sample.csv')
        self.cat = GeneralCsvCatalogue()
        self.cat.parse_csv(fname)

        # Create the temporary folder
        self.tmpd = tempfile.mkdtemp()

    def test_write_to_isf(self):
        """Write ISF catalogue"""
        # out is a catalogue in the ISF Format
        out = self.cat.write_to_isf_catalogue(catalogue_id='BB', name='BB')

        # Checking number of events
        self.assertEqual(3, len(out.events))

        # Check magnitude type and value of the first event
        self.assertEqual(out.events[0].magnitudes[0].value, 4.6)
        self.assertEqual(out.events[0].magnitudes[0].scale, 'mb')
