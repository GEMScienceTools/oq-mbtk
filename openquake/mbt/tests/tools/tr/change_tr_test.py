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
import h5py
import shutil
import numpy
import unittest

from openquake.mbi.ccl.classify import classify
from openquake.mbt.tools.tr.change_class import change

BASE_PATH = os.path.dirname(__file__)


class ChangeTrTestCase(unittest.TestCase):
    """
    This class tests manual change from one class to another
    """

    def setUp(self):
        self.root_folder = os.path.join(BASE_PATH)
        self.tmp = os.path.join(BASE_PATH, '..', '..', 'tmp')
        if os.path.exists(self.tmp):
            shutil.rmtree(self.tmp)
        os.makedirs(self.tmp)

    def tearDown(self):
        #
        # removing tmp folder
        shutil.rmtree(self.tmp)

    def testcase01(self):
        """
        Testing TR change from crustal to stable crustal
        """
        ini_fname = os.path.join(
            BASE_PATH, '..', '..', 'data', 'tr', 'acr_scr.ini')
        cat = os.path.join(
            BASE_PATH, '..', '..', 'data', 'tr', 'catalogue.pkl')
        treg_filename = os.path.join(
            BASE_PATH, '..', '..', 'tmp', 'test02.hdf5')
        event_change = os.path.join(
            BASE_PATH, '..', '..', 'data', 'tr', 'ev_change.csv')

        # classify
        classify(ini_fname, True, self.root_folder)
        f = h5py.File(treg_filename, 'r')

        # manually change an event
        change(cat, treg_filename, event_change)

        # testing new file to see if the swap worked
        treg_filename2 = os.path.join(
            BASE_PATH, '..', '..', 'tmp', 'test02_up.hdf5')
        f2 = h5py.File(treg_filename2, 'r')

        expected = [1, 0, 0, 0, 0]
        numpy.testing.assert_array_equal(f2['crustal_active'][:], expected)

        # testing crustal stable
        expected = [0, 0, 1, 0, 1]
        numpy.testing.assert_array_equal(f2['crustal_stable'][:], expected)
        #
        f.close()
