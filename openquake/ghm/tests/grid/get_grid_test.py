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

"""
Module create_grid_test
"""

import os
import unittest
import tempfile
import numpy as np

import openquake.ghm.grid.get_sites as get_sites

DATA = os.path.join(os.path.dirname(__file__))


class GetSitesTestCase(unittest.TestCase):
    """ Test the creation of a grid of sites """

    def test_get_sites_eur(self):
        """ Test creation of sites for the EUR model """
        model = 'eur'
        folder_out = tempfile.mkdtemp()
        fname_conf = os.path.join(DATA, 'data', 'conf.toml')
        get_sites.main(model, folder_out, fname_conf)
        fname_expected = os.path.join(DATA, 'data', 'eur_FIPS_CNTRY.csv')
        expected = np.loadtxt(fname_expected, delimiter=',')
        fname_computed = os.path.join(folder_out, 'eur_res5.csv')
        computed = np.loadtxt(fname_computed, delimiter=',')
        msg = f'Content of files {fname_computed} and {fname_expected} does'
        msg += ' not match'
        np.testing.assert_almost_equal(computed, expected, err_msg=msg)
