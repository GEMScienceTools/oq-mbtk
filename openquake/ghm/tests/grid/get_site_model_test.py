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
import toml
import unittest
import tempfile
import numpy as np

from openquake.ghm.grid.get_sites import _get_sites
# from openquake.ghm.grid.get_site_model import _get_site_model

DATA = os.path.join(os.path.dirname(__file__))


class GetSitesModelTestCase(unittest.TestCase):
    """ Test the creation of a grid of sites """

    def test_get_model(self):
        """ Test creation of site model for the CEA region """

        model = 'cea'
        folder_out = tempfile.mkdtemp()
        fname_conf = os.path.join(DATA, 'data', 'conf.toml')

        # Get the configuration
        conf = toml.load(fname_conf)

        root_path = os.path.dirname(fname_conf)
        sites, _, _, _ = _get_sites(model, folder_out, conf, root_path)

        #fname_expected = os.path.join(DATA, 'data', 'cea.csv')
        #expected = np.loadtxt(fname_expected, delimiter=',')
        #fname_computed = os.path.join(folder_out, 'cea.csv')
        #computed = np.loadtxt(fname_computed, delimiter=',')
        #np.testing.assert_almost_equal(computed, expected)
