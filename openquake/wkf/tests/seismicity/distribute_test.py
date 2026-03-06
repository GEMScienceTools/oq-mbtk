#!/usr/bin/env python
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
import tempfile
import unittest
import numpy
import pandas as pd
import numpy as np

from openquake.wkf.seismicity.distribute import _distribute_rates as distribute

HERE = os.path.dirname(__file__)
CWD = os.getcwd()
DATA = os.path.relpath(os.path.join(HERE, 'data', 'distribute'), CWD)


class RatesDistributeTestCase(unittest.TestCase):
    """ Tests the calculation of rates """

    def setUp(self):
        self.out_folder = tempfile.mkdtemp()
        self.conf = os.path.join(DATA, 'conf.toml')

    def test_distribute_rates(self):
        """ Test the original approach """
        # In this test the configuration file contains only the agr and bgr
        # values. With this configuration it's not possible to add uncertainty.

        # Run the code
        distribute(DATA, self.conf, self.out_folder)

        # Test results
        res = pd.read_csv(os.path.join(self.out_folder, '00.csv'))
        expected = numpy.array([3.200, 3.5010, 3.6771, 3.8020])
        computed = res.agr.to_numpy()
        numpy.testing.assert_almost_equal(expected, computed, decimal=4)


    def test_distribute_rates_error(self):
        """ Test the original approach """
        # In this test the configuration file contains only the agr and bgr
        # values. With this configuration it's not possible to add uncertainty.

        # Run the code
        conf = os.path.join(DATA, 'conf01.toml')
        with self.assertRaises(ValueError):
            distribute(DATA, conf, self.out_folder, eps_rate=-1)


    def test_distribute_rates_delta(self):
        """ Test the mean value + 1std for b and rate """

        # Run the code
        distribute(
                DATA,
                self.conf,
                self.out_folder,
                eps_rate=1.0,
                eps_b=1.0)

        # Test results. The expected total agr is 4.671150
        res = pd.read_csv(os.path.join(self.out_folder, '00.csv'))
        expected = numpy.array([3.671158, 3.97219, 4.14828, 4.27322])
        computed = res.agr.to_numpy()
        numpy.testing.assert_almost_equal(computed, expected, decimal=4)

    def test_distribute_rates_deltaA(self):
        """ Test the mean value + 2std for rate """

        # Run the code
        distribute(
                DATA,
                self.conf,
                self.out_folder,
                eps_rate=2.0)

        # Test results. The expected total agr is 4.671150
        res = pd.read_csv(os.path.join(self.out_folder, '00.csv'))
        expected = numpy.array([3.241309, 3.542339, 3.718430, 3.843369])
        computed = res.agr.to_numpy()
        numpy.testing.assert_almost_equal(expected, computed, decimal=4)

    def test_distribute_rates_flat(self):
        """ Test the original approach + 10% flat """

        # Run the code
        distribute(
                DATA,
                self.conf,
                self.out_folder,
                fraction_flat=0.1)

        # Test results. The expected total agr is 4.671150
        res = pd.read_csv(os.path.join(self.out_folder, '00.csv'))
        expected = numpy.array([3.2607, 3.5118, 3.6698, 3.7855])
        agr = np.log10(np.sum(10**expected))
        computed = res.agr.to_numpy()
        numpy.testing.assert_almost_equal(computed, expected, decimal=4)
        numpy.testing.assert_almost_equal(agr, 4.2, decimal=4)
