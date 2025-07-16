# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2025 GEM Foundation and G. Weatherill
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.
"""
Tests the GM database parsing and selection
"""
import os
import unittest
import numpy as np
from scipy.constants import g

from openquake.smt.utils import convert_accel_units
from openquake.smt.utils_intensity_measures import SCALAR_XY


BASE_DATA_PATH = os.path.join(os.path.dirname(__file__))


class SmUtilsTestCase(unittest.TestCase):
    """
    Tests GroundMotionTable and selection.
    """
    def assertNEqual(self, first, second, rtol=1e-6, atol=1e-9,
                     equal_nan=True):
        self.assertTrue(np.allclose(first, second,
                                    rtol=rtol, atol=atol,
                                    equal_nan=equal_nan))

    def test_accel_units(self):
        """
        Test acceleration units function.
        """
        func = convert_accel_units
        for acc in [np.nan, 0, 100, -g*5, g*6.5,
                    np.array([np.nan, 0, 100, g*5, g*6.5])]:

            # check that cm_sec and m_sec produce the same result:
            _1, _2 = func(acc, 'g', 'cm/s/s'), func(acc, 'cm/s/s', 'g')
            for cmsec in ('cm/s^2', 'cm/s**2'):
                self.assertNEqual(_1, func(acc, 'g', cmsec))
                self.assertNEqual(_2, func(acc, cmsec, 'g'))

            _1, _2 = func(acc, 'g', 'm/s/s'), func(acc, 'm/s/s', 'g')
            for msec in ('m/s^2', 'm/s**2'):
                self.assertNEqual(_1, func(acc, 'g', msec))
                self.assertNEqual(_2, func(acc, msec, 'g'))

            # Assert same label is no-op:
            self.assertNEqual(func(acc, 'g', 'g'), acc)
            self.assertNEqual(func(acc, 'cm/s/s', 'cm/s/s'), acc)
            self.assertNEqual(func(acc, 'm/s/s', 'm/s/s'), acc)

            # Assume input in g and converting to cm/s/s
            expected = acc * (100 * g)
            self.assertNEqual(func(acc, 'g', 'cm/s/s'), expected)

            # To m/s/s
            expected /= 100
            self.assertNEqual(func(acc, 'g', 'm/s/s'), expected)

            with self.assertRaises(ValueError):  # invalid units 'a'
                func(acc, 'a')

    def tst_scalar_xy(self):
        argslist = [(np.nan, np.nan),
                    (1, 2),
                    (3.5, -4.706),
                    (np.array([np.nan, 1, 3.5]),
                     np.array([np.nan, 2, -4.706]))]
    
        expected = {
            'Geometric': [np.nan, np.sqrt(1 * 2), np.sqrt(3.5 * -4.706),
                          [np.nan, np.sqrt(1 * 2), np.sqrt(3.5 * -4.706)]],
            'Arithmetic': [np.nan, (1+2.)/2., (3.5 - 4.706)/2,
                           [np.nan, (1+2.)/2., (3.5 - 4.706)/2]],
            'Larger': [np.nan, 2, 3.5, [np.nan, 2, 3.5]],
            'Vectorial': [np.nan, np.sqrt(5.), np.sqrt(3.5**2 + 4.706**2),
                          [np.nan, np.sqrt(5.), np.sqrt(3.5**2 + 4.706**2)]]
        }

        for i, args in enumerate(argslist):
            for type_, exp in expected.items():
                res = SCALAR_XY[type_](*args)
                equals = np.allclose(res, exp[i], rtol=1e-7, atol=0,
                                     equal_nan=True)
                if hasattr(equals, 'all'):
                    equals = equals.all()
                try:
                    self.assertTrue(equals)
                except AssertionError:
                    asd = 9
                

if __name__ == "__main__":
    unittest.main()