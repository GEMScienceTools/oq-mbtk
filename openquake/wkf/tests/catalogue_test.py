# ------------------- The OpenQuake Model Building Toolkit --------------------
# Copyright (C) 2022 GEM Foundation
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

import os
import unittest
import numpy as np
import pandas as pd
from openquake.wkf.catalogue import extract

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


class ExtractCatalogueTest(unittest.TestCase):
    """ Tests the filtering of a catalogue """

    def test_extract(self):
        """ Filtering catalogue by depth """
        fname = os.path.join(DATA_PATH, 'catalogue_01.csv')
        kwargs = {'min_depth': 5, 'max_depth': 20}
        computed = extract(fname, **kwargs)
        data = {'eventID': [3, 4],
                'year': [2000, 2000],
                'month': [1, 1],
                'day': [1, 1],
                'magnitude': [5.3, 5.4],
                'longitude': [13.0, 24.0],
                'latitude': [23.0, 24.0],
                'depth': [10.0, 20.0]}
        expected = pd.DataFrame(data)
        np.array_equal(expected.values, computed.values)
