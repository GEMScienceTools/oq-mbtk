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
import numpy as np
import unittest
import tempfile
import toml
import matplotlib.pyplot as plt

from openquake.wkf.compute_gr_params import weichert_analysis

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'compute_gr_param')
PLOT = False


class ComputeGRParametersTest(unittest.TestCase):
    """ Tests the calculation of GR parameters """

    def setUp(self):
        tmp_folder = tempfile.mkdtemp()
        self.fname_input_pattern = os.path.join(DATA_PATH, '*.csv')
        ref_config = os.path.join(DATA_PATH, 'config.toml')
        conf_txt = toml.load(ref_config)
        self.fname_config = os.path.join(tmp_folder, 'config.toml')
        with open(self.fname_config, 'w', encoding='utf-8') as tmpf:
            toml.dump(conf_txt, tmpf)
        self.folder_out = tmp_folder

    def test_compute_gr_param(self):
        """ Testing the calculation """

        # Compute GR params
        weichert_analysis(self.fname_input_pattern, self.fname_config,
                          self.folder_out)

        # Load updated configuration file
        conf = toml.load(self.fname_config)

        # Tests
        expected = 5.217742014665241
        computed = conf['sources']['00']['agr_weichert']
        self.assertAlmostEqual(computed, expected, msg='aGR', places=5)

        expected = 1.1531979338923517
        computed = conf['sources']['00']['bgr_weichert']
        self.assertAlmostEqual(computed, expected, msg='bGR', places=5)

        expected = 4.4
        computed = conf['sources']['00']['rmag']
        self.assertAlmostEqual(computed, expected, msg='rmag', places=5)

        expected = 1.3921021542700736
        computed = conf['sources']['00']['rmag_rate']
        self.assertAlmostEqual(computed, expected, msg='rmag_rate', places=5)

        expected = 0.2500289278698006
        computed = conf['sources']['00']['rmag_rate_sig']
        self.assertAlmostEqual(computed, expected, msg='rmag_rate_sig',
                               places=5)

        if PLOT:
            plt.show()
