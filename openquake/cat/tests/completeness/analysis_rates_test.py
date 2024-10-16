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
import numpy as np

from openquake.cat.completeness.generate import (get_completenesses,
                                                 _get_completenesses)
from openquake.cat.completeness.analysis import (completeness_analysis,
                                                 clean_completeness, _make_ctab)

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data',
                         'completeness_rates')
PLOT = True


def has_duplicates(iterable):
    """
    checks for duplicates in list of array completeness tables
    """
    seen = []
    for x in iterable:
        if x in seen:
            return True
        seen.append(x)
    return False


class ComputeGRParametersTest(unittest.TestCase):
    """ Tests the calculation of GR parameters """

    def setUp(self):

        # Temp folder
        tmp_folder = tempfile.mkdtemp()

        # Folder with the catalogue
        self.fname_input_pattern = os.path.join(DATA_PATH, 'subcat_00*.csv')
        ref_config = os.path.join(DATA_PATH, 'config.toml')

        # Load the config template
        self.conf_txt = toml.load(ref_config)

        # Create the config file for the first test
        self.fname_config = os.path.join(tmp_folder, 'config.toml')
        with open(self.fname_config, 'w', encoding='utf-8') as tmpf:
            toml.dump(self.conf_txt, tmpf)

        # Output folder
        self.folder_out = tmp_folder

        # Create completeness files
        get_completenesses(self.fname_config, self.folder_out)

    def test_compute_gr_param(self):
        """ Testing the calculation """

        completeness_analysis(self.fname_input_pattern,
                              self.fname_config,
                              self.folder_out,
                              self.folder_out,
                              self.folder_out)

        # Load updated configuration file
        conf = toml.load(self.fname_config)

        # Tests

        expected = 5.2725
        computed = conf['sources']['00c']['agr_weichert']
        self.assertAlmostEqual(computed, expected, msg='aGR', places=5)

        expected = 0.97468
        computed = conf['sources']['00c']['bgr_weichert']
        self.assertAlmostEqual(computed, expected, msg='bGR', places=5)

        expected = 5.0
        computed = conf['sources']['00c']['rmag']
        self.assertAlmostEqual(computed, expected, msg='rmag', places=5)

        expected = 2.50674
        computed = conf['sources']['00c']['rmag_rate']
        self.assertAlmostEqual(computed, expected, msg='rmag_rate', places=5)

        expected = 0.2627786
        computed = conf['sources']['00c']['rmag_rate_sig']
        self.assertAlmostEqual(computed, expected, msg='rmag_rate_sig',
                               places=5)

    def test_filter_completeness(self):
        """
        tests cleaning based on original completeness table
        """
        # disposition configs
        mags_in = np.array([3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0])
        years_in = np.array([1980.0, 1990.0, 2000.0])
        cref = [[2000.0, 4.9], [1990.0, 5.3], [1980.0, 5.6]]

        # get all dispositions
        all_disps, _, _ = _get_completenesses(mags_in, years_in)


        # dispositions filtered by reference table 
        filt_disps, mags, years = _get_completenesses(mags_in, years_in,
                                                      completeness_ref=cref)

        # check there are fewer disps than in unfiltered   
        assert len(filt_disps) < len(all_disps)

        # put the dispositions into completeness tables
        ctabs = []
        years_considered = []
        magsA = []
        magsB = []
        magsC = []
        for iper, prm in enumerate(filt_disps):
            ctab = _make_ctab(prm, years, mags)
            if not isinstance(ctab, str):
                ctabs.append(ctab.tolist())
                years_considered.extend([c[0] for c in ctab.tolist()])

                for c in ctab:
                    if c[0] == cref[0][0]:
                        magsA.append(c[1])
                    elif c[0] == cref[1][0]:
                        magsB.append(c[1])
                    elif c[0] == cref[2][0]:
                        magsC.append(c[1])
                    else:
                        raise ValueError('Invalid magnitude included')

        ctabs_ref = [[[1980.0, 6.5]], [[1990.0, 6.0], [1980.0, 6.5]], [[1980.0, 6.0]], [[2000.0, 5.5], [1980.0, 6.5]], [[2000.0, 5.5], [1990.0, 6.0], [1980.0, 6.5]], [[2000.0, 5.5], [1980.0, 6.0]], [[1990.0, 5.5], [1980.0, 6.5]], [[1990.0, 5.5], [1980.0, 6.0]], [[1980.0, 5.5]], [[2000.0, 5.0], [1980.0, 6.5]], [[2000.0, 5.0], [1990.0, 6.0], [1980.0, 6.5]], [[2000.0, 5.0], [1980.0, 6.0]], [[2000.0, 5.0], [1990.0, 5.5], [1980.0, 6.5]], [[2000.0, 5.0], [1990.0, 5.5], [1980.0, 6.0]], [[2000.0, 5.0], [1980.0, 5.5]], [[1990.0, 5.0], [1980.0, 6.5]], [[1990.0, 5.0], [1980.0, 6.0]], [[1990.0, 5.0], [1980.0, 5.5]], [[1980.0, 5.0]], [[2000.0, 4.5], [1980.0, 6.5]], [[2000.0, 4.5], [1990.0, 6.0], [1980.0, 6.5]], [[2000.0, 4.5], [1980.0, 6.0]], [[2000.0, 4.5], [1990.0, 5.5], [1980.0, 6.5]], [[2000.0, 4.5], [1990.0, 5.5], [1980.0, 6.0]], [[2000.0, 4.5], [1980.0, 5.5]], [[2000.0, 4.5], [1990.0, 5.0], [1980.0, 6.5]], [[2000.0, 4.5], [1990.0, 5.0], [1980.0, 6.0]], [[2000.0, 4.5], [1990.0, 5.0], [1980.0, 5.5]], [[2000.0, 4.5], [1980.0, 5.0]], [[1990.0, 4.5], [1980.0, 6.5]], [[1990.0, 4.5], [1980.0, 6.0]], [[1990.0, 4.5], [1980.0, 5.5]], [[1990.0, 4.5], [1980.0, 5.0]], [[2000.0, 4.0], [1980.0, 6.5]], [[2000.0, 4.0], [1990.0, 6.0], [1980.0, 6.5]], [[2000.0, 4.0], [1980.0, 6.0]], [[2000.0, 4.0], [1990.0, 5.5], [1980.0, 6.5]], [[2000.0, 4.0], [1990.0, 5.5], [1980.0, 6.0]], [[2000.0, 4.0], [1980.0, 5.5]], [[2000.0, 4.0], [1990.0, 5.0], [1980.0, 6.5]], [[2000.0, 4.0], [1990.0, 5.0], [1980.0, 6.0]], [[2000.0, 4.0], [1990.0, 5.0], [1980.0, 5.5]], [[2000.0, 4.0], [1980.0, 5.0]], [[2000.0, 4.0], [1990.0, 4.5], [1980.0, 6.5]], [[2000.0, 4.0], [1990.0, 4.5], [1980.0, 6.0]], [[2000.0, 4.0], [1990.0, 4.5], [1980.0, 5.5]], [[2000.0, 4.0], [1990.0, 4.5], [1980.0, 5.0]]]
        assert ctabs == ctabs_ref

        # check for no duplicates
        assert has_duplicates(ctabs) == False

        # check only specified years included (technically also checked before)
        assert sorted(set(years_considered)) == sorted([c[0] for c in cref])

        # check only magnitudes within range allowed (hardcoded to 1.0)
        assert max(set(magsA)) <= cref[0][1] + 1.0
        assert min(set(magsA)) >= cref[0][1] - 1.0
        assert max(set(magsB)) <= cref[1][1] + 1.0
        assert min(set(magsB)) >= cref[1][1] - 1.0
