
# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2025 GEM Foundation
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
 
import os
import shutil
import unittest
import tempfile
import numpy as np

from openquake.calculators.tests import open8
from openquake.calculators.export import export
from openquake.calculators.base import run_calc

import openquake.man.general_utils.csv_output_utils as csv


BASE_DATA_PATH = os.path.join(os.path.dirname(__file__), 'data_csv_output_test')
BASE_EXP_PATH = os.path.join(BASE_DATA_PATH, 'expected')
BASE_CASE8 = os.path.join(BASE_DATA_PATH, 'case_8')

OVERWRITE = False


class TestMeanMDE(unittest.TestCase):

    def test_output_mre(self):
        """
        test reorg of one instance of an MDE file; one rlz
        """
        fname = os.path.join(BASE_DATA_PATH, 'Mag_Dist_Eps-1.csv')
        fout = 'test-1.csv'
        csv.mean_mde_for_gmt(fname, fout, 0.002105, 'SA(0.1)', 1e-10)
        expected = os.path.join(BASE_EXP_PATH, 'site_0.002105_SA01_mde-1.csv')
        expected_lines = [line for line in open8(expected)]
        actual_lines = [line for line in open8(fout)]
        assert expected_lines == actual_lines
        os.remove(fout)

    def test_output_mre_2(self):
        """
        test reorg of one instance of an MDE file; two rlzs
        """
        fname = os.path.join(BASE_DATA_PATH, 'Mag_Dist_Eps-2.csv')
        fout = 'test-2.csv'
        csv.mean_mde_for_gmt(fname, fout, 0.002105, 'SA(0.1)', 1e-10)
        expected = os.path.join(BASE_EXP_PATH, 'site_0.002105_SA01_mde-2.csv')
        expected_lines = [line for line in open8(expected)]
        actual_lines = [line for line in open8(fout)]

        actual_lines = [[float(j) for j in i.split()] for i in actual_lines]
        expected_lines = [[float(j) for j in i.split()] for i in expected_lines]
        actual_lines = np.array(actual_lines)
        expected_lines = np.array(expected_lines)
        aae = np.testing.assert_almost_equal
        aae(actual_lines, expected_lines, decimal=4)

        os.remove(fout)

    def test_output_mre_3(self):
        """
        tests that if a MDE output file includes more than one IMT, the
        function mean_mde_for_gmt considers only the specified IMT when
        creating the file that will be plotted by GMT
        """
        fname1 = os.path.join(BASE_DATA_PATH, 'Mag_Dist_Eps-mean-0.csv')
        fout1, path1 = tempfile.mkstemp()
        fname2 = os.path.join(BASE_CASE8, 'expected/Mag_Dist_Eps-mean-0.csv')
        fout2, path2 = tempfile.mkstemp()

        if OVERWRITE:
            shutil.copy(path1, fname1)
            shutil.copy(path2, fname2)

        csv.mean_mde_for_gmt(fname1, path1, 0.002105, 'SA(0.1)', 1e-10)
        csv.mean_mde_for_gmt(fname2, path2, 0.002105, 'SA(0.1)', 1e-10)
        expect_lines1 = [[float(j) for j in i.split()] for i in open8(path1)]
        expect_lines2 = [[float(j) for j in i.split()] for i in open8(path2)]
        expect_lines1 = np.array(expect_lines1)
        expect_lines2 = np.array(expect_lines2)
        aae = np.testing.assert_almost_equal
        aae(expect_lines1, expect_lines2, decimal=4)

    def test_output_llt(self):
        """
        test reorg of one instance of an MDE file; one rlz
        """
        fname = os.path.join(BASE_DATA_PATH, 'TRT_Lon_Lat-mean-0.csv')
        fout = 'test-1.csv'
        csv.mean_llt_for_gmt(fname, fout, 0.002105, 'SA(0.1)', 1e-10)
        expected = os.path.join(BASE_EXP_PATH, 'site_0.002105_SA01_llt.csv')
        expected_lines = [line for line in open8(expected)]
        actual_lines = [line for line in open8(fout)]
        assert expected_lines == actual_lines
        os.remove(fout)


class OutputTestCase(unittest.TestCase):

    def test_mde_llt_format(self):
        """
        will fail if the output format changes
        """
        # Run test job
        calc1 = run_calc(os.path.join(BASE_CASE8, 'job_mde.ini'))
        calc2 = run_calc(os.path.join(BASE_CASE8, 'job_llt.ini'))

        # Test mre results output format
        [fname_mde] = export(('disagg-stats', 'csv'), calc1.datastore)
        exp_mde = os.path.join(BASE_CASE8, 'expected/Mag_Dist_Eps-mean-0.csv')
        [fname_llt] = export(('disagg-stats', 'csv'), calc2.datastore)
        exp_llt = os.path.join(BASE_CASE8, 'expected/TRT_Lon_Lat-mean-0.csv')

        if OVERWRITE:
            shutil.copyfile(fname_llt, exp_mde)
            shutil.copyfile(fname_mde, exp_llt)

        # Test MDE format
        actual_lines = []
        for i, line in enumerate(open8(fname_mde)):
            if i == 0:
                hea1_comp = line
            elif i == 1:
                hea2_comp = line
            else:
                actual_lines.append([float(j) for j in line.split()[1:]])

        expected_lines = []
        for i, line in enumerate(open8(exp_mde)):
            if i == 0:
                hea1_exp = line
            elif i == 1:
                hea2_exp = line
            if i < 2:
                continue
            expected_lines.append([float(j) for j in line.split()[1:]])

        assert hea2_comp == hea2_exp

        actual_lines = np.array(actual_lines)
        expected_lines = np.array(expected_lines)
        aae = np.testing.assert_almost_equal
        aae(actual_lines, expected_lines, decimal=4)

        os.remove(fname_mde)

        # Test LLT format
        actual_lines_floats = []
        actual_lines_strings = []

        inds = [1, 2, 4, 5, 6]
        for i, line in enumerate(open8(fname_llt)):
            if i == 0:
                hea1_comp = line
            elif i == 1:
                hea2_comp = line
            else:
                line_tmp = line.split(',')
                line = [line_tmp[i] for i in inds]
                actual_lines_floats.append([float(j) for j in line])
                actual_lines_strings.append([line_tmp[0],line_tmp[3]])

        expected_lines_floats = []
        expected_lines_strings = []
        for i, line in enumerate(open8(exp_llt)):
            if i == 0:
                hea1_exp = line
            elif i == 1:
                hea2_exp = line
            if i < 2:
                continue
            line_tmp = line.split(',')
            line = [line_tmp[i] for i in inds]
            expected_lines_floats.append([float(j) for j in line])
            expected_lines_strings.append([line_tmp[0],line_tmp[3]])

        assert hea2_comp == hea2_exp

        actual_lines_floats = np.array(actual_lines_floats)
        expected_lines_floats = np.array(expected_lines_floats)
        aae = np.testing.assert_almost_equal
        aae(actual_lines, expected_lines, decimal=4)

        assert actual_lines_strings == expected_lines_strings

        os.remove(fname_llt)


class TestReadHeader(unittest.TestCase):
    """
    Test reading the contect of the hazard curve file header
    """
    def testcase01(self):
        """ Reading hcurve header """
        fname = os.path.join(BASE_DATA_PATH, 'hazard_curve-mean-PGA_22071.csv')
        fhandle = open(fname, 'r')
        header = csv._get_header1(fhandle.readline())
        tmpstr = 'result_type'
        msgstr = 'The result types do not match'
        self.assertEqual("mean", header[tmpstr], msgstr)
        tmpstr = 'investigation_time'
        msgstr = 'The investigation times do not match'
        self.assertEqual(1.0, header[tmpstr], msgstr)
        tmpstr = 'imt'
        msgstr = 'The IMTs do not match'
        self.assertEqual("PGA", header[tmpstr], msgstr)

    def testcase02(self):
        """ Reading hcurve header """
        fname = os.path.join(BASE_DATA_PATH, 'hazard_curve-mean-PGA_22072.csv')
        fhandle = open(fname, 'r')
        header = csv._get_header1(fhandle.readline())
        tmpstr = 'investigation_time'
        msgstr = 'The investigation times do not match'
        self.assertEqual(50.0, header[tmpstr], msgstr)
        tmpstr = 'imt'
        msgstr = 'The IMTs do not match'
        self.assertEqual("SA(0.2)", header[tmpstr], msgstr)

    def testcase03(self):
        """ Reading hcurve header - csv version 3.6"""
        tmpstr = 'hazard_curve-mean-PGA_23538_v3.6.csv'
        fname = os.path.join(BASE_DATA_PATH, tmpstr)
        fhandle = open(fname, 'r')
        header = csv._get_header1(fhandle.readline())
        tmpstr = 'investigation_time'
        msgstr = 'The investigation times do not match'
        self.assertEqual(1.0, header[tmpstr], msgstr)
        #
        tmpstr = 'imt'
        msgstr = 'The IMTs do not match'
        self.assertEqual("PGA", header[tmpstr], msgstr)
        #
        msgstr = 'Wrong engine version'
        expected = 'OpenQuake engine 3.6.0-git3c85fde84e'
        self.assertEqual(expected, header['engine'], msgstr)


class CatalogueFromSESTest(unittest.TestCase):
    def testcase01(self):
        fname = os.path.join(BASE_DATA_PATH, 'ruptures.csv')
        catalogue = csv.get_catalogue_from_ses(fname, 10)
        msg = 'Total number of events not matching'
        self.assertEqual(catalogue.get_number_events(), 51, msg)
        dat = [[6.05, 21], [6.15, 11], [6.25, 8], [6.35, 11]]
        for mag, expected in dat:
            fmt = 'Number of events with magnitude {:.2f} not matching'
            msg = fmt.format(mag)
            computed = sum(abs(catalogue.data['magnitude'] - mag) < 1e-8)
        self.assertEqual(computed, expected, msg)
