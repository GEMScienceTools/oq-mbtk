
import os
import shutil
import unittest
import tempfile
import numpy as np

import openquake.man.tools.csv_output as csv
from openquake.man.tools.csv_output import mean_mde_for_gmt
from openquake.calculators.tests import open8
from openquake.calculators.export import export
from openquake.calculators.base import run_calc

BASE_DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
BASE_EXP_PATH = os.path.join(os.path.dirname(__file__), 'expected')
BASE_CASE8 = os.path.join(os.path.dirname(__file__), 'case_8')

OVERWRITE = False


class TestMeanMDE(unittest.TestCase):

    def test_output_mre(self):
        """
        test reorg of one instance of an MDE file; one rlz
        """
        fname = os.path.join(BASE_DATA_PATH, 'Mag_Dist_Eps-1.csv')
        fout = 'test-1.csv'
        mean_mde_for_gmt(fname, fout, 0.002105, 'SA(0.1)', 1e-10)
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
        mean_mde_for_gmt(fname, fout, 0.002105, 'SA(0.1)', 1e-10)
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

        mean_mde_for_gmt(fname1, path1, 0.002105, 'SA(0.1)', 1e-10)
        mean_mde_for_gmt(fname2, path2, 0.002105, 'SA(0.1)', 1e-10)
        expect_lines1 = [[float(j) for j in i.split()] for i in open8(path1)]
        expect_lines2 = [[float(j) for j in i.split()] for i in open8(path2)]
        expect_lines1 = np.array(expect_lines1)
        expect_lines2 = np.array(expect_lines2)
        aae = np.testing.assert_almost_equal
        aae(expect_lines1, expect_lines2, decimal=4)


class OutputTestCase(unittest.TestCase):

    def test_mde_format(self):
        """
        will fail if the output format changes
        """
        # run test job
        calc = run_calc(os.path.join(BASE_CASE8, 'job.ini'))

        # test mre results output format
        [fname] = export(('disagg-stats', 'csv'), calc.datastore)
        expected = os.path.join(BASE_CASE8, 'expected/Mag_Dist_Eps-mean-0.csv')

        if OVERWRITE:
            shutil.copyfile(fname, expected)

        expected_lines = [line for line in open8(expected)]
        actual_lines = [line for line in open8(fname)]
        assert expected_lines[1:] == actual_lines[1:]
        os.remove(fname)


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
