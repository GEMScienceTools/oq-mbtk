import os
import unittest

import openquake.man.tools.csv_output as csv
from openquake.man.tools.csv_output import mean_mde_for_gmt
from openquake.calculators.tests import open8, CalculatorTestCase
from openquake.calculators.export import export
import case_8

BASE_DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
BASE_EXP_PATH = os.path.join(os.path.dirname(__file__), 'expected')



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
        self.assertEqual(len(expected_lines), len(actual_lines))
        for ii in range(len(actual_lines)):
            self.assertEqual(expected_lines[ii], actual_lines[ii])
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
        self.assertEqual(len(expected_lines), len(actual_lines))
        for ii in range(len(actual_lines)):
            self.assertEqual(expected_lines[ii], actual_lines[ii])
        os.remove(fout)

class TestOutputFormat(CalculatorTestCase):

    def test_mde_format(self):
        """
        will fail if the output format changes
        """
        BASE_CASE8 = os.path.join(os.path.dirname(__file__), 'case_8')
        # run test job
        self.run_calc('',  'case_8/job.ini')
        # test mre results output format
        [fname] = export(('disagg-stats', 'csv'), self.calc.datastore)
        self.assertEqualFiles(os.path.join(BASE_CASE8, 'expected/Mag_Dist_Eps-mean-0.csv'), fname)


class TestMDeOutput(unittest.TestCase):

    def test_read_mre(self):
        fname = os.path.join(BASE_DATA_PATH, 'mde.csv')


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
        #
        dat = [[6.05, 21], [6.15, 11], [6.25, 8], [6.35, 11]]
        for mag, expected in dat:
            fmt = 'Number of events with magnitude {:.2f} not matching'
            msg = fmt.format(mag)
            computed = sum(abs(catalogue.data['magnitude'] - mag) < 1e-8)
        self.assertEqual(computed, expected, msg)
