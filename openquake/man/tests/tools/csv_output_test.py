import os
import unittest

import openquake.man.tools.csv_output as csv

BASE_DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')




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
