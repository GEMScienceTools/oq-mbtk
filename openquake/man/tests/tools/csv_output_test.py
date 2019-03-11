import os
import unittest

import openquake.man.tools.csv_output as csv

BASE_DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


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

