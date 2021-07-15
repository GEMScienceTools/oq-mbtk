import os
import numpy
import pandas as pd
import filecmp
import unittest
import tempfile
import matplotlib.pyplot as plt
from openquake.sub.cross_sections import Trench
from openquake.hazardlib.geo.geodetic import point_at
from openquake.sub.create_multiple_cross_sections import get_cs

PLOT = False
BASE_PATH = os.path.dirname(__file__)


class CrossSectionIntersectionTest(unittest.TestCase):

    def setUp(self):
        trench_fname = os.path.join(BASE_PATH, 'data', 'traces', 'trench.xyz')
        data = numpy.loadtxt(trench_fname)
        self.trench = Trench(data)

    def test01(self):
        tmp_fname = 'trash'
        cs_length = 400
        cs_depth = 100
        intd = 100
        handle, tmp_fname = tempfile.mkstemp()
        print(tmp_fname)
        get_cs(self.trench, 'tmp.txt', cs_length, cs_depth, intd, 0, tmp_fname)

        if PLOT:
            _ = plt.figure()
            columns = ['lon', 'lat', 'dep', 'len', 'azim', 'id', 'fname']
            df = pd.read_csv(tmp_fname, names=columns, delimiter=' ')
            for i, row in df.iterrows():
                ex, ey = point_at(row.lon, row.lat, row.azim, row.len)
                plt.plot([row.lon], [row.lat], 'o')
                plt.text(row.lon, row.lat, '{:d}'.format(row.id))
                plt.plot([row.lon, ex], [row.lat, ey], '-')
            plt.show()

        expected = os.path.join(BASE_PATH, 'data', 'traces', 'expected.txt')
        msg = 'The two files do not match'
        self.assertTrue(filecmp.cmp(tmp_fname, expected), msg)
