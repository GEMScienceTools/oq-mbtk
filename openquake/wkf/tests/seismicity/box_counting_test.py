import os
import tempfile
import unittest
import subprocess

import h3
import numpy as np
import pandas as pd

from openquake.wkf.seismicity.box_counting import (
    boxcounting, get_rate_from_completeness, get_rate_double_truncated_gr)

CODE = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'mbi', 'wkf',
                    'wkf_boxcounting_h3.py')

H3RES = 5
CAT = """eventID,year,month,day,magnitude,longitude,latitude,depth
1,1990,1,1,5.05,10.0,45.0,10
2,1960,1,1,6.2,10.0,45.0,10
3,2010,1,1,4.55,12.0,46.0,10
4,2010,1,1,5.00,20.0,50.0,10
"""
CONFIG = """
[default]
completeness_table = [[1980, 4.0], [1950, 6.0]]

[sources.1]
agr = 4.0
bgr = 1.0

[sources.2]
agr = 3.0
bgr = 1.0
completeness_table = [[2000, 4.5]]
"""
CELL_A = h3.latlng_to_cell(45.0, 10.0, H3RES)
CELL_C = h3.latlng_to_cell(46.0, 12.0, H3RES)
CELL_D = h3.latlng_to_cell(50.0, 20.0, H3RES)


class BoxCountingTestCase(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.fcat = os.path.join(self.tmp, 'cat.csv')
        self.fmap = os.path.join(self.tmp, 'mapping.csv')
        self.fconf = os.path.join(self.tmp, 'config.toml')
        with open(self.fcat, 'w') as fou:
            fou.write(CAT)
        with open(self.fmap, 'w') as fou:
            fou.write(f'{CELL_A},1\n{CELL_C},2\n')
        with open(self.fconf, 'w') as fou:
            fou.write(CONFIG)

    def _run(self, weighting):
        df = boxcounting(self.fcat, H3RES, self.fmap, self.fconf, '', 2020,
                         weighting)
        return {h3.int_to_str(int(k)): v
                for k, v in zip(df.h3idx, df["count"])}

    def test_one(self):
        res = self._run('one')
        self.assertEqual(res, {CELL_A: 2.0, CELL_C: 1.0, CELL_D: 1.0})

    def test_completeness(self):
        res = self._run('completeness')
        self.assertEqual(set(res), {CELL_A, CELL_C})
        np.testing.assert_allclose(res[CELL_A], 1 / 40 + 1 / 70)
        np.testing.assert_allclose(res[CELL_C], 1 / 20)

    def test_mfd(self):
        res = self._run('mfd')
        self.assertEqual(set(res), {CELL_A, CELL_C})
        expA = (10**-1.0 - 10**-1.1) + (10**-2.2 - 10**-2.3)
        np.testing.assert_allclose(res[CELL_A], expA)
        np.testing.assert_allclose(res[CELL_C], 10**-1.5 - 10**-1.6)

    def test_helpers(self):
        compl = [[1980, 4.0], [1950, 6.0]]
        self.assertEqual(get_rate_from_completeness(compl, 3.0, 1990, 2020), 0)
        self.assertEqual(get_rate_from_completeness(compl, 5.0, 1970, 2020), 0)
        np.testing.assert_allclose(
            get_rate_double_truncated_gr(4.0, 1.0, 5.0), 10**-1 - 10**-1.1)

    def test_cli(self):
        fout = os.path.join(self.tmp, 'out')
        cmd = ['python', CODE, self.fcat, self.fmap, self.fconf, str(H3RES),
               fout, '-y', '2020', '-w', 'completeness']
        subprocess.run(cmd, check=True)
        df = pd.read_csv(os.path.join(fout, 'box_counting_cat.csv'))
        self.assertEqual(list(df.columns), ['lon', 'lat', 'count'])
        dfh = pd.read_csv(os.path.join(fout, 'box_counting_h3_cat.csv'))
        self.assertEqual(list(dfh.columns), ['h3idx', 'lon', 'lat', 'count'])
        self.assertIn(h3.str_to_int(CELL_A), dfh.h3idx.astype('uint64').values)
        np.testing.assert_allclose(dfh['count'].sum(), 1/40 + 1/70 + 1/20)
