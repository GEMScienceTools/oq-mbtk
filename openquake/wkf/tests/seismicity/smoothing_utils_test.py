import os
import sys
import tempfile
import unittest
import subprocess

import h3
import numpy as np
import pandas as pd

from openquake.wkf.seismicity.smoothing_utils import (
    smoothing, smooth_counts, normal_pdf, get_kernel_weights,
    get_cell_distances, get_avg_edge_length_km)

CODE = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'mbi', 'wkf',
                    'wkf_smoothing.py')

H3RES = 5
SIGMAS = [[0.6, 10.0], [0.4, 25.0]]
MAXDISTKM = 40.0
CONFIG = """
kernel_maximum_distance = 40.0
kernel_smoothing = [[0.6, 10.0], [0.4, 25.0]]
"""
CELL_A = h3.latlng_to_cell(45.0, 10.0, H3RES)
CELL_B = h3.latlng_to_cell(45.3, 10.4, H3RES)
CELL_C = h3.latlng_to_cell(50.0, 20.0, H3RES)


def _write_count(fname, cells_counts):
    rows = []
    for cell, cnt in cells_counts:
        lat, lon = h3.cell_to_latlng(cell)
        rows.append((h3.str_to_int(cell), lon, lat, cnt))
    df = pd.DataFrame(rows, columns=['h3idx', 'lon', 'lat', 'count'])
    df.to_csv(fname, index=False)


def _to_cells(df):
    """ Returns a dictionary cell -> nocc """
    return {h3.latlng_to_cell(lat, lon, H3RES): v
            for lon, lat, v in zip(df.lon, df.lat, df.nocc)}


class SmoothingTestCase(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.fcount = os.path.join(self.tmp, 'box_counting_h3_cat.csv')
        self.fconf = os.path.join(self.tmp, 'config.toml')
        _write_count(self.fcount, [(CELL_A, 3.0), (CELL_B, 1.5),
                                   (CELL_C, 2.0)])
        with open(self.fconf, 'w') as fou:
            fou.write(CONFIG)
        self.edge = get_avg_edge_length_km(H3RES)
        self.maxk = int(np.ceil(MAXDISTKM / self.edge))

    def test_helpers(self):
        # Normal pdf
        np.testing.assert_allclose(normal_pdf(0.0, 1.0), 1 / np.sqrt(2*np.pi))
        np.testing.assert_allclose(normal_pdf(2.0, 2.0),
                                   np.exp(-0.5) / (2 * np.sqrt(2 * np.pi)))
        # Kernel weights
        dsts = np.array([1.0, 10.0, 20.0, 30.0])
        wei = get_kernel_weights(dsts, SIGMAS)
        exp = (0.6 * normal_pdf(dsts, 10.0) + 0.4 * normal_pdf(dsts, 25.0))
        np.testing.assert_allclose(wei, exp / exp.sum())
        self.assertTrue(np.all(np.diff(wei) < 0))
        # Distances: the minimum is 1 km and they scale with the grid
        # distance
        idxs = [CELL_A] + list(h3.grid_ring(CELL_A, 2))
        dsts = get_cell_distances(CELL_A, idxs, self.edge)
        self.assertEqual(dsts[0], 1.0)
        np.testing.assert_allclose(dsts[1:], 2 * self.edge)

    def test_single_cell(self):
        # A single cell: the count is distributed over the cells within
        # `maxk` rings using weights that depend only on the ring
        _write_count(self.fcount, [(CELL_A, 3.0)])
        res = _to_cells(smooth_counts(self.fcount, SIGMAS, MAXDISTKM))
        disk = set(h3.grid_disk(CELL_A, self.maxk))
        self.assertEqual(set(res), disk)
        np.testing.assert_allclose(sum(res.values()), 3.0)

        dsts = np.maximum(
            np.array([h3.grid_distance(CELL_A, c) for c in disk]) *
            self.edge, 1.0)
        wei = 0.6 * normal_pdf(dsts, 10.0) + 0.4 * normal_pdf(dsts, 25.0)
        wei = 3.0 * wei / wei.sum()
        exp = dict(zip(disk, wei))
        for cell in disk:
            np.testing.assert_allclose(res[cell], exp[cell], rtol=1e-10)

    def test_multiple_cells(self):
        # Results are the sum of the contributions of each cell
        smo = smooth_counts(self.fcount, SIGMAS, MAXDISTKM)
        res = _to_cells(smo)
        self.assertEqual(len(res), len(smo))
        np.testing.assert_allclose(smo.nocc.sum(), 6.5)

        exp = {}
        for cell, cnt in [(CELL_A, 3.0), (CELL_B, 1.5), (CELL_C, 2.0)]:
            fname = os.path.join(self.tmp, 'single.csv')
            _write_count(fname, [(cell, cnt)])
            for k, v in _to_cells(
                    smooth_counts(fname, SIGMAS, MAXDISTKM)).items():
                exp[k] = exp.get(k, 0.0) + v
        self.assertEqual(set(res), set(exp))
        for cell in exp:
            np.testing.assert_allclose(res[cell], exp[cell], rtol=1e-10)

    def test_smoothing_config(self):
        fout = os.path.join(self.tmp, 'smooth.csv')
        smoothing(self.fcount, self.fconf, fout)
        df = pd.read_csv(fout)
        self.assertEqual(list(df.columns), ['lon', 'lat', 'nocc'])
        exp = smooth_counts(self.fcount, SIGMAS, MAXDISTKM)
        np.testing.assert_allclose(df.values, exp.values)

    def test_cli(self):
        fout = os.path.join(self.tmp, 'out', 'smooth.csv')
        cmd = [sys.executable, CODE, self.fcount, self.fconf, fout]
        subprocess.run(cmd, check=True)
        df = pd.read_csv(fout)
        self.assertEqual(list(df.columns), ['lon', 'lat', 'nocc'])
        np.testing.assert_allclose(df.nocc.sum(), 6.5)
