"""
"""

import os

import numpy as np
import unittest

from openquake.hazardlib.geo.geodetic import distance

from openquake.sub.misc.edge import _read_edge, _resample_edge

BASE_DATA_PATH = os.path.dirname(__file__)


class TrapezoidalCellsSurfaceTest(unittest.TestCase):

    def setUp(self):
        filename = os.path.join(BASE_DATA_PATH, 'dat/edge/edge_000.csv')
        self.edge = _read_edge(filename)
