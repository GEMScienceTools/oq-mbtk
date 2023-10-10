#!/usr/bin/env python
# coding: utf-8

import unittest
import numpy as np
from openquake.fnm.rupture import get_ruptures_section


class TestRupturesOneSection(unittest.TestCase):

    def test_from_set_sections(self):

        tmp = np.array([[[0.,  0., 6., 12.],
                         [0.,  6., 6., 12.],
                         [0., 12., 6., 12.],
                         [0., 18., 6., 12.]]])
        rups = get_ruptures_section(tmp)
        expected = np.array([[0,  0,  6, 12, 0, 1, 0, 0],
                             [0,  0, 12, 12, 1, 1, 0, 1],
                             [0,  0, 18, 12, 2, 1, 0, 2],
                             [0,  0, 24, 12, 3, 1, 0, 3],
                             [0,  6,  6, 12, 4, 1, 0, 4],
                             [0,  6, 12, 12, 5, 1, 0, 5],
                             [0,  6, 18, 12, 6, 1, 0, 6],
                             [0, 12,  6, 12, 7, 1, 0, 7],
                             [0, 12, 12, 12, 8, 1, 0, 8],
                             [0, 18,  6, 12, 9, 1, 0, 9]])
        np.testing.assert_array_equal(rups, expected)
