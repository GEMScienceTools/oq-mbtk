#!/usr/bin/env python
# coding: utf-8

import pathlib
import unittest
import numpy as np

from openquake.fnm.mesh import get_mesh_bb
from openquake.fnm.bbox import get_bb_distance_matrix
from openquake.fnm.connections import get_connections
from openquake.fnm.fault_system import (
    get_fault_system, get_connection_rupture_table)
from openquake.fnm.tests.connection_test import _get_surfs
from openquake.fnm.rupture import (_get_ruptures_first_level,
                                   _check_rupture_has_connections)

PLOTTING = False
HERE = pathlib.Path(__file__).parent


class TestCreateRupturesConnectionTable(unittest.TestCase):

    def test_three_sections(self):
        """ Tests the construction of the rupcon table for three sections """

        surfs = _get_surfs()
        subs_size = [-0.5, -1]
        threshold = 20.0

        bboxes = [get_mesh_bb(surf.mesh) for surf in surfs]
        fsys = get_fault_system(surfs, subs_size)

        dmtx = get_bb_distance_matrix(bboxes)
        binm = np.zeros_like(dmtx)
        binm[dmtx < threshold] = 1

        # Get the connections
        key = 'threshold_distance'
        criteria = {'min_distance_between_subsections': {key: 15.}}
        conns, _, _ = get_connections(fsys, binm, criteria)

        aratios = np.array([0, 100], dtype=int)
        fsys = np.array(fsys, dtype=object)
        rups = _get_ruptures_first_level(fsys, aratios)

        computed = get_connection_rupture_table(rups, conns)

        # The expected set of connections must include all the ruptures in the
        # first section with UL origin in 0, 0. In the second section (the one
        # the middle) the ruptures required are the ones including either the
        # leftmost or rightmost subsection - or both. The third section must
        # include the leftmost subsection
        expected = np.array([[0, 0, 0, 0],
                             [0, 1, 0, 1],
                             [0, 2, 0, 2],
                             [0, 3, 0, 3],
                             [1, 0, 1, 10],
                             [1, 1, 1, 11],
                             [1, 2, 1, 12],
                             [1, 3, 0, 13],
                             [1, 3, 1, 13],
                             [1, 6, 0, 16],
                             [1, 8, 0, 18],
                             [1, 9, 0, 19],
                             [2, 3, 1, 23],
                             [2, 6, 1, 26],
                             [2, 8, 1, 28],
                             [2, 9, 1, 29]])

        aae = np.testing.assert_array_equal
        aae(computed, expected)


class TestGetConnection(unittest.TestCase):
    """ This class contains tests on the connections for ruptures """

    def test_rup_has_conn(self):
        """ Test the check for connections """
        conns = np.array([[0, 1, 0, 0, 6, 12, 0, 18, 6, 12],
                          [1, 2, 0, 0, 6, 12, 0, 18, 6, 12]])
        rup = np.array([0., 0., 24., 12., 23., 1., 2., 3.])
        computed = _check_rupture_has_connections(conns, rup)
        expected = np.array([[1, 1, 1, 1]])
        aae = np.testing.assert_array_equal
        aae(computed, expected)
