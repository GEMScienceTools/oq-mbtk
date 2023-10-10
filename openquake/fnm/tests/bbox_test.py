#!/usr/bin/env python
# coding: utf-8

import unittest
import numpy as np

from openquake.fnm.mesh import get_mesh_bb
from openquake.fnm.bbox import get_bb_distance_matrix
from openquake.hazardlib.geo import Line, Point
from openquake.hazardlib.geo.surface.kite_fault import (
    KiteSurface,
    get_profiles_from_simple_fault_data,
)

PLOTTING = False


class TestDistanceMatrix(unittest.TestCase):
    def setUp(self):
        mesh_spacing = 2.5
        profile_sd = 2.5
        edge_sd = 5.0

        # Create the Kite Fault Surface
        usd = 0
        lsd = 12.0
        dip = 80.0
        fault_trace = Line([Point(10.0, 45.0), Point(10.3, 45.0)])
        profiles = get_profiles_from_simple_fault_data(
            fault_trace, usd, lsd, dip, mesh_spacing
        )
        self.surf1 = KiteSurface.from_profiles(profiles, profile_sd, edge_sd)

        # Create the Kite Fault Surface
        usd = 0
        lsd = 12.0
        dip = 60.0
        fault_trace = Line([Point(9.9, 45.2), Point(10.2, 45.3)])
        profiles = get_profiles_from_simple_fault_data(
            fault_trace, usd, lsd, dip, mesh_spacing
        )
        self.surf2 = KiteSurface.from_profiles(profiles, profile_sd, edge_sd)

        # Create the Kite Fault Surface
        usd = 0
        lsd = 12.0
        dip = 90.0
        fault_trace = Line([Point(10.3, 45.4), Point(10.2, 45.7)])
        profiles = get_profiles_from_simple_fault_data(
            fault_trace, usd, lsd, dip, mesh_spacing
        )
        self.surf3 = KiteSurface.from_profiles(profiles, profile_sd, edge_sd)

        # Create the Kite Fault Surface
        usd = 0
        lsd = 12.0
        dip = 90.0
        fault_trace = Line([Point(10.1, 45.25), Point(10.3, 45.25)])
        profiles = get_profiles_from_simple_fault_data(
            fault_trace, usd, lsd, dip, mesh_spacing
        )
        self.surf4 = KiteSurface.from_profiles(profiles, profile_sd, edge_sd)

    def test_get_dist_matrix(self):
        """ """
        bboxes = [
            get_mesh_bb(self.surf1.mesh),
            get_mesh_bb(self.surf2.mesh),
            get_mesh_bb(self.surf3.mesh),
            get_mesh_bb(self.surf4.mesh),
        ]
        dmtx = get_bb_distance_matrix(bboxes)

        if PLOTTING:
            import matplotlib.pyplot as plt

            fig, axs = plt.subplots(1, 1)
            surfs = [self.surf1, self.surf2, self.surf3, self.surf4]
            colors = ["red", "green", "blue", "orange"]
            for i, surf in enumerate(surfs):
                plt.plot(surf.mesh.lons, surf.mesh.lats, ".", color=colors[i])
            for i, bbox in enumerate(bboxes):
                plt.hlines([bbox[2], bbox[3]], bbox[0], bbox[1], linewidths=0.5)
                plt.vlines([bbox[0], bbox[1]], bbox[2], bbox[3], linewidths=0.5)
                plt.text(bbox[0], bbox[2], f"{i}")
            plt.show()

        # Testing the distance matrix - Using the service available at NOAA
        # https://www.nhc.noaa.gov/gccalc.shtml the distance between (45, 10)
        # and (45.2, 10) is 22 km. This corresponds (with some minor
        # differences) to the distance between bbox 0 and bbox 1. The distance
        # between bbox 1 and 3 is 0, as expected since they intersect.
        expected = np.array([[0.0, 16.5839325, 44.4673135, 27.7674462],
                             [0.0,  0.00000000, 11.5560295,  0.00000000],
                             [0.0,  0.00000000,  0.00000000, 16.6747524],
                             [0.0,  0.00000000,  0.00000000,  0.00000000]])
        np.testing.assert_almost_equal(expected, dmtx)
