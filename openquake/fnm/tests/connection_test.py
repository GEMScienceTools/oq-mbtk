#!/usr/bin/env python
# coding: utf-8

import pathlib
import unittest
import numpy as np

from openquake.fnm.plot import plot
from openquake.fnm.mesh import get_mesh_bb
from openquake.fnm.connections import get_connections, check_neighbors
from openquake.fnm.fault_system import get_fault_system
from openquake.fnm.bbox import get_bb_distance_matrix
from openquake.fnm.importer import kite_surfaces_from_geojson

from openquake.hazardlib.geo import Line, Point
from openquake.hazardlib.geo.surface.kite_fault import (
    KiteSurface, get_profiles_from_simple_fault_data)

PLOTTING = False
HERE = pathlib.Path(__file__).parent


def _get_surfs():

    mesh_spacing = 2.0
    profile_sd = 1.0
    edge_sd = 1.0

    # Create the Kite Fault Surface
    usd = 0
    lsd = 12.0
    dip = 80.0
    fault_trace = Line([Point(10.3, 45.0), Point(10.0, 45.0)])
    profiles = get_profiles_from_simple_fault_data(
        fault_trace, usd, lsd, dip, mesh_spacing)
    surf0 = KiteSurface.from_profiles(profiles, profile_sd, edge_sd)

    # Create the Kite Fault Surface
    usd = 0
    lsd = 12.0
    dip = 80.0
    fault_trace = Line([Point(10.6, 45.1), Point(10.3, 45.1)])
    profiles = get_profiles_from_simple_fault_data(
        fault_trace, usd, lsd, dip, mesh_spacing)
    surf1 = KiteSurface.from_profiles(profiles, profile_sd, edge_sd)

    # Create the Kite Fault Surface
    usd = 0
    lsd = 12.0
    dip = 80.0
    fault_trace = Line([Point(10.9, 45.22), Point(10.6, 45.22)])
    profiles = get_profiles_from_simple_fault_data(
        fault_trace, usd, lsd, dip, mesh_spacing)
    surf2 = KiteSurface.from_profiles(profiles, profile_sd, edge_sd)

    return [surf0, surf1, surf2]


class TestCheckNeighbors(unittest.TestCase):

    def test_lower_right_subsec(self):
        """ Test subsection located in the lower right part of the surface """
        mesh = np.zeros((8, 10, 3))
        cell = np.array([4, 5, 4, 3])
        computed = check_neighbors(mesh, cell)
        # Since the subsection is in the lower right part of the surface it has
        # neighbors on top and left.
        expected = 9
        np.testing.assert_equal(computed, expected)

    def test_full_subsection(self):
        """ Test subsection covering the entire surface """
        mesh = np.zeros((8, 10, 3))
        cell = np.array([0, 0, 10, 8])
        computed = check_neighbors(mesh, cell)
        # Since the subsection is in the lower right part of the surface it has
        # neighbors on top and left.
        expected = 0
        np.testing.assert_equal(computed, expected)


class TestFindConnections(unittest.TestCase):

    def test_connection_filter_angle(self):
        """Test select connections by angle"""
        # The fault system used in this test is an idealised one.
        fname = HERE / 'data' / 'test_system.geojson'
        surfs = kite_surfaces_from_geojson(fname, 2)
        surfs = [surfs[4], surfs[5]]

        subs_size = [-0.5, -1]
        fsys = get_fault_system(surfs, subs_size)

        criteria = {'max_connection_angle': {'threshold': 60.}}
        binm = np.ones((len(fsys), len(fsys)))
        _, _, _ = get_connections(fsys, binm, criteria)

        # TODO add a proper test

    def test_connection_by_distance(self):
        """Test connections by distance """

        # Set the size of subsections, get the surfaces representing sections,
        # compute the BBox of each section and create the fault system
        subs_size = [-0.5, -1]
        surfs = _get_surfs()
        bboxes = [get_mesh_bb(surf.mesh) for surf in surfs]
        fsys = get_fault_system(surfs, subs_size)

        # Get the bboxes distance matrix. The binary matrix `binm` is true when
        # the distance between the bounding boxes for two sections is shorter
        # than the threshold distance
        dmtx = get_bb_distance_matrix(bboxes)
        binm = np.zeros_like(dmtx)
        threshold = 20.0  # This is a threshold distance in km
        binm[dmtx < threshold] = 1

        # Set the criteria
        key = 'threshold_distance'
        criteria = {'min_distance_between_subsections': {key: 20.}}

        # Get the connections
        conns, _, _ = get_connections(fsys, binm, criteria)

        expected = np.array([[0., 1., 0., 0., 6., 12., 0., 18., 6., 12.],
                             [1., 2., 0., 0., 6., 12., 0., 18., 6., 12.]])
        np.testing.assert_array_equal(conns, expected)

        if PLOTTING:
            meshes = [s.mesh for s in surfs]
            plot(meshes, connections=conns)

    def test_connection_by_distance_only_closest(self):
        """Test connections by distance: only the two closest subsections """

        # Set the size of subsections, get the surfaces representing sections,
        # compute the BBox of each section and create the fault system
        subs_size = [-0.5, -1]
        surfs = _get_surfs()
        bboxes = [get_mesh_bb(surf.mesh) for surf in surfs]
        fsys = get_fault_system(surfs, subs_size)

        # Get the bboxes distance matrix. The binary matrix `binm` is true when
        # the distance between the bounding boxes for two sections is shorter
        # than the threshold distance
        dmtx = get_bb_distance_matrix(bboxes)
        binm = np.zeros_like(dmtx)
        threshold = 30.0  # This is a threshold distance in km
        binm[dmtx < threshold] = 1

        # Get the connections
        key = 'threshold_distance'
        criteria = {'min_distance_between_subsections': {key: 20.}}
        conns, _, _ = get_connections(fsys, binm, criteria)

        if PLOTTING:
            meshes = [s.mesh for s in surfs]
            plot(meshes, connections=conns)

    def test_connection_kunlun_2_sections(self):
        """Test connections for 2 faults in Kunlun """

        subs_size = [-0.5, -1]
        fname = HERE / 'data' / 'kunlun_faults.geojson'
        surfs = kite_surfaces_from_geojson(fname, 2)
        surfs = [surfs[9], surfs[4]]
        bboxes = [get_mesh_bb(surf.mesh) for surf in surfs]
        fsys = get_fault_system(surfs, subs_size)

        # Get the bboxes distance matrix. The binary matrix `binm` is true when
        # the distance between the bounding boxes for two sections is shorter
        # than the threshold distance
        dmtx = get_bb_distance_matrix(bboxes)
        binm = np.zeros_like(dmtx)
        threshold = 10.0  # This is a threshold distance in km
        binm[dmtx < threshold] = 1

        # Get the connections
        key = 'threshold_distance'
        criteria = {'min_distance_between_subsections': {key: 20.}}
        conns, _, _ = get_connections(fsys, binm, criteria)

        if PLOTTING:
            meshes = [s.mesh for s in surfs]
            plot(meshes, connections=conns)

    def test_connection_kunlun_all(self):
        """Test connections forKunlun faults """

        # Set the size of subsections, get the surfaces representing sections,
        # compute the BBox of each section and create the fault system
        subs_size = [-0.5, -1]
        fname = HERE / 'data' / 'kunlun_faults.geojson'
        surfs = kite_surfaces_from_geojson(fname, 2)
        bboxes = [get_mesh_bb(surf.mesh) for surf in surfs]
        fsys = get_fault_system(surfs, subs_size)

        if PLOTTING:
            _plot(surfs, bboxes)

        # Get the bboxes distance matrix. The binary matrix `binm` is true when
        # the distance between the bounding boxes for two sections is shorter
        # than the threshold distance
        dmtx = get_bb_distance_matrix(bboxes)
        binm = np.zeros_like(dmtx)
        threshold = 10.0  # This is a threshold distance in km
        binm[dmtx < threshold] = 1

        # Get the connections
        key = 'threshold_distance'
        criteria = {'min_distance_between_subsections': {key: 20.}}
        conns, _, _ = get_connections(fsys, binm, criteria)

        if PLOTTING:
            meshes = [s.mesh for s in surfs]
            plot(meshes, connections=conns)

    def test_connection_kunlun_triple(self):
        """Test connections for Kunlun triple junction"""

        # Set the size of subsections, get the surfaces representing sections,
        # compute the BBox of each section and create the fault system
        subs_size = [-0.5, -1]
        fname = HERE / 'data' / 'kunlun_faults.geojson'
        surfs = kite_surfaces_from_geojson(fname, 2)
        surfs = [surfs[3], surfs[8], surfs[9]]

        bboxes = [get_mesh_bb(surf.mesh) for surf in surfs]
        fsys = get_fault_system(surfs, subs_size)

        # Get the bboxes distance matrix. The binary matrix `binm` is true when
        # the distance between the bounding boxes for two sections is shorter
        # than the threshold distance
        dmtx = get_bb_distance_matrix(bboxes)
        binm = np.zeros_like(dmtx)
        threshold = 10.0  # This is a threshold distance in km
        binm[dmtx < threshold] = 1

        # Get the connections
        key = 'threshold_distance'
        criteria = {'min_distance_between_subsections': {key: 20.}}
        conns, _, _ = get_connections(fsys, binm, criteria)
        self.assertEqual(len(conns), 3)

        # Get the connections
        criteria = {'min_distance_between_subsections': {key: 20.},
                    'max_connection_angle': {'threshold': 60.}}
        conns, _, _ = get_connections(fsys, binm, criteria)
        # self.assertEqual(len(conns), 2)

        if PLOTTING:
            meshes = [s.mesh for s in surfs]
            plot(meshes, connections=conns)

    def test_connection_kunlun_double(self):
        """Test connections for Kunlun junction"""

        # Set the size of subsections, get the surfaces representing sections,
        # compute the BBox of each section and create the fault system
        subs_size = [-0.5, -1]
        fname = HERE / 'data' / 'kunlun_faults.geojson'
        surfs = kite_surfaces_from_geojson(fname, 2)
        # surfs = [surfs[3], surfs[8], surfs[9]]
        surfs = [surfs[3], surfs[9]]

        bboxes = [get_mesh_bb(surf.mesh) for surf in surfs]
        fsys = get_fault_system(surfs, subs_size)

        # Get the bboxes distance matrix. The binary matrix `binm` is true when
        # the distance between the bounding boxes for two sections is shorter
        # than the threshold distance
        dmtx = get_bb_distance_matrix(bboxes)
        binm = np.zeros_like(dmtx)
        threshold = 10.0  # This is a threshold distance in km
        binm[dmtx < threshold] = 1

        # Get the connections
        key = 'threshold_distance'
        criteria = {'min_distance_between_subsections': {key: 20.},
                    'max_connection_angle': {'threshold': 60.}}
        conns, _, _ = get_connections(fsys, binm, criteria)
        # self.assertEqual(len(conns), 2)

        if PLOTTING:
            meshes = [s.mesh for s in surfs]
            plot(meshes, connections=conns)

    @unittest.skip('takes a lot of time')
    def test_connection_atf_haiyuan_all(self):
        """Test connections for atf haiyuan fault system """

        # Set the size of subsections, get the surfaces representing sections,
        # compute the BBox of each section and create the fault system
        subs_size = [-0.5, -1]

        # Error between 20 and 23
        fname = HERE / 'data' / 'atf_haiyuan_fault_system.geojson'

        idxs = [i for i in range(0, 50)]
        idxs.extend([i for i in range(55, 91)])
        skip = [21, 22]
        iplot = []

        # Create fault surfaces
        surfs = kite_surfaces_from_geojson(fname, 2, idxs=idxs, skip=skip,
                                           iplot=iplot)
        bboxes = [get_mesh_bb(surf.mesh) for surf in surfs]
        fsys = get_fault_system(surfs, subs_size)

        if PLOTTING:
            _plot(surfs, bboxes)

        # Get the bboxes distance matrix. The binary matrix `binm` is true when
        # the distance between the bounding boxes for two sections is shorter
        # than the threshold distance
        dmtx = get_bb_distance_matrix(bboxes)
        binm = np.zeros_like(dmtx)
        threshold = 10.0  # This is a threshold distance in km
        binm[dmtx < threshold] = 1

        # Get the connections
        key = 'threshold_distance'
        criteria = {'min_distance_between_subsections': {key: 20.}}
        conns, _, _ = get_connections(fsys, binm, criteria)

        # TODO add a check for repeatability

        if PLOTTING:
            meshes = [s.mesh for s in surfs]
            plot(meshes, connections=conns)


def _plot(surfs, bboxes):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 1)
    colors = np.random.rand(len(surfs), 3)
    for i, surf in enumerate(surfs):
        plt.plot(surf.mesh.lons, surf.mesh.lats, '.', color=colors[i])
    for i, bbox in enumerate(bboxes):
        plt.hlines([bbox[2], bbox[3]], bbox[0], bbox[1], linewidths=0.5)
        plt.vlines([bbox[0], bbox[1]], bbox[2], bbox[3], linewidths=0.5)
        plt.text(bbox[0], bbox[2], f'{i}')
    plt.show()
