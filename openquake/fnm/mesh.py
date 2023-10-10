#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numba import jit, njit

from openquake.hazardlib.geo.mesh import Mesh

PI = 3.1415926535
EARTH_RADIUS = 6371.0


def get_min_distance(mesh1: Mesh, mesh2: Mesh) -> float:
    """
    Computes the minimum distance between two meshes. Assumes the coordinates
    are in radians. Reuses code available in the OQ engine.

    :param mesh1:
        A :class:`openquake.hazardlib.geo.mesh.Mesh` instance
    :param mesh2:
        A :class:`openquake.hazardlib.geo.mesh.Mesh` instance
    :returns:
        A float with the minimum distance in km between the two meshes
    """
    return _get_min_distance(mesh1.lons, mesh1.lats, mesh1.depths,
                             mesh2.lons, mesh2.lats, mesh2.depths)


@njit
def _get_min_distance(lon1: np.ndarray, lat1: np.ndarray, dep1: np.ndarray,
                      lon2: np.ndarray, lat2: np.ndarray, dep2: np.ndarray) -> float:

    lo1r = lon1.flatten() * PI / 180.0
    la1r = lat1.flatten() * PI / 180.0
    lo2r = lon2.flatten() * PI / 180.0
    la2r = lat2.flatten() * PI / 180.0
    de1f = dep1.flatten()
    de2f = dep2.flatten()

    mind = 1e100
    for lon, lat, dep in zip(lo1r, la1r, de1f):

        hdists = np.arcsin(np.sqrt(
            np.sin((lat - la2r) / 2.0) ** 2 +
            np.cos(lat) * np.cos(la2r) * np.sin((lon - lo2r) / 2.0) ** 2
        ))
        vdists = dep - de2f
        dists = np.sqrt(hdists ** 2 + vdists ** 2)
        mind = np.min(np.array([mind, np.min(dists)]))

    return mind * 2. * EARTH_RADIUS


def get_mesh_polygon(mesh: Mesh) -> np.ndarray:
    """
    Creates the polygon describing the boundary of the section from the mesh
    coordinates.

    :param lons:
        The mesh longitudes
    :param lats:
        The mesh latitudes
    :param depths:
        The mesh depths
    :returns:
        A :class:`numpy.ndarray` instance
    """
    return _get_mesh_polygon(np.array(mesh.lons), np.array(mesh.lats),
                             np.array(mesh.depths))


@njit
def _get_mesh_polygon(lons: np.ndarray, lats: np.ndarray, deps: np.ndarray) -> np.ndarray:
    # Get the number of points needed to describe the perimeter
    num_points = lons.shape[1] * 2 + lons.shape[0] * 2 - 2
    out = np.zeros((num_points, 3))
    cnt = 0
    for i in np.arange(0, lons.shape[1]):
        out[cnt, 0] = lons[0, i]
        out[cnt, 1] = lats[0, i]
        out[cnt, 2] = deps[0, i]
        cnt += 1
    for i in np.arange(1, lons.shape[0]):
        out[cnt, 0] = lons[i, -1]
        out[cnt, 1] = lats[i, -1]
        out[cnt, 2] = deps[i, -1]
        cnt += 1
    for i in np.arange(lons.shape[1]-1, 0, -1):
        out[cnt, 0] = lons[-1, i]
        out[cnt, 1] = lats[-1, i]
        out[cnt, 2] = deps[-1, i]
        cnt += 1
    for i in np.arange(lons.shape[0]-1, -1, -1):
        out[cnt, 0] = lons[i, 0]
        out[cnt, 1] = lats[i, 0]
        out[cnt, 2] = deps[i, 0]
        cnt += 1
    return out


def get_mesh_bb(mesh):
    """
    Returns a list with the mininum and max longitude and the mininum and max
    latitude.
    """
    return [np.min(mesh.lons), np.max(mesh.lons),
            np.min(mesh.lats), np.max(mesh.lats)]
