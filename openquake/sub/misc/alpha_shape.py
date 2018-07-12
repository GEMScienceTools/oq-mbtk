"""
:mod:`ccar18.utils.alpha_shape` module. Tool for computing the alpha shape of
a cloud of points
"""

import math
import numpy as np
import shapely.geometry as geometry

from shapely.ops import cascaded_union, polygonize
from scipy.spatial import Delaunay


def _add_edge(edges, edge_points, coords, i, j):
    """
    Add a line between the i-th and j-th points,
    if not in the list already
    """
    if (i, j) in edges or (j, i) in edges:
        # already added
        return
    edges.add((i, j))
    edge_points.append(coords[[i, j]])


def alpha_shape(xco, yco, alpha):
    """
    Compute the alpha shape (concave hull) of a set
    of points.

    Code from:
        http://blog.thehumangeo.com/2014/05/12/drawing-boundaries-in-python/

    :param points:
        A numpy array nx2
    :param alpha:
        Alpha value to influence the gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers. Too large, and you lose
        everything!
    """
    #
    # create points
    points = [geometry.Point(x, y) for x, y in zip(xco, yco)]
    #
    #
    if len(points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return geometry.MultiPoint(list(points)).convex_hull

    coords = np.array([point.coords[0] for point in points])
    tri = Delaunay(coords)
    edges = set()
    edge_points = []
    #
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]
        #
        # Lengths of sides of triangle
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)
        #
        # Semiperimeter of triangle
        s = (a + b + c)/2.0
        #
        # Area of triangle by Heron's formula
        area = math.sqrt(s*(s-a)*(s-b)*(s-c))
        circum_r = a*b*c/(4.0*area)
        #
        # Here's the radius filter.
        if circum_r < 1.0/alpha:
            _add_edge(edges, edge_points, coords, ia, ib)
            _add_edge(edges, edge_points, coords, ib, ic)
            _add_edge(edges, edge_points, coords, ic, ia)
    #
    #
    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    #
    #
    return cascaded_union(triangles), edge_points
