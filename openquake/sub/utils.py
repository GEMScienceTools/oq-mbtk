# ------------------- The OpenQuake Model Building Toolkit --------------------
# Copyright (C) 2022 GEM Foundation
#           _______  _______        __   __  _______  _______  ___   _
#          |       ||       |      |  |_|  ||  _    ||       ||   | | |
#          |   _   ||   _   | ____ |       || |_|   ||_     _||   |_| |
#          |  | |  ||  | |  ||____||       ||       |  |   |  |      _|
#          |  |_|  ||  |_|  |      |       ||  _   |   |   |  |     |_
#          |       ||      |       | ||_|| || |_|   |  |   |  |    _  |
#          |_______||____||_|      |_|   |_||_______|  |___|  |___| |_|
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------
# vim: tabstop=4 shiftwidth=4 softtabstop=4
# coding: utf-8

import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt

from pyproj import Proj, CRS

from mpl_toolkits.mplot3d import Axes3D
from openquake.hazardlib.geo import Line, Point
from openquake.hazardlib.geo.surface import ComplexFaultSurface, KiteSurface
from openquake.hazardlib.scalerel.wc1994 import WC1994
from openquake.hazardlib.geo.utils import plane_fit
from openquake.sub.profiles import ProfileSet


def mecclass(plungt, plungb, plungp):
    """
    This is taken from the FMC package.
    See https://josealvarezgomez.wordpress.com/

    It provides a classification of the rupture mechanism based on the
    Kaverina et al. (1996) methodology.

    :parameter plungt:
    :parameter plungb:
    :parameter plungp:

    """
    plunges = np.asarray((plungp, plungb, plungt))
    P = plunges[0]
    B = plunges[1]
    T = plunges[2]
    maxplung, axis = plunges.max(0), plunges.argmax(0)
    if maxplung >= 67.5:
        if axis == 0:  # P max
            clase = 'N'  # normal faulting
        elif axis == 1:  # B max
            clase = 'SS'  # strike-slip faulting
        elif axis == 2:  # T max
            clase = 'R'  # reverse faulting
    else:
        if axis == 0:  # P max
            if B > T:
                clase = 'N-SS'  # normal - strike-slip faulting
            else:
                clase = 'N'  # normal faulting
        if axis == 1:  # B max
            if P > T:
                clase = 'SS-N'  # strike-slip - normal faulting
            else:
                clase = 'SS-R'  # strike-slip - reverse faulting
        if axis == 2:  # T max
            if B > P:
                clase = 'R-SS'  # reverse - strike-slip faulting
            else:
                clase = 'R'  # reverse faulting
    return clase


def get_direction_cosines(strike, dip):
    """
    Compute the direction cosines of the plane defined by the strike-dip tuple.

    :parameter strike:
        Strike of the plane. Defined using the right hand rule
    :parameter dip:
        Dip of the plane. Defined using the right hand rule
    :return:
        A 3x1 array containing the direction cosines of the normal to the plane
    """
    if dip < 89.99:
        c = np.cos(np.radians(dip))
        h = np.sin(np.radians(dip))
    else:
        c = 0.
        h = 1.
    a = h * np.sin(np.radians(strike + 90.))
    b = h * np.cos(np.radians(strike + 90.))
    den = np.sqrt(a**2. + b**2. + c**2.)
    a /= den
    b /= den
    c /= den
    return a, b, c


def plane_intersection(pln1, pln2):
    """
    Given two planes defined in the Hessian form
    (see http://mathworld.wolfram.com/HessianNormalForm.html) each one
    represented by 4x1 numpy array (nx, ny, nz, p) compute the line formed
    by the intersection between the two planes.

    :parameter pln1:
        A 4x1 array with direction cosines of the first plane
    :parameter pln2:
        A 4x1 array with direction cosines of the second plane
    :return:
        An array with the direction cosines of the line
    """
    dirc = np.cross(pln1[:-1], pln2[:-1])
    nrm = (sum(dirc**2))**.5
    return dirc / nrm


def get_line_of_intersection(strike1, dip1, strike2, dip2):
    """
    Find the direction cosines of the line obtained by the intersection between
    two planes defined in terms of strike and dip.

    :parameter strike1:
    :parameter dip1:
    :parameter strike2:
    :parameter dip2:
    """
    a, b, c = get_direction_cosines(strike1, dip1)
    acs, bcs, ccs = get_direction_cosines(strike2, dip2)
    pln1 = np.array([a, b, c, 0])
    pln2 = np.array([acs, bcs, ccs, 0])
    # inter contains the direction cosines of the line obtained by the
    # intersection between the two planes
    return plane_intersection(pln1, pln2)


def plot_planes_at(x, y, strikes, dips, magnitudes, strike_cs, dip_cs,
                   aratio=1.0, msr=None, ax=None, zorder=20, color='None',
                   linewidth=1, axis=None):
    """
    This plots a cross-section and number of rupture planes defined in terms
    of a strike and a dip.

    :parameter x:
        Coordinates x on the cross-section
    :parameter y:
        Coordinates y on the cross-section (it corresponds to a depth)
    :parameter strikes:
        Strike values of the planes
    :parameter dips:
        Dip values of the planes
    :parameter strike_cs:
        Strike angle of the cross-section plane [in degrees]
    :parameter dip_cs:
        Dip angle of the cross-section plane [in degrees]
    """

    if axis is None:
        _ = plt.gca()
    else:
        plt.sca(axis)

    if msr is None:
        msr = WC1994()

    cols = ['red', 'blue', 'green']

    for strike, dip, col, mag in zip(strikes, dips, cols, magnitudes):

        area = msr.get_median_area(mag, None)
        width = (area / aratio)**.5
        t = np.arange(-width / 2, width / 2, 0.1)

        inter = get_line_of_intersection(strike, dip, strike_cs, dip_cs)
        xl = t * inter[0]
        yl = t * inter[1]
        zl = t * inter[2]
        ds = -np.sign(t) * (xl**2 + yl**2)**.5 + x

        if color is not None:
            col = color

        plt.plot(ds, zl + y, zorder=zorder, color=col, linewidth=linewidth)


def _read_edge_file(filename):
    """
    :parameter str filename:
        The name of the edge file
    :return:
        An instance of :class:`openquake.hazardlib.geo.line.Line`
    """
    points = []
    for line in open(filename, 'r'):
        aa = re.split('\\s+', line)
        points.append(Point(float(aa[0]),
                            float(aa[1]),
                            float(aa[2])))
    return Line(points)


def _read_edges(foldername):
    """
    :parameter foldername:
        The folder containing the `edge_*` files
    :return:
        A list of :class:`openquake.hazardlib.geo.line.Line` instances
    """
    path = os.path.join(foldername, 'edge*.*')
    tedges = []
    for fle in sorted(glob.glob(path)):
        tedges.append(_read_edge_file(fle))
    return tedges

def _read_edge_file(filename):
    """
    :parameter str filename:
        The name of the edge file
    :return:
        An instance of :class:`openquake.hazardlib.geo.line.Line`
    """
    points = []
    for line in open(filename, 'r'):
        aa = re.split('\\s+', line)
        points.append(Point(float(aa[0]),
                            float(aa[1]),
                            float(aa[2])))
    return Line(points)

def _read_profiles(foldername):
    """
    :parameter foldername:
        The folder containing the `cs_*` files
    :return:
        A list of :class:`openquake.hazardlib.geo.line.Line` instances
    """
    path = os.path.join(foldername, 'cs*.*')
    tprofiles = []
    for fle in sorted(glob.glob(path)):
        tprofiles.append(_read_pro_file(fle))
    return tprofiles


def _get_array(tedges):
    """
    :parameter list tedges:
        A list of :class:`openquake.hazardlib.geo.line.Line` instances
    :return:
    """
    edges = np.zeros((len(tedges), len(tedges[0]), 3))
    for i, edge in enumerate(tedges):
        coo = [(edge.points[i].longitude,
                edge.points[i].latitude,
                edge.points[i].depth) for i in range(len(edge.points))]
        xx = np.array(coo)
        edges[i] = xx


def _check_edges(edges):
    """
    This checks that all the edges follow the right hand rule
    :param list edges:
        The list of edges to be analysed.
    :return:
        An instance of :class:`numpy.ndarray` of cardinality equal to the
        number of edges. Where integers are positive, the edges need to be
        flipped.
    """

    # Check the input
    if len(edges) < 1:
        return None

    # Create a matrix of points
    pnts = []
    for edge in edges:
        pnts += [[pnt.longitude, pnt.latitude, pnt.depth]
                 for pnt in edge.points]
    pnts = np.array(pnts)

    # Project the points using Lambert Conic Conformal
    fmt = "+proj=lcc +lon_0={:f} +lat_1={:f} +lat_2={:f}"
    mla = np.mean(pnts[:, 1])
    srs = CRS.from_proj4(fmt.format(np.mean(pnts[:, 0]), mla - 10, mla + 10))
    p = Proj(srs)

    # From m to km
    x, y = p(pnts[:, 0], pnts[:, 1])
    x = x / 1e3  # m -> km
    y = y / 1e3  # m -> km

    # Fit the plane
    tmp = np.vstack((x.flatten(), y.flatten(), pnts[:, 2].flatten())).T
    _, ppar = plane_fit(tmp)

    # Analyse the edges
    chks = []
    for edge in edges:

        epnts = np.array([[pnt.longitude, pnt.latitude, pnt.depth] for pnt in
                          edge.points[0:2]])
        ex, ey = p(epnts[:, 0], epnts[:, 1])
        ex = ex / 1e3
        ey = ey / 1e3

        # Check the edge direction Vs the perpendicular to the plane
        idx = [0, -1]
        edgv = np.array([np.diff(ex[idx])[0], np.diff(ey[idx])[0]])
        chks.append(np.sign(np.cross(ppar[:2], edgv)))

    return np.array(chks)


def build_complex_surface_from_edges(foldername):
    """
    :parameter str foldername:
        The folder containing the `edge_*` files
    :return:
        An instance of :class:`openquake.hazardlib.geo.surface`
    """

    # Read edges
    tedges = _read_edges(foldername)

    # Check edges
    try:
        chks = _check_edges(tedges)
    except ValueError:
        msg = 'Error while checking the edges in {.s}'.format(foldername)
        print(msg)

    # Fix edges
    if np.any(chks > 0.):
        for i, chk in enumerate(chks):
            if chk < 0:
                tedges[i] = tedges[i].flip()
                print('flipping')

    # Build complex fault surface
    surface = ComplexFaultSurface.from_fault_data(tedges, mesh_spacing=5.0)

    return surface


def build_kite_surface_from_profiles(foldername):
    """
    :parameter str foldername:
        The folder containing the `edge_*` files
    :return:
        An instance of :class:`openquake.hazardlib.geo.surface`
    """

    # Read edges
    profiles = ProfileSet.from_files(foldername)

    # Build kite fault surface
    surface = KiteSurface.from_profiles(profiles.profiles, 5, 5)

    return surface


def plot_complex_surface(tedges):
    """
    :parameter list tedges:
        A list of :class:`openquake.hazardlib.geo.line.Line` instances
    """

    # create the figure
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')

    # plotting edges
    for edge in tedges:
        coo = [(edge.points[i].longitude,
                edge.points[i].latitude,
                edge.points[i].depth) for i in range(len(edge.points))]
        coo = np.array(coo)
        #
        # plot edges
        ax.plot(coo[:, 0], coo[:, 1], coo[:, 2])
        #
        # shallow part of the subduction surface
        k = np.nonzero(coo[:, 2] < 50.)
        if len(k[0]):
            ax.plot(coo[k[0], 0], coo[k[0], 1], coo[k[0], 2], 'or',
                    markersize=2)
    #
    # set axes
    ax.set_zlim([0, 300])
    ax.invert_zaxis()
    ax.view_init(50, 10)

    return fig, ax
