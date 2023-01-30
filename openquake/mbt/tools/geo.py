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

import shapely
import numpy as np

from pyproj import Transformer

from openquake.mbt.tools.mfd import get_moment_from_mfd
from openquake.hazardlib.geo.polygon import Polygon
from openquake.hazardlib.geo.line import Line
from openquake.hazardlib.source import SimpleFaultSource


def get_line_inside_polygon(pnt_lon, pnt_lat, poly_lon, poly_lat):
    """
    :parameter pnt_lon:
        A list with the longitude values of the fault trace
    :parameter pnt_lat:
        A list with the latitude values of the fault trace
    :parameter poly_lon:
        A list with the longitude values of the polygon
    :parameter poly_lat:
        A list with the latitude values of the polygon
    :return:
        Indexes of the points inside the polygon
    """
    # Fix the projections
    transformer = Transformer.from_proj("EPSG:4326",
                                        {"proj": 'lcc', "lat_1": poly_lat[0]})

    # Create polygon
    poly_xy = []
    for lo, la in zip(poly_lon, poly_lat):
        x, y = transformer.transform(lo, la)
        poly_xy.append((x, y))
    polygon = shapely.geometry.Polygon(poly_xy)

    # Create linesting
    line_xy = []
    for lo, la in zip(pnt_lon, pnt_lat):
        x, y = transformer.transform(lo, la)
        line_xy.append((x, y))
    line = shapely.geometry.LineString(line_xy)

    # Intersection
    if line.intersects(polygon):
        tmpl = line.intersection(polygon)
        return tmpl.length/line.length
    else:
        return None


def get_faults_in_polygon(polygon, faults):
    """
    Finds the faults inside a polygon

    :parameter polygon:
        An instance of :class:`~openquake.hazardlib.geo.polygon.Polygon`
    :parameter faults:
        A list of :class:`~openquake.hazardlib.source.SimpleFaultSource`
        instances
    """

    # Checks
    assert isinstance(polygon, Polygon)
    assert isinstance(faults, list)

    # Init
    sel_flts = {}
    mo_tot = 0

    # Processing faults
    for fault in faults:
        assert isinstance(fault, SimpleFaultSource)
        xf, yf, _ = from_trace_to_xy(fault.fault_trace)

        # Compute the fraction of the fault trace inside the polygon
        frac = get_line_inside_polygon(xf, yf, polygon.lons, polygon.lats)
        if frac is not None:
            mo = get_moment_from_mfd(fault.mfd)
            sel_flts[fault.source_id] = (mo, frac)
            mo_tot += frac * mo

    return sel_flts, mo_tot


def from_trace_to_xy(trace):
    """
    Extracts longitude and latitude values from the trace used to define the
    geometry of a :class:`~openquake.hazardlib.source.SimpleFaultSource`
    instance.

    :parameter trace:
        A :class:`~openquake.hazardlib.geo.line.Line` instance
    :return:
        Three lists containing lons, lats and deps
    """
    assert isinstance(trace, Line)
    lons = []
    lats = []
    deps = []
    for pnt in trace.points:
        lons.append(pnt.longitude)
        lats.append(pnt.latitude)
        deps.append(pnt.depth)
    return lons, lats, deps


def get_idx_points_inside_polygon(plon, plat, poly_lon, poly_lat,
                                  pnt_idxs, buff_distance=10.):
    """
    :parameter plon:
        Points longitude list
    :parameter plat:
        Points latitude list
    :parameter poly_lon:
        A list containing the longitude coordinates of the polygon vertexes
    :parameter poly_lat:
        A list containing the latitude coordinates of the polygon vertexes
    :return:
        Indexes of the points inside the polygon
    """
    selected_idx = []

    # Fix the projections
    transformer = Transformer.from_proj("EPSG:4326",
                                        {"proj": 'lcc',
                                         "lat_1": poly_lat[0],
                                         "lon_0": poly_lon[0]})

    # Create polygon
    poly_xy = []
    for lo, la in zip(poly_lon, poly_lat):
        x, y = transformer.transform(la, lo)
        poly_xy.append((x, y))
    poly_xy = np.array(poly_xy)

    # Shapely polygon
    polygon = shapely.geometry.Polygon(poly_xy)

    # Add buffer
    buff = polygon.buffer(buff_distance)

    # Find points inside
    cxy = []
    for lo, la, jjj in zip(plon, plat, pnt_idxs):
        x, y = transformer.transform(la, lo)
        cxy.append((x, y))
        point = shapely.geometry.Point((x, y))
        if point.within(buff):
            selected_idx.append(jjj)

    return selected_idx
