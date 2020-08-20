"""
"""

# MN: 'np' imported but not used
import numpy as np
import shapely

from pyproj import Proj, transform

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
    # MN: 'selected_idx' assigned but never used
    selected_idx = []
    # Fix the projections
    inProj = Proj('epsg:4326')
    outProj = Proj('+proj=lcc +lon_0={:f}'.format(poly_lon[0]))
    
    # Create polygon
    poly_xy = []
    for lo, la in zip(poly_lon, poly_lat):
        x, y = transform(inProj, outProj, lo, la)
        poly_xy.append((x, y))
    polygon = shapely.geometry.Polygon(poly_xy)
    # Create linesting
    line_xy = []
    for lo, la in zip(pnt_lon, pnt_lat):
        x, y = transform(inProj, outProj, lo, la)
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
    #
    # Fix the projections
    inProj = Proj('epsg:4326')
    outProj = Proj('+proj=lcc +lon_0={:f}'.format(poly_lon[0]))
    #
    # Create polygon
    poly_xy = []
    for lo, la in zip(poly_lon, poly_lat):
        x, y = transform(inProj, outProj, lo, la)
        poly_xy.append((x, y))
    #
    # Shapely polygon
    polygon = shapely.geometry.Polygon(poly_xy)
    #
    # Add buffer if requested
    buff = polygon.buffer(buff_distance)
    #
    # Find points inside
    cxy = []
    for lo, la, jjj in zip(plon, plat, pnt_idxs):
        x, y = transform(inProj, outProj, lo, la)
        cxy.append((x, y))
        point = shapely.geometry.Point((x, y))
        if point.within(buff):
            selected_idx.append(jjj)
    return selected_idx


def find_points_close_to_multisegment(plon, plat, mseg_lon, mseg_lat, pnt_idxs,
                                      buff_distance=10.):
    """
    :parameter plon:
        Points longitude
    :parameter plat:
        Points latitude
    :parameter mseg_lon:
        A list containing the longitude coordinates of the multi-segment
        vertexes
    :parameter mseg_lat:
        A list containing the latitude coordinates of the multi-segment
        vertexes
    :return:
        Indexes of the points nearby the multi-segmented line
    """
    selected_idx = []
    # Fix the projections
    inProj = Proj('epsg:4326')
    outProj = Proj('+proj=lcc +lon_0={:f}'.format(poly_lon[0]))
    # Create polygon
    mseg_xy = []
    for lo, la in zip(mseg_lon, mseg_lat):
        x, y = transform(inProj, outProj, lo, la)
        mseg_xy.append((x, y))
    # Create polygon
    line = shapely.geometry.LineString(mseg_xy)
    buff = line.buffer(buff_distance)
    # Find close points
    for lo, la, jjj in zip(plon, plat, pnt_idxs):
        x, y = transform(inProj, outProj, lo, la)
        point = shapely.geometry.Point((x, y))
        if point.within(buff):
            selected_idx.append(jjj)
    return selected_idx
