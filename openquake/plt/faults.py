"""
module :mod:`openquake.plt.faults` provides functions for plotting in 3D fault
surfaces and ruptures.
"""


import numpy as np
import pyvista as pv
import geopandas as gpd

from shapely.geometry import LineString, Polygon
from openquake.hazardlib.geo.geodetic import npoints_towards
from openquake.hazardlib.source import SimpleFaultSource
from openquake.hazardlib.geo.surface import SimpleFaultSurface


def get_trace_linestring(src):
    """
    Returns a line string from the fault trace

    :param src:
        A :class:`openquake.hazardlib.source.simplefault.SimpleFaultSource`
        instance
    """
    coo = [(p.longitude, p.latitude, p.depth) for p in src.fault_trace]
    return LineString(coo)


def get_fault_surface_mesh_coordinates(src):
    """
    """
    print(src.__name__)
    if src.__name__ == 'SimpleFaultSource':
        sfc = SimpleFaultSurface.from_fault_data(
                src.fault_trace,
                src.upper_seismogenic_depth, src.lower_seismogenic_depth,
                src.dip, src.rupture_mesh_spacing)
    else:
        raise ValueError('This source type is not supported')
    return get_mesh_coordinates(sfc.mesh)


def get_mesh_coordinates(mesh):
    """
    """
    coo = np.stack([mesh.lons.reshape(-1), mesh.lats.reshape(-1),
                    mesh.depths.reshape(-1)])
    coo = np.moveaxis(coo, 0, -1)
    coo = coo[np.isfinite(coo[:, 0]), :]
    return coo


def get_fault_surface_coordinates(src):
    """
    Returns the coordinates of the vertexes representing the polygons
    describing the surface of the fault

    :param src:
        An instance of :class:`openquake.hazardlib.source.SimpleFaultSource`
    :returns:
        A numpy array with the coordinates of the vertexes in clockwise
        order. This array can be used to directly build a Shapely Polygon
        with Polygon(coo).
    """
    upp_z = src.upper_seismogenic_depth
    upp_h = upp_z * np.tan(np.radians(90-src.dip))

    low_z = src.lower_seismogenic_depth
    low_h = low_z * np.tan(np.radians(90-src.dip))

    sfc = SimpleFaultSurface.from_fault_data(
            src.fault_trace,
            src.upper_seismogenic_depth, src.lower_seismogenic_depth,
            src.dip, src.rupture_mesh_spacing)
    dir = sfc.get_strike() + 90

    coo = [(p.longitude, p.latitude, p.depth) for p in src.fault_trace]
    upp = np.array(coo)
    out = []
    for cc in upp:
        tmp = npoints_towards(cc[0], cc[1], cc[2], dir, upp_h, 0, 2)
        out.append((tmp[0][1], tmp[1][1], upp_z))
    for cc in upp[::-1]:
        tmp = npoints_towards(cc[0], cc[1], cc[2], dir, low_h, 0, 2)
        out.append((tmp[0][1], tmp[1][1], low_z))
    return np.array(out), np.array(coo)


def get_fault_surface_meshgrid(src):
    """

    :param src:
        An instance of :class:`openquake.hazardlib.source.SimpleFaultSource`
    """
    # Get the coordinates of the polygon and prepare the output grid
    coo, _ = get_fault_surface_coordinates(src)
    half = int(len(coo)/2)
    out = np.zeros((2, half, 3))
    out[0, :, :] = coo[:half, :]
    out[1, :, :] = coo[:half-1:-1, :]
    return out


def get_pv_line(coo, close=False):
    """
    Create a pv.PolyData instance
    """
    pdata = pv.PolyData(coo)
    dlt = 1 if close else 0
    aa = [len(coo)+dlt]
    aa.extend(range(0, len(coo)))
    if close:
        aa.extend([0])
    pdata.lines = aa
    return pdata


def get_pv_points(coo):
    pdata = pv.PolyData(coo)
    return pdata


def get_gdf_3d_polygons(ssm):
    """
    """
    ids = []
    trts = []
    geoms = []
    for grp in ssm:
        for src in grp:
            if isinstance(src, SimpleFaultSource):
                ids.append(src.source_id)
                trts.append(src.tectonic_region_type)
                geoms.append(Polygon(get_fault_surface_coordinates(src)))
    d = {'id': ids, 'trt': trts, 'geometry': geoms}
    gdf = gpd.GeoDataFrame(d, crs="EPSG:4326")
    return gdf


def get_gdf_fault_traces(ssm):
    ids = []
    trts = []
    geoms = []
    for grp in ssm:
        for src in grp:
            if isinstance(src, SimpleFaultSource):
                ids.append(src.source_id)
                trts.append(src.tectonic_region_type)
                geoms.append(get_trace_linestring(src))
    d = {'id': ids, 'trt': trts, 'geometry': geoms}
    gdf = gpd.GeoDataFrame(d, crs="EPSG:4326")
    return gdf


def get_line_coo(line):
    coo = np.array([[p.longitude, p.latitude, p.depth] for p in line.points])
    return coo
