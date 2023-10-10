# coding: utf-8

import geojson
import numpy as np

from pyproj import Proj

try:
    import pygmt
except:
    pygmt = None

from openquake.hazardlib.geo import Point, Line
from openquake.hazardlib.geo.surface.kite_fault import (
    get_profiles_from_simple_fault_data,
    KiteSurface,
)

import logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


def fix_right_hand(trace, dip_dir):
    azi = trace.average_azimuth()
    if np.abs((azi + 90) % 360 - dip_dir) < 60:
        return trace
    else:
        trace.flip()
        return trace


def plot_profiles(profiles, trace, mesh=None):

    min_lo = 400.0
    min_la = 400.0
    max_lo = -400.0
    max_la = -400.0

    # Computing extent of the region
    dlt = 0.2
    problem = []
    for i_pro, pro in enumerate(profiles):
        if np.any(np.isnan(pro.coo[:, 0])):
            problem.append(i_pro)
            continue
        min_lo = np.min([min_lo, np.min(pro.coo[:, 0])])
        min_la = np.min([min_la, np.min(pro.coo[:, 1])])
        max_lo = np.max([max_lo, np.max(pro.coo[:, 0])])
        max_la = np.max([max_la, np.max(pro.coo[:, 1])])
    min_lo -= dlt
    min_la -= dlt
    max_lo += dlt
    max_la += dlt
    region = [min_lo, max_lo, min_la, max_la]

    cells = []
    if mesh is not None:
        for i in range(0, mesh.lons.shape[0] - 1):
            for j in range(0, mesh.lons.shape[1] - 1):
                if np.all(np.isfinite(mesh.lons[i:i + 1, j:j + 1])):
                    cells.append([[mesh.lons[i, j], mesh.lats[i, j]],
                                  [mesh.lons[i, j + 1], mesh.lats[i, j + 1]],
                                  [mesh.lons[i + 1, j + 1],
                                   mesh.lats[i + 1, j + 1]],
                                  [mesh.lons[i + 1, j], mesh.lats[i + 1, j]],
                                  [mesh.lons[i, j], mesh.lats[i, j]]])
        cells = np.array(cells)

    fig = pygmt.Figure()
    fig.basemap(region=region, projection="M15c", frame=True)
    fig.coast(land="lightgray", water="skyblue")
    fig.plot(x=trace.coo[:, 0], y=trace.coo[:, 1], pen="0.2p,red")

    for pro in profiles:
        fig.plot(x=pro.coo[:, 0], y=pro.coo[:, 1], pen="0.05p,black")

    for idx in problem:
        pro = profiles[idx]
        fig.plot(x=pro.coo[:, 0], y=pro.coo[:, 1], pen="0.08p,green")

    if mesh is not None:
        for cell in cells:
            fig.plot(x=cell[:, 0], y=cell[:, 1], pen="0.08p,green")

    fig.show()


def get_dip_dir(prop: dict):
    """
    :param prop:
        A dictionary with the properties of a feature
    """

    if isinstance(prop["dip_dir"], float) or isinstance(prop["dip_dir"], int):
        return float(prop["dip_dir"])
    else:
        if prop["dip_dir"] == "N":
            return 0
        elif prop["dip_dir"] == "E":
            return 90
        elif prop["dip_dir"] == "S":
            return 180
        elif prop["dip_dir"] == "W":
            return 270
        elif prop["dip_dir"] == "NE":
            return 45
        elif prop["dip_dir"] == "SE":
            return 135
        elif prop["dip_dir"] == "SW":
            return 225
        elif prop["dip_dir"] == "NW":
            return 315
        else:
            msg = "Unknown definition of dir direction"
            raise ValueError(msg)


def create_surfaces(
    data,
    edge_sd: float = 2.0,
    idxs: list = [],
    skip: list = [],
    iplot: list = [],
) -> list:
    """
    :param data:
        A dictionary with the content of a .geojson file that describes the
        geometry of the faults
    :returns:
        A list of
        :class:`openquake.hazardlib.geo.surface.kite_fault.KiteSurface`
        instances.
    """
    # Creates the surfaces
    surfs = []
    edge_sd = 2.0
    for i_fea, fea in enumerate(data['features']):

        # Get info
        geom = fea['geometry']
        prop = fea['properties']
        fid = prop.get("fid", None)

        if len(idxs) and i_fea not in idxs:
            continue

        # Skip feature if requested
        if i_fea in skip:
            msg = f'Skipping feature with fid = {fid}'
            logging.info(msg)
            continue

        dip_dir = get_dip_dir(prop)
        dip = prop.get("dip", None)

        # Create the fault trace
        fault_trace = Line([Point(c[0], c[1]) for c in geom["coordinates"]])
        fault_trace = fix_right_hand(fault_trace, dip_dir)

        # Create the fault trace
        coo = np.array([[p.longitude, p.latitude] for p in fault_trace])
        coo_p = np.zeros((coo.shape[0], 2))
        m_lon = np.mean(coo[:, 0])
        m_lat = np.mean(coo[:, 1])
        proj = Proj(proj='lcc', lon_0=m_lon,
                    lat_1=m_lat - 10., lat_2=m_lat + 10.)
        coo_p[:, 0], coo_p[:, 1] = proj(coo[:, 0], coo[:, 1])

        # Smoothing fault trace
        # if interpolate:
        #     from scipy.interpolate import splprep, splev
        #     tck, u = splprep(coo_p.T, s=0.01)
        #     u_new = np.linspace(u.min(), u.max(), 200)
        #     # Evaluate a B-spline
        #     x_new, y_new = splev(u_new, tck)
        #     smo_lo, smo_la = proj(x_new, y_new, inverse=True)
        # else:
        smo_lo = coo[:, 0]
        smo_la = coo[:, 1]
        fault_trace = Line([Point(*c) for c in zip(smo_lo, smo_la)])

        # Check the length of the trace wrt the edge sampling
        msg = f'Fault id: {fid} - Trace lenght shorter than 2 * edge_sd'
        if fault_trace.get_length() < edge_sd * 2:
            logging.warning(msg)

        # Adjust the sampling distance along the edges
        num = np.round(fault_trace.get_length() / edge_sd)
        edge_sd_res = (fault_trace.get_length() / num) * 0.98

        # Get profiles
        upp_sd = prop.get("usd", None)
        low_sd = prop.get("lsd", None)
        profs = get_profiles_from_simple_fault_data(
            fault_trace, upp_sd, low_sd, dip, 2.0)

        try:
            surf = KiteSurface.from_profiles(
                profs, profile_sd=2.0, edge_sd=edge_sd_res)
        except ValueError('Cannot build kite Surface'):
            import matplotlib.pyplot as plt
            plt.plot(smo_lo, smo_la, 'r')
            plt.plot(coo[:, 0], coo[:, 1], 'b')
            for pro in profs:
                plt.plot(pro.coo[:, 0], pro.coo[:, 1], '-')
            plt.title(f'Feature {i_fea}')
            plt.show()

        # Check the number of columns composing this surface
        msg = f'Fault id: {fid} - Surface with less than 2 columns'
        if surf.mesh.lons.shape[1] < 2:
            logging.warning(msg)
        msg = f'Fault id: {fid} - Surface with less than 2 rows'
        if surf.mesh.lons.shape[0] < 2:
            logging.warning(msg)

        if np.any(np.isnan(surf.mesh.lons)) and pygmt is not None:
            plot_profiles(profs, fault_trace, surf.mesh)
            plt.title(f'Feature {i_fea}')

        # Update the list of surfaces created
        surfs.append(surf)

        if i_fea in iplot and pygmt is not None:
            plot_profiles(profs, fault_trace)
            plt.title(f'Feature {i_fea}')

    return surfs


def kite_surfaces_from_geojson(fname: str, edge_sd: float = 2.0,
                               idxs: list = [], skip: list = [],
                               iplot: list = []) -> list:
    """
    Create OQ kite surfaces from a geojson file.

    :returns:
        A list of :class:`openquake.hazardlib.geo.surface.KiteSurface`
        instances.
    """
    # Read .geojson file with fault info
    with open(fname) as f:
        data = geojson.load(f)
    surfs = create_surfaces(data, edge_sd, idxs=idxs, skip=skip, iplot=iplot)
    return surfs
