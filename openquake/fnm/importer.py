# ------------------- The OpenQuake Model Building Toolkit --------------------
# ------------------- FERMI: Fault nEtwoRks ModellIng -------------------------
# Copyright (C) 2023 GEM Foundation
#         .-.
#        /    \                                        .-.
#        | .`. ;    .--.    ___ .-.     ___ .-. .-.   ( __)
#        | |(___)  /    \  (   )   \   (   )   '   \  (''")
#        | |_     |  .-. ;  | ' .-. ;   |  .-.  .-. ;  | |
#       (   __)   |  | | |  |  / (___)  | |  | |  | |  | |
#        | |      |  |/  |  | |         | |  | |  | |  | |
#        | |      |  ' _.'  | |         | |  | |  | |  | |
#        | |      |  .'.-.  | |         | |  | |  | |  | |
#        | |      '  `-' /  | |         | |  | |  | |  | |
#       (___)      `.__.'  (___)       (___)(___)(___)(___)
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


import geojson
import numpy as np
import matplotlib.pyplot as plt

from pyproj import Proj

try:
    import pygmt
except:
    pygmt = None

from openquake.hazardlib.geo import Point, Line
from openquake.hazardlib.geo.surface import SimpleFaultSurface
from openquake.hazardlib.geo.surface.kite_fault import (
    get_profiles_from_simple_fault_data,
    KiteSurface,
)

import logging

logging.basicConfig(
    format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S'
)


def fix_right_hand(trace, dip_dir):
    azi = trace.average_azimuth()
    if np.abs((azi + 90) % 360 - dip_dir) < 60:
        return trace
    else:
        trace.flip()
        return trace


def plot_profiles_plt(profiles, trace, mesh=None):

    problem = []
    for i_pro, pro in enumerate(profiles):
        if np.any(np.isnan(pro.coo[:, 0])):
            problem.append(i_pro)
            continue

    cells = []
    if mesh is not None:
        for i in range(0, mesh.lons.shape[0] - 1):
            for j in range(0, mesh.lons.shape[1] - 1):
                if np.all(np.isfinite(mesh.lons[i : i + 1, j : j + 1])):
                    cells.append(
                        [
                            [mesh.lons[i, j], mesh.lats[i, j]],
                            [mesh.lons[i, j + 1], mesh.lats[i, j + 1]],
                            [mesh.lons[i + 1, j + 1], mesh.lats[i + 1, j + 1]],
                            [mesh.lons[i + 1, j], mesh.lats[i + 1, j]],
                            [mesh.lons[i, j], mesh.lats[i, j]],
                        ]
                    )
        cells = np.array(cells)

    fig = plt.figure()
    plt.plot(trace.coo[:, 0], trace.coo[:, 1], 'r')

    for pro in profiles:
        plt.plot(pro.coo[:, 0], pro.coo[:, 1], "k", lw=0.25)

    for idx in problem:
        pro = profiles[idx]
        plt.plot(pro.coo[:, 0], pro.coo[:, 1], "g", lw=0.5)

    if mesh is not None:
        for cell in cells:
            plt.plot(cell[:, 0], cell[:, 1], "orange", lw=0.5)

    plt.show()


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
                if np.all(np.isfinite(mesh.lons[i : i + 1, j : j + 1])):
                    cells.append(
                        [
                            [mesh.lons[i, j], mesh.lats[i, j]],
                            [mesh.lons[i, j + 1], mesh.lats[i, j + 1]],
                            [mesh.lons[i + 1, j + 1], mesh.lats[i + 1, j + 1]],
                            [mesh.lons[i + 1, j], mesh.lats[i + 1, j]],
                            [mesh.lons[i, j], mesh.lats[i, j]],
                        ]
                    )
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
        proj = Proj(
            proj='lcc', lon_0=m_lon, lat_1=m_lat - 10.0, lat_2=m_lat + 10.0
        )
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
        msg = f'Fault id: {fid} - Trace length shorter than 2 * edge_sd'
        if fault_trace.get_length() < edge_sd * 2:
            logging.warning(msg)

        # Adjust the sampling distance along the edges
        num = np.round(fault_trace.get_length() / edge_sd)
        edge_sd_res = (fault_trace.get_length() / num) * 0.98

        # Get profiles
        upp_sd = prop.get("usd", None)
        low_sd = prop.get("lsd", None)
        profs = get_profiles_from_simple_fault_data(
            fault_trace, upp_sd, low_sd, dip, edge_sd
        )

        try:
            surf = KiteSurface.from_profiles(
                profs, align=True, profile_sd=2.0, edge_sd=edge_sd_res
            )

            if np.any(np.isnan(surf.mesh.array)):
                if pygmt is not None:
                    plot_profiles(profs, fault_trace, surf.mesh)
                    plt.title(f'Feature {i_fea}')
                else:
                    print(f'Feature {i_fea} has NaNs in the mesh')
                    # plt.plot(surf.mesh.lons, surf.mesh.lats, 'b.')
                    # plt.plot(fault_trace.coo[:,0], fault_trace.coo[:,1], 'r')
                    # plt.show()
                    # plot_profiles_plt(profs, fault_trace, surf.mesh)
            else:
                # plot_profiles_plt(profs, fault_trace, surf.mesh)
                pass

        except ValueError('Cannot build kite Surface'):
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

        # Update the list of surfaces created
        surfs.append(surf)

        if i_fea in iplot and pygmt is not None:
            plot_profiles(profs, fault_trace)
            plt.title(f'Feature {i_fea}')

    return surfs


def kite_surfaces_from_geojson(
    fname: str,
    edge_sd: float = 2.0,
    idxs: list = [],
    skip: list = [],
    iplot: list = [],
) -> list:
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


def simple_fault_surface_from_feature(
    feature: dict,
    lsd_default=20.0,
    usd_default=0.0,
    edge_sd: float = 2.0,
    min_sd=0.5,
) -> SimpleFaultSurface:
    geom = feature['geometry']
    prop = feature['properties']
    dip = prop.get("dip", None)
    lsd = prop.get("lsd", lsd_default)
    usd = prop.get("usd", usd_default)

    fault_trace = Line([Point(c[0], c[1]) for c in geom["coordinates"]])

    while edge_sd > min_sd:
        try:
            return SimpleFaultSurface.from_fault_data(
                fault_trace, usd, lsd, dip, edge_sd
            )
        except (ValueError, AssertionError):
            edge_sd /= 2.0


def simple_fault_surfaces_from_geojson(
    fname: str,
    edge_sd: float = 2.0,
    idxs: list = [],
    skip: list = [],
) -> list:
    """
    Create OQ simple fault surfaces from a geojson file.

    :returns:
        A list of :class:`openquake.hazardlib.geo.surface.SimpleFaultSurface`
        instances.
    """
    # Read .geojson file with fault info
    with open(fname) as f:
        data = geojson.load(f)
    # surfs = create_surfaces(data, edge_sd, idxs=idxs, skip=skip, iplot=iplot)
    surfs = []

    for i_fea, feature in enumerate(data['features']):
        fid = feature['properties'].get("fid", None)

        if len(idxs) and i_fea not in idxs:
            continue

        # Skip feature if requested
        if i_fea in skip:
            msg = f'Skipping feature with fid = {fid}'
            logging.info(msg)
            continue

        surf = simple_fault_surface_from_feature(feature, edge_sd=edge_sd)
        surfs.append(surf)

    return surfs
