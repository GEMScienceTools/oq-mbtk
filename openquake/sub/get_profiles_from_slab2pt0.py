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

import pygmt
import netCDF4
import numpy as np
from numba import njit
from openquake.hazardlib.geo.geodetic import (
    point_at, npoints_towards, geodetic_distance, azimuth)
from openquake.sub.cross_sections import CrossSection, Slab2pt0


@njit
def get_mean_azimuth(azims):
    """
    :param azims:
    """
    sins = np.mean(np.sin(np.radians(azims)))
    coss = np.mean(np.cos(np.radians(azims)))
    mean_azimuth = np.degrees(np.arctan2(sins, coss)) % 360
    return mean_azimuth


@njit
def get_bounding_box(lons, lats, delta=0.0):
    """
    :param lons:
    :param lats:
    """
    milo = np.min(lons)
    milo = milo if milo <= 180 else milo - 360
    malo = np.max(lons)
    malo = malo if malo <= 180 else malo - 360
    return [milo-delta, malo+delta, np.min(lats)-delta, np.max(lats)+delta]


def get_initial_traces(bb, dip_dir, spacing):
    """
    :param bb:
    :param dip_dir:
    :param spacing:
    """

    # List of profiles
    profiles = []

    # Max length
    max_length = geodetic_distance(bb[0], bb[2], bb[1], bb[3])

    # Compute spacing
    spacing_lon = np.abs(spacing / np.sin(dip_dir))
    spacing_lat = np.abs(spacing / np.cos(dip_dir))
    print(f'Spacing lon: {spacing_lon:.2f} lat: {spacing_lat:.2f}')

    # tmp = point_at(bb[0], bb[1], dip_dir, spacing)
    # spacing_lon = geodetic_distance(bb[0], bb[1], tmp[0], bb[1])
    # spacing_lat = geodetic_distance(bb[0], bb[1], bb[0], tmp[1])
    # print(f'Spacing lon: {spacing_lon:.2f} lat: {spacing_lat:.2f}' )

    # Dip towards 3rd or 4th quadrants
    if dip_dir > 180:

        # Top distance and azimuth
        distance = geodetic_distance(bb[1], bb[3], bb[0], bb[3])
        edge_azimuth = azimuth(bb[1], bb[3], bb[0], bb[3])

        # Number of samples
        num_samples = np.ceil(distance/spacing_lon)
        distance = spacing_lon * num_samples

        # Sampling first edge
        coords = npoints_towards(
            bb[1], bb[3], 0, edge_azimuth, distance, 0, num_samples)

        # Get profiles
        profiles = _get_profiles(coords, dip_dir, max_length)

        # Right distance and azimuth
        distance = geodetic_distance(bb[1], bb[3], bb[1], bb[2])
        edge_azimuth = azimuth(bb[1], bb[3], bb[1], bb[2])

        # Number of samples
        num_samples = np.ceil(distance/spacing_lat)
        distance = spacing_lat * num_samples

        # Sampling first edge
        coords = npoints_towards(
            bb[0], bb[2], 0, edge_azimuth, distance, 0, num_samples)

        # Create profiles
        tmp_profiles = _get_profiles(coords, dip_dir, max_length)
        tmp_profiles.reverse()
        profiles.extend(tmp_profiles)

        return profiles

    # Top distance and azimuth
    distance = geodetic_distance(bb[0], bb[2], bb[1], bb[2])
    edge_azimuth = azimuth(bb[0], bb[2], bb[1], bb[2])

    # Number of samples
    num_samples = np.ceil(distance/spacing_lon)
    distance = spacing_lon * num_samples

    # Sampling first edge
    coords = npoints_towards(
        bb[0], bb[2], 0, edge_azimuth, distance, 0, num_samples)

    # Get profiles
    profiles = _get_profiles(coords, dip_dir, max_length)

    # Left distance and azimuth
    distance = geodetic_distance(bb[0], bb[2], bb[0], bb[3])
    edge_azimuth = azimuth(bb[0], bb[2], bb[0], bb[3])

    # Number of samples
    num_samples = np.ceil(distance/spacing_lat)
    distance = spacing_lat * num_samples

    # Sampling first edge
    coords = npoints_towards(
        bb[0], bb[2], 0, edge_azimuth, distance, 0, num_samples)

    # Create profiles
    tmp_profiles = _get_profiles(coords, dip_dir, max_length)
    tmp_profiles.reverse()
    profiles.extend(tmp_profiles)

    return profiles, distance


def _get_profiles(coords, dip_dir, max_length):
    tmp_profiles = []
    for icoo in range(len(coords[0])):
        xco = coords[0][icoo]
        yco = coords[1][icoo]
        tmp = point_at(xco, yco, dip_dir, max_length)
        arr = np.array([[xco, yco], [tmp[0], tmp[1]]])
        tmp_profiles.append(arr)
    return tmp_profiles


def get_profiles(fname_str: str, fname_dep: str, spacing: float, fname_fig:
                 str = ''):
    """
    :param fname_str:
        The name of the Slab2.0 .grd file with the values of depth
    :param fname_dep:
        The name of the Slab2.0 .grd file with the values of depth
    :param spacing:
        The separation distance between traces
    """

    # Reading file with strike values
    f_strike = netCDF4.Dataset(fname_str)
    strikes = np.array(f_strike.variables['z'])
    mask = np.where(np.isfinite(strikes))
    strikes = strikes[mask]

    # Compute the mean strike
    strike_dir = get_mean_azimuth(strikes.flatten())
    dip_dir = (strike_dir + 90) % 360

    # Compute the bounding box
    x = np.array(f_strike.variables['x'])
    y = np.array(f_strike.variables['y'])
    xx, yy = np.meshgrid(x, y)
    bb = get_bounding_box(xx[mask], yy[mask], delta=1.)

    # Get traces
    traces, plen = get_initial_traces(bb, dip_dir, spacing)

    # Create cross-sections
    css = []
    for pro in traces:
        xlo = pro[0, 0]
        xla = pro[0, 1]
        xlo = xlo if xlo < 180 else xlo - 360
        cs = CrossSection(xlo, xla, plen, dip_dir)
        css.append(cs)

    # Reading file with depth values
    f_dep = netCDF4.Dataset(fname_dep)
    depths = np.array(f_dep.variables['z'])
    mask = np.where(np.isfinite(depths))

    # Filter
    depths = depths[mask]
    xx = xx[mask]
    yy = yy[mask]

    # Coords
    tmp = zip(xx.flatten(), yy.flatten(), depths.flatten())
    depths = [[x, y, z] for x, y, z in tmp]
    depths = np.array(depths)
    mask = depths[:, 0] > 180
    depths[mask, 0] = depths[mask, 0] - 360

    milo = np.min(depths[:, 0])
    mila = np.min(depths[:, 1])
    print(f'Min lon {milo:.2f} Max lon {np.max(depths[:, 0]):.2f}')
    print(f'Min lat {mila:.2f} Max lat {np.max(depths[:, 1]):.2f}')

    # Slab 2.0
    slb = Slab2pt0(depths, css)
    slb.compute_profiles(30.0)

    if len(fname_fig) > 0:
        fig = pygmt.Figure()
        pygmt.makecpt(cmap="jet", series=[0.0, 600])
        fig.basemap(region=bb, projection="M20c", frame=True)
        fig.coast(land="gray", water="skyblue")
        for pro in traces:
            fig.plot(x=pro[:, 0], y=pro[:, 1], pen="red")
        fig.plot(x=depths[:, 0], y=depths[:, 1], style='c0.025c', pen="green")
        for key in slb.profiles:
            pro = slb.profiles[key]
            if pro.shape[0] > 0:
                fig.plot(x=pro[:, 0],
                         y=pro[:, 1],
                         color=pro[:, 2],
                         cmap=True,
                         style="h0.025c")
        fig.savefig(fname_fig)

    return slb
