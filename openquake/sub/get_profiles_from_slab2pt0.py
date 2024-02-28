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


import pyproj
import netCDF4
import numpy as np
import geopandas as gpd
from numba import njit
from openquake.hazardlib.geo.geodetic import (
    point_at, npoints_towards, geodetic_distance, azimuth)
from openquake.sub.cross_sections import CrossSection, Slab2pt0

pygmt_available = False
#try:
#    import pygmt
#    pygmt_available = True
#except ImportError:
#    pass


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
    This computes the bounding box. Output longitudes are in the range
    [0, 360].

    :param lons:
        A :object:`numpy.ndarray` with the longitude coordinates
    :param lats:
        A :object:`numpy.ndarray` with the latitude coordinates
    :returns:
        A tuple with a list containing the bounding box in the format
        [min_lon, max_lon, min_lat, max_lat]
    """
    milo = np.min(lons)
    malo = np.max(lons)
    bbox = [milo - delta, malo + delta, np.min(lats) - delta,
            np.max(lats) + delta]
    idl = False
    if milo < 0:
        if malo > 0:
            idl = True
        # this means we are crossing the IDL so we convert coordinates to
        # [0, 360] centered at Greenwich
        # 0   90   180  181  270   359
        # 0   90   180  -179 -90   -1 <- input
        new_lons = np.copy(lons)
        new_lons[lons < 0] = 360 + lons[lons < 0]
        milo = np.min(new_lons)
        malo = np.max(new_lons)
        bbox = [milo - delta, malo + delta, np.min(lats) - delta,
                np.max(lats) + delta]

    # bbox = [milo-delta, malo+delta, np.min(lats)-delta, np.max(lats)+delta]
    return bbox, idl


def get_initial_traces(box, boy, dip_dir, spacing):
    """
    Computes initial traces for the subduction profiles

    :param box: x limits of the bounding box
    :param boy: y limits of the bounding box
    :param dip_dir: dip direction
    :param spacing: spacing between profiles
    """
    max_length = geodetic_distance(box[0], boy[0], box[3], boy[3])
    distance = geodetic_distance(box[0], boy[0], box[1], boy[1])

    edge_azimuth = azimuth(box[0], boy[0], box[1], boy[1])
    num_samples = np.ceil(distance / spacing)

    coords = npoints_towards(box[0], boy[0], 0, edge_azimuth, distance,
                             0, num_samples)
    profiles = _get_profiles(coords[0], coords[1], dip_dir, max_length)
    # reverse profiles to satisfy right hand rule
    profiles.reverse()

    return np.array(profiles), max_length


def aa_get_initial_traces(bb, dip_dir, spacing):

    spacing *= 1e3

    import pyproj
    from shapely.geometry import Point, LineString

    g = pyproj.Geod(ellps="WGS84")

    # Max length
    line_string = LineString([Point(bb[0], bb[2]), Point(bb[1], bb[3])])
    max_length = g.geometry_length(line_string)

    # Spacing - TODO need to add correction for latitude
    angle = ((np.floor(dip_dir / 90) + 1) * 90.0 - dip_dir)
    spacing_ver = np.abs(spacing / np.sin(angle))
    spacing_hor = np.abs(spacing / np.cos(dip_dir))

    # Compute distance of the edges and number of samples
    _, az21_right, distance_right = g.inv(bb[1], bb[2], bb[1], bb[3])
    az12_low, _, distance_low = g.inv(bb[0], bb[2], bb[1], bb[2])

    num_samples_low = np.floor(distance_low / spacing_hor)
    distance_remaining = distance_low - num_samples_low * spacing_hor
    num_samples_right = np.floor((distance_right - distance_remaining) /
                                 spacing_ver)

    points = g.fwd_intermediate(bb[0], bb[2], az12_low, num_samples_low + 1,
                                spacing_hor, initial_idx=0, terminus_idx=0)
    profiles = _get_profiles(np.array(points.lons), np.array(points.lats),
                             dip_dir, max_length)

    points = g.fwd_intermediate(bb[1], bb[3], az21_right,
                                num_samples_right + 1, spacing_ver,
                                initial_idx=0, terminus_idx=0)
    tmp_profiles = _get_profiles(np.array(points.lons), np.array(points.lats),
                                 dip_dir, max_length)
    profiles.extend(tmp_profiles)

    return np.array(profiles), max_length


def _get_profiles(lons, lats, dip_dir, max_length):
    tmp_profiles = []
    for xco, yco in zip(lons, lats):
        tmp = point_at(xco, yco, dip_dir, max_length)
        arr = np.array([[xco, yco], [tmp[0], tmp[1]]])
        tmp_profiles.append(arr)
    return tmp_profiles


"""
def _get_profiles(coords, dip_dir, max_length):
    tmp_profiles = []
    for icoo in range(len(coords[0])):
        xco = coords[0][icoo]
        yco = coords[1][icoo]
        tmp = point_at(xco, yco, dip_dir, max_length)
        arr = np.array([[xco, yco], [tmp[0], tmp[1]]])
        tmp_profiles.append(arr)
    return tmp_profiles
"""


def tmp_get_initial_traces(bb, dip_dir, spacing):
    """
    :param bb:
    :param dip_dir:
    :param spacing:
    """

    idl = False
    if bb[0] < 180 and bb[1] > 180:
        idl = True

    # List of profiles
    profiles = []

    # Max length
    max_length = geodetic_distance(bb[0], bb[2], bb[1], bb[3])

    # Compute spacing
    angle = ((np.floor(dip_dir / 90) + 1) * 90.0 - dip_dir)
    spacing_lon = np.abs(spacing / np.sin(angle))
    spacing_lat = np.abs(spacing / np.cos(dip_dir))
    print(f'Spacing lon: {spacing_lon:.2f} lat: {spacing_lat:.2f}')

    # tmp = point_at(bb[0], bb[1], dip_dir, spacing)
    # spacing_lon = geodetic_distance(bb[0], bb[1], tmp[0], bb[1])
    # spacing_lat = geodetic_distance(bb[0], bb[1], bb[0], tmp[1])
    # print(f'Spacing lon: {spacing_lon:.2f} lat: {spacing_lat:.2f}' )

    # Dip towards 3rd or 4th quadrants
    if dip_dir > 180:

        # Right edge distance and azimuth
        distance = geodetic_distance(bb[1], bb[3], bb[1], bb[2])
        edge_azimuth = azimuth(bb[1], bb[3], bb[1], bb[2])

        # Number of samples
        num_samples = np.ceil(distance / spacing_lat)
        spacing_lat = distance / num_samples
        distance = spacing_lon * num_samples

        # Sample the first edge (the right one) moving bottom up and compute
        # the profiles. We reverse the list to comply with the right hand rule
        coords = npoints_towards(
            bb[1], bb[3], 0, edge_azimuth, distance, 0, num_samples)
        tmp_profiles = _get_profiles(coords, dip_dir, max_length)
        tmp_profiles.reverse()

        # Top edge distance and azimuth
        distance = geodetic_distance(bb[1], bb[3], bb[0], bb[3])
        edge_azimuth = azimuth(bb[1], bb[3], bb[0], bb[3])

        # Number of samples
        num_samples = np.ceil(distance / spacing_lon)
        spacing_lon = distance / num_samples
        distance = spacing_lon * num_samples

        # Sample the second edge (the top one) from rigth to left and get the
        # profiles
        coords = npoints_towards(
            bb[1], bb[3], 0, edge_azimuth, distance, 0, num_samples)
        tmp_profiles = _get_profiles(coords, dip_dir, max_length)

        # Update the profile list
        profiles.extend(tmp_profiles)
        profiles = np.array(profiles)
        mask = profiles[:, :, 0] > 180
        profiles[mask, 0] = profiles[mask, 0] - 360
        return profiles, distance

    # Bottom edge distance and azimuth. The latter is taken from the bottom
    # right to the bottom left corner
    distance = geodetic_distance(bb[0], bb[2], bb[1], bb[2])
    edge_azimuth = azimuth(bb[1], bb[2], bb[0], bb[2])

    # Number of samples
    num_samples = np.round(distance / spacing_lon)
    spacing_lon = distance / num_samples
    distance = spacing_lon * num_samples

    # Sampling first edge going from right to left. 0 is the vertical distance
    # in this case
    coords = npoints_towards(
        bb[1], bb[2], 0, edge_azimuth, distance, 0, num_samples)

    # Get profiles
    profiles = _get_profiles(coords, dip_dir, max_length)

    # Left distance and azimuth. The latter is taken bottom up.
    distance = geodetic_distance(bb[0], bb[2], bb[0], bb[3])
    edge_azimuth = azimuth(bb[0], bb[2], bb[0], bb[3])

    # Number of samples
    num_samples = np.ceil(distance / spacing_lat)
    spacing_lat = distance / num_samples
    distance = spacing_lat * num_samples

    # Sampling first edge
    coords = npoints_towards(
        bb[0], bb[2], 0, edge_azimuth, distance, 0, num_samples)

    # Create profiles
    tmp_profiles = _get_profiles(coords, dip_dir, max_length)
    return profiles, distance


def get_profiles_geojson(geojson: str, fname_dep: str, spacing: float,
                         fname_fig: str = ''):
    """
    :param fname_str:
        The name of the Slab2.0 .grd file with the values of strike
    :param fname_dep:
        The name of the Slab2.0 .grd file with the values of depth
    :param spacing:
        The separation distance between traces
    :param fname_fig:
        String specifiying location in which to save output figure
    """
    f_strike = netCDF4.Dataset(fname_dep)
    strikes = np.array(f_strike.variables['z'])
    mask = np.where(np.isfinite(strikes))
    strikes = strikes[mask]

    # Mesh
    x = np.array(f_strike.variables['x'])
    y = np.array(f_strike.variables['y'])
    xx, yy = np.meshgrid(x, y)
    css = []
    gdf = gpd.read_file(geojson)
    gdf['coords'] = gdf.geometry.apply(lambda geom: list(geom.coords))

    # Create cross-sections
    min_lo = 180.0
    min_la = 90.
    max_lo = -180.0
    max_la = -90.0
    for index, row in gdf.iterrows():
        coo = np.array(row.coords)
        min_lo = np.min([min_lo, np.min(coo[:, 0])])
        min_la = np.min([min_la, np.min(coo[:, 1])])
        max_lo = np.max([max_lo, np.max(coo[:, 0])])
        max_la = np.max([max_la, np.max(coo[:, 1])])
    lon_c = min_lo + (max_lo - min_lo) / 2
    lat_c = min_la + (max_la - min_la) / 2

    # Define the forward projection
    aeqd = pyproj.Proj(proj='aeqd', ellps='WGS84', datum='WGS84',
                       lat_0=lat_c, lon_0=lon_c).srs
    gdf_pro = gdf.to_crs(crs=aeqd)

    # Create cross-sections
    for index, row in gdf.iterrows():
        print(gdf.coords[index][0][0], gdf.coords[index][0][1])
        cs = CrossSection(gdf.coords[index][0][0], gdf.coords[index][0][1],
                          (gdf_pro.length[index] / 1000), gdf.dipdir[index])
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
    slb.compute_profiles(spacing / 2)
    if len(str(fname_fig)) > 0:
        bb = np.array([125, 5, 160, 20])
        dlt = 5.0
        reg = [bb - dlt, bb[1] + dlt, bb[2] - dlt, bb[3] + dlt]
        clo = np.mean([bb[0], bb[1]])
        cla = np.mean([bb[2], bb[3]])
        """
        if pygmt_available:
            fig = pygmt.Figure()
            pygmt.makecpt(cmap="jet", series=[0.0, 800])
            # fig.basemap(region=reg, projection="M20c", frame=True)
            fig.basemap(region=reg, projection=f"T{clo}/{cla}/12c", frame=True)
            fig.coast(land="gray", water="skyblue")
            # Profile traces
            for i, pro in enumerate(traces):
                fig.plot(x=pro[:, 0], y=pro[:, 1], pen="red")
                fig.text(x=pro[0, 0], y=pro[0, 1], text=f'{i}', font="4p")
            # Grid
            fig.plot(x=depths[:, 0], y=depths[:, 1],
                     color=-depths[:, 2],
                     style='c0.025c',
                     cmap=True)
            # Profiles
            for key in slb.profiles:
                pro = slb.profiles[key]
                if pro.shape[0] > 0:
                    fig.plot(x=pro[:, 0],
                             y=pro[:, 1],
                             color=pro[:, 2],
                             cmap=True,
                             style="h0.025c",
                             pen='black')
            fig.savefig(fname_fig)
            fig.show()
        else:
            from matplotlib import pyplot as plt
            plt.scatter(depths[:, 0], depths[:, 1], c=-depths[:, 2])
            for i, pro in enumerate(traces):
                plt.plot(pro[:, 0], pro[:, 1], 'k')
                plt.text(pro[0, 0], pro[0, 1], f'{i}')
            for key in slb.profiles:
                pro = slb.profiles[key]
                if pro.shape[0] > 0:
                    plt.plot(pro[:, 0], pro[:, 1], c='r')
            if max(reg[0], reg[1]) > 180:
                xmin = reg[0]-360; xmax = reg[1]-360
            else:
                xmin = reg[0]; xmax = reg[1]
            plt.xlim([xmin, xmax])
            plt.colorbar(label='depth to slab (km)')
            plt.savefig(fname_fig)
        """
    return slb


def get_profiles(fname_str: str, fname_dep: str, spacing: float, fname_fig:
                 str = ''):
    """
    :param fname_str:
        The name of the Slab2.0 .grd file with the values of strike
    :param fname_dep:
        The name of the Slab2.0 .grd file with the values of depth
    :param spacing:
        The separation distance between traces
    :param fname_fig:
        String specifiying location in which to save output figure
    """

    # Reading file with strike values
    f_strike = netCDF4.Dataset(fname_str)
    strikes = np.array(f_strike.variables['z'])
    mask = np.where(np.isfinite(strikes))
    strikes = strikes[mask]

    # Compute the mean strike
    strike_dir = get_mean_azimuth(strikes.flatten())
    dip_dir = (strike_dir + 90) % 360

    # Mesh
    x = np.array(f_strike.variables['x'])
    y = np.array(f_strike.variables['y'])
    xx, yy = np.meshgrid(x, y)

    # Compute the initial bounding box
    tmp_bb, _ = get_bounding_box(xx[mask], yy[mask], delta=1.)
    cx = np.mean([tmp_bb[0:2]])
    cy = np.mean([tmp_bb[2:4]])

    # Rotate the grid with the fault information and get the bounding box
    rx, ry = rotate(xx[mask].flatten(), yy[mask].flatten(), cx, cy, -dip_dir)
    bb = tmp_bb
    r_bb, _ = get_bounding_box(rx, ry, delta=1.)

    # Compute the rotated and buffered bounding box
    dlt = 3.0
    coox = [r_bb[0] - dlt, r_bb[1] + dlt, r_bb[1] + dlt, r_bb[0] - dlt]
    cooy = [r_bb[2] - dlt, r_bb[2] - dlt, r_bb[3] + dlt, r_bb[3] + dlt]
    nbbx, nbby = rotate(coox, cooy, cx, cy, dip_dir)

    # Get traces
    traces, plen = get_initial_traces(nbbx, nbby, dip_dir, spacing)

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
    slb.compute_profiles(spacing / 2)

    if len(str(fname_fig)) > 0:

        dlt = 5.0
        reg = [bb[0] - dlt, bb[1] + dlt, bb[2] - dlt, bb[3] + dlt]
        clo = np.mean([bb[0], bb[1]])
        cla = np.mean([bb[2], bb[3]])

        if pygmt_available:

            fig = pygmt.Figure()
            pygmt.makecpt(cmap="jet", series=[0.0, 800])
            # fig.basemap(region=reg, projection="M20c", frame=True)
            fig.basemap(region=reg, projection=f"T{clo}/{cla}/12c", frame=True)
            fig.coast(land="gray", water="skyblue")

            # Profile traces
            for i, pro in enumerate(traces):
                fig.plot(x=pro[:, 0], y=pro[:, 1], pen="red")
                fig.text(x=pro[0, 0], y=pro[0, 1], text=f'{i}', font="4p")

            # Grid
            fig.plot(x=depths[:, 0], y=depths[:, 1],
                     color=-depths[:, 2],
                     style='c0.025c',
                     cmap=True)

            # Profiles
            for key in slb.profiles:
                pro = slb.profiles[key]
                if pro.shape[0] > 0:
                    fig.plot(x=pro[:, 0],
                             y=pro[:, 1],
                             color=pro[:, 2],
                             cmap=True,
                             style="h0.025c",
                             pen='black')
            fig.savefig(fname_fig)
            fig.show()

        else:
            from matplotlib import pyplot as plt
            plt.scatter(depths[:, 0], depths[:, 1], c=-depths[:, 2])
            for i, pro in enumerate(traces):
                plt.plot(pro[:, 0], pro[:, 1], 'k')
                plt.text(pro[0, 0], pro[0, 1], f'{i}')

            for key in slb.profiles:
                pro = slb.profiles[key]
                if pro.shape[0] > 0:
                    plt.plot(pro[:, 0], pro[:, 1], c='r')

            if max(reg[0], reg[1]) > 180:
                xmin = reg[0]-360; xmax = reg[1]-360
            else:
                xmin = reg[0]; xmax = reg[1]
            plt.xlim([xmin, xmax])
            plt.colorbar(label='depth to slab (km)')
            plt.savefig(fname_fig)



    return slb


def rotate(x, y, offset_x, offset_y, degrees):
    radians = np.deg2rad(degrees)
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = np.cos(radians)
    sin_rad = np.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y
    return qx, qy
