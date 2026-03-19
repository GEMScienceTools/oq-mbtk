#!/usr/bin/env python
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

""" module create_buffers """

import pathlib
import time
import toml
import shapely
import datetime
import logging
import multiprocessing
import numpy as np
import geopandas as gpd

import matplotlib.pyplot as plt
from shapely.plotting import plot_polygon

from openquake.ghm.utils_debug import ForkedPdb
from openquake.baselib import sap
from openquake.hazardlib.geo.geodetic import npoints_towards
from openquake.hazardlib.geo.geodetic import (
    geodetic_distance, npoints_between
)

# Set the log format
logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(asctime)s | %(message)s"
    )


def _get_buffer_from_poly(lons, lats, dist_km, num_points=16, res_dist=10):
    """
    :param lons:
        An instance of a :class:`numpy.ndarray` with lons
    :param lats:
        An instance of a :class:`numpy.ndarray` with lats
    :param dist_km:
        The buffer distance [km]
    :param num_points:
        Number of points used to discretize a circle
    :param res_dist:
        A resampling distance [km] used to regularize the boundary of the
        polygon
    """

    # Resampling
    out = _resample(lons, lats, res_dist)

    # Create circles
    circles = []
    for _, (lon, lat) in enumerate(zip(out[:, 0], out[:, 1])):

        tmp_circle = get_circle(lon, lat, dist_km, num_points)
        tmp_circle = shapely.make_valid(
            tmp_circle, method='structure'
        )

        # Check if the circle is valid
        if not tmp_circle.is_valid:

            _, ax = plt.subplots()
            plot_polygon(
                tmp_circle,
                ax=ax,
                add_points=True,
                facecolor='lightblue',
                edgecolor='blue')
            ax.set_aspect('equal', adjustable='box')
            plt.show()

        # Update the collection of circles
        circles.append(tmp_circle)

    # Combine all the circles to obtain the buffer
    tmp_buffer = shapely.union_all(circles)

    return tmp_buffer


def _resample(lons, lats, dist_sampling):
    """
    Resample the boundary of a polygon

    :param lons:
    :param lats:
    :param dist_sampling:
    """
    dists = geodetic_distance(lons[:-1], lats[:-1], lons[1:], lats[1:])
    out = []
    for ic, (lo, la) in enumerate(zip(lons[:-1], lats[:-1])):
        if dists[ic] > dist_sampling:
            npo = int(np.floor(dists[ic] / dist_sampling)) + 1
            newp = npoints_between(
                lo, la, 0.0, lons[ic + 1], lats[ic + 1], 0.0, npo
            )
            for tlo, tla in zip(newp[0], newp[1]):
                out.append([float(tlo), float(tla)])
        else:
            out.append([lo, la])
    return np.array(out)


def get_circle(lon, lat, dist_km, num_points=16):
    """
    Creates a set of points on a circle at a certain distance from the center
    defined by input `lon` and 'lat` coordinates.

    :param lon:
        Longitude of circle's center [decimal degrees]
    :param lat:
        Latitude of circle's center [decimal degrees]
    :param dist_km:
        Circle radius [km]
    :param num_points:
        Number of points used to represent in a discretized form the circle
    :returns:
        A :class:`shapely.Polygon` instance
    """
    shift = False
    delta = 360 / num_points
    azimuths = np.linspace(0, 360 - delta, num=num_points)
    circle = np.zeros((num_points, 2))

    # Checking if the circle crosses the IDL
    azim = 90.0
    if np.sign(lon) < 0:
        azim = -90.0
    coo = npoints_towards(
        lon,
        lat,
        depth=0.0,
        azimuth=azim,
        hdist=dist_km,
        vdist=0.0,
        npoints=2
    )
    # Shifting the origin
    if np.sign(coo[0][1]) != np.sign(coo[0][0]):
        shift = True
        sgn = np.sign(lon)
        lon = lon - 180 * sgn

    # Create the circle
    for i, azim in enumerate(azimuths):
        coo = npoints_towards(
            lon,
            lat,
            depth=0.0,
            azimuth=azim,
            hdist=dist_km,
            vdist=0.0,
            npoints=2)
        circle[i, 0] = coo[0][1]
        circle[i, 1] = coo[1][1]

    if shift:
        circle[:, 0] = circle[:, 0] + 180 * sgn
        shift = True

    return shapely.Polygon(circle)


def process_model(indata):
    """
    :param row:
        An instance of :class:`geopandas.GeoSeries`
    """

    row = indata[0]
    poly_filter = indata[1]
    key_column = indata[2]
    dist_km = indata[3]

    # Process individual polygons
    init_buffer = False
    tmp = gpd.GeoSeries(row.geometry).explode()
    tmp = tmp.set_crs("EPSG:4326")

    for i_poly, poly in enumerate(tmp):
        tpoly = shapely.make_valid(poly, method='structure')
        for mpoly in gpd.GeoSeries(tpoly).explode():

            # Filter multipolygon
            if poly_filter is not None:
                if np.any(poly_filter['model'].str.contains(row[key_column])):
                    tgeo = poly_filter[poly_filter.model ==
                                       row[key_column]].geometry
                    flag = poly.within(tgeo)
                    if not np.any(flag):
                        continue

            tmps = "Number of polygons processed: "
            print(f"{tmps} {i_poly+1:06d}/{len(tmp):06d}", end="\r")
            lons = np.array(mpoly.exterior.coords.xy[0])
            lats = np.array(mpoly.exterior.coords.xy[1])
            lons = (lons + 180.0) % 360 - 180.0

            # Get the buffer for the current polygon
            tmp_buffer = _get_buffer_from_poly(
                lons, lats, dist_km, num_points=16, res_dist=10)

            # Combine the original polygon with the buffer
            tmp_buffer = shapely.union_all([tmp_buffer, tpoly])

        if init_buffer is False:
            buffer = tmp_buffer
            init_buffer = True
        else:
            buffer = shapely.union(tmp_buffer, buffer)

    # Cleaning holes
    data = _remove_holes(buffer)
    # ForkedPdb().set_trace()

    return [data, row[key_column]]


def _remove_holes_multipolygon(buf):
    tmp = shapely.MultiPolygon(shapely.Polygon(p.exterior) for p in buf.geoms)
    return tmp


def _remove_holes(buffer):
    """
    Given a polygon or multipolygon, remove every interior polygon and keep
    only the exteriors

    :param buffer:
        An instance of :class:`shapely.Polygon` or
        :class:`shapely.MultiPolygon`
    :returns:
        A list of tuples each one containing one instance of
        :class:`shapely.Polygon` or :class:`shapely.MultiPolygon` and the
        model key
    """

    if isinstance(buffer, shapely.GeometryCollection):
        # TODO
        # This is for filtering out LineStrings found while processing `oat`
        # with a buffering distance of 100 km. I was not able to understand
        # why only in this case the code was generating LineStrings (and
        # therefore a GeometryCollection)
        first = True
        for geo in buffer.geoms:
            if isinstance (geo, shapely.MultiPolygon):
                if first:
                    out = _remove_holes_multipolygon(buf)
                    first = False
                else:
                    tmp = _remove_holes_multipolygon(buf)
                    out = shapely.union(tmp, out)
            elif isinstance (geo, shapely.Polygon):
                if first:
                    out = shapely.Polygon(geo.exterior)
                    first = False
                else:
                    tmp = shapely.Polygon(geo.exterior)
                    out = shapely.union(tmp, out)
            else:
                continue
    elif isinstance(buffer, shapely.MultiPolygon):
        out = _remove_holes_multipolygon(buffer)
    elif isinstance(buffer, shapely.Polygon):
        out = shapely.Polygon(buffer.exterior)
    else:
        raise ValueError('Unhandled case')

    return out


def run(indata):

    # Number of CPU cores to use for multiprocessing
    num_cores = multiprocessing.cpu_count()

    # Create a multiprocessing Pool with the number of cores
    pool = multiprocessing.Pool(processes=num_cores)

    # Parallelize the for loop using the map function of the Pool
    results = pool.map(process_model, indata)

    # Close the pool to free resources
    pool.close()
    pool.join()

    return results


def main(fname_config):
    """
    Create a set of files with new polygons representing the original plus
    a buffer of a certain distance.
    """
    """
    Processes the zonation file and creates the buffers

    :param fname_config:
        The name of the configuration file
    """

    packet_size = 1000

    print(f"\nStart time: {datetime.datetime.now()} [s]")

    # Read configuration file
    config = toml.load(fname_config)
    fname_config_path = pathlib.Path(fname_config).parent.resolve()
    fname_zonation = pathlib.Path(config['zonation_fname'])
    output_dir = fname_config_path / 'out'
    if 'output_dir' in config.keys():
        output_dir = config['output_dir']
    output_dir = pathlib.Path(output_dir)

    num_points = config['num_points_in_circle']
    dist_km = config['buffer_distance']
    keys = []
    if 'models_keys' in config.keys():
        keys = config['models_keys']
    key_column = getattr(config, 'key_column', 'code')

    # Name of the file with polygons to filter input shapefile
    fname_model_filter = None
    if 'model_filter' in config.keys():
        fname_model_filter = config['model_filter']
        fname_model_filter = pathlib.Path(fname_model_filter)

    # Fixing the path of the output folder
    if not output_dir.is_absolute():
        output_dir = fname_config_path / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output file
    out_fname = f'buffer_{dist_km:.0f}km.geojson'
    out_fname = output_dir / out_fname

    # Fixing the path to the file with the zonation
    if not fname_zonation.is_absolute():
        fname_zonation = fname_config_path / fname_zonation

    # Read zonation
    gdf = gpd.read_file(fname_zonation)
    gdf = gdf.sort_values(by=[key_column])

    # Read filter
    poly_filter = None
    if fname_model_filter is not None:
        poly_filter = gpd.read_file(fname_model_filter)
        poly_filter = poly_filter.set_crs("EPSG:4326")
    data = []

    # Preparing data
    indata = []
    for _, row in gdf.iterrows():

        logging.info(f"Working on {row[key_column]}\n")

        if len(keys) > 0 and row[key_column] not in keys:
            continue

        if len(row.geometry.geoms) > packet_size:
            for trow in split_geoseries(row, packet_size):
                indata.append([trow, poly_filter, key_column, dist_km])
        else:
            indata.append([row, poly_filter, key_column, dist_km])

        # NOTE: This is for debugging purposes: sequential execution.
        #       Uncomment the following two lines and comment out the one
        #       row below running parallel calculation
        # out = process_model([row, poly_filter, key_column, dist_km])
        # data.append(out)

    # Compute buffers
    start_time = time.time()
    data = run(indata)
    print(f"\nExecution time: {time.time() - start_time:.2f}s")

    # Aggregating buffers for the same model
    print(f"Aggregating buffers")
    final_data = aggregate_data(data)

    # Create geodataframe with the results
    columns = ['geometry', 'name']
    tmpgdf = gpd.GeoDataFrame(final_data, columns=columns)

    # Save buffer to folder
    print(f"Saving file")
    tmpgdf = tmpgdf.set_crs("EPSG:4326")
    tmpgdf.to_file(out_fname, driver='GeoJSON')
    print(f"Created file: {out_fname} with {len(tmpgdf)} buffers")
    print(f"\nStart time: {datetime.datetime.now()} [s]")


def aggregate_data(data):
    collector = {}
    for dat in data:
        if dat[1] in collector:
            collector[dat[1]] = collector[dat[1]].union(dat[0])
        else:
            collector[dat[1]] = dat[0]
    out = []
    for key in sorted(collector.keys()):
        out.append([collector[key], key])
    return out


def split_geoseries(row, packet_size=1000):
    polys = []
    packets = []
    count = 0
    for geo in row.geometry.geoms:

        if count == packet_size:
            new = row.copy()
            new.geometry = shapely.MultiPolygon(polys)
            packets.append(new)
            polys = []
            count = 0

        polys.append(geo)
        count += 1

    if count > 0:
        new = row.copy()
        new.geometry = shapely.MultiPolygon(polys)
        packets.append(new)

    return packets


main.fname_config = "Configuration file"

if __name__ == '__main__':
    sap.run(main)
