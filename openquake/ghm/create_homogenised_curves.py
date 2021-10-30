#!/usr/bin/env python3
#
# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (c) 2019 GEM Foundation
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake.  If not, see <http://www.gnu.org/licenses/>.

import os
import re
import sys
import glob
import copy
import pickle
import pandas
import warnings
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.speedups

from rtree import index
from datetime import datetime
from shapely.geometry import Point
from openquake.ghm import mosaic
from openquake.baselib import sap
from openquake.ghm.utils import create_query, explode
from openquake.man.tools.csv_output import _get_header1


def get_poly_from_str(tstr):
    """
    Get the coordinates of a polygon from a string.

    :param str tstr:
        A string with a sequence of lon, lat tuples
    :return:
        A :class:`numpy.ndarray` instance containing the coordinates of a
        polygon.
    """
    li = re.split('\\s+', tstr)
    coo = []
    for i in range(0, len(li), 2):
        coo.append([float(li[i]), float(li[i+1])])
    coo = np.array(coo)
    return coo


def find_hazard_curve_file(datafolder, key, imt_str):
    """
    Searches for a file in a folder given a key

    :param str datafolder:
        The name of the folder where to search
    :param str key:
        The pattern to be used for searching the file
    :param imt_str:
        String specifying the desired intensity measure type
    :return:
        A list with the files matching the pattern
    """
    # First search for mean results
    tmps = 'hazard_curve-mean-{:s}*.csv'.format(imt_str)
    key = re.sub('[0-9]', '', key)
    data_path = os.path.join(datafolder, key.upper(), 'out*', tmps)
    data_fname = glob.glob(data_path)
    if len(data_fname) == 0:
        tmps = 'hazard_curve-rlz-*-{:s}*.csv'.format(imt_str)
        data_path = os.path.join(datafolder, key.upper(), 'out*', tmps)
        data_fname = glob.glob(data_path)
    return data_fname


def homogenise_curves(dat, poes, buf):
    """
    :param dat:
    :param poes:
    :return:
        Return the hazard curve
    """
    tmp = np.zeros_like((dat[:, 0]))
    # points inside a model
    tmp[dat[:, 1] == 0] = buf + dat[dat[:, 1] == 0, 0]
    # points outside a model
    tmp[dat[:, 1] == 1] = buf - dat[dat[:, 1] == 1, 0]
    meanhc = np.zeros((poes.shape[1]))
    for i in range(poes.shape[0]):
        meanhc += poes[i, :] * tmp[i]/sum(tmp)
    return meanhc


def recompute_probabilities(df, old_ivt, new_ivt):
    """
    :param df:
    :param old_ivt:
    :param new_ivt:
    """
    for key, val in df.iteritems():
        if re.search('poe', key):
            dat = val.values
            dat[dat > 0.99999999] = 0.99999999
            df[key] = dat
            rate = -np.log(1.-val)/old_ivt
            df[key] = 1.-np.exp(-rate*new_ivt)
    return df


def get_hcurves_geodataframe(fname):
    """
    :param fname:
        Name of the file with the hazard curves
    """
    header = _get_header1(open(fname, 'r').readline())
    inv_time = header['investigation_time']
    imt_str = header['imt']
    res_type = header['result_type']
    # Load hazard curve data
    df = pandas.read_csv(fname, skiprows=1)
    df['Coordinates'] = list(zip(df.lon, df.lat))
    df['Coordinates'] = df['Coordinates'].apply(Point)
    map_gdf = gpd.GeoDataFrame(df, geometry='Coordinates')
    # Homogenise hazard curves to the same investigation period
    if inv_time != 1.0:
        map_gdf = recompute_probabilities(map_gdf, inv_time, 1.0)
    return map_gdf, (res_type, inv_time, imt_str)


def print_model_info(i, key):
    """ """
    dt = datetime.now()
    tmps = dt.strftime('%H:%M:%S')
    tmps = '[@{:s} - #{:d}] Working on {:s}'.format(tmps, i, key)
    print(tmps)


def print_model_read(i, data_fname):
    """ """
    dt = datetime.now()
    tmps = dt.strftime('%H:%M:%S')
    tmps = '[@{:s} - #{:d}] Reading {:s}'.format(tmps, i, data_fname)
    print(tmps)


def get_imtls(poes):
    """
    Returns a numpy.ndarray with the intensity measure levels

    :param poes:
        A list of strings
    """
    imtls = []
    for tmps in poes:
        imtls.append(float(re.sub('poe-', '', tmps)))
    return np.array(imtls)


def process_maps(contacts_shp, outpath, datafolder, sidx_fname, boundaries_shp,
                 imt_str, inland_shp, models_list=None,
                 only_buffers=False):
    """
    This function processes all the models listed in the mosaic.DATA
    dictionary. The code creates for the models in contact with other models
    a file with the points outside of the buffer area

    :param str contacts_shp:
        The shapefile containing the contacts between models
    :param str outpath:
        The folder where results are stored
    :param str datafolder:
        The path to the folder containing the mosaic data
    :param str sidx_fname:
        The name of the file containing the rtree spatial index
    :param str boundaries_shp:
        The name of the shapefile containing the polygons of the countries
    :param str imt_str:
        The string defininig the IMT used for the homogenisation of results
    :param str inland_shp:
        The name of the shapefile defining inland areas
    :param str:
        [optional] A list of models IDs
    """
    shapely.speedups.enable()
    #
    # Checking output directory
    if os.path.exists(outpath):
        lst = glob.glob(os.path.join(outpath, '*.json'))
        lst += glob.glob(os.path.join(outpath, '*.txt'))
        if len(lst):
            raise ValueError('The code requires an empty folder')
    else:
        os.mkdir(outpath)
    # Read the shapefile with the contacts between models
    contacts_df = gpd.read_file(contacts_shp)
    # Read the shapefile with inland areas
    inland_df = gpd.read_file(inland_shp)
    # Load the spatial index
    sidx = index.Rtree(sidx_fname)
    #
    # Get the list of the models from the data folder
    if models_list is None:
        models_list = []
        for key in mosaic.DATA.keys():
            models_list.append(re.sub('[0-9]+', '', key))
    #
    # Loop over the various models
    buf = 0.6
    header_save = None
    imts_save = None
    for i, key in enumerate(sorted(mosaic.DATA)):

        buffer_data = {}
        buffer_poes = {}
        coords = {}

        # Skip models not included in the list
        if re.sub('[0-9]+', '', key) not in models_list:
            continue
        # Find name of the file with hazard curves
        print_model_info(i, key)
        data_fname = find_hazard_curve_file(datafolder, key, imt_str)
        # Read hazard curves
        map_gdf, header = get_hcurves_geodataframe(data_fname[0])
        # Check the stability of information used. TODO we should also check
        # that the IMTs are always the same
        if header_save is None:
            header_save = header
        else:
            for obtained, expected in zip(header, header_save):
                # print(obtained, expected)
                # assert obtained == expected
                pass
        # Create the list of column names with hazard curve data. These are
        # the IMLs
        poelabs = [l for l in map_gdf.columns.tolist() if re.search('^poe', l)]
        imts = get_imtls(poelabs)
        if len(poelabs) < 1:
            raise ValueError('Empty list of column headers')
        # Check the IMLs used
        if imts_save is None:
            imts_save = imts
        else:
            np.testing.assert_allclose(imts_save, imts, rtol=1e-5)
        # Fixing an issue at the border between waf and ssa
        # TODO can we remove this now?
        if key in ['waf18', 'ssa18']:
            from shapely.geometry import Polygon
            coo = get_poly_from_str(mosaic.SUBSETS[key]['AO'][0])
            df = pd.DataFrame({'name': ['tmp'], 'geo': [Polygon(coo)]})
            dft = gpd.GeoDataFrame(df, geometry='geo')
            idx = map_gdf.geometry.intersects(dft.geometry[0])
            xdf = copy.deepcopy(map_gdf[idx])
            map_gdf = xdf
        # Read the shapefile with the polygons of countries. The explode
        # function converts multipolygons into a single multipolygon.
        tmpdf = gpd.read_file(boundaries_shp)
        inpt = explode(tmpdf)
        inpt['MODEL'] = key
        # Select polygons composing the given model and merge them into a
        # single multipolygon.
        selection = create_query(inpt, 'FIPS_CNTRY', mosaic.DATA[key])
        one_polygon = selection.dissolve(by='MODEL')
        # Now we process the polygons composing the selected model
        for poly in one_polygon.geometry:
            # Checking the contacts between the current model and the
            # surrounding ones as specified in the contacts_df geodataframe
            c = 0
            for la, lb, geo in zip(contacts_df.modelA, contacts_df.modelB,
                                   contacts_df.geometry):
                if key.upper() in [la, lb]:
                    print('    ', la, lb)
                    # Index of the points in the buffer. The buffer
                    # includes the country boundary + buffer distance.
                    # map_gdf is a dataframe with the hazard data.
                    idx = map_gdf.geometry.intersects(geo.buffer(buf))
                    # Key defining the second model
                    other = lb
                    if key.upper() == lb:
                        other = la
                    # Create the polygon covering the second model
                    selection = create_query(inpt, 'FIPS_CNTRY',
                                             mosaic.DATA[other.lower()])
                    other_polygon = selection.dissolve(by='MODEL')
                    if not len(other_polygon):
                        raise ValueError('Empty dataframe')
                    # Create a dataframe with just the points in the buffer
                    # and save the distance of each point from the border
                    tmpdf = copy.deepcopy(map_gdf[idx])
                    tmpdf.crs = {'init': 'epsg:4326'}
                    tmpdf = gpd.sjoin(tmpdf, inland_df, how='inner',
                                      op='intersects')
                    dst = tmpdf.distance(geo)
                    tmpdf = tmpdf.assign(distance=dst)
                    # Select the points contained in the buffer and belonging
                    # to the other model. These points are labelled.
                    g = other_polygon.geometry[0]
                    idx_other = tmpdf.geometry.intersects(g)
                    tmpdf = tmpdf.assign(outside=idx_other)
                    tmpdf.outside = tmpdf.outside.astype(int)
                    # Update the polygon containing just internal points i.e.
                    # points within the model but outside of the buffers. The
                    # points in the buffer but outside the model are True.
                    poly = poly.difference(geo.buffer(buf))
                    # Write the data in the buffer between the two models
                    fname = 'buf{:d}_{:s}.json'.format(c, key)
                    fname = os.path.join(outpath, fname)
                    if len(tmpdf):
                        tmpdf.to_file(fname, driver='GeoJSON')
                    else:
                        warnings.warn('Empty dataframe', RuntimeWarning)
                    # Update the counter of the points in the buffer and
                    # store hazard curves and their position (i.e. inside
                    # or outside the polygon of a model)
                    c += 1
                    for iii, (p, d, o) in enumerate(zip(tmpdf.geometry,
                                                        tmpdf['distance'],
                                                        tmpdf['outside'])):
                        # pidx = tmpdf.index.values[iii]
                        # get only poes for the various IMLs
                        tmp = tmpdf[poelabs]
                        poe = tmp.iloc[iii].values
                        # Using rtree we find the closest point on the
                        # reference grid. Check that there is a single index.
                        res = list(sidx.nearest((p.x, p.y, p.x, p.y), 1))
                        if len(res) > 1:
                            msg = 'The number of indexes found is larger '
                            msg += 'than 1'
                            print('Indexes:', res)
                            raise ValueError(msg)
                        # Update the information for the reference point
                        # found. The buffer_data dictionary contains
                        # distance and position information of the point
                        # in the buffer
                        if res[0] in buffer_data:
                            buffer_data[res[0]].append([d, o])
                            buffer_poes[res[0]].append(poe)
                        else:
                            buffer_data[res[0]] = [[d, o]]
                            buffer_poes[res[0]] = [poe]
                            coords[res[0]] = [p.x, p.y]
            # idx is a series of booleans
            if not only_buffers:
                df = pandas.DataFrame({'Name': [key], 'Polygon': [poly]})
                gdf = gpd.GeoDataFrame(df, geometry='Polygon')
                within = gpd.sjoin(map_gdf, gdf, op='within')
                fname = os.path.join(outpath, 'map_{:s}.json'.format(key))
                within.to_file(fname, driver='GeoJSON')
        #
        # Storing temporary files
        tmpdir = os.path.join(outpath, 'temp')
        if not os.path.exists(tmpdir):
            os.mkdir(tmpdir)
        #
        fname = os.path.join(tmpdir, '{:s}_data.pkl'.format(key))
        fou = open(fname, "wb")
        pickle.dump(buffer_data, fou)
        fou.close()
        #
        fname = os.path.join(tmpdir, '{:s}_poes.pkl'.format(key))
        fou = open(fname, "wb")
        pickle.dump(buffer_poes, fou)
        fou.close()
        #
        fname = os.path.join(tmpdir, '{:s}_coor.pkl'.format(key))
        fou = open(fname, "wb")
        pickle.dump(coords, fou)
        fou.close()

    buffer_processing(outpath, datafolder, sidx_fname, imt_str,
                      models_list, poelabs, buf)


def buffer_processing(outpath, datafolder, sidx_fname, imt_str, models_list,
                      poelabs, buf):
    """
    Buffer processing
    """

    buffer_data = {}
    buffer_poes = {}
    coords = {}

    tmpdir = os.path.join(outpath, 'temp')
    for i, key in enumerate(sorted(mosaic.DATA)):

        # Skip models not included in the list
        if re.sub('[0-9]+', '', key) not in models_list:
            continue
        #
        print('Loading {:s}'.format(key))
        #
        fname = os.path.join(tmpdir, '{:s}_data.pkl'.format(key))
        fou = open(fname, 'rb')
        tbuffer_data = pickle.load(fou)
        fou.close()
        #
        fname = os.path.join(tmpdir, '{:s}_poes.pkl'.format(key))
        fou = open(fname, 'rb')
        tbuffer_poes = pickle.load(fou)
        fou.close()
        #
        fname = os.path.join(tmpdir, '{:s}_coor.pkl'.format(key))
        fou = open(fname, 'rb')
        tcoords = pickle.load(fou)
        fou.close()

        for k in tbuffer_data.keys():
            if k not in buffer_data:
                buffer_data[k] = []
                buffer_poes[k] = []

            coords[k] = tcoords[k]
            for d in tbuffer_data[k]:
                buffer_data[k].append(d)
            for d in tbuffer_poes[k]:
                buffer_poes[k].append(d)

    # Here we process the points in the buffer
    msg = 'Final processing'
    logging.info(msg)
    fname = os.path.join(outpath, 'buf.txt')
    fou = open(fname, 'w')
    # TODO
    header = 'i,lon,lat'
    for l in poelabs:
        header += ','+l
    fou.write(header)
    # This is the file with points that has only one value (in theory this is
    # impossible)
    fname = os.path.join(outpath, 'buf_unique.txt')
    fuu = open(fname, 'w')
    fuu.write(header)
    c = 0
    # This is the array we use to store the hazard curves for the points within
    # a buffer
    buffer_array = np.empty((len(buffer_data.keys()), len(poelabs)+2))
    # Process information within the buffers
    for key in buffer_data.keys():
        c += 1
        dat = np.array(buffer_data[key])
        if dat.shape[0] > 1:
            poe = np.array(buffer_poes[key])
            meanhc = homogenise_curves(dat, poe, buf)
        else:
            RuntimeWarning('Zero values')
            meanhc = buffer_poes[key][0]
            tmps = '{:d},{:f},{:f}'.format(c, coords[key][0], coords[key][1])
            for p in meanhc:
                tmps += ',{:f}'.format(p)
            if key not in coords:
                continue
            fuu.write(tmps+'\n')
        # Checking key for the point
        if key not in coords:
            raise ValueError('missing coords: {:s}'.format(key))
        # Writing poes
        tmps = '{:d},{:f},{:f}'.format(c, coords[key][0], coords[key][1])
        for p in meanhc:
            tmps += ',{:f}'.format(p)
        fou.write(tmps+'\n')

        if coords[key][0] > 180 or coords[key][0] < -180:
            raise ValueError('out of bounds')
        buffer_array[c-1, :] = [coords[key][0], coords[key][1]] + \
            list(meanhc)

    columns = ['lon', 'lat'] + poelabs
    bdf = pandas.DataFrame(buffer_array, columns=columns)
    bdf['Coordinates'] = list(zip(bdf.lon, bdf.lat))
    bdf['Coordinates'] = bdf['Coordinates'].apply(Point)
    gbdf = gpd.GeoDataFrame(bdf, geometry='Coordinates')
    fname = os.path.join(outpath, 'map_buffer.json')
    gbdf.to_file(fname, driver='GeoJSON')
    fou.close()
    fuu.close()


def process(contacts_shp, outpath, datafolder, sidx_fname, boundaries_shp,
             imt_str, inland_shp, models_list=None,
             only_buffers=False):
    """
    This function processes all the models listed in the mosaic.DATA dictionary
    and creates homogenised curves.
    """
    process_maps(contacts_shp, outpath, datafolder, sidx_fname, boundaries_shp,
        imt_str, inland_shp, models_list, only_buffers)

process.contacts_shp = 'Name of shapefile with contacts'
process.outpath = 'Output folder'
process.datafolder = 'Folder with the mosaic repository'
process.sidx_fname = 'Rtree spatial index file with ref. grid'
process.boundaries_shp = 'Name of shapefile with boundaries'
process.imt_str = 'String with the intensity measure type'
process.inland_shp = 'Name of shapefile with inland territories'
process.models_list = 'List of models to be processed'

if __name__ == "__main__":
    sap.run(process)
