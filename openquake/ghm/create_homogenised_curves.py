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

import os
import re
import glob
import copy
import shutil
import pickle
import warnings
import logging
from datetime import datetime
import h3
import pyproj
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.speedups

from shapely.geometry import Point
from openquake.ghm import mosaic
from openquake.baselib import sap
from openquake.ghm.utils import create_query
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


def find_hazard_curve_file(datafolder, vs30_flag, key, imt_str):
    """
    Searches for a file in a folder given a key

    :param str datafolder:
        The name of the folder where to search

    :param str key:
        The pattern to be used for searching the file

    :param str imt_str:
        String specifying the desired intensity measure type

    :param bool vs30_flag:
        True (1) if building vs30 maps

    :return:
        A list with the files matching the pattern
    """
    # First search for mean results
    tmps = 'hazard_curve-mean-{:s}*.csv'.format(imt_str)
    key = re.sub('[0-9]', '', key)

    if float(vs30_flag)==1:
        data_path = os.path.join(datafolder, key.upper(), 'out/vs30*', tmps)
    else:
        data_path = os.path.join(datafolder, key.upper(), 'out*', tmps)

    data_fname = glob.glob(data_path)

    if len(data_fname) == 0:
        tmps = 'hazard_curve-rlz-*-{:s}*.csv'.format(imt_str)

        if float(vs30_flag)==1:
            data_path = os.path.join(datafolder, key.upper(), 'out/vs30*', tmps)
        else:
            data_path = os.path.join(datafolder, key.upper(), 'out*', tmps)

        data_fname = glob.glob(data_path)

    return data_fname


def homogenise_curves(dat, poes, buf):
    """
    Homogenise the hazard curves within a buffer zone.

    :param dat:
        A :class:`numpy.ndarray` instance with two columns. The first one
        contains an integer that can be either 0 or 1. In the former case, the
        point is within the domain of a given model (and inside a buffer) in
        the latter case the point is in the buffer but outside the domain of
        a hazard model. The second column contains the distance to the boundary
        between two models.
    :param poes:
        The probabilities of exceedance to homogenise
    :return:
        Returns the homogenised hazard curve
    """

    # Initialize array with weights
    tmp = np.zeros_like((dat[:, 0]))

    # Points inside a model
    tmp[dat[:, 1] == 0] = buf + dat[dat[:, 1] == 0, 0]

    # Points outside a model
    tmp[dat[:, 1] == 1] = buf - dat[dat[:, 1] == 1, 0]

    # Compute mean curve
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
    for key, val in df.items():
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
    df = pd.read_csv(fname, skiprows=1)
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


def proc(contacts_shp, outpath, datafolder, sidx_fname, boundaries_shp,
         imt_str, inland_shp, models_list=None, only_buffers=False,
         buf=50, h3_resolution=6, mosaic_key='GID_0',vs30_flag=False,
         overwrite=False, sub=False):
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
    :param str models_list:
        [optional] A list of models IDs
    :param buf:
        [optional] Buffer distance
    :param h3_resolution:
        [optional] The h3 resolution
    :param mosaic_key:
        [optional] The key used to identify models
    :param bool vs30_flag:
        True (1) if building vs30 maps
    :param bool overwrite:
        True (1) to overwrite existing files
    :param bool sub:
        True (1) to create buffer map only for models in models_list
    """
    shapely.speedups.enable()
    # Buffer distance in [m]
    buf = float(buf) * 1000

    # Load mosaic data
    mosaic_data = mosaic.DATA[mosaic_key]

    # Checking output directory
    if os.path.exists(outpath):
        lst = glob.glob(os.path.join(outpath, '*.json'))
        lst += glob.glob(os.path.join(outpath, '*.txt'))
        if len(lst):
            if overwrite==True:
                print('Warning: overwriting existing files in {}'.format(outpath))
            else:
                raise ValueError(f'The code requires an empty folder\n{outpath}')
    else:
        os.mkdir(outpath)
    # Read the shapefile with the contacts between models
    contacts_df = gpd.read_file(contacts_shp)

    # Read the shapefile with inland areas
    inland_df = gpd.read_file(boundaries_shp)

    # Load the spatial index
    # sidx = index.Rtree(sidx_fname)

    # Get the list of the models from the data folder
    if models_list is None:
        models_list = []
        for key in mosaic_data.keys():
            if vs30_flag and key=='gld':
                continue
            models_list.append(re.sub('[0-9]+', '', key))

    # Loop over the various models. TODO the value of the buffer here must
    # be converted into a distance in km.
    header_save = None
    imts_save = None
    for i, key in enumerate(sorted(mosaic_data)):

        buffer_data = {}
        buffer_poes = {}
        coords = {}
        # Skip models not included in the list
        if re.sub('[0-9]+', '', key) not in models_list:
            continue
        # Find name of the file with hazard curves
        print_model_info(i, key)
        data_fname = find_hazard_curve_file(datafolder, vs30_flag, key, imt_str)
        print(data_fname[0])


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
        if key in ['waf', 'ssa']:
            from shapely.geometry import Polygon
            coo = get_poly_from_str(mosaic.SUBSETS['GID_0'][key]['AGO'][0])
            df = pd.DataFrame({'name': ['tmp'], 'geo': [Polygon(coo)]})
            dft = gpd.GeoDataFrame(df, geometry='geo')
            idx = map_gdf.geometry.intersects(dft.geometry[0])
            xdf = copy.deepcopy(map_gdf[idx])
            map_gdf = xdf

        # Read the shapefile with the polygons of countries. The explode
        # function converts multipolygons into a single multipolygon.
        tmpdf = gpd.read_file(boundaries_shp)

        # inpt = explode(tmpdf)
        inpt = tmpdf.explode(index_parts=True)
        inpt['MODEL'] = key

        # Select polygons composing the given model and merge them into a
        # single multipolygon.
        selection = create_query(inpt, mosaic_key, mosaic_data[key])
        one_polygon = selection.dissolve(by='MODEL')

        # PROJECTING
        aeqd = pyproj.Proj(proj='aeqd', ellps='WGS84',
                           datum='WGS84', lat_0=map_gdf.lat.mean(),
                           lon_0=map_gdf.lon.mean()).srs
        p4326 = pyproj.CRS.from_string("epsg:4326")
        map_gdf = map_gdf.set_crs('epsg:4326')
        map_gdf_pro = map_gdf.to_crs(crs=aeqd)

        # Now we process the polygons composing the selected model
        for poly in one_polygon.geometry:

            tmp = gpd.GeoSeries([poly], crs='epsg:4326')
            poly_pro = tmp.to_crs(crs=aeqd)

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

                    tmp_geo_gse = gpd.GeoSeries([geo], crs='epsg:4326')
                    geo_pro = tmp_geo_gse.to_crs(crs=aeqd)
                    tpoly = geo_pro.geometry.values
                    idx = map_gdf_pro.geometry.intersects(tpoly.buffer(buf)[0])

                    # Key defining the second model
                    other = lb
                    if key.upper() == lb:
                        other = la

                    # Create the polygon covering the second model
                    selection = create_query(inpt, mosaic_key,
                                             mosaic_data[other.lower()])
                    other_polygon = selection.dissolve(by='MODEL')
                    if not len(other_polygon):
                        raise ValueError('Empty dataframe')

                    # Create a dataframe with just the points in the buffer
                    # and save the distance of each point frotmpdfm the border
                    tmpdf = copy.deepcopy(map_gdf[idx])
                    tmpdf = tmpdf.set_crs('epsg:4326')
                    tmpdf = gpd.sjoin(tmpdf, inland_df, how='inner',
                                      predicate='intersects')
                    p_geo = gpd.GeoDataFrame({'geometry': [geo]})
                    p_geo = p_geo.set_crs('epsg:4326')

                    # Computing the distances
                    aeqd_local = pyproj.Proj(proj='aeqd', ellps='WGS84',
                                             datum='WGS84',
                                             lat_0=tmpdf.lat.mean(),
                                             lon_0=tmpdf.lon.mean()).srs
                    tmpdf_pro = tmpdf.to_crs(crs=aeqd_local)
                    p_geo_pro = p_geo.to_crs(crs=aeqd_local)

                    # Original distance is in [m]
                    dst = tmpdf_pro.distance(p_geo_pro.iloc[0].geometry)
                    # dst = tmpdf.distance(geo)
                    tmpdf = tmpdf.assign(distance=dst)

                    # Create a geodataframe with the geometry of the polygon
                    # for the second model
                    g = other_polygon.geometry[0]
                    xgdf = gpd.GeoDataFrame(gpd.GeoSeries(g))
                    xgdf = xgdf.rename(columns={0: 'geometry'}).set_geometry('geometry')
                    xgdf = xgdf.set_crs('epsg:4326')

                    # Rename to avoid raising an error in the sjoin
                    tmpdf = tmpdf.rename(columns={"index_right": "old_index_right"})

                    # Select the points contained in the buffer and belonging
                    # to the other model. 'tmpdf' contains the points in the
                    # buffer. These points are labelled.
                    #  idx_other = tmpdf.geometry.intersects(g)
                    res = gpd.sjoin(tmpdf, xgdf)

                    # Assign a new column to the dataframe
                    #  tmpdf = tmpdf.assign(outside=idx_other)
                    tmpdf = tmpdf.assign(outside=False)
                    tmpdf.loc[res.index, 'outside'] = 1
                    tmpdf.outside = tmpdf.outside.astype(int)

                    # Update the polygon containing just internal points i.e.
                    # points within the model but outside of the buffers. The
                    # points in the buffer but outside the model are True.
                    poly_pp = poly_pro.buffer(0)
                    poly_pp = poly_pp.difference(tpoly.buffer(buf)[0])
                    poly_pro = poly_pp

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
                        # res = list(sidx.nearest((p.x, p.y, p.x, p.y), 1))

                        # Handling the v4.0 Vs. v3.0 synthax
                        try:
                            res = h3.latlng_to_cell(p.y, p.x, h3_resolution)
                        except:
                            res = h3.geo_to_h3(p.y, p.x, h3_resolution)

                        # if len(res) > 1:
                        #    msg = 'The number of indexes found is larger '
                        #    msg += 'than 1'
                        #    print('Indexes:', res)
                        #    raise ValueError(msg)

                        # Update the information for the reference point
                        # found. The buffer_data dictionary contains
                        # distance and position information of the point
                        # in the buffer
                        if res in buffer_data:
                            buffer_data[res].append([d, o])
                            buffer_poes[res].append(poe)
                        else:
                            buffer_data[res] = [[d, o]]
                            buffer_poes[res] = [poe]
                            coords[res] = [p.x, p.y]
            #  Write information outside the buffers
            if not only_buffers:
                #df = pd.DataFrame({'Name': [key], 'Polygon': [poly_pro]})
                #gdf = gpd.GeoDataFrame(df, geometry='Polygon')
                #gdf = gdf.set_crs('epsg:4326')
                #gdf_pro = gdf.to_crs(crs=aeqd)
                tmp = gpd.GeoDataFrame(geometry=poly_pro)
                within = gpd.sjoin(map_gdf_pro, tmp, predicate='within')
                # Write results after going back to geographic projection
                fname = os.path.join(outpath, 'map_{:s}.json'.format(key))
                final = within.to_crs(crs=p4326)
                final.to_file(fname, driver='GeoJSON')
        # Store temporary files
        tmpdir = os.path.join(outpath, 'temp')
        if not os.path.exists(tmpdir):
            os.mkdir(tmpdir)

        print('saving everything to {}'.format(tmpdir))

        # Save data
        fname = os.path.join(tmpdir, f'{key:s}_data.pkl')
        fou = open(fname, "wb")
        pickle.dump(buffer_data, fou)
        fou.close()

        # Save poes
        fname = os.path.join(tmpdir, f'{key:s}_poes.pkl')
        fou = open(fname, "wb")
        pickle.dump(buffer_poes, fou)
        fou.close()

        # Save coordinates
        fname = os.path.join(tmpdir, f'{key:s}_coor.pkl')
        fou = open(fname, "wb")
        pickle.dump(coords, fou)
        fou.close()

    buffer_processing(outpath, imt_str, models_list, poelabs, buf, vs30_flag, sub)


def buffer_processing(outpath, imt_str, models_list, poelabs, buf, vs30_flag, sub=True):
    """
    Buffer processing

    :param outpath:
        Output path
    :param imt_str:
        String with the IMT name
    :param models_list:
        A list with the IDs of the models
    :param poelabs:
        A list with the column labels used in the .csv file produced by OQ
        and containing the hazard curves
    :param buf:
        The buffer distance in km
    :param bool vs30_flag:
        True (1) if building vs30 maps
    """

    print('Buffer processing')
    mosaic_data = mosaic.DATA['GID_0']
    buf = float(buf)

    buffer_data = {}
    buffer_poes = {}
    coords = {}

    tmpdir = os.path.join(outpath, 'temp')
    for i, key in enumerate(sorted(mosaic_data)):

        # Skip models not included in the list.
        # comment out these lines if wanting to join 
        # all the models, but some have been produced in former runs
        if re.sub('[0-9]+', '', key) not in models_list and sub==True:
            continue
        if key == 'gld' and vs30_flag == 1:
            continue
        print(f'  Loading {key:s}')

        fname = os.path.join(tmpdir, f'{key:s}_data.pkl')
        fou = open(fname, 'rb')
        # tbuffer_data is a dictionary where the key is the ID of the site
        # i.e. the geohash created by H3 in the most recent version of this
        # code
        tbuffer_data = pickle.load(fou)
        fou.close()

        fname = os.path.join(tmpdir, f'{key:s}_poes.pkl')
        fou = open(fname, 'rb')
        tbuffer_poes = pickle.load(fou)
        fou.close()

        fname = os.path.join(tmpdir, f'{key:s}_coor.pkl')
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
    msg = '\n   Final processing'
    logging.info(msg)
    fname = os.path.join(outpath, 'buf.txt')
    fou = open(fname, 'w')

    # TODO
    header = 'i,lon,lat'
    for lab in poelabs:
        header += ','+lab
    fou.write(header)

    # This is the file with points that have only one value (in theory this is
    # impossible)
    fname = os.path.join(outpath, 'buf_unique.txt')
    fuu = open(fname, 'w')
    fuu.write(header)

    # This is the array we use to store the hazard curves for the points within
    # a buffer
    buffer_array = np.empty((len(buffer_data.keys()), len(poelabs)+2))

    # Process information within the buffers
    c = 0
    for key in buffer_data.keys():
        c += 1
        dat = np.array(buffer_data[key])

        if dat.shape[0] > 1:
            poe = np.array(buffer_poes[key])
            meanhc = homogenise_curves(dat, poe, buf)
        else:
            RuntimeWarning('Zero values')
            meanhc = buffer_poes[key][0]
            tmps = f'{c:d},{coords[key][0]:f},{coords[key][1]:f}'
            for prob in meanhc:
                tmps += f',{prob:f}'
            if key not in coords:
                continue
            fuu.write(tmps+'\n')

        # Check key for the point
        if key not in coords:
            raise ValueError(f'missing coords: {key:s}')

        # Write poes
        tmps = f'{c:d},{coords[key][0]:f},{coords[key][1]:f}'
        for prob in meanhc:
            tmps += f',{prob:f}'
        fou.write(tmps+'\n')

        if coords[key][0] > 180 or coords[key][0] < -180:
            raise ValueError('out of bounds')
        buffer_array[c-1, :] = [coords[key][0], coords[key][1]] + \
            list(meanhc)

    columns = ['lon', 'lat'] + poelabs
    bdf = pd.DataFrame(buffer_array, columns=columns)
    bdf['Coordinates'] = list(zip(bdf.lon, bdf.lat))
    bdf['Coordinates'] = bdf['Coordinates'].apply(Point)
    gbdf = gpd.GeoDataFrame(bdf, geometry='Coordinates')
    fname = os.path.join(outpath, 'map_buffer.json')

    if len(gbdf):
        gbdf.to_file(fname, driver='GeoJSON')
    else:
        print('Empty buffer')

    fou.close()
    fuu.close()

def process(contacts_shp, outpath, datafolder, sidx_fname, boundaries_shp,
            imt_str, inland_shp, buf, vs30_flag, *, models_list=None, only_buffers=False,
            h3_resolution=6, mosaic_key='GID_0', foverwrite=False, sub=False):
    """
    This function processes all the models listed in the mosaic.DATA dictionary
    and creates homogenised curves.

    Example use that recreates the curves (model and buffer regions) for EUR and MIE models, 
    overwriting them in their existing folder (/home/hazard/mosaic/../ghm/PGA-rock) and 
    generating the buffer shapefiles for the full globe

    ./create_homogenised_curves.py ./../data/gis/contacts_between_models.shp 
    /home/hazard/mosaic/../ghm/PGA-rock /home/hazard/mosaic ../GGrid/trigrd_split_9_spacing_13 
    /home/hazard/mosaic/../gadm_410_level_0.gpkg PGA ./../data/gis/inland.shp 50.0 0 
    -m "eur,mie" -f 1
    """
    proc(contacts_shp, outpath, datafolder, sidx_fname, boundaries_shp,
         imt_str, inland_shp, models_list, only_buffers, buf, h3_resolution,
         mosaic_key, vs30_flag, float(foverwrite), sub)


process.contacts_shp = 'Name of shapefile with contacts'
process.outpath = 'Output folder'
process.datafolder = 'Folder with the mosaic repository'
process.sidx_fname = 'Rtreespatial index file with ref. grid'
process.boundaries_shp = 'Name of shapefile with boundaries'
process.imt_str = 'String with the intensity measure type'
process.inland_shp = 'Name of shapefile with inland territories'
process.buf = 'Buffer distance'
process.vs30_flag = 'Boolean flag to set path for reading hazard curves'
process.models_list = 'List of models to be processed'
process.h3_resolution = 'H3 resolution used to create the grid of sites'
process.mosaic_key = 'The key used to specify countries'
process.foverwrite = 'Boolean to allow overwriting of files'
process.sub = 'Boolean to create subset according to models_list'

if __name__ == "__main__":
    sap.run(process)

