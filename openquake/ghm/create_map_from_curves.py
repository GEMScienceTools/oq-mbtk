# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (c) 2018 GEM Foundation
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
from openquake.ghm.utils import create_query, explode


def get_poly_from_str(tstr):
    """
    Get the coordinates of a polygon from a string.

    :param str tstr:
        A string with a sequence of lon, lat tuples
    :return:
        A :class:`numpy.ndarray` instance containing the coordinates of a 
        polygon.
    """
    li = re.split('\s+', tstr)
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

def process_maps(contacts_shp, outpath, datafolder, sidx_fname, shapefile,
                 imt_str, inland_shapefile, models_list=None, 
                 only_buffers=False):
    """
    This function process all the models listed in the mosaic.DATA dictionary. 
    The code creates for the models in contact with other models a file with 
    the points outside of the buffer area

    :param contacts_shp:
        The shapefile containing the contacts between models
    :param str outpath:
        The folder where results are stored
    :param str datafolder:
        The path to the folder containing the mosaic data 
    :param str sidx_fname:
        The name of the file containing the rtree spatial index
    :param str shapefile:
        The name of the shapefile containing the polygons of the countries
    :param str only_buffers:
    """
    
    # TODO check that the GM values used to compute the poes are all the same

    shapely.speedups.enable()
    # reading the shapefile with the contacts between models
    contacts_df = gpd.read_file(contacts_shp)
    # reading the shapefile with inland area
    inland_df = gpd.read_file(inland_shapefile)
    #
    # load the spatial index
    sidx = index.Rtree(sidx_fname)
    #
    # 
    if models_list is None:
        models_list = mosaic.DATA.keys()
    #
    # Loop over the various models
    buf = 0.6
    buffer_data = {}
    buffer_poes = {}
    coords = {}
    for i, key in enumerate(sorted(mosaic.DATA)):
        #
        # skipping models
        if re.sub('[0-9]+', '', key) not in models_list:
            continue
        # printing info 
        dt = datetime.now()
        tmps = dt.strftime('%H:%M:%S')
        tmps = '[@{:s} - #{:d}] Working on {:s}'.format(tmps, i, key)
        print(tmps)
        # load data from csv file
        data_fname = find_hazard_curve_file(datafolder, key, imt_str)
        # printing info
        dt = datetime.now()
        tmps = dt.strftime('%H:%M:%S')
        tmps = '[@{:s} - #{:d}] Reading {:s}'.format(tmps, i, data_fname[0])
        print(tmps)
        # read the filename with the hazard map and create a geopandas 
        # geodataframe
        df = pandas.read_csv(data_fname[0], skiprows=1)
        df['Coordinates'] = list(zip(df.lon, df.lat))
        df['Coordinates'] = df['Coordinates'].apply(Point)
        map_gdf = gpd.GeoDataFrame(df, geometry='Coordinates')
        # create the list of column names with hazard curve data
        poelabs = [l for l in map_gdf.columns.tolist() if re.search('^poe', l)]
        if len(poelabs) < 1:
            raise ValueError('Empty list of column headers')


        # fixing an issue at the border between waf and ssa
        # TODO can we remove this?
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
        tmpdf = gpd.read_file(shapefile)
        inpt = explode(tmpdf)
        inpt['MODEL'] = key
        # Select polygons composing the given model and merge them into a
        # single multipolygon.
        selection = create_query(inpt, 'FIPS_CNTRY', mosaic.DATA[key])
        one_polygon = selection.dissolve(by='MODEL')
        # Now we process the polygons composing the selected model
        for poly in one_polygon.geometry:
            # Checking the contacts between the current model and the 
            # surrounding one as specified in the contacts_df geodataframe            
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
                    # Write the data in the buffer
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
                        pidx = tmpdf.index.values[iii]
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
                        # Update the information for the reference point on 
                        # found
                        if res[0] in buffer_data:
                            buffer_data[res[0]].append([d, o])
                            buffer_poes[res[0]].append(poe)
                        else:
                            buffer_data[res[0]] = [[d, o]]
                            buffer_poes[res[0]] = [poe]
                            coords[res[0]] = [p.x, p.y]
            #
            # idx is a series of booleans
            if not only_buffers:
                df = pandas.DataFrame({'Name': [key], 'Polygon': [poly]})
                gdf = gpd.GeoDataFrame(df, geometry='Polygon')
                within = gpd.sjoin(map_gdf, gdf, op='within')
                fname = os.path.join(outpath, 'map_{:s}.json'.format(key))
                within.to_file(fname, driver='GeoJSON')
    #
    # Here we process the points in the buffer
    msg = 'Final processing'
    logging.info(msg)
    fname = os.path.join(outpath, 'buf.txt')
    fou = open(fname, 'w')
    # TODO
    header = 'i,lon,lat'
    for l in poelabs:
        header += l
    fou.write(header)
    fname = os.path.join(outpath, 'buf_unique.txt')
    fuu = open(fname, 'w')
    # TODO
    fuu.write('i,lon,lat,PGA-0.002107\n')
    c = 0
    # This is the array we use to store the hazard curves for the points within
    # a buffer
    buffer_array = np.empty((len(buffer_data.keys()), len(poelabs)+2))
    for key in buffer_data.keys():
        c += 1
        dat = np.array(buffer_data[key])
        if dat.shape[0] > 1:
            poe = np.array(buffer_poes[key])
            meanhc = homogenise_curves(dat, poe, buf)
        else:
            RuntimeWarning('Zero values')
            meanhc = dat[0, 1]
            if key not in coords:
                continue
            fuu.write('{:d},{:f},{:f},{:f}\n'.format(c, coords[key][0],
                                                     coords[key][1], meanhc))
        # Checking key
        if key not in coords:
            raise ValueError('missing coords: {:s}'.format(key))
        # Writing poes
        tmps = '{:d},{:f},{:f}\n'.format(c, coords[key][0], coords[key][1])
        for p in meanhc:
            tmps += '{:f}'.format(p)
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


def main(argv):
    #
    # Logging
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.ERROR)
    #
    # Settings
    # contacts_df, outpath, datafolder, sidx_fname, shapefile
    argv.append('./data/gis/contacts_between_models.shp')
    argv.append('out')
    argv.append('/Users/mpagani/Documents/2018/diary/11/13_ucf/maps')
    argv.append('./data/global_grid/trigrd_split_9_spacing_13')
    path = '/Users/mpagani/NC/Hazard_Charles/Data/Administrative/'
    name = 'world_country_admin_boundary_with_fips_codes_mosaic_eu_russia.shp'
    argv.append(os.path.join(path, name))
    argv.append('SA(0.1)')
    #
    # Checking output directory
    lst = glob.glob(os.path.join(argv[5], '*.json'))
    lst += glob.glob(os.path.join(argv[5], '*.txt'))
    if len(lst):
        raise ValueError('The code requires an empty folder')
    #
    # Set Name of the shapefile containing the contacts between the models
    shapefile = argv[0]
    contacts_df = gpd.read_file(shapefile)
    #
    # Processing
    process_maps(contacts_df, argv[1], argv[2], argv[3], argv[4], argv[5],
                 argv[6])


if __name__ == "__main__":
    # argv[0] - Shapefile with contacts between models
    # argv[1] - Output folder
    # argv[2] - Folder of the main repository
    # argv[3] - Path to the global grid spatial index
    main(sys.argv[1:])
