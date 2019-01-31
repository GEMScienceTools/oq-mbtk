
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
    :param str tstr:
        A string with a sequence of lon, lat tuples
    """
    li = re.split('\s+', tstr)
    coo = []
    for i in range(0, len(li), 2):
        coo.append([float(li[i]), float(li[i+1])])
    coo = np.array(coo)
    return coo


def find_filename(datafolder, key):
    #
    # Set input filename
    tmps = 'hazard_map-mean_*.csv'
    key = re.sub('[0-9]', '', key)
    data_path = os.path.join(datafolder, key.upper(), 'out*', tmps)
    data_fname = glob.glob(data_path)
    if len(data_fname) == 0:
        tmps = 'hazard_map-rlz*.csv'
        data_path = os.path.join(datafolder, key.upper(), 'out*', tmps)
        data_fname = glob.glob(data_path)
    return data_fname


def process_maps(contacts_df, outfname, datafolder, sidx_fname, shapefile,
                 out_path='.', only_buffers=False):
    """
    This process all the models listed in the mosaic.DATA dictionary. The code
    creates for the models in contact with other models a file with the points
    outside of the buffer area
    """
    SKIPLIST = []

    shapely.speedups.enable()
    #
    # load the spatial index
    sidx = index.Rtree(sidx_fname)
    #
    # Loop over the various models
    buf = 0.6
    buffer_data = {}
    coords = {}
    for i, key in enumerate(sorted(mosaic.DATA)):

        dt = datetime.now()
        tmps = dt.strftime('%H:%M:%S')
        tmps = '[@{:s} - #{:d}] Working on {:s}'.format(tmps, i, key)
        print(tmps)

        # 29 is just the last model
        #if key not in ['ssa18', 'waf18', 'zaf18']:
        #   continue
        # if key not in ['eur13']:
        #    continue
        #if key not in ['sea18', 'idn17']:
        #    continue
        #if key not in ['sam18']:
        #    continue

        #if key in ['waf18']:
        #    buf = 0.52
        #else:
        #    buf = 0.6

        #
        # skip models
        if key in SKIPLIST:
            continue
        #
        # load data from csv file
        data_fname = find_filename(datafolder, key)

        dt = datetime.now()
        tmps = dt.strftime('%H:%M:%S')
        tmps = '[@{:s} - #{:d}] Reading {:s}'.format(tmps, i, data_fname[0])
        print(tmps)

        df = pandas.read_csv(data_fname[0], skiprows=1)
        df['Coordinates'] = list(zip(df.lon, df.lat))
        df['Coordinates'] = df['Coordinates'].apply(Point)
        map_gdf = gpd.GeoDataFrame(df, geometry='Coordinates')

        if key in ['waf18', 'ssa18']:
            from shapely.geometry import Polygon
            coo = get_poly_from_str(mosaic.SUBSETS[key]['AO'][0])
            df = pd.DataFrame({'name': ['tmp'], 'geo': [Polygon(coo)]})
            dft = gpd.GeoDataFrame(df, geometry='geo')
            idx = map_gdf.geometry.intersects(dft.geometry[0])
            xdf = copy.deepcopy(map_gdf[idx])
            map_gdf = xdf
        #
        # read polygon file and set MODEL attribute
        tmpdf = gpd.read_file(shapefile)
        inpt = explode(tmpdf)
        inpt['MODEL'] = key
        #
        # Select polygons composing the given model and merge them into a
        # single multipolygon
        selection = create_query(inpt, 'FIPS_CNTRY', mosaic.DATA[key])
        one_polygon = selection.dissolve(by='MODEL')
        #
        # Processing
        for poly in one_polygon.geometry:

            """
            #
            # idx is a series of booleans
            idx = map_gdf.geometry.intersects(p)
            pin = map_gdf[idx]
            map_gdf[idx].to_file('map_{:s}.json'.format(key), driver='GeoJSON')
            """

            #
            # Check all the contacts between models and process the ones
            # including this model
            c = 0
            for la, lb, geo in zip(contacts_df.modelA, contacts_df.modelB,
                                   contacts_df.geometry):
                if key.upper() in [la, lb]:
                    print('    ', la, lb)
                    #
                    # Index of the points in the buffer. The buffer
                    # includes the country boundary + buffer distance.
                    # map_gdf is a dataframe with the points of the hazard.
                    idx = map_gdf.geometry.intersects(geo.buffer(buf))
                    #
                    # Key defining the other model
                    other = lb
                    if key.upper() == lb:
                        other = la
                    #
                    # Load the polygon with the other model
                    selection = create_query(inpt, 'FIPS_CNTRY',
                                             mosaic.DATA[other.lower()])
                    other_polygon = selection.dissolve(by='MODEL')
                    if not len(other_polygon):
                        raise ValueError('Empty dataframe')
                    #
                    # Create a dataframe with just the points in the buffer
                    tmpdf = copy.deepcopy(map_gdf[idx])
                    dst = tmpdf.distance(geo)
                    tmpdf = tmpdf.assign(distance=dst)
                    #
                    # Select the points in the other model
                    g = other_polygon.geometry[0]
                    idx_other = tmpdf.geometry.intersects(g)
                    tmpdf = tmpdf.assign(outside=idx_other)
                    tmpdf.outside = tmpdf.outside.astype(int)
                    #
                    # Update the polygon containing just internal points i.e.
                    # points within the model but outside of the possible
                    # buffers
                    poly = poly.difference(geo.buffer(buf))
                    #
                    # Write the data in the buffer
                    fname = 'buf{:d}_{:s}.json'.format(c, key)
                    fname = os.path.join(out_path, fname)
                    if len(tmpdf):
                        tmpdf.to_file(fname, driver='GeoJSON')
                    else:
                        warnings.warn('Empty dataframe', RuntimeWarning)
                    #
                    # Update the container for the points in the buffers
                    c += 1
                    for p, gm, d, o in zip(tmpdf.geometry,
                                           tmpdf['PGA-0.002107'],
                                           tmpdf['distance'],
                                           tmpdf['outside']):
                        res = list(sidx.nearest((p.x, p.y, p.x, p.y), 1))
                        if res[0] in buffer_data:
                            buffer_data[res[0]].append([d, gm, o])
                        else:
                            buffer_data[res[0]] = [[d, gm, o]]
                            coords[res[0]] = [p.x, p.y]
            #
            # idx is a series of booleans
            if not only_buffers:
                df = pandas.DataFrame({'Name': [key], 'Polygon': [poly]})
                gdf = gpd.GeoDataFrame(df, geometry='Polygon')
                within = gpd.sjoin(map_gdf, gdf, op='within')
                fname = os.path.join(out_path, 'map_{:s}.json'.format(key))
                within.to_file(fname, driver='GeoJSON')

    #
    # Here we process the points in the buffer
    msg = 'Final processing'
    logging.info(msg)
    fou = open('./out/buf.txt', 'w')
    fou.write('i,lon,lat,PGA-0.002107\n')
    fuu = open('./out/buf_unique.txt', 'w')
    fuu.write('i,lon,lat,PGA-0.002107\n')
    c = 0
    buffer_array = np.empty((len(buffer_data.keys()), 3))
    for key in buffer_data.keys():
        c += 1
        dat = np.array(buffer_data[key])
        if dat.shape[0] > 1:
            tmp = np.zeros_like(dat[:, 0])
            tmp[dat[:, 2] == 0] = buf + dat[dat[:, 2] == 0, 0]
            tmp[dat[:, 2] == 1] = buf - dat[dat[:, 2] == 1, 0]
            meangm = sum(dat[:, 1] * tmp/sum(tmp))
        else:
            # meangm = dat[0, 1]
            meangm = dat[0, 1]
            if key not in coords:
                continue
            fuu.write('{:d},{:f},{:f},{:f}\n'.format(c, coords[key][0],
                                                     coords[key][1], meangm))
        #
        # Checking key
        if key not in coords:
            raise ValueError('missing coords: {:s}'.format(key))
        #
        # Writing files
        fou.write('{:d},{:f},{:f},{:f}\n'.format(c, coords[key][0],
                                                 coords[key][1], meangm))

        if coords[key][0] > 180 or coords[key][0] < -180:
            raise ValueError('out of bounds')
        buffer_array[c-1, :] = [coords[key][0], coords[key][1], meangm]

    bdf = pandas.DataFrame(buffer_array,
                           columns=['lon', 'lat', 'PGA-0.002107'])
    bdf['Coordinates'] = list(zip(bdf.lon, bdf.lat))
    bdf['Coordinates'] = bdf['Coordinates'].apply(Point)
    gbdf = gpd.GeoDataFrame(bdf, geometry='Coordinates')
    fname = os.path.join(out_path, 'map_buffer.json')
    gbdf.to_file(fname, driver='GeoJSON')
    fou.close()
    fuu.close()


def main(argv):
    #
    # Logging
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.ERROR)
    #
    # Settings
    argv.append('./data/gis/contacts_between_models.shp')
    argv.append('out')

    # argv.append('/Users/mpagani/Repos/mosaic/mosaic')
    argv.append('/Users/mpagani/Documents/2018/diary/11/13_ucf/maps')

    argv.append('./data/global_grid/trigrd_split_9_spacing_13')
    path = '/Users/mpagani/NC/Hazard_Charles/Data/Administrative/'
    # name = 'world_country_admin_boundary_shapefile_with_fips_codes_mosaic.shp'
    # name = 'world_country_admin_boundary_with_fips_codes_mosaic.shp'
    name = 'world_country_admin_boundary_with_fips_codes_mosaic_eu_russia.shp'
    argv.append(os.path.join(path, name))
    argv.append('out')
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
    process_maps(contacts_df, argv[1], argv[2], argv[3], argv[4], argv[5])


if __name__ == "__main__":
    # argv[0] - Shapefile with contacts between models
    # argv[1] - Output folder
    # argv[2] - Folder of the main repository
    # argv[3] - Path to the global grid spatial index
    main(sys.argv[1:])
