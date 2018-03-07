#!/usr/bin/env python
# coding: utf-8
import code
import os
import re
import sys
import h5py
import configparser

from openquake.baselib import sap

from oqmbt.tools.tr.set_crustal_earthquakes import SetCrustalEarthquakes
from oqmbt.tools.tr.set_subduction_earthquakes import SetSubductionEarthquakes


def str_to_list(tmps):
    return re.split('\,', re.sub('\s*', '', re.sub(r'\[|\]', '', tmps)))


def classify(ini_fname, rf, compute_distances):
    #
    # Parse .ini file
    config = configparser.ConfigParser()
    config.read(ini_fname)
    #
    # set root folder
    distance_folder = os.path.join(rf, config['general']['distance_folder'])
    catalogue_fname = os.path.join(rf, config['general']['catalogue_filename'])
    #
    # read priority list
    priorityl = str_to_list(config['general']['priority'])
    print(priorityl)
    #
    # tectonic regionalisation fname
    tmps = config['general']['treg_filename']
    treg_filename = os.path.join(rf, tmps)
    if not os.path.exists(treg_filename):
        f = h5py.File(treg_filename, "w")
        f.close()
    #
    # process the input information
    remove_from = []
    for key in priorityl:
        print('--->', key)
        #
        # subduction earthquakes
        if re.search('^slab', key) or re.search('^int', key):
            print('Classifying: {:s}'.format(key))
            edges_folder = os.path.join(rf, config[key]['folder'])
            distance_buffer_below = None
            if 'distance_buffer_below' in config[key]:
                tmps = config[key]['distance_buffer_below']
                distance_buffer_below = float(tmps)
            distance_buffer_above = None
            if 'distance_buffer_above' in config[key]:
                tmps = config[key]['distance_buffer_above']
                distance_buffer_above = float(tmps)
            lower_depth = None
            if 'lower_depth' in config[key]:
                lower_depth = float(config[key]['lower_depth'])
            #code.interact(local=locals())
            sse = SetSubductionEarthquakes(key,
                                           treg_filename,
                                           distance_folder,
                                           edges_folder,
                                           distance_buffer_below,
                                           distance_buffer_above,
                                           lower_depth,
                                           catalogue_fname)
            sse.classify(compute_distances, remove_from)
        #
        # crustal earthquakes
        elif re.search('^crustal', key):
            print('Classifying: {:s}'.format(key))
            tmps = config['crustal']['crust_filename']
            distance_delta = config['crustal']['distance_delta']
            crust_filename = os.path.join(rf, tmps)
            sce = SetCrustalEarthquakes(crust_filename,
                                        catalogue_fname,
                                        treg_filename,
                                        distance_delta,
                                        label='crustal')
            sce.classify(remove_from)
        #
        #
        else:
            raise ValueError('Undefined option')
        #
        #
        remove_from.append(key)


def main(argv):
    """
    """
    p = sap.Script(classify)
    p.arg(name='ini_fname', help='Configuration fname')
    p.arg(name='rf', help='Root folder (path are relative to this in the .ini')
    tmps = 'Flag for the calculation of distances'
    p.flg(name='compute_distances', help=tmps)
    if len(argv) < 1:
        print(p.help())
    else:
        p.callfunc()


if __name__ == "__main__":
    main(sys.argv[1:])
