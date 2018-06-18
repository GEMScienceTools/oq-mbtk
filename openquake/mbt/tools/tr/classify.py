#!/usr/bin/env python
# coding: utf-8

import os
import re
import sys
import h5py
import logging
import configparser

from openquake.baselib import sap
from openquake.mbt.tools.tr.catalogue import get_catalogue

from openquake.mbt.tools.tr.set_crustal_earthquakes import \
    SetCrustalEarthquakes
from openquake.mbt.tools.tr.set_subduction_earthquakes import \
    SetSubductionEarthquakes

logging.basicConfig(filename='classify.log', level=logging.DEBUG)


def str_to_list(tmps):
    return re.split('\,', re.sub('\s*', '', re.sub(r'\[|\]', '', tmps)))


def classify(ini_fname, compute_distances, rf):
    """
    :param str ini_fname:
        The path to the .ini file containing settings
    :param str rf:
        The root folder (all the path in the .ini file will use this as a
        reference
    :param bool compute_distances:
        A boolean controlling the calculation of distances between the slab
        surfaces and the earthquakes in the catalog
    """
    logger = logging.getLogger('classify')
    #
    #
    assert os.path.exists(ini_fname)
    #
    # Parse .ini file
    config = configparser.ConfigParser()
    config.read(ini_fname)
    #
    #
    if rf is False:
        assert 'root_folder' in config['general']
        rf = config['general']['root_folder']
    #
    # set root folder
    distance_folder = os.path.join(rf, config['general']['distance_folder'])
    catalogue_fname = os.path.join(rf, config['general']['catalogue_filename'])
    assert os.path.exists(catalogue_fname)
    #
    # Read priority list
    priorityl = str_to_list(config['general']['priority'])
    #
    # Tectonic regionalisation fname
    tmps = config['general']['treg_filename']
    treg_filename = os.path.join(rf, tmps)
    if not os.path.exists(treg_filename):
        logger.info('Creating: {:s}'.format(treg_filename))
        f = h5py.File(treg_filename, "w")
        f.close()
    else:
        logger.info('{:s} exists'.format(treg_filename))
    #
    # Log filename
    log_fname = 'log.hdf5'
    if os.path.exists(log_fname):
        os.remove(log_fname)
    logger.info('Creating: {:s}'.format(treg_filename))
    f = h5py.File(treg_filename)
    f.close()
    #
    # process the input information
    remove_from = []
    for key in priorityl:
        #
        # Set TR label
        if 'label' in config[key]:
            trlab = config[key]['label']
        else:
            trlab = key
        #
        # subduction earthquakes
        if re.search('^slab', key) or re.search('^int', key):
            #
            # Info
            logger.info('Classifying: {:s}'.format(key))
            #
            # Reading parameters
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
            #
            # Selecting earthquakes within a time period
            if 'low_year' in config[key]:
                low_year = float(config[key]['low_year'])
            else:
                low_year = -10000
            if 'upp_year' in config[key]:
                upp_year = float(config[key]['upp_year'])
            else:
                upp_year = 10000
            #
            # Selecting earthquakes within a magnitude range
            if 'low_mag' in config[key]:
                low_mag = float(config[key]['low_mag'])
            else:
                low_year = -5
            if 'upp_mag' in config[key]:
                upp_mag = float(config[key]['upp_mag'])
            else:
                upp_mag = 15
            #
            #
            sse = SetSubductionEarthquakes(trlab,
                                           treg_filename,
                                           distance_folder,
                                           edges_folder,
                                           distance_buffer_below,
                                           distance_buffer_above,
                                           lower_depth,
                                           catalogue_fname,
                                           log_fname,
                                           low_year,
                                           upp_year,
                                           low_mag,
                                           upp_mag)
            sse.classify(compute_distances, remove_from)
        #
        # crustal earthquakes
        elif re.search('^crustal', key) or re.search('^volcanic', key):
            #
            # info
            logger.info('Classifying: {:s}'.format(key))
            #
            # set data files
            tmps = config[key]['crust_filename']
            distance_delta = config[key]['distance_delta']
            #
            # set shapefile name
            shapefile = None
            if ('shapefile' in config[key]):
                shapefile = os.path.join(rf, config[key]['shapefile'])
                assert os.path.exists(shapefile)
            #
            # crust filename
            crust_filename = os.path.join(rf, tmps)
            #
            # classifying
            sce = SetCrustalEarthquakes(crust_filename,
                                        catalogue_fname,
                                        treg_filename,
                                        distance_delta,
                                        label=trlab,
                                        shapefile=shapefile,
                                        log_fname=log_fname)
            sce.classify(remove_from)
        #
        #
        else:
            raise ValueError('Undefined option')
        #
        # Updating the list of TR with lower priority
        if trlab not in remove_from:
            remove_from.append(trlab)
    #
    # reading filename
    c = get_catalogue(catalogue_fname)
    csvfname = 'classified_earthquakes.csv'
    fou = open(csvfname, 'w')
    f = h5py.File(treg_filename, 'r')
    fou.write('eventID,id,longitude,latitude,tr\n')
    for i, (eid, lo, la) in enumerate(zip(c.data['eventID'],
                                          c.data['longitude'],
                                          c.data['latitude'])):
        fnd = False
        for k in list(f.keys()):
            if f[k][i] and not fnd:
                fou.write('{:s},{:d},{:f},{:f},{:s}\n'.format(str(eid), i,
                                                              lo, la, k))
                fnd = True
        if not fnd:
            fou.write('{:s},{:d},{:f},{:f},{:s}\n'.format(eid, i, lo, la,
                                                          'unknown'))
    f.close()
    fou.close()


def main(argv):
    """
    This is the controlling script that can be used to subdivide an
    earthquake catalogue into many subsets, each one describing the seismicity
    generated by a specific tectonic region.

    The code supports the following classification criteria:
        - shallow crust
        - subduction (interface and inslab)

    The parameters required to complete the classification are specified in a
    .ini file (see https://en.wikipedia.org/wiki/INI_file for an explanation
    of the format) containing at least two sections.

    The first section - defined `general` - specifies global parameters such as
    the path to the catalogue filename (note that this one must follow the
    .hmtk format), the path to the .hdf5 file where the results will be stored,
    the folder where to store the files containing the computed distances
    between the subduction surfaces and the events included in the catalogue.
    A very important parameter in this section is the `priority` parameter.

    The second (and the following sections) specify the parameters and the
    data required to create tectonically uniform families of earthquakes.

    The selection of crustal earthquakes requires a set of points (described in
    terms of longitude, latitude and depth) describing the bottom surface of
    the crust. Ideally, this set of points should be homogenously distributed
    over a regular grid.

    The selection of subduction interface earthquakes is perfomed by selecting
    the ones located at less than a thershold distance [in km] from
    surface describing the top of the subducted slab - interface part.
    The threshold distance is defined in the `.ini` configuration file.

    The selection of subduction inslab earthquakes is perfomed in a manner
    similar to the one used for subduction interface seismicity.

    """
    p = sap.Script(classify)
    #
    # set arguments
    msg = 'Path to the configuration fname - typically a .ini file for tr'
    p.arg(name='ini_fname', help=msg)
    msg = 'Flag defining if the calculation of distances'
    p.flg(name='compute_distances', help=msg)
    msg = 'Root folder (path are relative to this in the .ini file)'
    p.opt(name='rf', help=msg)
    #
    # check command line arguments and run the code
    if len(argv) < 1:
        print(p.help())
    else:
        p.callfunc()


if __name__ == "__main__":
    main(sys.argv[1:])
