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
import logging
import configparser
import h5py
import pathlib
from decimal import Decimal, getcontext

from openquake.baselib import sap
from openquake.mbt.tools.tr.catalogue import get_catalogue

from openquake.mbt.tools.tr.set_crustal_earthquakes import \
    SetCrustalEarthquakes
from openquake.mbt.tools.tr.set_subduction_earthquakes import \
    SetSubductionEarthquakes

logging.basicConfig(filename='classify.log', level=logging.DEBUG)
getcontext().prec = 6 

def str_to_list(tmps):
    return re.split(',', re.sub('\\s*', '', re.sub('\\[|\\]', '', tmps)))


def classify(ini_fname, compute_distances, rf=''):
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
    assert os.path.exists(ini_fname)

    # Parse .ini file
    config = configparser.ConfigParser()
    config.read(ini_fname)

    # Set the root folder
    if rf is False or len(rf) < 1:
        assert 'root_folder' in config['general']
        rf = config['general']['root_folder']
    rf = os.path.normpath(rf)

    # Set distance folder
    distance_folder = os.path.join(rf, config['general']['distance_folder'])
    distance_folder = os.path.normpath(distance_folder)
    pth = pathlib.Path(distance_folder)
    pth.mkdir(parents=True, exist_ok=True)

    catalogue_fname = config['general']['catalogue_filename']
    if not re.search('^\\/', catalogue_fname):
        catalogue_fname = os.path.join(rf, catalogue_fname)
    catalogue_fname = os.path.normpath(catalogue_fname)
    msg = f'The file {catalogue_fname} does not exist'
    assert os.path.exists(catalogue_fname), msg

    # Read priority list
    priorityl = str_to_list(config['general']['priority'])

    # Tectonic regionalisation fname
    tmps = config['general']['treg_filename']
    treg_filename = os.path.join(rf, tmps)
    treg_filename = os.path.normpath(treg_filename)
    if not os.path.exists(treg_filename):
        # Create folder
        pth = pathlib.Path(treg_filename)
        dir_name = pth.parents[0]
        dir_name.mkdir(parents=True, exist_ok=True)
        # Create file
        logger.info(f'Creating: {treg_filename:s}')
        fle = h5py.File(treg_filename, "w")
        fle.close()
    else:
        logger.info(f'{treg_filename:s} exists')

    # Log filename
    log_fname = 'log.hdf5'
    if os.path.exists(log_fname):
        os.remove(log_fname)
    logger.info('Creating: {:s}'.format(log_fname))
    f = h5py.File(log_fname, 'w')
    f.close()

    # Process the input information
    remove_from = []
    for key in priorityl:

        # Set TR label
        if 'label' in config[key]:
            trlab = config[key]['label']
        else:
            trlab = key

        # Subduction earthquakes
        if re.search('^slab', key) or re.search('^int', key):

            # Info
            logger.info('Classifying {:s} events'.format(key))

            # Reading parameters
            edges_folder = os.path.join(rf, config[key]['folder'])
            distance_buffer_below = None
            if 'distance_buffer_below' in config[key]:
                tmps = config[key]['distance_buffer_below']
                distance_buffer_below = Decimal(tmps)
            distance_buffer_above = None
            if 'distance_buffer_above' in config[key]:
                tmps = config[key]['distance_buffer_above']
                distance_buffer_above = Decimal(tmps)
            lower_depth = None
            if 'lower_depth' in config[key]:
                lower_depth = Decimal(config[key]['lower_depth'])

            # Selecting earthquakes within a time period
            low_year = -10000
            if 'low_year' in config[key]:
                low_year = Decimal(config[key]['low_year'])
            upp_year = 10000
            if 'upp_year' in config[key]:
                upp_year = Decimal(config[key]['upp_year'])

            # Selecting earthquakes within a magnitude range
            low_mag = -5
            if 'low_mag' in config[key]:
                low_mag = Decimal(config[key]['low_mag'])
            upp_mag = 15
            if 'upp_mag' in config[key]:
                upp_mag = Decimal(config[key]['upp_mag'])

            # specifying surface type
            surftype = 'ComplexFault'
            if 'surface_type' in config[key]:
                surftype = config[key]['surface_type']

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
            sse.classify(compute_distances, remove_from, surftype)

        # crustal earthquakes
        elif re.search('^crustal', key) or re.search('^volcanic', key):

            # Set data files
            tmps = config[key]['crust_filename']
            if not re.search('^\\/', tmps):
                tmps = os.path.join(rf, tmps)
            distance_delta = config[key]['distance_delta']

            # Info
            logger.info(f'Classifying {key:s} events')
            logger.info(f'Reading file {tmps}')
            logger.info(f'Distance delta {distance_delta} [km]')

            # Set shapefile name
            shapefile = None
            if 'shapefile' in config[key]:
                shapefile = os.path.join(rf, config[key]['shapefile'])
                assert os.path.exists(shapefile)

            # Classify
            sce = SetCrustalEarthquakes(tmps,
                                        catalogue_fname,
                                        treg_filename,
                                        distance_delta,
                                        label=trlab,
                                        shapefile=shapefile,
                                        log_fname=log_fname)
            sce.classify(remove_from)

        else:
            raise ValueError('Undefined option')

        # Updating the list of TR with lower priority
        if trlab not in remove_from:
            remove_from.append(trlab)

    # reading filename
    c = get_catalogue(catalogue_fname)
    csvfname = os.path.join(rf, 'classified_earthquakes.csv')
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


def main(ini_fname, compute_distances, *, root_folder=''):
    classify(ini_fname, compute_distances, root_folder)


msg = 'Path to the configuration fname - typically a .ini file for tr'
classify.ini_fname = msg
msg = 'Root folder (path are relative to this in the .ini file)'
classify.root_folder = msg
msg = 'Flag defining if the calculation of distances'
classify.compute_distances = msg

if __name__ == "__main__":
    sap.run(main)
