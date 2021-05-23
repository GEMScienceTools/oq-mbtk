#!/usr/bin/env python
# coding: utf-8

import os
import h5py
import pickle
import numpy as np

from openquake.baselib import sap

from openquake.hmtk.seismicity.selector import CatalogueSelector
from openquake.hmtk.parsers.catalogue.csv_catalogue_parser import (
    CsvCatalogueWriter)


def get_treg(treg_fname):
    """
    Gets the labels defining the different tectonic regions.

    :param treg_filename:
        Name of the .hdf5 file with the classification of the catalogue
    :return:
        A list with the labels and the number of earthquakes in the catalogue
    """

    # List with the labels of the varioous tectonic regions
    aaa = []

    # Load TR
    if treg_fname is not None:
        f = h5py.File(treg_fname, 'r')
        for key in f.keys():
            aaa.append(key)
            alen = len(f[key])
            print(key)
        f.close()
    return aaa, alen


def create_sub_catalogue(alen, aaa, pck_fname, treg_fname, out_cata, out_path):
    """
    Creates .csv files with the subcatalogues

    :param alen:
        Number of earthquakes in the original catalogue
    :param aaa:
        List of the labels used to define the various tectonic regions
    :param pck_fname:
        Name of the file with the pickled catalogue
    :param treg_fname:
        Name of the .hdf5 file with the classification of the catalogue
    :param out_cata:
        Name of the .hdf5 file with the classification of the catalogue
    :param out_path:
        Name of the .hdf5 file with the classification of the catalogue
    :returns:
        A :class:`numpy.ndarray` vector of length N where N is the number of
        earthquakes in the original catalogue.
    """

    # The output vector
    tot_lab = np.zeros(alen)

    print(' ')
    fmt = '# earthquakes in the catalogue: {:d}'
    print(fmt.format(len(tot_lab)))

    # Loop over the tectonic regions
    for label in (aaa):

        # Output file name
        csv_filename = out_cata + "_TR_{:s}.csv".format(label)
        csv_filename = os.path.join(out_path, csv_filename)

        # Read the TR classification
        f = h5py.File(treg_fname, 'r')
        tr = f[label][:]
        f.close()

        if sum(tr) > 0:
            tmp_lab = tr*1
            tot_lab = tot_lab+tmp_lab
            catalogue = pickle.load(open(pck_fname, 'rb'))
            for lab in ['month', 'day', 'hour', 'minute', 'second']:
                idx = np.isnan(catalogue.data[lab])
                if lab == 'day' or lab == 'month':
                    catalogue.data[lab][idx] = 1
                elif lab == 'second':
                    catalogue.data[lab][idx] = 0.0
                else:
                    catalogue.data[lab][idx] = 0
            selector = CatalogueSelector(catalogue, create_copy=False)
            catalogue = selector.select_catalogue(tr)
            catalogue.data['hour'] = catalogue.data['hour'].astype(int)
            catalogue.data['minute'] = catalogue.data['minute'].astype(int)

            print(' ')
            fmt = '# earthquakes in this TR      : {:d}'
            print(fmt.format(len(catalogue.data['longitude'])))

            # Sub-catalogue
            print(csv_filename)
            csvcat = CsvCatalogueWriter(csv_filename)

            # Write the purged catalogue
            csvcat.write_file(catalogue)
            print("Catalogue successfully written to %s" % csv_filename)

    return tot_lab


def get_unclassified(tot_lab, pck_fname, out_cata, path_out):
    """
    Create a text file (.csv formatted) with the unclassified earthquakes

    :param tot_lab:

    :param pck_fname:
        Name of the pickle file with the catalogue
    :param out_cata:
        Name of the .csv output catalogue
    :param path_out:
        Path to output folder
    """

    # ID of the unclassified earthquakes
    tr_undef = abs(tot_lab-1)

    # Load the catalogue of unclassified earthquakes
    catalogue = pickle.load(open(pck_fname, 'rb'))

    # Select the unclassified
    selector = CatalogueSelector(catalogue, create_copy=False)
    catalogue = selector.select_catalogue(tr_undef)
    print('')
    print('# earthquakes: {:d}'.format(len(catalogue.data['longitude'])))

    # Sub-catalogue
    csv_filename = out_cata + "_TR_unclassified.csv"
    csv_filename = os.path.join(path_out, csv_filename)

    # Write the purged catalogue
    csvcat = CsvCatalogueWriter(csv_filename)
    csvcat.write_file(catalogue)
    print("Catalogue successfully written to %s" % csv_filename)


def create_sub_cata(pck_fname, treg_fname, *, out_cata='cat', path_out='.'):
    """
    Create a subcatalogue
    """
    aaa, alen = get_treg(treg_fname)
    tot_lab = create_sub_catalogue(alen, aaa, pck_fname, treg_fname, out_cata,
                                   path_out)
    get_unclassified(tot_lab, pck_fname, out_cata, path_out)


create_sub_cata.pck_fname = 'Name of the pickle file with the catalogue'
msg = 'Name of the .hdf5 file with the catalogue classification'
create_sub_cata.treg_fname = msg
create_sub_cata.out_cata = 'Prefix in for the output files [cat]'
create_sub_cata.path_out = 'Output path [./]'

if __name__ == "__main__":
    sap.run(create_sub_cata)
