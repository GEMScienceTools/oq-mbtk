#!/usr/bin/env python3

import os
import pandas as pd

from openquake.baselib import sap
from openquake.mbi.cat.create_csv import create_folder
from openquake.cat.hmg.plot import plot_histogram
from openquake.cat.hmg.map import plot_catalogue, write_gmt_file


def main(cat_fname: str, *, out_folder: str = './figs', mmin: float = 4.0):

    data = os.environ['GEM_DATA']

    # Read catalogue
    cat = pd.read_hdf(cat_fname)

    #
    # Create output folder
    create_folder(out_folder)

    #
    # Histogram of the homogenised catalogue
    fname = os.path.join(out_folder, 'hom_cat_histogram.pdf')
    fig, ax = plot_histogram(cat, xlim=[0, 9], ylim=[1e0, 1e5], fname=fname)

    #
    # Histogram of the homogenised catalogue - Magnitude larger than 4.5
    fname = os.path.join(out_folder, 'hom_cat_histogram_mmin_4pt5.pdf')
    fig, ax = plot_histogram(cat, xlim=[3.5, 9], ylim=[1e0, 1e5],
                             fname=fname, min_mag=4.5)

    #
    # Map with the homogenised catalogue
    fname = os.path.join(out_folder, './tmp/tmp.txt')
    topo = os.path.join(data, 'dem', 'globalGTOPO30.grd')

    fname_gmt = '/tmp/gmt.txt'
    write_gmt_file(cat, fname_gmt=fname_gmt, mmin=4.0)
    fname = os.path.join(out_folder, 'hom_cat_map')
    fname = plot_catalogue(fname_gmt, fformat="pdf",
                           fname_fig=fname, topography=topo,
                           title='Homogenised Catalog')


main.cat_fname = '.h5 file with origins'
main.out_folder = 'The folder where to store the figures'
main.mmin = 'Minimum magnitude for adding earthquakes to the map'

if __name__ == "__main__":
    """
    The function figs creates the figures describing basic data sets and
    homogenised catalog.
    """
    sap.run(main)
