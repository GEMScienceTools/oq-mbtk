#!/usr/bin/env python

import h3
import toml
import pandas as pd
import matplotlib.pyplot as plt
import openquake.mbt.tools.adaptive_smoothing as ak

from openquake.baselib import sap
from openquake.hmtk.parsers.catalogue import CsvCatalogueParser


def main(catalogue, h3_map, config, outputfile, plot=False):
    '''
    Runs an analysis of adaptive smoothed seismicity of Helmstetter et
    al. (2007).

    :param catalogue:
        An earthquake catalogue csv file containing the following columns -
        'longitude' - numpy.ndarray vector of longitudes
        'latitude' - numpy.ndarray vector of latitudes

    :param dict config:
        Location of toml file with model configuration. The following
        settings are necessary:
        * 'kernel' - Kernel choice for adaptive smoothing. Options are
            "Gaussian" or "PowerLaw" (string)
        * 'n_v' - number of nearest neighbour to use for smoothing
            distance (int). Use the Information Gain to calibrate this.
        * 'd_i_min' - minimum smoothing distance d_i, should be chosen
            based on location uncertainty. Default of 0.5 in Helmstetter
            et al. (float)

    :param output_file:
        String specifying location in which to save output.

    :param plot:
        Option to plot result to see smoothing

    :returns:
        Full smoothed seismicity data as np.ndarray, of the form
        [Longitude, Latitude, Smoothed], a plot if required.
    '''

    # Read h3 indices from mapping file
    h3_idx = pd.read_csv(h3_map)

    # Get lat/lon locations for each h3 cell, convert to seperate lat and
    # lon columns of dataframe
    h3_idx['latlon'] = h3_idx.iloc[:, 0].apply(h3.h3_to_geo)
    locations = pd.DataFrame(h3_idx['latlon'].tolist())
    locations.columns = ["lat", "lon"]

    # Load config file to get smoothing parameters
    config = toml.load(config)

    # Load catalogue csv. Uses CsvCatalogueParser for compatability with
    # hmtk catalogue outputs. Should work with any csv so long as
    # 'longitude' and 'latitude' cols are present
    cat = catalogue
    parser = CsvCatalogueParser(cat)
    cat = parser.read_file()
    cat.sort_catalogue_chronologically()

    # Run adaptive smoothing over chosen area, don't grid the data (h3
    # locs, already done!), don't use depths.
    smooth = ak.AdaptiveSmoothing([locations.lon, locations.lat],
                                  grid=False, use_3d=False)
    conf = {"kernel": config["kernel"], "n_v": config['n_v'],
            "d_i_min": config['d_i_min']}
    out = smooth.run_adaptive_smooth(cat, conf)
    # Make output into dataframe with named columns and write to a csv
    # file in specified loctaion
    out = pd.DataFrame(out)
    out.columns = ["lon", "lat", "nocc"]

    out["nocc"] = out["nocc"]
    out.to_csv(outputfile, header=True)

    if plot is True:
        plot_adaptive_smoothing(outputfile, cat)


def plot_adaptive_smoothing(smoothingfile, mor_cat):
    out = pd.read_csv(smoothingfile)
    plt.scatter(out["lon"], out["lat"], c=out["nocc"], cmap="viridis")
    plt.colorbar(label="event density")
    plt.scatter(mor_cat.data['longitude'],
                mor_cat.data['latitude'], s=0.5, c="k")
    plt.xlabel("longitude")
    plt.ylabel("latitude")
    plt.show()


descr = 'Instance of the openquake.hmtk.seismicity.catalogue.Catalogue class'
main.catalogue = descr
descr = 'h3 cells in which to calculate rate'
main.h3_locations = descr
descr = 'Config file defining the parameters for smoothing'
main.config = descr
descr = 'Name of file to save output to'
main.outputfile = descr

if __name__ == '__main__':
    sap.run(main)
