#!/usr/bin/env python

import h3
import toml
import pandas as pd
import matplotlib.pyplot as plt
import openquake.mbt.tools.adaptive_smoothing as ak

from openquake.baselib import sap
from openquake.hmtk.parsers.catalogue import CsvCatalogueParser
from openquake.wkf.utils import get_list


def main(catalogue:str, h3_map: str, config:str, outputfile:str,  use: str = []):
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
        * 'h3res' - h3 resolution for the model
        * 'maxdist' - maximum distance to consider a neighbour
        

    :param output_file:
        String specifying location in which to save output.

    :param use:
        Option to specify zone IDs to use in model

    :returns:
        Full smoothed seismicity data as np.ndarray, of the form
        [Longitude, Latitude, Smoothed], a plot if required.
    '''

    # Read h3 indices from mapping file
    #h3_idx = pd.read_csv(h3_map)
    h3_idx = pd.read_csv(h3_map, names = ("h3", "id"))
    
    if len(use) > 0:
        # Note that this should still consider events outside this zone, 
        # it will simply only return the adaptively smoothed values at 
        # locations within the zone (should be consistent with adjacent zones)
        l1 = use
        use = get_list(use)
        use = map(int, use)
        h3_idx = h3_idx[h3_idx['id'].isin(use)]
        print("Using zones ", l1)
    
    # Get lat/lon locations for each h3 cell, convert to seperate lat and
    # lon columns of dataframe
    h3_idx['latlon'] = h3_idx.loc[:,"h3"].apply(h3.h3_to_geo)
    locations = pd.DataFrame(h3_idx['latlon'].tolist())
    locations.columns = ["lat", "lon"]

    # Load config file to get smoothing parameters
    config = toml.load(config)
    config = config['smoothing']

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
                                  grid=False, use_3d=False, use_maxdist = True)
    conf = {"kernel": config['kernel'], "n_v": config['n_v'],
            "d_i_min": config['d_i_min'], "h3res": config['h3res'], 
            "maxdist": config['maxdist']}
    out = smooth.run_adaptive_smooth(cat, conf)
    # Make output into dataframe with named columns and write to a csv
    # file in specified loctaion
    out = pd.DataFrame(out)
    out.columns = ["lon", "lat", "nocc"]

    out["nocc"] = out["nocc"]
    out.to_csv(outputfile, header=True)


descr = 'Instance of the openquake.hmtk.seismicity.catalogue.Catalogue class'
main.catalogue = descr
descr = 'h3 cells in which to calculate rate'
main.h3_locations = descr
descr = 'Config file defining the parameters for smoothing'
main.config = descr
descr = 'Name of file to save output to'
main.outputfile = descr
descr = "Source IDs to use"
main.use = descr

if __name__ == '__main__':
    sap.run(main)
