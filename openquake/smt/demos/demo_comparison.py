"""
This is a demo script for comparing GMMs using the trellis plots, response
spectra, hierarchical clustering plots, Euclidean distance matrix plots and
Sammons Maps plotting functions available within the SMT's comparison module.
"""
import os
import shutil
import toml
import pandas as pd

from openquake.baselib import sap
from openquake.smt.comparison import compare_gmpes as comp

import warnings
warnings.filterwarnings("ignore")


BASE = os.path.abspath('')

# User input (can add more input tomls to run multiple analyses if required)
comparison_params = os.path.join(
    BASE, 'demo_input_files', 'demo_comparison_analysis_inputs.toml')

# Out dir
out_dir = os.path.join(BASE, 'outputs_demo_comparison')


def run_comparison(file):
    """
    Run the GMM comparison for the list of input tomls and return a dictionary
    storing the attenuation curves.
    
    The att_curves variable stores the attenuation curves for each
    imt-mag-depth combo per gmpe, per a config file (within which can be specified
    additionally the eq source properties, vs30 and other params). This allows the
    user to extract the predicted ground-motions and manipulate as they wish. Use
    the keys of this variable to understand how the predictions are stored.
    """
    # Create a config object
    config = toml.load(file)

    # Generate plots and retrieve attenuation curve data from config object
    att_curves = comp.plot_trellis(file, out_dir)
    comp.plot_spectra(file, out_dir)
    comp.plot_ratios(file, out_dir)
    comp.plot_cluster(file, out_dir)
    comp.plot_sammons(file, out_dir)
    comp.plot_euclidean(file, out_dir)

    return att_curves, config


def reformat_curves(att_curves, config, out_dir):
    """
    Export the (median) hazard curves into a CSV for the given
    config (i.e. run parameters).
    """
    # Get the distance type used
    R = config['general']['dist_type']+' (km)'

    # Get the key describing the vs30 + truncation level
    params_key = pd.Series(att_curves.keys()).values[0]

    # Then get the values per gmm (per imt-mag combination)
    vals = att_curves[params_key]['gmm att curves per imt-mag']

    # Set a store which we will turn into the dataframe
    store = {}

    # Get the distance values (same across the GMMs per run)
    store[R] = att_curves[params_key]['gmm att curves per imt-mag'][R]

    # Now get the curves into a dictionary format
    for imt in vals.keys():
        if imt == R:
            continue
        for scenario in vals[imt]:
            medians = vals[imt][scenario]
            for gmpe in medians:
                key = imt + ', ' +  scenario + ', ' + gmpe
                key = key.replace('\n', '')
                gmpe_medians = medians[gmpe]['median (g)']
                store[key] = gmpe_medians

    # Now into dataframe
    df = pd.DataFrame(store)

    # And export
    out_hc = os.path.join(out_dir, 'attenuation_curves.csv')
    df.to_csv(out_hc)

    return df # Might want to build on this so return the df...


def main():
    """
    Run the demo GMM comparison workflow
    """
    # Print that the analysis has started
    print("Starting GMM comparison analysis...")

    # Make a new directory for outputs
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    # Parse flatfile into metadata
    att_curves, config = run_comparison(comparison_params)

    # Reformat the att_curves dictionary into a csv
    df = reformat_curves(att_curves, config, out_dir)

    # Print that the analysis has finished
    print("GMM comparison analysis has successfully finished")


if __name__ == '__main__':
    sap.run(main)    