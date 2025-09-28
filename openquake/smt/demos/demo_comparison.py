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

# Inputs
demo_input = os.path.join(
    BASE, 'demo_input_files', 'demo_comparison_analysis_inputs.toml')

# Out dir
demo_out = os.path.join(BASE, 'outputs_demo_comparison')


def run_comparison(file, out_dir):
    """
    Run the GMM comparison the provided toml and return a dictionary
    storing the median attenuation curves.
    
    The att_curves variable stores the attenuation curves for each
    imt-mag-depth combo per gmpe for the given input scenario.
    
    The user can examine the keys of the att_curves variable to
    better understand the additional information stored within.
    """
    # Generate plots and retrieve attenuation curve data from config object
    att_curves = comp.plot_trellis(file, out_dir)
    comp.plot_spectra(file, out_dir)
    comp.plot_ratios(file, out_dir)
    comp.plot_cluster(file, out_dir)
    comp.plot_sammons(file, out_dir)
    comp.plot_euclidean(file, out_dir)

    return att_curves


def reformat_att_curves(att_curves, out_dir):
    """
    Export the (median) attenuation curves into a CSV for the given
    config (i.e. run parameters).
    """
    # Get the key describing the vs30 + truncation level
    params_key = pd.Series(att_curves.keys()).values[0]

    # Then get the values per gmm (per imt-mag combination)
    vals = att_curves[params_key]['gmm att curves per imt-mag']

    # Now get the curves into a dictionary format
    store = {}
    for imt in vals.keys():
        for scenario in vals[imt]:
            medians = vals[imt][scenario]
            for gmpe in medians: 
                # First per GMM
                if "(km)" not in gmpe:
                    key = imt + ', ' +  scenario + ', ' + gmpe
                    key = key.replace('\n', ' ')
                    gmpe_medians = medians[gmpe]['median (g)']
                    store[key] = gmpe_medians
                # Then get the distance for given scenario
                else:
                    dkey = f"values of {gmpe} for {scenario}"
                    if dkey not in store: # Skip if already in dict
                        store[dkey] = medians[gmpe]
                    
    # Now into dataframe
    df = pd.DataFrame(store)

    # And export
    fname = os.path.join(out_dir, 'attenuation_curves.csv')
    df.to_csv(fname)

    return df # Might want to build on this so return the df...


def main(input_toml=demo_input, out_dir=demo_out):
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
    att_curves = run_comparison(input_toml, out_dir)

    # Reformat the att_curves dictionary into a csv
    df = reformat_att_curves(att_curves, out_dir)

    # Print that the analysis has finished
    print("GMM comparison analysis has successfully finished")


if __name__ == '__main__':
    sap.run(main)    