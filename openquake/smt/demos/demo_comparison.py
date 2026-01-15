"""
This is a demo script for comparing GMMs using the trellis plots, response
spectra, hierarchical clustering plots, Euclidean distance matrix plots and
Sammons Maps plotting functions available within the SMT's comparison module.
"""
import os
import shutil

from openquake.smt.comparison import compare_gmpes as comp
from openquake.smt.comparison.utils_gmpes import (reformat_att_curves,
                                                  reformat_spectra)

import warnings
warnings.filterwarnings("ignore")


BASE = os.path.abspath('')

# Inputs
demo_input = os.path.join(BASE, 'demo_input_files', 'demo_comparison.toml')

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
    spectra = comp.plot_spectra(file, out_dir)
    comp.plot_ratios(file, out_dir)
    comp.plot_cluster(file, out_dir)
    comp.plot_sammons(file, out_dir)
    comp.plot_matrix(file, out_dir)

    return att_curves, spectra


def main(input_toml=demo_input, out_dir=demo_out):
    """
    Run the demo GMM comparison workflow.
    """
    # Print that the analysis has started
    print("Starting GMM comparison analysis...")

    # Make a new directory for outputs
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    # Parse flatfile into metadata
    att_curves, spectra = run_comparison(input_toml, out_dir)

    # Reformat the att_curves dictionary into a csv
    ac_df = reformat_att_curves(
        att_curves, os.path.join(out_dir, 'attenuation_curves.csv'))

    # Reformat the spectra dictionary into a csv too
    rs_df = reformat_spectra(
        spectra, os.path.join(out_dir, 'spectra.csv'))

    # Print that the analysis has finished
    print("GMM comparison analysis has successfully finished")


if __name__ == '__main__':
    main()