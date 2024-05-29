"""
This notebook provides a simple script for comparing candidate GMMs using
trellis plots, hierarchical clustering plots, Euclidean distance matrix plots
and Sammons Maps plotting functions available within the SMT's comparison module.
"""
import os
import toml
from openquake.smt.comparison import compare_gmpes as comp

import warnings
warnings.filterwarnings("ignore")

# User input (can add more input tomls to run multiple analyses if required)
file_list = ['demo_comparison_analysis_inputs.toml']

attenuation_curve_data = {}
for file in file_list:
    
    filename = file

    config_file = toml.load(filename)
    name_analysis = config_file['general']['name_analysis'] 
    output_directory = os.path.join(os.path.abspath(''),name_analysis)

    # set the output
    if not os.path.exists(output_directory): os.makedirs(output_directory)
    
    #Generate plots from config object
    attenuation_curve_data[file] = comp.plot_trellis(filename,output_directory)
    comp.plot_spectra(filename,output_directory)
    comp.plot_cluster(filename,output_directory)
    comp.plot_sammons(filename,output_directory)
    comp.plot_euclidean(filename,output_directory)
    
# The attenuation_curve_data variable stores the attenuation curves for each
# imt-mag-depth combo per gmpe, per a config file (within which can be specified
# additionally the eq source properties, vs30 and other params). This allows the
# user to extract the predicted ground-motions and manipulate as they wish. Use
# the keys of this variable to understand how the predictions are stored.