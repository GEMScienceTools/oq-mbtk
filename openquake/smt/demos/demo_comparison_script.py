"""
This notebook provides a simple and efficient script for comparing candidate 
GMPEs in terms of median predicted ground-motion using trellis plots,
hierarchical clustering plots, Euclidean distance matrix plots and Sammons Maps
plotting functions available within the SMT's comparison module.
"""
import os
import toml
from openquake.smt.comparison import compare_gmpes as comp


# User input (can add more input tomls to run multiple analyses if required)
file_list = ['demo_comparison_analysis_inputs.toml']

for file in file_list:
    
    filename = file

    config_file = toml.load(filename)
    name_analysis = config_file['general']['name_analysis'] 
    output_directory = os.path.join(os.path.abspath(''),name_analysis)

    # set the output
    if not os.path.exists(output_directory): os.makedirs(output_directory)
    
    #Generate plots from config object
    comp.plot_trellis(filename,output_directory)
    comp.plot_spectra(filename,output_directory)
    comp.plot_cluster(filename,output_directory)
    comp.plot_sammons(filename,output_directory)
    comp.plot_euclidean(filename,output_directory)