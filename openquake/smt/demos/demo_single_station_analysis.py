"""
This demo script runs a single stationresidual analysis using a subset of the
ESM flatfile filtered geographically to Albania + an approximately 100 km buffer.
"""
import os
import numpy as np
import pandas as pd
import pickle
import shutil
import toml

from openquake.baselib import sap
from openquake.smt.parsers.esm_url_flatfile_parser import ESMFlatfileParserURL
import openquake.smt.residuals.gmpe_residuals as res
import openquake.smt.residuals.residual_plotter as rspl
from openquake.smt.strong_motion_selector import rank_sites_by_record_count

import warnings
warnings.filterwarnings("ignore")

"""USER INPUTS"""

# Flatfile to use 
db = 'demo_flatfile.csv'

# Specify .toml file with GMPEs and imts to use
gmms_imts = 'demo_residual_analysis_inputs.toml'

# Specify results folder name
run_folder = 'results_single_station_analysis'

# Minimum number of records for a site to be considered in SSA
threshold = 45


def get_residuals():
    """
    Compute the residuals from the example flatfile, GMMs and imts
    """
    # Create metadata directory
    metadata_dir = 'metadata'
    if os.path.exists(metadata_dir):
        shutil.rmtree(metadata_dir)
            
    # Parse the metadata
    ESMFlatfileParserURL.autobuild("000", 'db', metadata_dir, db)
    
    # Get inputs
    metadata = os.path.join('metadata', 'metadatafile.pkl')
    sm_database = pickle.load(open(metadata,"rb")) 
    
    # If output directory for residuals exists remove and remake 
    if os.path.exists('residuals'):
        shutil.rmtree('residuals')
    
    # Get residuals
    residuals = res.Residuals.from_toml(gmms_imts)
    residuals.get_residuals(sm_database)
    
    # Create results folder for single station analysis
    if os.path.exists(run_folder):
       shutil.rmtree(run_folder)
    os.mkdir(run_folder)

    return sm_database


def single_station_analysis(sm_database):
    """
    Perform the analysis using the demo files
    """
    # Find sites with threshold minimum for number of records
    top_sites = rank_sites_by_record_count(sm_database, threshold)
    
    # For each station print some info
    msg = 'Sites with required threshold of at least %s records' %(threshold)
    print(msg)
    for idx, site_id in enumerate(top_sites.keys()):
        print(" Site ID: %s Name: %s, Number of Records: %s" %(
            site_id, top_sites[site_id]["Name"], top_sites [site_id]["Count"]))

    # Create SingleStationAnalysis object
    ssa1 = res.SingleStationAnalysis.from_toml(top_sites.keys(), gmms_imts)
    
    # Compute the total, inter-event and intra-event residuals for each site
    ssa1.get_site_residuals(sm_database)
    
    # Output for summary csv
    csv_output = os.path.join(run_folder, 'ssa_results.csv')
    
    # Get summary of statistics and output them
    ssa1.residual_statistics(True, csv_output)
    
    # Output plots
    for gmpe in ssa1.gmpe_list:
        for imt in ssa1.imts:
            
            # Create folder for the GMM
            gmpe_folder_name = str(ssa1.gmpe_list[gmpe]).split(
                '(')[0].replace('\n','_').replace(' ','').replace('"','')
        
            gmpe_folder = os.path.join(run_folder, gmpe_folder_name)
            
            if not os.path.exists(gmpe_folder):
                os.mkdir(gmpe_folder)
                    
            # Filenames
            output_all_res_with_site = os.path.join(
                gmpe_folder, gmpe_folder_name + '_' + imt + '_' +
                'AllResPerSite.jpg') 
            output_intra_res_components_with_site = os.path.join(
                gmpe_folder, gmpe_folder_name + '_' + imt + '_' +
                'IntraResCompPerSite.jpg') 
            
            # Create the plots and save
            rspl.ResidualWithSite(ssa1, gmpe, imt,
                                  output_all_res_with_site, filetype='jpg')
            rspl.IntraEventResidualWithSite(
                ssa1, gmpe, imt, output_intra_res_components_with_site, 
                iletype='jpg')


def main():
    """
    Run the demo for single station residual analysis
    """
    # Get residuals
    sm_database = get_residuals()
    
    # Run the single station analysis
    single_station_analysis(sm_database)
    
    
if __name__ == '__main__':
    sap.run(main)