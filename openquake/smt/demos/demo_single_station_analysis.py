"""
This demo script runs a single station residual analysis using
a subset of the ESM flatfile.
"""
import os
import pickle
import shutil

from openquake.baselib import sap
from openquake.smt.residuals.parsers.esm_url_flatfile_parser import ESMFlatfileParserURL
import openquake.smt.residuals.gmpe_residuals as res
import openquake.smt.residuals.residual_plotter as rspl

import warnings
warnings.filterwarnings("ignore")


BASE = os.path.abspath('')


"""USER INPUTS"""

# Flatfile to use 
demo_flatfile = os.path.join(BASE, 'demo_input_files', 'demo_flatfile.csv')

# Specify .toml file with GMPEs and imts to use
demo_inputs = os.path.join(BASE, 'demo_input_files', 'demo_residual_analysis_inputs.toml')

# Specify results folder name
demo_out = os.path.join(BASE, 'outputs_demo_station_analysis')

# Minimum number of records for a site to be considered in the SSA
demo_threshold = 10


def make_database(flatfile, out_dir):
    """
    Parse the flatfile to construct an SMT database.
    """
    # Create metadata directory
    metadata_dir = os.path.join(out_dir, 'metadata')
            
    # Parse the metadata
    ESMFlatfileParserURL.autobuild("000", 'db', metadata_dir, flatfile)
    
    # Get inputs
    metadata = os.path.join(metadata_dir, 'metadatafile.pkl')
    sm_database = pickle.load(open(metadata,"rb")) 

    return sm_database


def single_station_analysis(sm_database, gmms_imts, out_dir, threshold):
    """
    Perform the analysis using the demo files
    """
    # Print that workflow has begund
    print("Single station residual analysis workflow has begun...")

    # Find sites with threshold minimum for number of records
    top_sites = sm_database.rank_sites_by_record_count(threshold)
    
    # For each station print some info
    msg = 'Sites with required threshold of at least %s records:' %(threshold)
    print(msg)
    for _, site_id in enumerate(top_sites.keys()):
        print(" Site ID: %s Name: %s, Number of Records: %s" %(
            site_id, top_sites[site_id]["Name"], top_sites [site_id]["Count"]))

    # Create SingleStationAnalysis object
    ssa = res.SingleStationAnalysis.from_toml(top_sites.keys(), gmms_imts)
    
    # Compute the total, inter-event and intra-event residuals for each site
    ssa.get_site_residuals(sm_database)

    # Output for summary csv
    csv_output = os.path.join(out_dir, 'ssa_results.csv')
    
    # Get summary of statistics and output them
    ssa.station_residual_statistics(filename=csv_output)
    
    # Output plots
    for gmpe in ssa.gmpe_list:
        for imt in ssa.imts:
            
            # Create folder for the GMM
            gmpe_folder_name = str(ssa.gmpe_list[gmpe]).split(
                '(')[0].replace('\n','_').replace(' ','').replace('"','')
        
            gmpe_folder = os.path.join(out_dir, gmpe_folder_name)
            
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
            rspl.ResidualWithSite(
                ssa, gmpe, imt, output_all_res_with_site, filetype='jpg')
            rspl.IntraEventResidualWithSite(
                ssa, gmpe, imt, output_intra_res_components_with_site, filetype='jpg')

    # Print that workflow has finished
    print("Single station residual analysis workflow successfully completed.")


def main(flatfile=demo_flatfile,
         gmms_imts=demo_inputs,
         out_dir=demo_out,
         threshold=demo_threshold):
    """
    Run the demo single station residual analysis
    """
    # Make a new directory for outputs
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    # Get residuals
    sm_database = make_database(flatfile, out_dir)
    
    # Run the single station analysis
    single_station_analysis(sm_database, gmms_imts, out_dir, threshold)
    
    
if __name__ == '__main__':
    sap.run(main)