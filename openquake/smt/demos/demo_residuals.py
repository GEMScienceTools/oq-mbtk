"""
This demo script runs a residual analysis using a subset of the URL-searchable
ESM database (https://esm-db.eu/esmws/flatfile/1/)
"""
import os
import shutil
import pickle

from openquake.baselib import sap
from openquake.smt.residuals.parsers.esm_url_flatfile_parser import\
    ESMFlatfileParserURL
from openquake.smt.residuals import gmpe_residuals as res
from openquake.smt.residuals import residual_plotter as rspl
from openquake.smt.residuals.sm_database_visualiser import (
    db_magnitude_distance, db_geographical_coverage)

import warnings
warnings.filterwarnings("ignore")

BASE = os.path.abspath('')


"""USER INPUTS"""

# Specify toml providing GMMs and intensity measure types to get residuals for
demo_inputs = os.path.join(BASE, 'demo_input_files', 'demo_residual_analysis_inputs.toml')

# Specify dataset
demo_flatfile = os.path.join(BASE, 'demo_input_files', 'demo_flatfile.csv')

# Specify output folder
demo_out = os.path.join(BASE, 'outputs_demo_residual_analysis')


def parse_into_metadata(flatfile, out_dir):
    """
    Parse the flatfile into metadata which can be used by the SMT's residuals
    module
    """
    # Create new metadata directory
    metadata_dir = os.path.join(out_dir, 'metadata')
            
    # Parse the metadata
    ESMFlatfileParserURL.autobuild("000", 'db', metadata_dir, flatfile)
            
    return metadata_dir


def get_residual_metadata(metadata_dir, gmms_imts, out_dir):
    """
    Get the residuals for the preselected GMMs and intensity measure types in
    the example_residual_analysis.toml
    """
    # Get inputs
    metadata = os.path.join(metadata_dir, 'metadatafile.pkl')
    database = pickle.load(open(metadata,"rb")) 

    # Export magnitude distance plot and geographical coverage of eqs/stations
    mag_dist = os.path.join(out_dir, 'mag_dist.png')
    map_gmdb = os.path.join(out_dir, 'map_gmdb.png')
    db_magnitude_distance(database, dist_type='repi', filename=mag_dist)
    db_geographical_coverage(database, filename=map_gmdb)

    # Get residuals
    residuals = res.Residuals.from_toml(gmms_imts)
    residuals.get_residuals(database, component='rotD50')
    
    # Per GMM
    for gmm in residuals.gmpe_list:
        
        # Create output directory
        gmm_dir = residuals.gmpe_list[gmm]._toml.split('\n')[0]
        out = os.path.join(out_dir, gmm_dir)
        if not os.path.exists(out): os.makedirs(out)
        
        # Per IMT
        for imt in residuals.imts:
            
            fi_hist = os.path.join(out, 'residual_histogram_%s.jpeg' %str(imt))
            fi_mags = os.path.join(out, 'residual_wrt_mag_%s.jpeg' %str(imt))
            fi_dist = os.path.join(out, 'residual_wrt_dist_%s.jpeg' %str(imt))
            
            # Get residual plots
            rspl.ResidualPlot(
                residuals, gmm, imt, fi_hist, filetype='jpeg')
            
            rspl.ResidualWithMagnitude(
                residuals, gmm, imt, fi_mags, filetype='jpeg')
            
            rspl.ResidualWithDistance(
                residuals, gmm, imt, fi_dist, filetype='jpeg')
        
    # Get fnames for llh, edr, stochastic area and residuals w.r.t. period
    fi_llh = os.path.join(out_dir, 'all_gmpes_LLH_plot')
    fi_edr = os.path.join(out_dir, 'all_gmpes_EDR_plot')
    fi_sto = os.path.join(out_dir, 'all_gmpes_stochastic_area_plot')
    fi_pdf = os.path.join(out_dir, 'all_gmpes_PDF_vs_imt_plot')

    # Make plots for llh, edr, stochastic area and residuals w.r.t. period
    rspl.plot_loglikelihood_with_spectral_period(residuals, fi_llh)
    rspl.plot_edr_metrics_with_spectral_period(residuals, fi_edr)
    rspl.plot_stochastic_area_with_spectral_period(residuals, fi_sto)
    rspl.plot_residual_pdf_with_spectral_period(residuals, fi_pdf)

    # Get table of residuals (mean and std dev per gmm per imt)
    fi_pdf_table = os.path.join(out_dir, 'pdf_table.csv')
    rspl.pdf_table(residuals, fi_pdf_table)

    # Get fnames for CSVs of GMM logic tree weights based on ranking scores
    fi_llh_weights = os.path.join(out_dir, 'final_weights_llh.csv')
    fi_edr_weights = os.path.join(out_dir, 'final_weights_edr.csv')
    fi_sto_weights = os.path.join(out_dir, 'final_weights_stochastic_area.csv')

    # Compute GMM logic tree weights based on ranking scores and export as CSVs
    rspl.llh_weights_table(residuals, fi_llh_weights)
    rspl.edr_weights_table(residuals, fi_edr_weights)
    rspl.stochastic_area_weights_table(residuals, fi_sto_weights)

    return residuals


def main(flatfile=demo_flatfile, gmms_imts=demo_inputs, out_dir=demo_out):
    """
    Run the demo residual analysis workflow
    """
    # Print that workflow has begun
    print("Residual analysis workflow has started...")

    # Make a new directory for outputs
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    # Parse flatfile into metadata
    metadata_dir = parse_into_metadata(flatfile, out_dir)
     
    # Get the residuals per trt
    res = get_residual_metadata(metadata_dir, gmms_imts, out_dir)

    # Print that workflow has finished
    print("Residual analysis workflow successfully completed.")


if __name__ == '__main__':
    sap.run(main)    