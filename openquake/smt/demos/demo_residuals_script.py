"""
This demo script runs a residual analysis using a subset of the URL-searchable
ESM database (https://esm-db.eu/esmws/flatfile/1/)
"""
import os
import shutil
import pickle

from openquake.baselib import sap
from openquake.smt.parsers.esm_url_flatfile_parser import ESMFlatfileParserURL
from openquake.smt.residuals import gmpe_residuals as res
from openquake.smt.residuals import residual_plotter as rspl

import warnings
warnings.filterwarnings("ignore")

DATA = os.path.abspath('')

"""USER INPUTS"""

# Specify toml providing GMMs and intensity measure types to get residuals for
gmms_imts = 'demo_residual_analysis_inputs.toml'

# Specify dataset
db = 'demo_flatfile.csv'

# Specify output folder
out_dir = 'demo_run'


def parse_into_metadata():
    """
    Parse the flatfile into metadata which can be used by the SMT's residuals
    module
    """
    # Create metadata directory
    metadata_dir = os.path.join(DATA, out_dir + '_metadata')
    if os.path.exists(metadata_dir):
        shutil.rmtree(metadata_dir)
            
    # Parse the metadata
    ESMFlatfileParserURL.autobuild("000", 'db', metadata_dir, db)
            
    return metadata_dir


def get_residual_metadata(metadata_dir):
    """
    Get the residuals for the preselected GMMs and intensity measure types in
    the example_residual_analysis.toml
    """
    # Get inputs
    metadata = os.path.join(metadata_dir, 'metadatafile.pkl')
    database = pickle.load(open(metadata,"rb")) 

    # If output directory exists remove
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    # Get residuals
    residuals = res.Residuals.from_toml(gmms_imts)
    residuals.get_residuals(database, component='RotD50')
    
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
        
    # Get fipaths for llh, edr, stochastic area and residual plots + tables
    fi_llh = os.path.join(out_dir, 'all_gmpes_LLH_plot')
    fi_edr = os.path.join(out_dir, 'all_gmpes_EDR_plot')
    fi_sto = os.path.join(out_dir, 'all_gmpes_stochastic_area_plot')
    fi_pdf = os.path.join(out_dir, 'all_gmpes_PDF_vs_imt_plot')
    fi_pdf_table = os.path.join(out_dir, 'pdf_table.csv')

    # Get plots and tables
    rspl.plot_loglikelihood_with_spectral_period(residuals, fi_llh)
    rspl.plot_edr_metrics_with_spectral_period(residuals, fi_edr)
    rspl.plot_stochastic_area_with_spectral_period(residuals, fi_sto)
    rspl.plot_residual_pdf_with_spectral_period(residuals, fi_pdf)
    rspl.pdf_table(residuals, fi_pdf_table)
    
    return residuals


def main():
    """
    Run the demo workflow
    """
    # Parse flatfile into metadata
    metadata_dir = parse_into_metadata()
     
    # Get the residuals per trt
    res = get_residual_metadata(metadata_dir)
    
    breakpoint()


if __name__ == '__main__':
    sap.run(main)
    
    
    