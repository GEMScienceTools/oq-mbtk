"""
This demo script runs a residual analysis using a subset of the URL-searchable
ESM database (https://esm-db.eu/esmws/flatfile/1/)
"""
import os
import shutil
import pickle

from openquake.baselib import sap

from openquake.smt.residuals import gmpe_residuals as res
from openquake.smt.residuals import residual_plotter as rspl
from openquake.smt.residuals.parsers.esm_url_flatfile_parser import ESMFlatfileParserURL
from openquake.smt.residuals.sm_database_visualiser import db_magnitude_distance, db_geographical_coverage

import warnings
warnings.filterwarnings("ignore")


BASE = os.path.abspath('')


"""USER INPUTS"""

# Specify toml providing GMMs and intensity measure types to get residuals for
demo_inputs = os.path.join(BASE, 'demo_input_files', 'demo_residual_analysis_inputs.toml')

# Specify dataset
demo_flatfile = os.path.join(BASE, 'demo_input_files', 'demo_flatfile.csv')

# Specify horizontal component definition to use for observations
demo_comp = 'Geometric'

# Specify output folder
demo_out = os.path.join(BASE, 'outputs_demo_residual_analysis')


def parse_into_metadata(flatfile, out_dir):
    """
    Parse the flatfile into an SMT ground-motion database
    """
    # Create new metadata directory
    metadata_dir = os.path.join(out_dir, 'metadata')
            
    # Parse the metadata
    ESMFlatfileParserURL.autobuild("000", 'db', metadata_dir, flatfile)
            
    return metadata_dir


def get_residual_metadata(metadata_dir, gmms_imts, comp, out_dir):
    """
    Get the residuals for the requested GMMs and intensity
    measure types
    """
    # Get inputs
    metadata = os.path.join(metadata_dir, 'metadatafile.pkl')
    database = pickle.load(open(metadata,"rb")) 

    # Get residuals
    residuals = res.Residuals.from_toml(gmms_imts)
    residuals.compute_residuals(database, component=comp)

    # Export the residuals to a text file
    exp_dir = os.path.join(out_dir, f"residuals_hrz_comp_def_of_{comp}.txt")
    residuals.export_residuals(exp_dir)

    # Export magnitude distance plot and geographical coverage of eqs/stations
    mag_dist = os.path.join(out_dir, 'mag_dist.png')
    map_gmdb = os.path.join(out_dir, 'map_gmdb.png')
    db_magnitude_distance(database, dist_type='repi', filename=mag_dist)
    db_geographical_coverage(database, filename=map_gmdb)

    return residuals


def make_residual_plots(residuals, out_dir):
    """
    Generate various plots of the residual distributions and
    also plot them with respect to magnitude and distance
    """
    # Per GMM
    for gmm in residuals.gmpe_list:
        
        # Create output directory
        gmm_dir = residuals.gmpe_list[gmm]._toml.split('\n')[0]
        out = os.path.join(out_dir, gmm_dir)
        if not os.path.exists(out): os.makedirs(out)
        
        # Per IMT
        for imt in residuals.imts:
            
            # Get fnames
            fi_hist = os.path.join(out, 'residual_histogram_%s.png' %str(imt))
            fi_mags = os.path.join(out, 'residual_wrt_mag_%s.png' %str(imt))
            fi_dist = os.path.join(out, 'residual_wrt_dist_%s.png' %str(imt))
            
            # Get residual plots
            rspl.ResidualPlot(residuals, gmm, imt, filename=fi_hist)
            rspl.ResidualWithMagnitude(residuals, gmm, imt, filename=fi_mags)
            rspl.ResidualWithDistance(residuals, gmm, imt, filename=fi_dist, distance_type="rrup")
            

def calc_ranking_metrics(residuals, out_dir):
    """
    Compute LLH, EDR and Stochastic Area scores
    """
    # Compute llh, edr, stochastic area and residuals w.r.t. period
    residuals.get_llh_values()
    residuals.get_edr_wrt_imt()
    residuals.get_sto_wrt_imt()
    
    # Set fnames for llh, edr, stochastic area and residuals tables
    fi_llh_table = os.path.join(out_dir, 'llh_values.csv')
    fi_edr_table = os.path.join(out_dir, 'edr_values.csv')
    fi_sto_table = os.path.join(out_dir, 'sto_values.csv')
    fi_residual_means_and_stds_table = os.path.join(out_dir, 'means_and_stds_table.csv')

    # Make tables for llh, edr, stochastic area and residuals table
    rspl.llh_table(residuals, fi_llh_table)
    rspl.edr_table(residuals, fi_edr_table)
    rspl.sto_table(residuals, fi_sto_table)
    rspl.residual_means_and_stds_table(residuals, fi_residual_means_and_stds_table)

    # Set fnames for llh, edr, stochastic area and residuals plots w.r.t. period
    fi_llh_plot = os.path.join(out_dir, 'llh_vs_period.png')
    fi_edr_plot = os.path.join(out_dir, 'edr_vs_period.png')
    fi_sto_plot = os.path.join(out_dir, 'sto_vs_period.png')
    fi_pdf_plot = os.path.join(out_dir, 'means_and_stds_vs_period.png')
    
    # Make plots for llh, edr, stochastic area and residuals plots w.r.t. period
    rspl.plot_llh_with_period(residuals, fi_llh_plot)
    rspl.plot_edr_with_period(residuals, fi_edr_plot)
    rspl.plot_sto_with_period(residuals, fi_sto_plot)
    rspl.plot_residual_means_and_stds_with_period(residuals, fi_pdf_plot)

    # Set fnames for CSVs of GMM logic tree weights based on ranking scores
    fi_llh_weights = os.path.join(out_dir, 'weights_llh.csv')
    fi_edr_weights = os.path.join(out_dir, 'weights_edr.csv')
    fi_sto_weights = os.path.join(out_dir, 'weights_sto.csv')

    # Compute GMM logic tree weights based on ranking scores and export as CSVs
    rspl.llh_weights_table(residuals, fi_llh_weights)
    rspl.edr_weights_table(residuals, fi_edr_weights)
    rspl.sto_weights_table(residuals, fi_sto_weights)


def main(flatfile=demo_flatfile,
         gmms_imts=demo_inputs,
         comp=demo_comp,
         out_dir=demo_out):
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
     
    # Get the residuals
    res = get_residual_metadata(metadata_dir, gmms_imts, comp, out_dir)

    # Make the plots
    make_residual_plots(res, out_dir)

    # Compute ranking metrics
    calc_ranking_metrics(res, out_dir)

    # Print that workflow has finished
    print("Residual analysis workflow successfully completed.")


if __name__ == '__main__':
    sap.run(main)    