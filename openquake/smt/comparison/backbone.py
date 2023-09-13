"""
Functions for backbone approach GMC model development
"""
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot
from openquake.hazardlib import valid
from openquake.smt.comparison.utils_compare_gmpes import get_z1_z25, get_cols
from openquake.smt.comparison.utils_gmpes import (att_curves, _param_gmpes,
                                                  mgmpe_check)


def plot_abs_ratio_trellis_util(
        trt, ztor, rake, strike, dip, depth, Z1, Z25, Vs30, region, imt_list,
        mag_list, maxR, gmpe_list, aratio, Nstd, output_directory,
        custom_color_flag, custom_color_list, eshm20_region, dist_type,
        up_or_down_dip = None):
    """
    Generate absolute difference between the arithmetic mean of a suite of
    GMPEs and each GMPE individually vs distance for the given run configuration.
    
    This function is useful when a backbone model must be identified from a
    region for which no ground-motion data is available, and therefore the
    difference of each candidate GMPE and the mean of other appropriate GMPEs
    for the region may be used to guide model selection.
    
    :param gmpe_list:
        A list e.g. ['BooreEtAl2014', 'CauzziEtAl2014']. The entire list of
        GMPEs is considered in the plots (i.e. no distinction is made between
        if a GMPE has been assigned to GMC logic tree #1 or #2 when computing
        the mean here, and all GMPEs are compared against this mean)
    """
    # Setup
    fig = pyplot.figure(figsize=(len(mag_list)*5, len(imt_list)*4))
    Z1, Z25 = get_z1_z25(Z1, Z25, Vs30, region)
    colors = get_cols(custom_color_flag, custom_color_list)     
    step = 1
    
    for n, i in enumerate(imt_list):
        for l, m in enumerate(mag_list):
            fig.add_subplot(len(imt_list), len(mag_list), l+1+n*len(mag_list))
            mean_store = []
            for g, gmpe in enumerate(gmpe_list): 
                
                # Perform mgmpe check
                gsim = valid.gsim(gmpe)
                gmm_orig = gsim
                gmm = mgmpe_check(gsim)
                
                # ZTOR value
                if ztor is not None:
                    ztor_m = ztor[l]
                else:
                    ztor_m = None
                
                # Get gmpe params
                strike_g, dip_g, depth_g, aratio_g = _param_gmpes(
                    strike, dip, depth[l], aratio, rake, trt) 
                
                # Get attenuation curves
                mean, std, r_vals = att_curves(gmm, gmm_orig, depth[l],m, 
                                               aratio_g, strike_g, dip_g,
                                               rake,Vs30, Z1, Z25, maxR, 
                                               step, i, ztor_m, eshm20_region,
                                               dist_type, trt, up_or_down_dip)
                
                # Store the median per gmpe
                mean_store.append(mean[0][0])

                # update plots
                update_abs_ratio_plots(m, i, n, l, r_vals, imt_list, dist_type)

            pyplot.grid(axis='both', which='both', alpha=0.5)
            
            # Get abs diff and plot
            abs_ratio_data(mean_store, r_vals, gmpe_list, colors)
            
    # Finalise plots
    pyplot.legend(loc = "center left", bbox_to_anchor = (1.1, 1.05),
                  fontsize = '16')
    pyplot.savefig(os.path.join(output_directory, 'abs_ratio_plots.png'),
                   bbox_inches='tight', dpi=200, pad_inches=0.2)
    
    
### abs_ratio utils
def abs_ratio_data(mean_store, r_vals, gmpe_list, colors):
    """
    """
    # Get mean w.r.t. all the gmpes
    mean_df = pd.DataFrame(mean_store)
    mean_vals = (mean_df.mean(axis=0))
        
    # Get abs diff of mean and each gmpe
    for idx, means in enumerate(mean_store):
        abs_ratio = np.abs(mean_vals/means)
        pyplot.plot(r_vals, abs_ratio, color = colors[idx], linewidth=2,
                    linestyle='-', label=gmpe_list[idx])


def update_abs_ratio_plots(m, i, n, l, r_vals, imt_list, dist_type):
    """
    Add titles and axis labels to abs_ratio plots
    """
    if dist_type == 'rrup':
        label = 'Rrup (km)'
    if dist_type == 'rjb':
        label = 'Rjb (km)'
    if n == 0: #top row only
        pyplot.title('Mw = ' + str(m), fontsize='16') # Mod if required
    if n == len(imt_list)-1: #bottom row only
        pyplot.xlabel(label, fontsize='16')
    if l == 0: #left row only
        pyplot.ylabel('Abs. Ratio for ' + str(i), fontsize='16')
    pyplot.loglog()
    pyplot.ylim(0.5, 2.0) # Mod if required
    #pyplot.xlim(r_vals[2], max(r_vals)) # Mod if required