#!/usr/bin/env python
# coding: utf-8

# In[9]:


import re
import numpy as np
import os
import glob
import math
import pandas as pd


def _find_idcalc(path_hazard):
    # Find the first CSV file matching the pattern
    files = glob.glob(os.path.join(path_hazard, 'realizations_*.csv'))
    
    if not files:  # Check if no files are found
        raise FileNotFoundError("No files matching the pattern 'realizations_*.csv' found in the specified path.")
    
    # Extract the number from the first matched file
    match = re.search(r'realizations_(\d+)\.csv$', files[0])
    
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Filename '{files[0]}' does not match the expected pattern.")


    
def disaggrMDEmean(Disaggr_name, imt, poe_target, type_disaggr):
    # Read CSV data
    disaggr_csv = pd.read_csv(Disaggr_name, header=1)
    
    # Extract columns as NumPy arrays for efficiency
    SA_types = disaggr_csv['imt'].values
    poe_col = disaggr_csv['poe'].values
    iml = disaggr_csv['iml'].values
    mag_col = disaggr_csv['mag'].values
    dist_col = disaggr_csv['dist'].values
    eps_col = disaggr_csv['eps'].values
    wmean_poe_all = disaggr_csv['mean'].values
    
    # Initialize arrays
    mag, dist, eps, wmean_poe = [], [], [], []
    mag_mean_i, dist_mean_i, eps_mean_i = [], [], []

    # Apply conditions using masks
    if type_disaggr == 'iml':
        mask = SA_types == imt
    else:
        mask = (SA_types == imt) & (np.abs(poe_col - poe_target) <= 1e-3)

    # Apply the mask to extract relevant rows
    mag_selected = mag_col[mask]
    dist_selected = dist_col[mask]
    eps_selected = eps_col[mask]
    wmean_poe_selected = wmean_poe_all[mask]
    iml_selected = iml[mask]
    
    # Calculate weighted means directly
    mag_mean_i = mag_selected * wmean_poe_selected
    dist_mean_i = dist_selected * wmean_poe_selected
    eps_mean_i = eps_selected * wmean_poe_selected

    # Populate lists with selected values
    mag.extend(mag_selected)
    dist.extend(dist_selected)
    eps.extend(eps_selected)
    wmean_poe.extend(wmean_poe_selected)
    
    # Calculate aggregate values
    
    if np.any(iml_selected > 0):  # Check if there are any valid entries
        poe_mode = np.max(wmean_poe_selected)
        it = np.abs(wmean_poe_selected - poe_mode).argmin()

        mag_mode = mag_selected[it]
        dist_mode = dist_selected[it]
        eps_mode = eps_selected[it]

        mag_mean = np.sum(mag_mean_i) / poe_target
        dist_mean = np.sum(dist_mean_i) / poe_target
        eps_mean = np.sum(eps_mean_i) / poe_target

        disaggr_mean = [mag_mean, dist_mean, eps_mean]
        disaggr_mode = [mag_mode, dist_mode, eps_mode]
    else:
        disaggr_mean = [0, 0, 0]
        disaggr_mode = [0, 0, 0]
    
    return iml_selected, mag, dist, eps, wmean_poe, disaggr_mean, disaggr_mode
   


def main(file_sites_in: str, path_disaggregation: str):
    
    """
    Find the mean a mode of a disaggregation analysis given the output in csv

    """

    type_disaggr = 'poe'
    main.file_sites_in = "Path to the input site model .csv"
    main.path_disaggregation = "Path to the disaggregation output from OQ .csv file"
 
    # Read the input sites CSV
    sites_csv = pd.read_csv(file_sites_in)
    
    # Extract data from sites_csv
    custom_site_ids = sites_csv['custom_site_id'].values
    lons = sites_csv['lon'].values
    lats = sites_csv['lat'].values
    
    # Initialize lists to store outputs
    output_data = { "calc_id": [], "id": [], "site": [], "lon": [], "lat": [],
        "imt": [], "poe": [], "iml": [], "M mean": [], "R mean": [], "eps mean": [],
        "M mode": [], "R mode": [], "eps mode": [], "check": [], "pass": []
    }
    
    # Messages for readme.txt
    out_text = []

    # Iterate through each custom site
    for c, custom_site_id in enumerate(custom_site_ids):
        
        calc_id = _find_idcalc(path_disaggregation)
        fname_disaggr = os.path.join(path_disaggregation, f'Mag_Dist_Eps-mean-{c}_{calc_id}.csv')

        if os.path.isfile(fname_disaggr):
            with open(fname_disaggr, "r") as f:
                metadata = f.readline()
                
            # Extract coordinates from metadata
            lon_disagg = round(float(metadata.split('lon=')[1].split(',')[0].strip()), 4)
            lat_disagg =round(float(metadata.split('lat=')[1].split('"')[0].strip()), 4)
            
            # Check if coordinates match
            if np.isclose(lon_disagg, lons[c], atol=0.01) and np.isclose(lat_disagg, lats[c], atol=0.01):
                disaggr_csv = pd.read_csv(fname_disaggr, header=1)
                
                poe_target = [0.002107] if type_disaggr == 'iml' else disaggr_csv['poe'].unique()

                for imt in disaggr_csv['imt'].unique():
                    for poe in poe_target:
                        # Call external function and unpack results
                        iml_disaggr, mag, dist, eps, wmean_poe, disaggr_means_s, disaggr_mode = disaggrMDEmean(
                            fname_disaggr, imt, poe, type_disaggr)
                        
                        # Append data to the dictionary
                        
                        output_data["calc_id"].append(calc_id)
                        output_data["id"].append(c)
                        output_data["site"].append(custom_site_id)
                        output_data["lon"].append(lon_disagg)
                        output_data["lat"].append(lat_disagg)
                        output_data["poe"].append(poe)
                        output_data["imt"].append(imt)
                        output_data["iml"].append(iml_disaggr[0])
                        
                        output_data["M mean"].append(disaggr_means_s[0])
                        output_data["R mean"].append(disaggr_means_s[1])
                        output_data["eps mean"].append(disaggr_means_s[2])

                       
                        if iml_disaggr[0] > 0:
                            output_data["M mode"].append(f"{disaggr_mode[0]:.2f}")
                            output_data["R mode"].append(f"{disaggr_mode[1]:.2f}")
                            output_data["eps mode"].append(f"{disaggr_mode[2]:.2f}")
                        else:
                            output_data["M mode"].append(str(-999))
                            output_data["R mode"].append(str(-999))
                            output_data["eps mode"].append(str(-999))

                        check_val = sum(wmean_poe)
                        output_data["check"].append(check_val)
                        check_ratio = abs((check_val - poe) / poe)
                        output_data["pass"].append('yes' if check_ratio <= 0.1 else 'no')
            else:
                out_text.append(f'Coordinate mismatch for site id {c}: {custom_site_id}')
        else:
            out_text.append(f'No disaggregation data for site id {c}: {custom_site_id}')

    # Create DataFrames from the collected data
    df_disaggr = pd.DataFrame(output_data)
    df_disaggr.to_csv(path_disaggregation + 'res_disaggregation_mean_output.csv' ,index=False) 
    # Save messages to readme.txt
    with open(os.path.join(path_disaggregation, 'res_disaggregation_readme.txt'), 'w') as f:
        f.write('\n'.join(out_text))


if __name__ == '__main__':
        # Call main function with the required arguments
    from openquake.baselib import sap
        
    sap.run(main)
