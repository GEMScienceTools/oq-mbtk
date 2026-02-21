# Script Title: GMICEs and IGMCEs of Gallahue and Abrahamson (2023)'s work

# Description:
# This script implements the methodology described in the paper titled 
# "New Methodology for Unbiased Ground-Motion Intensity Conversion Equations" 
# by Gallahue, M., and Abrahamson, N., published in 2023. Ground-motion 
# intensity conversion equations (GMICEs) and intensity ground-motion 
# conversion equations (IGMCEs) developed in the work are coded to 
# obtain PGA and Intensity.

# Reference:
# Molly Gallahue, Norman Abrahamson; New Methodology for Unbiased Ground‐Motion 
# Intensity Conversion Equations. Bulletin of the Seismological Society of 
# America 2023; 113 (3): 1133–1151. doi: https://doi.org/10.1785/0120220224

import numpy as np
import pandas as pd
import os

# Coefficients
COEFFS = {
    "eq19": {"d1": 2.919, "d2": 0.356, "d3": 0.010, "d4": 1.041, "d5": -0.889, "sigma": 0.566},
    "eq20": {"h1": 8.622, "h2": 1.230, "h3": 0.056, "h4": -0.568, "sigma": 0.704},
    "eq22": {"f1": -2.808, "f2": 0.444, "f3": -0.061, "f4": -0.047, "f5": -0.458, "sigma": 0.618},
    "eq23": {"i1": -6.558, "i2": 0.754, "i3": -0.072, "i4": -0.187, "sigma": 0.667}
}

# Functions
def gmice(data, equation_id, epsilon=0):
    """
    Ground-motion to intensity conversion (PGA -> Intensity)
    Equations: 19 or 20
    Units: PGA [g], Rhyp [km]
    Epsilon (ϵ): It can be estimated using the mean ϵ from the disaggregation; 
    however, if disaggregation results are not available, then ϵ can be approximated 
    from the slope of the hazard curve at any particular site (Gallahue and Abrahamson, 2023).
    """
    df = pd.read_csv(data)
    c = COEFFS.get(equation_id)
    
    if equation_id == "eq19":
        required = ["PGA", "Mw", "Rhyp"]
        if not all(col in df.columns for col in required):
            raise ValueError(f"ERROR: {required} are not found for Equation 19!")
        
        ln_pga = np.log(df["PGA"])
        ln_rhyp = np.log(df["Rhyp"])
        df["Intensity_pred"] = (c["d1"] + c["d2"] * ln_pga + 
                           c["d3"] * (ln_pga - np.log(0.1))**2 + 
                           c["d4"] * df["Mw"] + c["d5"] * ln_rhyp)
        
    elif equation_id == "eq20":
        if "PGA" not in df.columns:
            raise ValueError("ERROR: PGA is not found for Equation 20!")
        
        ln_pga = np.log(df["PGA"])
        df["Intensity_pred"] = (c["h1"] + c["h2"] * ln_pga + 
                           c["h3"] * (ln_pga - np.log(0.1))**2 + 
                           c["h4"] * epsilon)
    else:
        raise ValueError("ERROR: Invalid equation ID! Please select 'eq19' or 'eq20' for GMICE.")

    save_results(df, "gmice_res.csv")

def igmce(data, equation_id, epsilon=0):
    """
    Intensity to ground-motion conversion (Intensity -> PGA)
    Equations: 22 or 23
    Units: PGA [g], Rjb [km]
    Epsilon (ϵ): It can be estimated using the mean ϵ from the disaggregation; 
    however, if disaggregation results are not available, then ϵ can be approximated 
    from the slope of the hazard curve at any particular site (Gallahue and Abrahamson, 2023).
    """
    df = pd.read_csv(data)
    c = COEFFS.get(equation_id)
    
    if equation_id == "eq22":
        required = ["Intensity", "Mw", "Rjb"]
        if not all(col in df.columns for col in required):
            raise ValueError(f"ERROR: {required} are not found for Equation 22!")
        
        ln_rjb = np.log(df["Rjb"])
        ln_pga = (c["f1"] + c["f2"] * df["Intensity"] + 
                  c["f3"] * (df["Intensity"] - 6)**2 + 
                  c["f4"] * df["Mw"] + c["f5"] * ln_rjb)
        df["PGA_pred"] = np.exp(ln_pga)
        
    elif equation_id == "eq23":
        if "Intensity" not in df.columns:
            raise ValueError("ERROR: Intensity is not found for Equation 23!")
            
        ln_pga = (c["i1"] + c["i2"] * df["Intensity"] + 
                  c["i3"] * (df["Intensity"] - 6)**2 + 
                  c["i4"] * epsilon)
        df["PGA_pred"] = np.exp(ln_pga)
    else:
        raise ValueError("ERROR: Invalid equation ID! Please select 'eq22' or 'eq23' for IGMCE.")

    save_results(df, "igmce_res.csv")

def save_results(df, filename):
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filepath = os.path.join(output_dir, filename)
    
    df = df.round(5)
    df.to_csv(filepath, index=False)
    print(f"Done! Saved to '{filepath}'")
