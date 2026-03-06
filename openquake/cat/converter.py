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
import pathlib


COEFFS = {
    "eq19": {"d1": 2.919, "d2": 0.356, "d3": 0.010, "d4": 1.041, "d5": -0.889, "sigma": 0.566},
    "eq20": {"h1": 8.622, "h2": 1.230, "h3": 0.056, "h4": -0.568, "sigma": 0.704},
    "eq22": {"f1": -2.808, "f2": 0.444, "f3": -0.061, "f4": -0.047, "f5": -0.458, "sigma": 0.618},
    "eq23": {"i1": -6.558, "i2": 0.754, "i3": -0.072, "i4": -0.187, "sigma": 0.667}
}


class GallahueAbrahamson2023Model1:
    """
    Ground-motion to intensity conversion (PGA -> Intensity)
    Equations: 19 or 20
    Units: pga [g], rhypo [km]
    Epsilon (ϵ): It can be estimated using the mean ϵ from the disaggregation; 
    however, if disaggregation results are not available, then ϵ can be approximated 
    from the slope of the hazard curve at any particular site (Gallahue and Abrahamson, 2023).
    """
    def __init__(self, data: np.ndarray):
        self.data = data
        self.mint = None

    def get_intensity(self, mode: str = 'eq19', epsilon: float = 0):
        if mode == 'eq19':

            required = ['pga', 'mag', 'rhypo']
            self._check_columns(required)
            
            c = COEFFS["eq19"]
            ln_pga = np.log(self.data['pga'])
            ln_rhypo = np.log(self.data['rhypo'])
            
            self.mint = (c["d1"] + c["d2"] * ln_pga + 
                         c["d3"] * (ln_pga - np.log(0.1))**2 + 
                         c["d4"] * self.data['mag'] + c["d5"] * ln_rhypo)
            
        elif mode == 'eq20':
            self._check_columns(['pga'])
            
            c = COEFFS["eq20"]
            ln_pga = np.log(self.data['pga'])
            
            self.mint = (c["h1"] + c["h2"] * ln_pga + 
                         c["h3"] * (ln_pga - np.log(0.1))**2 + 
                         c["h4"] * epsilon)
        else:
            raise ValueError("Invalid mode! Choose 'eq19' or 'eq20'.")
        
        return self.mint

    def save(self, filename: str):
        if self.mint is None:
            raise ValueError("Please run 'get_intensity' before saving.")
        _save_results(self.data, self.mint, 'intensity', filename)

    def _check_columns(self, required):
        missing = [col for col in required if col not in self.data.dtype.names]
        if missing:
            raise ValueError(f"Missing required columns in structured array: {missing}")


class GallahueAbrahamson2023Model2:
    """
    Intensity to ground motion conversion (Intensity -> PGA)
    Equations: 22 or 23
    Units: pga [g], rjb [km]
    Epsilon (ϵ): It can be estimated using the mean ϵ from the disaggregation; 
    however, if disaggregation results are not available, then ϵ can be approximated 
    from the slope of the hazard curve at any particular site (Gallahue and Abrahamson, 2023).
    """
    def __init__(self, data: np.ndarray):
        self.data = data
        self.pga = None

    def get_pga(self, mode: str = 'eq22', epsilon: float = 0):
        if mode == 'eq22':
            required = ['intensity', 'mag', 'rjb']
            self._check_columns(required)
            
            c = COEFFS["eq22"]
            ln_rjb = np.log(self.data['rjb'])
            
            ln_pga = (c["f1"] + c["f2"] * self.data['intensity'] + 
                      c["f3"] * (self.data['intensity'] - 6)**2 + 
                      c["f4"] * self.data['mag'] + c["f5"] * ln_rjb)
            self.pga = np.exp(ln_pga)
            
        elif mode == 'eq23':
            self._check_columns(['intensity'])
            
            c = COEFFS["eq23"]
            ln_pga = (c["i1"] + c["i2"] * self.data['intensity'] + 
                      c["i3"] * (self.data['intensity'] - 6)**2 + 
                      c["i4"] * epsilon)
            self.pga = np.exp(ln_pga)
        else:
            raise ValueError("Invalid mode! Choose 'eq22' or 'eq23'.")
            
        return self.pga

    def save(self, filename: str):
        if self.pga is None:
            raise ValueError("Please run 'get_pga' before saving.")
        _save_results(self.data, self.pga, 'pga', filename)

    def _check_columns(self, required):
        missing = [col for col in required if col not in self.data.dtype.names]
        if missing:
            raise ValueError(f"Missing required columns in structured array: {missing}")


def _save_results(data: np.ndarray, result_array: np.ndarray, result_name: str, filename: str):
    path = pathlib.Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    keys = data.dtype.names
    tmp = {k: data[k] for k in keys}
    tmp[result_name] = result_array
    
    df = pd.DataFrame(tmp)
    df = df.round(5)
    df.to_csv(path, index=False)
    print(f"Done! Saved to '{filename}'")
