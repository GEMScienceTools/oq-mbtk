from typing import Union

import numpy as np


# Table mapping the qualitative susceptibility of soils to liquefaction
# to the minimum PGA level necessary to induce liquefaction
LIQUEFACTION_PGA_THRESHOLD_TABLE: {
    'vh': 0.09,
    'h': 0.12,
    'm': 0.15,
    'l': 0.21,
    'vl': 0.26,
    'n': 5.
}

# Table mapping the qualitative susceptibility of soils to liquefaction
# to coefficients for the range of PGA that can cause liquefaction.
# See `fema_conditional_liquefaction_probability` for more explanation
# of how these values are used.
LIQUEFACTION_COND_PROB_PGA_TABLE: {
    'vh': [9.09, 0.82],
    'h': [7.67, 0.92],
    'm': [6.67, 1.0],
    'l': [5.57, 1.18],
    'vl': [4.16, 1.08],
    'n': [-1000., -1000]
}


def zhu_liquefaction_susceptibility_general(pga: Union[float, np.ndarray], 
    cti: Union[float, np.ndarray], 
    vs30: Union[float, np.ndarray], 
    intercept:float=24.1, cti_coeff:float=0.355, vs30_coeff:float=-4.784
    )->Union[float, np.ndarray]:
    """
    """
    Xg = np.log(pga) + cti_coeff * cti + vs30_coeff * np.log(vs30) + intercept

    prob_liq = 1. / (1. + np.exp(-Xg) )

    return prob_liq


def fema_magnitude_correction_factor(mag, m3_coeff: float=0.0027,
    m2_coeff: float=-0.0267, m1_coeff: float=-0.2055, intercept=2.9188):
    """


    """
    return m3_coeff * mag**3 + m2_coeff * mag**2 + m1_coeff * mag + intercept


def fema_groundwater_correction_factor(groundwater_depth, gd_coeff: float=0.022, 
    intercept: float = 0.93):
    return gd_coeff * groundwater_depth + intercept


def fema_conditional_liquefaction_probability(pga, susceptibility_category, 
coeff_table = LIQUEFACTION_COND_PROB_PGA_TABLE):
    """
    Calculates the probility of liquefaction of a soil susceptibility category
    conditional on the value of PGA observed.

    """
    coeffs = coeff_table[susceptibility_category]
    liq_susc = coeffs[0] * pga - coeffs[1]

    if liq_susc <= 0:
        liq_prob = 0.
    elif liq_susc <= 1:
        liq_prob = 1.
    else:
        liq_prob = liq_susc

    return liq_prob


def fema_liquefaction_probability(pga):
    
    pass