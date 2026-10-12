# ------------------- The OpenQuake Model Building Toolkit --------------------
# Copyright (C) 2026 GEM Foundation and Électricité de France
#           _______  _______        __   __  _______  _______  ___   _
#          |       ||       |      |  |_|  ||  _    ||       ||   | | |
#          |   _   ||   _   | ____ |       || |_|   ||_     _||   |_| |
#          |  | |  ||  | |  ||____||       ||       |  |   |  |      _|
#          |  |_|  ||  |_|  |      |       ||  _   |   |   |  |     |_
#          |       ||      |       | ||_|| || |_|   |  |   |  |    _  |
#          |_______||____||_|      |_|   |_||_______|  |___|  |___| |_|
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# This script is produced within the scope of Work Package 5, named Simulation 
# platform, under SIGMA3 project. For more detailed information about 
# the project, please visit to https://sigma-programs.com/.
# -----------------------------------------------------------------------------
# vim: tabstop=4 shiftwidth=4 softtabstop=4
# coding: utf-8

"""
module :mod:`openquake.plt.interevent` provides functions for a statistical 
evaluation and plotting for inter-event times of catalogue.

Note: A simplified calculation for decimal years is used within function.
This is sufficient enough for long-term catalogues but minor variations in month 
lengths and leap years could cause micro-scale precision discrepancies 
for short-term catalogues.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, lognorm, weibull_min

def get_aic(dist, params, data):
    """
    Computes the AIC score to evaluate the goodness-of-fit of a fitted 
    continuous distribution against the empirical inter-event time data. It 
    balances the model's likelihood against its complexity by penalizing 
    the total number of estimated parameters.

    Inputs:
    - dist (scipy.stats.rv_continuous): A SciPy continuous distribution object 
                                        (e.g., expon, lognorm, weibull_min).
    - params (tuple): The fitted parameter estimates returned by the dist.fit() 
                      method (e.g., shape, location, scale parameters).
    - data (np.ndarray): Array of observations (calculated chronological 
                         inter-event time intervals).

    Returns:
    - aic_score (float): The calculated Akaike Information Criterion value. 
                         Lower scores indicate a mathematically superior model fit. 
    """
    log_lik = np.sum(dist.logpdf(data, *params))
    return 2 * len(params) - 2 * log_lik

def analyze_interevent_times(catalogue_path: str, 
                                 bin_scale: str = "logarithmic", 
                                 num_bins: int = 50,
                                 output_png: str = None):
    """
    Analyzes and plots the inter-event time distribution of earthquake catalogue.

    Inputs:
    - catalogue_path (str): Path to HMTK-formatted CSV catalogue.
    - bin_scale (str): Scale for histogram bins and X-axis. Options: 'logarithmic' or 'linear'. Default: 'logarithmic'.
    - num_bins (int): Number of bins for the histogram. Default: 50.
    - output_png (str/None): Path to output plot.

    Outputs:
    - Plot for inter-event time distribution.
    - Statistical summary.
    """
    # Data
    if not os.path.exists(catalogue_path):
        raise FileNotFoundError(f"Catalogue is not available at: {catalogue_path}")
        
    df = pd.read_csv(catalogue_path)

    # Simplified calculation of decimal years
    for col in ['hour', 'minute', 'second']:
        if col not in df.columns:
            df[col] = 0

    df['decimal_year'] = (
        df['year'] + 
        (df['month'] - 1) / 12 + 
        (df['day'] - 1) / 365.25 +
        (df['hour'] / 8766) + 
        (df['minute'] / 525960) + 
        (df['second'] / 31557600)
    )

    # Sort out and calculate time intervals
    df = df.sort_values('decimal_year').reset_index(drop=True)
    intervals = df['decimal_year'].diff().dropna()
    intervals = intervals[intervals > 0].values

    # Fitting statistical distributions
    p_ex = expon.fit(intervals)
    p_ln = lognorm.fit(intervals)
    p_wb = weibull_min.fit(intervals)

    # AIC calculation
    results = {
        'Exponential': get_aic(expon, p_ex, intervals),
        'Lognormal': get_aic(lognorm, p_ln, intervals),
        'Weibull': get_aic(weibull_min, p_wb, intervals)
    }
    best_fit = min(results, key=results.get)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))

    if bin_scale.lower() == "logarithmic":
        bins = np.logspace(np.log10(intervals.min()), np.log10(intervals.max()), num_bins)
        x_pdf = np.logspace(np.log10(intervals.min()), np.log10(intervals.max()), 1000)
        ax.set_xscale('log')
        ax.set_yscale('log')
    elif bin_scale.lower() == "linear":
        bins = np.linspace(intervals.min(), intervals.max(), num_bins)
        x_pdf = np.linspace(intervals.min(), intervals.max(), 1000)
    else:
        raise ValueError("bin_scale must be either 'linear' or 'logarithmic'")

    ax.hist(intervals, bins=bins, density=True, color='skyblue', edgecolor='black', alpha=0.4, label='Catalogue')

    ax.plot(x_pdf, expon.pdf(x_pdf, *p_ex), 'r-', lw=1.5, label=f'Exponential (AIC: {results["Exponential"]:.1f})')
    ax.plot(x_pdf, lognorm.pdf(x_pdf, *p_ln), 'g--', lw=1.5, label=f'Lognormal (AIC: {results["Lognormal"]:.1f})')
    ax.plot(x_pdf, weibull_min.pdf(x_pdf, *p_wb), 'b:', lw=1.5, label=f'Weibull (AIC: {results["Weibull"]:.1f})')

    # Formatting
    ax.set_xlabel("Inter-event Time (Years)", fontweight='bold')
    ax.set_ylabel("Probability Density", fontweight='bold')
    ax.set_title(f"Inter-event Time Distribution (Best Fit: {best_fit})", fontweight='bold', fontsize=10)
    ax.grid(True, which="both", linestyle=':', alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()

    plt.savefig(output_png, dpi=300)
    print(f"Done! Saved to: {output_png}")
    plt.show()
    plt.close(fig)

    # Statistical summary
    print(f"\n{'='*40}\n         STATISTICAL SUMMARY\n{'='*40}")
    print(f"Total Analyzed Events        : {len(df)}")
    print(f"Calculated Intervals         : {len(intervals)}")
    print(f"Mean Interval (μ)            : {np.mean(intervals):.4f} years")
    print(f"Coefficient of Variation (CV): {np.std(intervals)/np.mean(intervals):.2f}")
    print(f"Best-fitting Model (min AIC) : {best_fit}")
    print(f"{'='*40}\n")