#!/usr/bin/env python
# coding: utf-8

import re
import os
import glob
import toml
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
from openquake.baselib import sap
from openquake.wkf.utils import create_folder, get_list


# TODO move outside of this module
def hypocentral_depth_analysis(fname: str, depth_min: float, depth_max: float,
                               depth_binw: float, figure_name_out: str = '',
                               show: bool = False, depth_bins=[], label='',
                               figure_format='png') -> Tuple[np.ndarray, np.ndarray]:    
    """
    :param fname:
        The name of the file containing the catalogue
    :param depth_min:
        The minimum depth [km]
    :param depth_max:
        The maximum depth [km]
    :param depth_binw:
        The width of the bins used [km]. Alternatively it's possible to use
        the bins by setting the `bins` variable.
    :param figure_name_out:
        The name of the figure to be created
    :param show:
        When true the show figures on screen
    :param depth_bins:
        The bins used to build the statistics. Overrides the `depth_min`,
        `depth_max`, `depth_binw` combination.
    :param label:
        A label used in the title of the figure
    :param figure_format:
        Format of the figure
    """

    # Read the file as a pandas Dataframe
    df = pd.read_csv(fname)

    # Set depth intervals
    if len(depth_bins) < 1:
        bins = np.arange(depth_min, depth_max+depth_binw*0.1, depth_binw)
    else:
        bins = np.array([float(a) for a in depth_bins])
        depth_max = max(bins)
        depth_min = min(bins)

    # Filter the catalogue
    df = df[(df.depth > depth_min) & (df.depth <= depth_max)]

    # Build the histogram
    hist, _ = np.histogram(df['depth'], bins=bins)

    if show or len(figure_name_out):

        # Create the figure
        fig, ax1 = plt.subplots(constrained_layout=True)
        heights = np.diff(bins)

        plt.barh(bins[:-1], width=hist, height=heights, align='edge', 
                 hatch='///', fc='none', ec='blue', alpha=0.5)
        
        ax1.set_ylim([depth_max, depth_min])
        ax1.invert_yaxis()
        ax1.grid(which='both')
        ax1.set_xlabel('Count')
        ax1.set_ylabel('Depth [km]')
        
        ax2 = ax1.twiny()
        ax2.invert_yaxis()
        ax2.set_ylim([depth_max, depth_min])
        ax2.set_xlim([0, 1.0])
        color = 'tab:red'
        ax2.set_xlabel('Normalized count', color=color)
        ax2.tick_params(axis='x', labelcolor=color)
        
        plt.barh(bins[:-1], width=hist/sum(hist), height=heights, color='none', 
                 edgecolor=color, linewidth=2.0, align='edge')

        # PMF labels
        import matplotlib.patheffects as pe
        path_effects=[pe.withStroke(linewidth=4, foreground="lightgrey")]
        for x, y in zip(hist/sum(hist), bins[:-1]+depth_binw*0.5):
            ax2.text(x, y, "{:.2f}".format(x), path_effects=path_effects)

        # Set the figure title
        ax2.set_title('Source: {:s}'.format(label), loc='left')

        # Save the figure (if requested)
        if len(figure_name_out):
            plt.savefig(figure_name_out, format=figure_format)

        # Show the figure (if requested)
        if show:
            plt.show()
        plt.close()

    return hist, bins


def analyze_hypocentral_depth(folder_subcat: str, *, depth_min: float = 0,
                              depth_max: float = 300.0, depth_binw: float = 10,
                              folder_out_figs: str = '', show: bool = False,
                              depth_bins: str = '', conf=''):
    """
    Analyses the distribution of hypocentral depths within a depth interval.
    """

    create_folder(folder_out_figs)
    path = os.path.join(folder_subcat, 'subcatalogue*.csv')
    print("Storing figures in: {:s}".format(folder_out_figs))

    if len(depth_bins) > 0:
        depth_bins = get_list(depth_bins)

    if len(conf) > 0:
        model = toml.load(conf)

    # Select point in polygon
    for fname in sorted(glob.glob(path)):

        match = re.search('.*subcatalogue_zone_(.*).csv', fname)
        source_id = match.group(1)
        figure_format = 'png'
        fmt = 'hypodepth_distribution_zone_{:s}.{:s}'
        tmp = fmt.format(source_id, figure_format)
        fname_figure_out = os.path.join(folder_out_figs, tmp)

        # Building the figure/statistics
        hist, depb = hypocentral_depth_analysis(fname, depth_min, depth_max,
                                          depth_binw, fname_figure_out,
                                          show, depth_bins, source_id,
                                          figure_format)
        
        THRESHOLD = 0.03
        if len(conf) > 0:
            
            midd = depb[:-1]+np.diff(depb)/2

            hist = hist / np.sum(hist)
            idx = hist > THRESHOLD
            hist = hist[idx]
            midd = midd[idx]

            wei = np.around(hist, 2)
            wei = wei / np.sum(wei)
            wei = np.around(wei, 2)

            swei = np.sum(wei)
            if abs(1.0-swei) > 1e-2:
                msg = "Weights do not sum to 1: {:f}\n{:s}".format(swei, fname)
                warnings.warn(msg)

            var = model['sources'][source_id] 
            tlist = []
            for w, m in zip(wei, midd):
                if w > 1e-10:
                    tlist.append([w, m])
            var['hypocenter_distribution'] = tlist

    if len(conf) > 0:
        # Saving results into the config file
        with open(conf, 'w') as fou:
            fou.write(toml.dumps(model))
            print('Updated {:s}'.format(conf))




analyze_hypocentral_depth.folder_subcat = 'The folder with the subcatalogues'
analyze_hypocentral_depth.depth_min = 'The minimum hypocentral depth [km]'
analyze_hypocentral_depth.depth_max = 'The maximum hypocentral depth [km]'
analyze_hypocentral_depth.depth_binw = 'The depth bin width [km]'
descr = "The name of the folder where to store figures"
analyze_hypocentral_depth.folder_out_figs = descr
descr = "[true/false] when true show figures on screen"
analyze_hypocentral_depth.show = descr
descr = "A string with the bins limits to be used. Overrides depth min, max, bin width"
analyze_hypocentral_depth.depth_bins = descr
descr = "A .toml formatted file. When a name is provided, it is updated automatically."
analyze_hypocentral_depth.conf = descr

if __name__ == '__main__':
    sap.run(analyze_hypocentral_depth)
