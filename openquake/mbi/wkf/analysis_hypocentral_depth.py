#!/usr/bin/env python
# coding: utf-8

import re
import os
import glob
import toml
import warnings
import numpy as np
from openquake.baselib import sap
from openquake.wkf.utils import create_folder, get_list
from openquake.wkf.seismicity.hypocentral_depth import (
    hypocentral_depth_analysis)


def analyze_hypocentral_depth(folder_subcat: str, *, depth_min: float = 0,
                              depth_max: float = 300.0, depth_binw: float = 10,
                              folder_out_figs: str = '', show: bool = False,
                              depth_bins: str = '', conf='', use: str = [],
                              skip: str = []):
    """
    Analyses the distribution of hypocentral depths within a depth interval.
    """

    if len(use) > 0:
        use = get_list(use)
    if len(skip) > 0:
        skip = get_list(skip)

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
        src_id = match.group(1)

        if (len(use) and src_id not in use) or (src_id in skip):
            continue

        figure_format = 'png'
        fmt = 'hypodepth_distribution_zone_{:s}.{:s}'
        tmp = fmt.format(src_id, figure_format)
        fname_figure_out = os.path.join(folder_out_figs, tmp)

        # Building the figure/statistics
        hist, depb = hypocentral_depth_analysis(
            fname, depth_min, depth_max, depth_binw, fname_figure_out, show,
            depth_bins, src_id, figure_format)

        if hist is None:
            continue

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
                # Fixing
                wei[-1] += 1.0-swei
                swei = np.sum(wei)
                if abs(1.0-swei) > 1e-2:
                    fmt = "Weights do not sum to 1: {:f}\n{:s}"
                    msg = fmt.format(swei, fname)
                    warnings.warn(msg)
                    exit()

            var = model['sources'][src_id]
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
descr = "String with the bins limits. Overrides depth-min, depth-max, "
descr += "depth-binw"
analyze_hypocentral_depth.depth_bins = descr
descr = "A .toml file. When provided, updated with new info"
analyze_hypocentral_depth.conf = descr
descr = "Source IDs to use"
analyze_hypocentral_depth.use = descr
descr = "Source IDs to skip"
analyze_hypocentral_depth.skip = descr

if __name__ == '__main__':
    sap.run(analyze_hypocentral_depth)
