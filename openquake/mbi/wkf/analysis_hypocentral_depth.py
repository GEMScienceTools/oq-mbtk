#!/usr/bin/env python
# coding: utf-8

import re
import os
import glob
import toml
import warnings
import numpy as np
import pandas as pd
from openquake.baselib import sap
from openquake.wkf.utils import create_folder, get_list
from openquake.wkf.seismicity.hypocentral_depth import (
    hypocentral_depth_analysis)


def analyze_hypocentral_depth(folder_subcat: str, depth_min: float = 0,
                              depth_max: float = 300.0, depth_binw: float = 10,
                              folder_out_figs: str = '', show: bool = False,
                              depth_bins: str = '', conf='', use: str = [],
                              skip: str = [], writecsv: bool = True):
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

    if writecsv:
        hy_out = folder_out_figs.replace('figs','dat')
        if not os.path.exists(hy_out):
            os.makedirs(hy_out)
        hy_out_fi = os.path.join(hy_out, f'hc_{src_id}.csv')
        pd.DataFrame({'depth': midd, 'weight': wei}).to_csv(hy_out_fi, index=False)

    if len(conf) > 0:
        # Saving results into the config file
        with open(conf, 'w') as fou:
            fou.write(toml.dumps(model))
            print('Updated {:s}'.format(conf))


def main(folder_subcat: str, *, depth_min: float = 0,
         depth_max: float = 300.0, depth_binw: float = 10,
         folder_out_figs: str = '', show: bool = False,
         depth_bins: str = '', conf='', use: str = [],
         skip: str = [], writecsv: bool = True):
    """
    Analyses the distribution of hypocentral depths within a depth interval.
    """
    analyze_hypocentral_depth(folder_subcat, depth_min, depth_max, depth_binw,
                              folder_out_figs, show, depth_bins, conf, use,
                              skip, writecsv)


main.folder_subcat = 'The folder with the subcatalogues'
main.depth_min = 'The minimum hypocentral depth [km]'
main.depth_max = 'The maximum hypocentral depth [km]'
main.depth_binw = 'The depth bin width [km]'
descr = "The name of the folder where to store figures"
main.folder_out_figs = descr
descr = "[true/false] when true show figures on screen"
main.show = descr
descr = "String with the bins limits. Overrides depth-min, depth-max, "
descr += "depth-binw"
main.depth_bins = descr
descr = "A .toml file. When provided, updated with new info"
main.conf = descr
descr = "Source IDs to use"
main.use = descr
descr = "Source IDs to skip"
main.skip = descr
descr = 'Write outputs to csv files as well as config'
main.writecsv = descr

if __name__ == '__main__':
    sap.run(main)
