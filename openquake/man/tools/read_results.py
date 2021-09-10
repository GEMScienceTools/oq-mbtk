import re
import os
import numpy as np
import pandas as pd
from glob import glob
from openquake.man.tools.csv_output import read_hazard_curve_csv


def get_rlzs(folder):
    fmt = 'realizations_*.csv'
    fname = glob(os.path.join(folder, fmt))
    print('Found: {:s}'.format(fname[0]))
    m = re.match(".*_(\\d*)\\.csv$", fname[0])
    calc_id = m.group(1)
    return pd.read_csv(fname[0]), calc_id


def get_mean_hc(folder, imt):
    """
    Reads the mean hazard curve found in `folder` for a given imt

    :param folder:
        Name of the folder
    :param imt:
        Name of the imt
    :returns:
        The same parameters of the function `read_hazard_curve_csv` i.e. a
        tuple with: lon, lat, poe, header, imls
    """
    fmt = 'hazard_curve-mean-{:s}*.csv'
    fname = glob(os.path.join(folder, fmt.format(imt)))
    print('Found: {:s}'.format(fname[0]))
    return read_hazard_curve_csv(fname[0])


def get_quantile_hc(folder, imt, quantile):
    tmps = 'quantile_curve-{:s}-{:s}*.csv'.format(str(quantile), imt)
    fname = glob(os.path.join(folder, tmps))
    return read_hazard_curve_csv(fname[0])


def get_rlz_hcs(folder, imt):
    fmt = 'hazard_curve-rlz*-{:s}_*.csv'
    poes = []
    pattern = os.path.join(folder, fmt.format(imt))
    for i, fname in enumerate(sorted(glob(pattern))):
        if i == 0:
            m = re.match(".*_(\\d*)\\.csv$", fname)
            calc_id = m.group(1)
        lo, la, poe, hea, iml = read_hazard_curve_csv(fname)
        poes.append(poe)
    return lo, la, np.array(poes), hea, np.squeeze(iml), calc_id
