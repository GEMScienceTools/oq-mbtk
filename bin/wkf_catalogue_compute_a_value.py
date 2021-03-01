#!/usr/bin/env python
# coding: utf-8

import os
from glob import glob
import toml
import numpy
import pandas as pd
from openquake.wkf.utils import _get_src_id, create_folder, get_list
from openquake.baselib import sap
from openquake.mbt.tools.model_building.dclustering import _add_defaults
from openquake.mbt.tools.model_building.plt_tools import _load_catalogue
from openquake.hmtk.seismicity.occurrence.utils import get_completeness_counts


def get_exrs(df: pd.DataFrame, bgr: str):
    """
    Computes annual exceedence rates using eq. 10 in Weichert (1980; BSSA).

    :param df:
        A table
    :param bgr:
        The b-value of the Gutenberg-Richer relatioship
    :returns:
        Annual exceedance rate for all the magnitude values in the dataframe.
    """
    beta = bgr * numpy.log(10.0)
    exr = []
    for m in df.mag:
        cond = (df.nobs > 0) & (df.mag >= m)
        N = sum(df.nobs[cond])
        tmp = numpy.exp(-beta*df.mag[cond])
        num = numpy.sum(tmp)
        den = numpy.sum(tmp*df.deltaT[cond])
        exr.append(N * num / den)
    return numpy.array(exr)


def get_agr(mag, bgr, rate):
    """
    :param mag:
        The magnitude to which the parameter `rate` refers to. If the rates
        are binned this should be the lower limit of the bin with `mag`
    :param bgr:
        The b-value of the Gutenberg-Richer relatioship
    :param rate:
        The rate of occurrence of earthquakes larger than `mag`
    :returns:
        The a-value of the GR relationship
    """
    return numpy.log10(rate) + bgr * (mag)


def compute_a_value(fname_input_pattern: str, bval: float, fname_config: str,
                    folder_out: str, use: str = ''):
    """
    This function assignes an a-value to each source with a file selected by
    the provided `fname_input_pattern`.
    """

    if len(use) > 0:
        use = get_list(use)

    # Processing input parameters
    bval = float(bval)
    create_folder(folder_out)

    if isinstance(fname_input_pattern, str):
        # fname_list = [f for f in glob(fname_input_pattern)]
        fname_list = glob(fname_input_pattern)
    else:
        fname_list = fname_input_pattern

    # Parsing config
    model = toml.load(fname_config)
    binw = model['bin_width']

    # Processing files
    for fname in sorted(fname_list):

        # Get source ID
        src_id = _get_src_id(fname)
        if len(use) > 0 and src_id not in use:
            continue
        print(fname)

        if 'sources' in model:
            if (src_id in model['sources'] and
                    'completeness_table' in model['sources'][src_id]):
                tmp = model['sources'][src_id]['completeness_table']
                ctab = numpy.array(tmp)
            else:
                ctab = numpy.array(model['default']['completeness_table'])

        # Processing catalogue
        tcat = _load_catalogue(fname)
        tcat = _add_defaults(tcat)
        tcat.data["dtime"] = tcat.get_decimal_time()
        cent_mag, t_per, n_obs = get_completeness_counts(tcat, ctab, binw)

        df = pd.DataFrame()
        df['mag'] = cent_mag
        df['deltaT'] = t_per
        df['nobs'] = n_obs
        fout = os.path.join(folder_out, 'occ_count_zone_{:s}'.format(src_id))
        df.to_csv(fout, index=False)

        # Computing GR a
        if 'sources' not in model:
            model['sources'] = {}
        if src_id not in model['sources']:
            model['sources'][src_id] = {}

        exrs = get_exrs(df, bval)
        aval = get_agr(df.mag[0]-binw/2, bval, exrs[0])

        tmp = "{:.5e}".format(aval)
        model['sources'][src_id]['agr_counting'] = float(tmp)

        tmp = "{:.5e}".format(bval)
        model['sources'][src_id]['bgr_counting'] = float(tmp)

    # Saving results into the config file
    with open(fname_config, 'w') as fou:
        fou.write(toml.dumps(model))
        print('Updated {:s}'.format(fname_config))


descr = 'Pattern to select input files or list of files'
compute_a_value.fname_input_pattern = descr
compute_a_value.bval = 'GR b-value'
descr = 'Name of the .toml file with configuration parameters'
compute_a_value.fname_config = descr
descr = 'Name of the output folder where to store occurrence counts'
compute_a_value.folder_out = descr
compute_a_value.use = 'A list with the ID of sources that should be considered'

if __name__ == '__main__':
    sap.run(compute_a_value)
