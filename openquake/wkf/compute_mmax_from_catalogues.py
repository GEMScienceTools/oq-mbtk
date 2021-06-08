#!/usr/bin/env python
# coding: utf-8

import toml
import numpy as np
from glob import glob
from openquake.wkf.utils import _get_src_id
from openquake.mbt.tools.model_building.plt_tools import _load_catalogue


def compute_mmax(fname_input_pattern: str, fname_config: str, label: str):
    """
    This function assignes an mmax value to each source with a catalogue
    file as selected by the provided `fname_input_pattern`.
    """

    if isinstance(fname_input_pattern, str):
        fname_list = glob(fname_input_pattern)
    else:
        fname_list = fname_input_pattern

    # Parsing config
    model = toml.load(fname_config)

    # Processing files
    for fname in sorted(fname_list):

        src_id = _get_src_id(fname)

        # Processing catalogue
        tcat = _load_catalogue(fname)

        if tcat is None or len(tcat.data['magnitude']) < 2:
            continue

        tmp = "{:.5e}".format(np.max(tcat.data['magnitude']))
        model['sources'][src_id]['mmax_{:s}'.format(label)] = float(tmp)

    # Saving results into the config file
    with open(fname_config, 'w') as fou:
        fou.write(toml.dumps(model))
        print('Updated {:s}'.format(fname_config))
