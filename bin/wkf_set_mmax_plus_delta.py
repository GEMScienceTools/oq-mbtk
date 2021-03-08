#!/usr/bin/env python
# coding: utf-8

import re
import os
import h3
import json
import toml
import copy
import pandas as pd
import numpy as np
import geopandas as gpd
from openquake.baselib import sap


def set_mmax_plus_delta(fname_conf: str, mmax_delta: float, min_mmax: None):

    # Parsing config
    model = toml.load(fname_conf)
    output = copy.copy(model)

    if min_mmax is not None:
        min_mmax = float(min_mmax)
    mmax_delta = float(mmax_delta)

    # Iterate over sources
    for src_id in model['sources']:
        mmax = 0.
        for param in model['sources'][src_id]:
            if re.search('^mmax_', param):
                mmax = max(mmax, model['sources'][src_id][param])
        if min_mmax is not None:
            mmax = max(mmax, min_mmax)
        output['sources'][src_id]['mmax'] = mmax + mmax_delta

    # Saving results into the config file
    with open(fname_conf, 'w') as f:
        f.write(toml.dumps(output))
        print('Updated {:s}'.format(fname_conf))


descr = 'The name of configuration file'
set_mmax_plus_delta.fname_conf = descr
descr = 'The increment to apply to mmax observed'
set_mmax_plus_delta.mmax_delta = descr
descr = 'The minimum mmax assigned'
set_mmax_plus_delta.min_mmax = descr

if __name__ == '__main__':
    sap.run(set_mmax_plus_delta)
