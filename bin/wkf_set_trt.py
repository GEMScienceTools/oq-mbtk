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

def get_list(tmps):
    aa = re.split('\\,', tmps)
    aa = [re.sub('(^\\s*|\\s^)', '', a) for a in aa]
    return aa

def set_trt(fname_conf: str, use: str, trt: str):

    # Parsing config
    model = toml.load(fname_conf)
    output = copy.copy(model)

    if use != "*":
        use = get_list(use)

    # Iterate over sources
    for src_id in model['sources']:
        if use == "*" or src_id in use:
            output['sources'][src_id]['tectonic_region_type'] = trt

    # Saving results into the config file
    with open(fname_conf, 'w') as f:
        f.write(toml.dumps(output))
        print('Updated {:s}'.format(fname_conf))

def main(fname_conf: str, use: str, trt: str):
    set_trt(fname_conf, use, trt)

descr = 'The name of configuration file'
main.fname_conf = descr
descr = 'Can be either a string with * or with source IDs separated by commas'
main.use = descr
descr = 'The TRT label e.g. Active Shallow Crust'
main.method = descr

if __name__ == '__main__':
    sap.run(main)
