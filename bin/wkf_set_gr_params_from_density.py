#!/usr/bin/env python
# coding: utf-8

import re
import toml
import copy
import numpy as np
import geopandas as gpd
from openquake.baselib import sap


def get_list(tmps):
    aa = re.split('\\,', tmps)
    aa = [re.sub('(^\\s*|\\s^)', '', a) for a in aa]
    return aa


def set_gr_params_from_density(fname_conf: str, fname_shp: str, sid: str,
                               agr_per_1m_sqkm: float, bgr: float):
    """
    Compute the aGR for the sources given the density of earthquakes per
    100.000 km2
    """

    # Parsing config
    model = toml.load(fname_conf)
    output = copy.copy(model)

    # Iterate over sources
    method = 'assigned'
    labb = "bgr_{:s}".format(method)
    laba = "agr_{:s}".format(method)

    # Calculate the area in km2
    poly = gpd.read_file(fname_shp)
    poly = poly.loc[poly['id'] == sid]
    poly = poly.to_crs(32633)
    poly['area'] = poly['geometry'].area / 10**6

    agr = np.log10(10**agr_per_1m_sqkm * poly['area'].iloc[0] / 1e5)

    if sid not in output['sources']:
        output['sources'][sid] = {}
    output['sources'][sid][labb] = float(np.round(bgr, 5))
    output['sources'][sid][laba] = float(np.round(agr, 5))

    # Saving results into the config file
    with open(fname_conf, 'w') as f:
        f.write(toml.dumps(output))
        print('Updated {:s}'.format(fname_conf))


fun = set_gr_params_from_density
descr = 'The name of configuration file'
fun.fname_conf = descr
descr = 'Can be either a string with * or with source IDs separated by commas'
fun.use = descr
descr = 'The label with the method used to infer these parameters: '
descr += 'e.g. weichert, counting'
fun.method = descr
descr = 'A string with source IDs separated by commas'
fun.skip = descr

if __name__ == '__main__':
    sap.run(set_gr_params_from_density)
