#!/usr/bin/env python
# coding: utf-8

import re
import toml
import copy
from openquake.baselib import sap


def get_list(tmps):
    aa = re.split('\\,', tmps)
    aa = [re.sub('(^\\s*|\\s^)', '', a) for a in aa]
    return aa


def check_weights(tmps):
    smm = 0
    for tstr in eval(tmps):
        smm += tstr[0]
    msg = 'Weights does not sum to 1'
    assert abs(smm-1.0) < 1e-5, msg


def set_hypodepth_dist(fname_conf: str, *, use: str = '*',
                       hddstr: str = None):
    """
    Set the hypocentral depth in the model file given a pattern or a set of
    source IDs.
    """

    # Parsing config
    model = toml.load(fname_conf)
    output = copy.copy(model)

    if hddstr is None:
        hdd = model['default']['hypocenter_distribution']
    else:
        check_weights(hddstr)
        hdd = hddstr

    # Create list
    if use != "*":
        use = get_list(use)

    # Iterate over sources
    for src_id in model['sources']:
        if use == "*" or src_id in use:
            output['sources'][src_id]['hypocenter_distribution'] = hdd

    # Saving results into the config file
    with open(fname_conf, 'w') as f:
        f.write(toml.dumps(output))
        print('Updated {:s}'.format(fname_conf))


descr = 'The name of configuration file'
set_hypodepth_dist.fname_conf = descr
descr = 'Can be either a string with * or with source IDs separated by commas'
set_hypodepth_dist.use = descr
descr = 'The string with the hypodepth distribution'
set_hypodepth_dist.hddstr = descr

if __name__ == '__main__':
    sap.run(set_hypodepth_dist)
