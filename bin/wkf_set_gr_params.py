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


def set_gr_params(fname_conf: str, *, use: str = "*", method: str = "weichert",
                  exclude: str = None):
    """
    Choose the GR parameters to be used for constructing the sources. The
    supported options are 'weichert' and 'counting'
    """

    # Parsing config
    model = toml.load(fname_conf)
    output = copy.copy(model)

    if exclude is not None and use != "*":
        raise ValueError("Please choose one between 'use' or 'exclude'")

    if use != "*":
        use = get_list(use)

    if exclude is not None:
        exclude = get_list(exclude)

    if method not in ['weichert', 'counting', 'assigned']:
        raise ValueError("The {:s} option is not supported".format(method))

    # Iterate over sources
    labb = "bgr_{:s}".format(method)
    laba = "agr_{:s}".format(method)

    for sid in model['sources']:
        if exclude is not None and sid in exclude:
            continue
        else:
            if use == "*" or sid in use:
                print("sid: {:>6s} {:s}".format(sid, method))
                output['sources'][sid]['bgr'] = output['sources'][sid][labb]
                output['sources'][sid]['agr'] = output['sources'][sid][laba]

    # Saving results into the config file
    with open(fname_conf, 'w') as f:
        f.write(toml.dumps(output))
        print('Updated {:s}'.format(fname_conf))


descr = 'The name of configuration file'
set_gr_params.fname_conf = descr
descr = 'Can be either a string with * or with source IDs separated by commas'
set_gr_params.use = descr
descr = 'The label with the method used to infer these parameters: '
descr += 'e.g. weichert, counting'
set_gr_params.method = descr
descr = 'A string with source IDs separated by commas'
set_gr_params.exclude = descr

if __name__ == '__main__':
    sap.run(set_gr_params)
