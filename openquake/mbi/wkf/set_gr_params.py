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


def set_gr_params(fname_conf: str, use: str = "*", method: str = "weichert",
                  exclude: str = None, only_ab: bool = False):
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

    if method not in ['weichert', 'counting', 'basel']:
        raise ValueError("The {:s} option is not supported".format(method))

    # Iterate over sources
    labb = "bgr_{:s}".format(method)
    laba = "agr_{:s}".format(method)
    labas = "agr_sig_{:s}".format(method)
    labbs = "bgr_sig_{:s}".format(method)

    for src_id in model['sources']:
        if exclude is not None and src_id in exclude:
            continue
        if use != "*" and src_id not in use:
            continue
        else:
            print("src_id:", src_id, " ", method)
            if use == "*" or src_id in use:
                output['sources'][src_id]['bgr'] = \
                    output['sources'][src_id][labb]
                output['sources'][src_id]['agr'] = \
                    output['sources'][src_id][laba]
                if not only_ab:
                    output['sources'][src_id]['agr_sig'] = \
                        output['sources'][src_id][labas]
                    output['sources'][src_id]['bgr_sig'] = \
                        output['sources'][src_id][labbs]

    # Saving results into the config file
    with open(fname_conf, 'w') as f:
        f.write(toml.dumps(output))
        print('Updated {:s}'.format(fname_conf))


def main(fname_conf: str, *, use: str = "*", method: str = "weichert",
         exclude: str = None, only_ab: bool = False):
    set_gr_params(fname_conf, use, method, exclude, only_ab)


descr = 'The name of configuration file'
main.fname_conf = descr
descr = 'Can be either a string with * or with source IDs separated by commas'
main.use = descr
descr = 'The label with the method used to infer these parameters: '
descr += 'e.g. weichert, counting'
main.method = descr
descr = 'A string with source IDs separated by commas'
main.skip = descr
descr = 'Only a and b'
main.only_ab = descr


if __name__ == '__main__':
    sap.run(main)
