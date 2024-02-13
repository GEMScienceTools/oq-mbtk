#!/usr/bin/env python
# coding: utf-8

import toml
import copy
from openquake.baselib import sap
from openquake.wkf.utils import get_list


def main(fname_conf: str, fname_defaults: str,  use: str = [], skip: str = []):

    if len(use) > 0:
        use = get_list(use)
    if len(skip) > 0:
        skip = get_list(skip)
        
    # Parsing config
    model = toml.load(fname_conf)
    output = copy.copy(model)

    # Parsing defaults
    defaults = toml.load(fname_defaults)

    # Adding fields
    #if 'sources' not in output:
    #    output['sources'] = {}

    for src_id in defaults['sources']:
        if (len(use) and src_id not in use) or (src_id in skip):
            continue
        if src_id not in output['sources']:
            output['sources'][src_id] = {}
        for key in defaults['sources'][src_id]:
            default = defaults['sources'][src_id][key]
            output['sources'][src_id][key] = default

    # Saving results into the config file
    with open(fname_conf, 'w') as f:
        f.write(toml.dumps(output))
        print('Updated {:s}'.format(fname_conf))


descr = 'The .toml filename of configuration file'
main.fname_conf = descr
descr = 'The .toml filename with default properties'
main.fname_defaults = descr

if __name__ == '__main__':
    sap.run(main)
