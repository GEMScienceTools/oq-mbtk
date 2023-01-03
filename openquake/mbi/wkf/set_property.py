#!/usr/bin/env python
# coding: utf-8

import toml
import copy
from openquake.baselib import sap
from openquake.wkf.utils import get_list


def main(fname_conf: str, key: str, value, *, use: str = [], skip: str = []):

    if len(use) > 0:
        use = get_list(use)
    if len(skip) > 0:
        skip = get_list(skip)

    # Parsing config
    model = toml.load(fname_conf)
    output = copy.copy(model)

    # Iterate over sources
    for src_id in model['sources']:

        if (len(use) and src_id not in use) or (src_id in skip):
            continue

        output['sources'][src_id][key] = value

    # Saving results into the config file
    with open(fname_conf, 'w') as f:
        f.write(toml.dumps(output))
        print('Updated {:s}'.format(fname_conf))


descr = 'The name of configuration file'
main.fname_conf = descr
descr = 'Key defining the property'
main.key = descr
descr = 'A list of source IDs to be used'
main.use = descr
descr = 'A list of source IDs to be skipped'
main.skip = descr

if __name__ == '__main__':
    sap.run(main)
