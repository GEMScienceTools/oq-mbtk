#!/usr/bin/env python
# coding: utf-8

import toml
import copy
from openquake.baselib import sap
from openquake.wkf.utils import get_list


def main(fname_conf: str, rmag: float, ratio:float, *, use: str = [],
         skip: str = []):

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

        dat = output['sources'][src_id]
        rmag = float(rmag)
        ratio = float(ratio)
        rmag_rate = 10**(dat['agr'] - dat['bgr']*rmag)

        output['sources'][src_id]['rmag'] = rmag
        output['sources'][src_id]['rmag_rate'] = rmag_rate
        output['sources'][src_id]['rmag_rate_sig'] = rmag_rate * ratio

    # Saving results into the config file
    with open(fname_conf, 'w') as f:
        f.write(toml.dumps(output))
        print('Updated {:s}'.format(fname_conf))


descr = 'The name of configuration file'
main.fname_conf = descr
descr = 'The reference magnitude'
main.rmag = descr
descr = 'The ratio between the std of the rate and the rate'
main.ratio = descr
descr = 'A list of source IDs to be used'
main.use = descr
descr = 'A list of source IDs to be skipped'
main.skip = descr

if __name__ == '__main__':
    sap.run(main)
