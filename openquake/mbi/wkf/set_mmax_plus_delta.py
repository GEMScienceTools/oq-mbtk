#!/usr/bin/env python
# coding: utf-8

import re
import toml
import copy
from openquake.baselib import sap
from openquake.wkf.utils import get_list


def set_mmax_plus_delta(fname_conf: str, mmax_delta: float, min_mmax: None,
                        use: str = [], skip: str = []):

    if len(use) > 0:
        use = get_list(use)
    if len(skip) > 0:
        skip = get_list(skip)

    # Parsing config
    model = toml.load(fname_conf)
    output = copy.copy(model)

    if min_mmax is not None:
        min_mmax = float(min_mmax)
    mmax_delta = float(mmax_delta)

    # Iterate over sources
    for src_id in model['sources']:

        if (len(use) and src_id not in use) or (src_id in skip):
            continue

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


def main(fname_conf: str, mmax_delta: float, min_mmax: None, *, use: str=[],
         skip: str=[]):
    set_mmax_plus_delta(fname_conf, mmax_delta, min_mmax, use, skip)


descr = 'The name of configuration file'
main.fname_conf = descr
descr = 'The increment to apply to mmax observed'
main.mmax_delta = descr
descr = 'The minimum mmax assigned'
main.min_mmax = descr
descr = 'A list of source IDs to be used'
main.use = descr
descr = 'A list of source IDs to be skipped'
main.skip = descr

if __name__ == '__main__':
    sap.run(main)
