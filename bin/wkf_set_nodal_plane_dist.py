#!/usr/bin/env python
# coding: utf-8

import re
import toml
import copy
from openquake.baselib import sap
from wkf_set_hypodepth_dist import check_weights


def get_list(tmps):
    aa = re.split('\\,', tmps)
    aa = [re.sub('(^\\s*|\\s^)', '', a) for a in aa]
    return aa


def set_nodal_plane_dist(fname_conf: str, *, use: str = '*',
                         npdstr: str = None):
    """
    Set the hypocentral depth in the model file given a pattern or a set of
    source IDs.
    """

    # Parsing config
    model = toml.load(fname_conf)
    output = copy.copy(model)

    if npdstr is None:
        npd = model['default']['hypocenter_distribution']
    else:
        check_weights(npdstr)
        npd = npdstr

    # Create list
    if use != "*":
        use = get_list(use)

    # Iterate over sources
    for src_id in model['sources']:
        if use == "*" or src_id in use:
            output['sources'][src_id]['hypocenter_distribution'] = npd

    # Saving results into the config file
    with open(fname_conf, 'w') as f:
        f.write(toml.dumps(output))
        print('Updated {:s}'.format(fname_conf))


descr = 'The name of configuration file'
set_nodal_plane_dist.fname_conf = descr
descr = 'Can be either a string with * or with source IDs separated by commas'
set_nodal_plane_dist.use = descr
descr = 'The string with the nodal plane distribution'
set_nodal_plane_dist.npdstr = descr

if __name__ == '__main__':
    sap.run(set_nodal_plane_dist)
