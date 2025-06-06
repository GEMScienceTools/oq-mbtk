#!/usr/bin/env python
# coding: utf-8

import os
import sys

from openquake.baselib import sap
from openquake.sub.build_complex_surface import build_complex_surface


def main(in_path, max_sampl_dist, out_path, upper_depth=0,
         lower_depth=1000, *, from_id='.*', to_id='.*'):
    """
    Builds edges that can be used to generate a complex fault surface
    starting from a set of profiles
    """
    # check if output folder is empty
    if os.path.exists(out_path):
        tmps = f'\nError: {out_path} already exists and contains profiles and '
        tmps += '\n edges from a previous run! Specify empty output directory.'
        print(tmps)
        sys.exit()
    # otherwise create directory
    else:
        os.makedirs(out_path)

    build_complex_surface(in_path, max_sampl_dist, out_path, upper_depth,
                          lower_depth, from_id, to_id)


main.in_path = 'Path to the input folder'
main.max_sampl_dist = 'Maximum profile sampling distance'
main.out_path = 'Path to the output folder'
main.upper_depth = 'Upper depth'
main.lower_depth = 'lower depth'
main.from_id = 'Index profile where to start the sampling'
main.to_id = 'Index profile where to stop the sampling'

if __name__ == "__main__":
    sap.run(main)
