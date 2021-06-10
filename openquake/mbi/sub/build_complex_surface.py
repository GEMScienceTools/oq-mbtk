#!/usr/bin/env python
# coding: utf-8

from openquake.baselib import sap
from openquake.sub.build_complex_surface import build_complex_surface


def build_complex_fault_surface(in_path, max_sampl_dist, out_path,
                                upper_depth=0, lower_depth=1000, *,
                                from_id='.*', to_id='.*'):
    """
    Builds edges that can be used to generate a complex fault surface
    starting from a set of profiles
    """
    build_complex_surface(in_path, max_sampl_dist, out_path, upper_depth,
                          lower_depth, from_id, to_id)


build_complex_fault_surface.in_path = 'Path to the input folder'
msg = 'Maximum profile sampling distance'
build_complex_fault_surface.max_sampl_dist = msg
build_complex_fault_surface.out_path = 'Path to the output folder'
build_complex_fault_surface.upper_depth = 'Upper depth'
build_complex_fault_surface.lower_depth = 'lower depth'
msg = 'Index profile where to start the sampling'
build_complex_fault_surface.from_id = msg
build_complex_fault_surface.to_id = 'Index profile where to stop the sampling'

#if __name__ == "__main__":
#    sap.run(main)
