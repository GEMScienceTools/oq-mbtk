#!/usr/bin/env python
# coding: utf-8

import pathlib
import cProfile
import pstats
from openquake.fnm.importer import kite_surfaces_from_geojson
from openquake.fnm.tests.rupture_fsys_test import _get_rups

HERE = pathlib.Path(__file__).parent


def run():
    """
    """

    # Set the size of subsections, get the surfaces representing sections,
    fname = HERE / '..' / 'tests' / 'data' / 'kunlun_faults.geojson'
    tmp = kite_surfaces_from_geojson(fname, 2)
    # surfs = [tmp[4], tmp[5], tmp[6], tmp[7], tmp[9], tmp[13], tmp[14], tmp[15], tmp[20]]
    # surfs = [tmp[5], tmp[6], tmp[7], tmp[9], tmp[13], tmp[15], tmp[20]]
    # surfs = [tmp[5], tmp[6], tmp[7], tmp[15], tmp[20]]
    surfs = [tmp[3], tmp[8], tmp[9]]  # large sections - should be slow
    # surfs = tmp

    subs_size = [-0.5, -1]
    key = 'threshold_distance'
    criteria = {'min_distance_between_subsections': {key: 10.}}
    _ = _get_rups(surfs, subs_size, criteria, max_connection_level=4)


if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()
    run()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(30)
