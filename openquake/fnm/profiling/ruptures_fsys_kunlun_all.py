#!/usr/bin/env python
# coding: utf-8

import pathlib
from openquake.fnm.importer import kite_surfaces_from_geojson
from openquake.fnm.tests.rupture_fsys_test import _get_rups

HERE = pathlib.Path(__file__).parent


def run():
    # Set the size of subsections, get the surfaces representing sections,
    fname = HERE / '..' / 'tests' / 'data' / 'kunlun_faults.geojson'
    surfs = kite_surfaces_from_geojson(fname, 2)

    subs_size = [-0.5, -1]
    threshold = 10.0
    rups = _get_rups(surfs, subs_size, threshold, max_connection_level=2)


if __name__ == '__main__':
    import cProfile, pstats
    profiler = cProfile.Profile()
    profiler.enable()
    run()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()
