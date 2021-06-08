#!/usr/bin/env python

"""
Module :mod:`openquake.sub.build_complex_surface` creates a complex fault
surface from a set of profiles
"""

import os
import sys
import logging
import numpy
from openquake.baselib import sap
from openquake.sub.create_2pt5_model import (read_profiles_csv,
                                             get_profiles_length,
                                             get_interpolated_profiles,
                                             write_edges_csv,
                                             write_profiles_csv)

logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')


def build_complex_surface(in_path, max_sampl_dist, out_path, upper_depth=0,
                          lower_depth=1000, from_id='.*', to_id='.*'):
    """
    :param str in_path:
        Folder name. It contains files with the prefix 'cs_'
    :param str float max_sampl_dist:
        Sampling distance [km]
    :param str out_path:
        Folder name
    :param float upper_depth:
        The depth above which we cut the profiles
    :param float lower_depth:
        The depth below which we cut the profiles
    :param str from_id:
        The ID of the first profile to be considered
    :param str to_id:
        The ID of the last profile to be considered
    """

    # Check input and output folders
    if in_path == out_path:
        tmps = '\nError: the input folder cannot be also the output one\n'
        tmps += '    input : {0:s}\n'.format(in_path)
        tmps += '    output: {0:s}\n'.format(out_path)
        sys.exit()

    # Read the profiles
    sps, dmin, dmax = read_profiles_csv(in_path,
                                        float(upper_depth),
                                        float(lower_depth),
                                        from_id, to_id)

    # Check
    logging.info('Number of profiles: {:d}'.format(len(sps)))
    if len(sps) < 1:
        fmt = 'Did not find cross-sections in {:s}\n exiting'
        msg = fmt.format(os.path.abspath(in_path))
        logging.error(msg)
        sys.exit(0)

    # Compute length of profiles
    lengths, longest_key, shortest_key = get_profiles_length(sps)
    logging.info('Longest profile (id: {:s}): {:2f}'.format(
        longest_key, lengths[longest_key]))
    logging.info('Shortest profile (id: {:s}): {:2f}'.format(
        shortest_key, lengths[shortest_key]))
    logging.info('Depth min: {:.2f}'.format(dmin))
    logging.info('Depth max: {:.2f}'.format(dmax))

    # Info
    number_of_samples = numpy.ceil(lengths[longest_key] / float(
        max_sampl_dist))
    tmps = 'Number of subsegments for each profile: {:d}'
    logging.info(tmps.format(int(number_of_samples)))
    tmp = lengths[shortest_key]/number_of_samples
    logging.info('Shortest sampling [%s]: %.4f' % (shortest_key, tmp))
    tmp = lengths[longest_key]/number_of_samples
    logging.info('Longest sampling  [%s]: %.4f' % (longest_key, tmp))

    # Resampled profiles
    rsps = get_interpolated_profiles(sps, lengths, number_of_samples)

    # Store new profiles
    write_profiles_csv(rsps, out_path)

    # Store computed edges
    write_edges_csv(rsps, out_path)


build_complex_surface.in_path = 'Path to the input folder'
build_complex_surface.max_sampl_dist = 'Maximum profile sampling distance'
build_complex_surface.out_path = 'Path to the output folder'
build_complex_surface.upper_depth = 'Upper depth'
build_complex_surface.lower_depth = 'lower depth'
build_complex_surface.from_id = 'Index profile where to start the sampling'
build_complex_surface.to_id = 'Index profile where to stop the sampling'

if __name__ == "__main__":
    sap.run(build_complex_surface)
