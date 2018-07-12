#!/usr/bin/env python
import sys
import numpy
import logging

from openquake.baselib import sap

from openquake.sub.create_2pt5_model import (read_profiles_csv,
                                             get_profiles_length,
                                             get_interpolated_profiles,
                                             write_edges_csv,
                                             write_profiles_csv)


def build_complex_surface(in_path, max_sampl_dist, out_path, upper_depth=0,
                          lower_depth=1000, from_id='.*', to_id='.*'):
    """
    :param str in_path:
        Folder name
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
    #
    # Check input and output folders
    if in_path == out_path:
        tmps = '\nError: the input folder cannot be also the output one\n'
        tmps += '    input : {0:s}\n'.format(in_path)
        tmps += '    output: {0:s}\n'.format(out_path)
        logging.warning(tmps.format(lab, len(allrup[lab])))
        exit(0)
    #
    # read profiles
    sps, dmin, dmax = read_profiles_csv(in_path,
                                        float(upper_depth),
                                        float(lower_depth),
                                        from_id, to_id)
    logging.info('Number of profiles: {:d}'.format(len(sps)))
    #
    # compute length of profiles
    lengths, longest_key, shortest_key = get_profiles_length(sps)
    logging.info('Longest profile (id: {:s}): {:2f}'.format(longest_key,
                                                     lengths[longest_key]))
    logging.info('Shortest profile (id: {:s}): {:2f}'.format(shortest_key,
                                                      lengths[shortest_key]))
    logging.info('Depth min: {:.2f}'.format(dmin))
    logging.info('Depth max: {:.2f}'.format(dmax))
    #
    #
    number_of_samples = numpy.ceil(lengths[longest_key] / float(max_sampl_dist))
    tmps = 'Number of subsegments for each profile: {:d}'
    logging.info(tmps.format(int(number_of_samples)))
    tmp = lengths[shortest_key]/number_of_samples
    logging.info('Shortest sampling [%s]: %.4f' % (shortest_key, tmp))
    tmp = lengths[longest_key]/number_of_samples
    logging.info('Longest sampling  [%s]: %.4f' % (longest_key, tmp))
    #
    # resampled profiles
    rsps = get_interpolated_profiles(sps, lengths, number_of_samples)
    #
    # store new profiles
    write_profiles_csv(rsps, out_path)
    #
    # store computed edges
    write_edges_csv(rsps, out_path)


def main(argv):

    p = sap.Script(build_complex_surface)
    p.arg(name='in_path', help='Path to the input folder')
    p.arg(name='max_sampl_dist', help='Maximum profile sampling distance')
    p.arg(name='out_path', help='Path to the output folder')
    p.opt(name='upper_depth', help='Upper depth')
    p.opt(name='lower_depth', help='Lower depth')
    p.opt(name='from_id', help='Index profile where to start the sampling')
    p.opt(name='to_id', help='Index profile where to stop the sampling')

    if len(argv) < 1:
        print(p.help())
    else:
        p.callfunc()


if __name__ == "__main__":
    main(sys.argv[1:])
