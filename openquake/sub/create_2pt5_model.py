#!/usr/bin/env python
import os
import re
import sys
import glob
import numpy

from pyproj import Geod
from openquake.hazardlib.geo.geodetic import distance


def get_profiles_length(sps):
    """
    :parameter dict sps:
        A dictionary containing the subduction profiles
    :returns:
        A dictionary where key is the ID of the profile and value is the length
        and, a string identifying the longest profile
    """
    lengths = {}
    longest_key = None
    shortest_key = None
    longest_length = 0.
    shortest_length = 1e10
    for key in sorted(sps.keys()):
        dat = sps[key]
        total_length = 0
        for idx in range(0, len(dat)-1):
            dst = distance(dat[idx, 0], dat[idx, 1], dat[idx, 2],
                           dat[idx+1, 0], dat[idx+1, 1], dat[idx+1, 2])
            total_length += dst
        lengths[key] = total_length
        if longest_length < total_length:
            longest_length = total_length
            longest_key = key
        if shortest_length > total_length:
            shortest_length = total_length
            shortest_key = key
    return lengths, longest_key, shortest_key


def get_interpolated_profiles(sps, lengths, number_of_samples):
    """
    :parameter dict sps:
        A dictionary containing the subduction profiles key is a string and
        value is an instance of :class:`numpy.ndarray`
    :parameter dict lengths:
        A dictionary containing the subduction profiles lengths
    :parameter float number_of_samples:
        Number of subsegments to be created
    :returns:
        A dictionary
    """
    ssps = {}
    for key in sorted(sps.keys()):
        #
        # calculate the sampling distance
        samp = lengths[key] / number_of_samples
        #
        # set data for the profile
        dat = sps[key]
        #
        # projecting profile coordinates
        g = Geod(ellps='WGS84')
        #
        # horizontal 'slope'
        az_prof, _, _ = g.inv(dat[0, 0], dat[0, 1], dat[-1, 0], dat[-1, 1])
        #
        # initialise
        idx = 0
        cdst = 0
        spro = [[dat[0, 0], dat[0, 1], dat[0, 2]]]
        #
        # process the segments composing the profile
        while idx < len(dat)-1:
            #
            # segment length
            _, _, dst = g.inv(dat[idx, 0], dat[idx, 1],
                              dat[idx+1, 0], dat[idx+1, 1])
            dst /= 1e3
            dst = (dst**2 + (dat[idx, 2] - dat[idx+1, 2])**2)**.5
            #
            # calculate total distance i.e. cumulated + new segment
            total_dst = cdst + dst
            #
            # number of new points
            num_new_points = int(numpy.floor(total_dst/samp))
            #
            # take samples if possible
            if num_new_points > 0:
                dipr = numpy.arcsin((dat[idx+1, 2]-dat[idx, 2])/dst)
                hfact = numpy.cos(dipr)
                vfact = numpy.sin(dipr)
                #
                #
                for i in range(0, num_new_points):
                    tdst = (i+1) * samp - cdst
                    hdst = tdst * hfact
                    vdst = tdst * vfact
                    # tlo, tla = p((x[idx] + hdst*xfact)*1e3,
                    #              (y[idx] + hdst*yfact)*1e3, inverse=True)
                    tlo, tla, _ = g.fwd(dat[idx, 0], dat[idx, 1], az_prof,
                                        hdst*1e3)
                    spro.append([tlo, tla, dat[idx, 2]+vdst])
                    #
                    # check that the h and v distances are coherent with
                    # the original distance
                    assert abs(tdst-(hdst**2+vdst**2)**.5) < 1e-4
                    #
                    # check distance with the previous point and depths Vs
                    # previous points
                    if i > 0:
                        check = distance(tlo, tla, dat[idx, 2]+vdst,
                                         spro[-2][0], spro[-2][1], spro[-2][2])
                        if abs(check - samp) > samp*0.15:
                            msg = 'Distance between consecutive points'
                            msg += ' is incorrect: {:.3f} {:.3f}'.format(check,
                                                                         samp)
                            raise ValueError(msg)
                        # new depth larger than previous
                        if numpy.any(numpy.array(spro)[:-1, 2] > spro[-1][2]):
                            raise ValueError('')

                #
                # new distance left over
                cdst = (dst + cdst) - num_new_points * samp
            else:
                cdst += dst
            #
            # updating index
            idx += 1
        #
        # Saving results
        if len(spro):
            ssps[key] = numpy.array(spro)
        else:
            print('length = 0')
    return ssps


def read_profiles_csv(foldername, upper_depth=0, lower_depth=1000,
                      from_id=".*", to_id=".*"):
    """
    :param str foldername:
        The name of the folder containing the set of digitized profiles
    :param float upper_depth:
        The depth from where to cut profiles
    :param float lower_depth:
        The depth until where to sample profiles
    :param str from_id:
        The profile key from where to read profiles (included)
    :param str to_id:
        The profile key until where to read profiles (included)
    """
    dmin = +1e100
    dmax = -1e100
    sps = {}
    #
    # reading files
    read_file = False
    for filename in sorted(glob.glob(os.path.join(foldername, 'cs*.csv'))):
        #
        # get the filename ID
        sid = re.sub('^cs_', '', re.split('\.', os.path.basename(filename))[0])
        if not re.search('[a-zA-Z]', sid):
            sid = '%03d' % int(sid)
        if not from_id == '.*' and not re.search('[a-zA-Z]', from_id):
            from_id = '%03d' % int(from_id)
        if not to_id == '.*' and not re.search('[a-zA-Z]', to_id):
            to_id = '%03d' % int(to_id)
        #
        # check the file key
        if (from_id == '.*') and (to_id == '.*'):
            read_file = True
        elif (from_id == '.*') and (sid <= to_id):
            read_file = True
        elif (sid >= from_id) and (to_id == '.*'):
            read_file = True
        elif (sid >= from_id) and (sid <= to_id):
            read_file = True
        else:
            read_file = False
        #
        # reading data
        if read_file:
            tmpa = numpy.loadtxt(filename)
            #
            # selecting depths within the defined range
            j = numpy.nonzero((tmpa[:, 2] >= upper_depth) &
                              (tmpa[:, 2] <= lower_depth))
            #
            # upper depth
            pntt = False
            if len(j[0]) > 1 and min(j[0]) == 0:
                # start from top
                pass
            elif max(tmpa[:, 2]) < upper_depth:
                continue
            else:
                idx = min(j[0])
                pntt = _get_point_at_depth(tmpa[idx-1, :], tmpa[idx, :],
                                           upper_depth)
            #
            # upper depth
            pntb = False
            if len(j[0]) > 1 and max(j[0]) == len(tmpa[:, 2])-1:
                # reached bottom
                pass
            else:
                idx = max(j[0])
                pntb = _get_point_at_depth(tmpa[idx, :], tmpa[idx+1, :],
                                           lower_depth)
            #
            # final profile
            if len(j[0]) > 1:
                tmpl = tmpa[j[0], :].tolist()
                if pntb:
                    tmpl.append(pntb)
                if pntt:
                    tmpl = [pntt] + tmpl
                #
                # updating the output array for the current profile
                sps[sid] = numpy.array(tmpl)
                dmin = min(min(sps[sid][:, 2]), dmin)
                dmax = max(max(sps[sid][:, 2]), dmax)
    return sps, dmin, dmax


def _get_point_at_depth(coo1, coo2, depth):
    g = Geod(ellps='WGS84')
    az12, az21, dist = g.inv(coo1[0], coo1[1], coo2[0], coo2[1])
    grad = (dist*1e-3) / (coo2[2] - coo1[2])
    dx = (depth - coo1[2]) * grad * 1e3
    lon, lat, _ = g.fwd(coo1[0], coo1[1], az12, dx)
    return [lon, lat, depth]


def write_profiles_csv(sps, foldername):
    """
    :parameter dic sps:
        A dictionary with the sampled profiles
    :parameter str foldername:
        The name of the folder where we write the files with the interpolated
        profiles
    """
    if not os.path.exists(foldername):
        os.mkdir(foldername)
    for key in sorted(sps):
        dat = numpy.array(sps[key])
        fname = os.path.join(foldername, 'cs_%s.csv' % (key))
        numpy.savetxt(fname, dat)


def write_edges_csv(sps, foldername):
    """
    :parameter dic sps:
        A dictionary where keys are the profile labels and values are
        :class:`numpy.ndarray` instances
    :parameter str foldername:
        The name of the file which contains the interpolated profiles
    """
    if not os.path.exists(foldername):
        os.mkdir(foldername)
    #
    # run for all the edges i.e. number of
    max_num = len(sps[list(sps.keys())[0]])
    for idx in range(0, max_num-1):
        dat = []
        for key in sorted(sps):
            dat.append(sps[key][idx, :])
        fname = os.path.join(foldername, 'edge_%03d.csv' % (idx))
        numpy.savetxt(fname, numpy.array(dat))


def main(argv):
    """
    argv[0] - Folder name
    argv[1] - Sampling distance [km]
    argv[2] - Output folder name
    argv[3] - Maximum sampling distance
    """
    in_path = os.path.abspath(argv[0])
    out_path = os.path.abspath(argv[2])
    #
    # Check input
    if len(argv) < 3:
        tmps = 'Usage: create_2pt5_model.py <in_folder>'
        tmps += ' <ini_filename> <out_folder>'
        print(tmps)
        exit(0)
    #
    # Sampling distance [km]
    if len(argv) < 4:
        maximum_sampling_distance = 25.
    else:
        maximum_sampling_distance = float(argv[3])
    #
    # Check folders
    if in_path == out_path:
        tmps = '\nError: the input folder cannot be also the output one\n'
        tmps += '    input: {0:s}\n'.format(in_path)
        tmps += '    input: {0:s}\n'.format(out_path)
        print(tmps)
        exit(0)
    #
    # Read profiles
    sps, dmin, dmax = read_profiles_csv(in_path)
    #
    # Compute lengths
    lengths, longest_key, shortest_key = get_profiles_length(sps)
    number_of_samples = numpy.ceil(lengths[longest_key] /
                                   maximum_sampling_distance)
    print('Number of subsegments:', number_of_samples)
    tmp = lengths[shortest_key]/number_of_samples
    print('Shortest sampling [%s]: %.4f' % (shortest_key, tmp))
    tmp = lengths[longest_key]/number_of_samples
    print('Longest sampling  [%s]: %.4f' % (longest_key, tmp))
    #
    # Resampled profiles
    rsps = get_interpolated_profiles(sps, lengths, number_of_samples)
    #
    # Store profiles
    write_profiles_csv(rsps, out_path)
    #
    # Store edges
    write_edges_csv(rsps, out_path)


if __name__ == "__main__":
    main(sys.argv[1:])
