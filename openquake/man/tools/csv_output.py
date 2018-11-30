import re
import numpy
from scipy import interpolate

import matplotlib.pyplot as plt


def mde_for_gmt(filename, fout):
    """
    This simple function converts the information in the .csv file (m-d-e) into
    a format suitable to be used by GMT.

    :param str filename:
        Name of the file containing the original information
    :param str fout:
        The name of the file where the information must be stored
    """
    fou = open(fout, 'w')
    base_dic = {}
    cnt = 0
    for line in open(filename, 'r'):
        if cnt > 2:
            #
            # Splitting the row
            aa = re.split('\,', line)
            key = '{0:s}_{1:s}'.format(aa[0], aa[1])
            #
            # Updating the base level of the bin
            if key in base_dic:
                base = base_dic[key]
            else:
                base = 0.
                base_dic[key] = 0.
            base_dic[key] += float(aa[3])
            #
            # Formatting the output
            fmt = '{0:7.5e} {1:7.5e} {2:7.5e} {3:7.5e} {4:7.5e}'
            outs = fmt.format(float(aa[0]), float(aa[1]), base+float(aa[3]),
                              float(aa[2]), base)
            if float(aa[3]) > 1e-8:
                fou.write(outs+'\n')
        cnt += 1


def read_dsg_ll(fname):
    """
    :param fname:
        Name of the file containing the results
    :return:
        A tuple with longitude, latitude and probabilities of exceedance
    """
    lons = []
    lats = []
    poes = []
    for i, line in enumerate(open(fname, 'r')):
        if i == 0:
            head1 = line
        elif i == 1:
            head2 = line
        else:
            aa = re.split('\,', re.sub('^\s*', '', line))
            lons.append(float(aa[0]))
            lats.append(float(aa[1]))
            poes.append(float(aa[2]))
    return numpy.array(lons), numpy.array(lats), numpy.array(poes)


def read_dsg_m(fname):
    """
    :param fname:
        Name of the file containing the results
    :return:
        A tuple with magnitudes and probabilities of exceedance
    """
    mags = []
    poes = []
    for i, line in enumerate(open(fname, 'r')):
        if i == 0:
            head1 = line
        elif i == 1:
            head2 = line
        else:
            aa = re.split('\,', re.sub('^\s*', '', line))
            mags.append(float(aa[0]))
            poes.append(float(aa[1]))
    return numpy.array(mags), numpy.array(poes)


def get_map_from_curves(imls, poes, pex):
    """
    :parameter imls:
    :parameter poes:
    :parameter pex:
    """
    dat = []
    for idx in range(0, poes.shape[0]):
        dval = 0
        if (any(poes[idx, :] > 0.0) and
            min(poes[idx, poes[idx, :] > 0.0]) < pex and
                max(poes[idx, :]) > pex):
            f2 = interpolate.interp1d(poes[idx, poes[idx, :] > 0],
                                      imls[poes[idx, :] > 0],
                                      kind='linear')
            dval = f2(pex)

            if dval > 1.0:
                plt.plot(imls[poes[idx, :] > 0],
                         poes[idx, poes[idx, :] > 0], '-')
                plt.xscale('log')
                plt.yscale('log')
                plt.show()
        else:
            dval = 0.0
        dat.append(dval)
    return numpy.asarray(dat)


def _get_header_curve2(line):
    imls = []
    aa = re.split('\,', line)
    for bb in aa[3:]:
        tmp = re.sub('^[a-zA-Z]*\(*[0-9]*\.*[0-9]*\)*-', '',
                     re.sub(':float32', '', bb))
        imls.append(float(tmp))
    return imls


def _get_header_uhs2(line):
    rps = []
    per = []
    aa = re.split('\,', line)
    for bb in aa[2:]:
        mtc = re.match('(\d+\.+\d+)\~([A-Z]+\((.*)\)|PGA)', bb)
        rps.append(float(mtc.group(1)))
        if mtc.group(3) is None:
            per.append(0.0)
        else:
            per.append(float(mtc.group(3)))
    return numpy.array(rps), numpy.array(per)


def read_uhs_csv(filename):
    """
    Read a .csv file containing a number of UHSs
    """
    lats = []
    lons = []
    uhss = []
    for idx, line in enumerate(open(filename, 'r')):
        if idx == 0:
            header1 = _get_header1(line)
        elif idx == 1:
            rps, prs = _get_header_uhs2(line)
        else:
            aa = re.split('\,', line)
            lons.append(float(aa[0]))
            lats.append(float(aa[1]))
            uhss.append([float(bb) for bb in aa[2:]])
    return numpy.array(lons), numpy.array(lats), numpy.array(uhss), header1, \
        rps, prs


def read_hazard_curve_csv(filename):
    """
    Read a csv file containing hazard curves.
    :param str filename:
        Name of the .csv file containing the data
    :return:
        A tuple with the following information:
            - Longitudes
            - Latitudes 
            - PoEs
            - String with the header 
            - IMLs
    """
    lats = []
    lons = []
    imls = []
    curs = []
    for idx, line in enumerate(open(filename, 'r')):
        if idx == 0:
            header1 = _get_header1(line)
        elif idx == 1:
            imls = _get_header_curve2(line)
        else:
            aa = re.split('\,', line)
            lons.append(float(aa[0]))
            lats.append(float(aa[1]))
            curs.append([float(bb) for bb in aa[3:]])
    assert len(lons) == len(lats) == len(curs)
    return numpy.array(lons), numpy.array(lats), numpy.array(curs), header1, \
        numpy.array(imls)


def _get_header1(line):
    pass


def _get_header2(line):
    imls = []
    aa = re.split('\,', line)
    for bb in aa[3:]:
        imls.append(re.sub('^poe-', '', bb))
    return imls


def read_hazard_map(filename):
    """
    Reads a .csv file with hazard maps created by the OpenQuake engine
    """
    lats = []
    lons = []
    maps = []
    for idx, line in enumerate(open(filename, 'r')):
        if not re.search('^#', line):
            if idx == 0:
                header1 = _get_header1(line)
            elif idx == 1:
                header2 = _get_header2(line)
            else:
                aa = re.split('\,', line)
                lons.append(float(aa[0]))
                lats.append(float(aa[1]))
                maps.append([float(bb) for bb in aa[2:]])
    return numpy.array(lons), numpy.array(lats), numpy.array(maps), header1, \
        header2
