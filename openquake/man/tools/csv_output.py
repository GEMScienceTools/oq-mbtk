# ------------------- The OpenQuake Model Building Toolkit --------------------
# Copyright (C) 2022 GEM Foundation
#           _______  _______        __   __  _______  _______  ___   _
#          |       ||       |      |  |_|  ||  _    ||       ||   | | |
#          |   _   ||   _   | ____ |       || |_|   ||_     _||   |_| |
#          |  | |  ||  | |  ||____||       ||       |  |   |  |      _|
#          |  |_|  ||  |_|  |      |       ||  _   |   |   |  |     |_
#          |       ||      |       | ||_|| || |_|   |  |   |  |    _  |
#          |_______||____||_|      |_|   |_||_______|  |___|  |___| |_|
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------
# vim: tabstop=4 shiftwidth=4 softtabstop=4
# coding: utf-8

import re
import numpy
import pandas as pd
from scipy import interpolate
from openquake.hmtk.seismicity.catalogue import Catalogue


def get_disagg_header_info(header, var, fl=False):
    """
    This function gets information about disagg by MDE from the header
    of the output file (bins, weights, etc.).
    Returns a list of the values 

    : param str header: 
        header line of disagg output
    : param str var:
        desired value from header 
    : param boolean fl:
        if true, first converts the list values to floats
    """
    # get term from header
    
    first_break = header.index(var)
    strt = header[first_break+len(var):]
    end_first = strt.index(']')
    
    if fl==True:
        return [float(i) for i in strt[:end_first].split(', ')]
    else:
        return strt[:end_first].split(', ')


def get_rlzs_mde(header):
    """
    This function returns dictionary of realizations and weights from 
    a disaggregation by MDE

    :param str header:      
        header of file with results for disagg by Mag_Dist_Eps- ... 
    """
        
    rlz = get_disagg_header_info(header, 'rlz_ids=[')
    wei = get_disagg_header_info(header, 'weights=[')
    
    each_rlz = {}
    for w,r in zip(wei,rlz):
        each_rlz['rlz'+r] = float(w)
        
    return each_rlz


def get_mean_mde(fname, poe, imt):
    """
    gets the mean disagg by mde by weighting all realazations using
    information from the header
    
    :param str fname:      
        name of file with results, usually includes Mag_Dist_Eps- ... 
    :param float poe:        
        poe to be isolated/plotted, corresponding to investigation 
        time specified in job
    :param str imt:        
        imt to be isolated/plotted
    """

    # read in the rest of the outputs
    df = pd.read_csv(fname, skiprows = 1)

    # take only the rows of interest based on poe, imt
    df_sub = df.loc[(df['poe']==float(poe)) & (df['imt']==imt)].reset_index()

    # create dataframe for mean results
    df_mean = pd.DataFrame(columns=['mag','dist','eps','poe_c'])

    if 'mean' not in df:

        # get header from disagg output
        with open(fname) as f:
            header = f.readline()
    
        each_rlz = get_rlzs_mde(header)
        rlzkeys = [*each_rlz]
    

    
        new_poe = []
        for r in rlzkeys:
            poes = numpy.array([float(f) for f in df_sub[r].values])
            tmp_array = each_rlz[r] * poes
            new_poe.append(list(tmp_array))

        df_mean['poe_c'] = numpy.sum(numpy.array(new_poe), axis=0)

    else:
        df_mean['poe_c'] = df['mean']
    
    for key in ['mag', 'eps', 'dist']:
        df_mean[key] = df_sub[key]
        
    return df_mean

def mean_mde_for_gmt(fname, fout, poe, imt, threshold):
    """
    puts mean disagg outputs into csv file format to be plotted by GMT;
    may supersede mde_for_gmt
    
    :param str fname:      
        name of file with results, usually includes Mag_Dist_Eps- ... 
    :param float poe:        
        poe to be isolated/plotted, corresponding to investigation 
        time specified in job
    :param str imt:        
        imt to be isolated/plotted
    :param str fout:       
        root of output filename
    :param float threshold:  
        contribution included in output if above this value
    """
    
    df_mean = get_mean_mde(fname, poe, imt)
    
    fou = open(fout,'w')

    base_dic = {}
    #import pdb; pdb.set_trace()
    for ind in list(df_mean.index):
        line = df_mean.loc[ind]

        key = '{0:s}_{1:s}'.format(str(line.mag), str(line.dist))
        #
        # Updating the base level of the bin
        if key in base_dic:
            base = base_dic[key]
        else:
            base = 0.
            base_dic[key] = 0.
        base_dic[key] += line.poe_c
        #
        # Formatting the output
        fmt = '{0:7.5e} {1:7.5e} {2:7.5e} {3:7.5e} {4:7.5e}'
        outs = fmt.format(line.mag, line.dist, base+line.poe_c,
                          line.eps, base)

        if float(line.poe_c) > threshold:
            fou.write(outs+'\n')
        
    print('Written to {}'.format(fout))


def mde_for_gmt(filename, froot):
    """
    This simple function converts the information in the .csv file (m-d-e) into
    a format suitable to be used by GMT.

    :param str filename:
        Name of the file containing the original information
    :param str fout:
        Root path (including file prefix) for the output files
    """
    flist = []

    # Read input
    df = pd.read_csv(filename, comment='#')

    # Find the unique combinations of IMT and poe
    sips = set()
    for group_name, df_group in df.groupby(['imt', 'poe']):
        if group_name not in sips:
            sips.add(group_name)

    # Column names with the realizations
    rlz_cols = [col for col in df.columns if 'rlz' in col]

    # For each imt + poe
    for sip in list(sips):
        imt = sip[0]

        # For each rlz
        for rlz in rlz_cols:

            base_dic = {}
            name = f'{froot}_{imt}_{rlz}.txt'
            flist.append(name)
            fou = open(name, 'w')

            for i, row in df[df.imt == imt].iterrows():
                key = '{0:.2f}_{1:.2f}'.format(row.mag, row.dist)

                # Updating the base level of the bin
                if key in base_dic:
                    base = base_dic[key]
                else:
                    base = 0.
                    base_dic[key] = 0.
                base_dic[key] += row[rlz]

                # Formatting the output:
                # magnitude, distance, z, height, upp,
                fmt = '{:7.5e} {:7.5e} {:7.5e} {:7.5e} {:7.5e}'
                outs = fmt.format(row.mag, row.dist, base+row[rlz], row.eps,
                                  base)

                if row[rlz] > 1e-8:
                    fou.write(outs+'\n')
            fou.close()
    return flist


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
            _ = line
        elif i == 1:
            _ = line
        else:
            aa = re.split('\\,', re.sub('^\\s*', '', line))
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
            _ = line
        elif i == 1:
            _ = line
        else:
            aa = re.split('\\,', re.sub('^\\s*', '', line))
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
        else:
            dval = 0.0
        dat.append(dval)
    return numpy.asarray(dat)


def _get_header_curve2(line):
    imls = []
    aa = re.split('\\,', line)
    for bb in aa[3:]:
        tmp = re.sub('^[a-zA-Z]*\\(*[0-9]*\\.*[0-9]*\\)*-', '',
                     re.sub(':float32', '', bb))
        imls.append(float(tmp))
    return imls


def _get_header_uhs2(line):
    rps = []
    per = []
    aa = re.split('\\,', line)
    for bb in aa[2:]:
        mtc = re.match('(\\d+\\.+\\d+)\\~([A-Z]+\\((.*)\\)|PGA)', bb)
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
            aa = re.split('\\,', line)
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
            aa = re.split('\\,', line)
            lons.append(float(aa[0]))
            lats.append(float(aa[1]))
            curs.append([float(bb) for bb in aa[3:]])
    assert len(lons) == len(lats) == len(curs)
    return numpy.array(lons), numpy.array(lats), numpy.array(curs), header1, \
        numpy.array(imls)


def _get_header1(line):

    header = {}

    tmpstr = "imt"
    if re.search('generated_by', line):
        # version 3.6
        imt_pattern = r'{:s}=\'([^\']*)\''.format(tmpstr)
        # engine
        tmpstr = "generated_by"
        pattern = r'{:s}=\'([^\']*)\''.format(tmpstr)
        mtc = re.search(pattern, line)
        header["engine"] = mtc.group(1)
    else:
        # version 3.5 and before
        imt_pattern = r'{:s}=\"([^\']*)\"'.format(tmpstr)

    # result type
    aa = re.split('\\,', re.sub('#', '', line))
    header['result_type'] = re.sub('^\\s*', '', re.sub('\\s*$', '', aa[0]))
    # investigation time
    tmpstr = "investigation_time"
    pattern = "{:s}=(\\d*\\d.\\d*)".format(tmpstr)
    mtc = re.search(pattern, line)
    header[tmpstr] = float(mtc.group(1))
    # IMT
    mtc = re.search(imt_pattern, line)
    header["imt"] = mtc.group(1)
    return header


def _get_header2(line):
    imls = []
    aa = re.split('\\,', line)
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
                aa = re.split('\\,', line)
                lons.append(float(aa[0]))
                lats.append(float(aa[1]))
                maps.append([float(bb) for bb in aa[2:]])
    return numpy.array(lons), numpy.array(lats), numpy.array(maps), header1, \
        header2


def get_catalogue_from_ses(fname, duration):
    """
    Converts a set of ruptures into an instance of
    :class:`openquake.hmtk.seismicity.catalogue.Catalogue`.

    :param fname:
        Name of the .csv file
    :param float duration:
        Duration [in years] of the SES
    :returns:
        A :class:`openquake.hmtk.seismicity.catalogue.Catalogue` instance
    """
    # Read the set of ruptures
    ses = pd.read_csv(fname, sep='\t', skiprows=1)
    if len(ses.columns) < 2:
        ses = pd.read_csv(fname, sep=',', skiprows=1)
    # Create an empty catalogue
    cat = Catalogue()
    # Set catalogue data
    cnt = 0
    year = []
    eventids = []
    mags = []
    lons = []
    lats = []
    deps = []
    print(ses['rup_id'])
    print('Columns:', ses.columns)
    for i in range(len(ses)):
        nevents = ses['multiplicity'][i]
        for j in range(nevents):
            eventids.append(f'{cnt:d}')
            mags.append(ses['mag'].values[i])
            lons.append(ses['centroid_lon'].values[i])
            lats.append(ses['centroid_lat'].values[i])
            deps.append(ses['centroid_depth'].values[i])
            cnt += 1
            year.append(numpy.random.random_integers(1, duration, 1))

    data = {}
    year = numpy.array(year, dtype=int)
    data['year'] = year
    data['month'] = numpy.ones_like(year, dtype=int)
    data['day'] = numpy.ones_like(year, dtype=int)
    data['hour'] = numpy.zeros_like(year, dtype=int)
    data['minute'] = numpy.zeros_like(year, dtype=int)
    data['second'] = numpy.zeros_like(year)
    data['magnitude'] = numpy.array(mags)
    data['longitude'] = numpy.array(lons)
    data['latitude'] = numpy.array(lats)
    data['depth'] = numpy.array(deps)
    data['eventID'] = eventids
    cat.data = data
    cat.end_year = duration
    cat.start_year = 0
    cat.data['dtime'] = cat.get_decimal_time()
    return cat
