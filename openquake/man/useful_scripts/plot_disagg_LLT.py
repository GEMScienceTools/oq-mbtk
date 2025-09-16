# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2025 GEM Foundation
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.

import os
import re
import toml
import subprocess

from openquake.baselib import sap

from openquake.man.useful_scripts.csv_output_utils import (get_disagg_header_info,
                                                           mean_llt_for_gmt)

"""
used to plot mean disaggregation by MDE results using GMT

usage:

./plot_disagg_LLT.py <filename> <probability of exceedance> <IMT>

example of usage (filename is relative path):

./plot_disagg_LLT.py ../tests/tools/case_8/expected/TRT_Lon_Lat-mean-0.csv 0.002105 SA\(0.1\)

"""


def get_plt_settings(settings_fname, stt_plot, stt_default):
    """
    gets settings for gmt plot

    :param str settings_fname:
        name of file with gmt settings; optional
    :param dict stt_plot:
        settings for this particular plot. currently only
        supports VIEW but can add others
    :param dict stt_default:
        gmt default settings, i.e. font sizes
    """
    settings = toml.load(settings_fname)

    if 'main' in settings.keys():
        for key in settings['main'].keys():
            stt_default[key] = settings['main'][key]

    if 'plot' in settings.keys():
        for key in settings['plot'].keys():
            stt_plot[key] = settings['plot'][key]

    return stt_plot, stt_default


def plot_gmt(fname, fout, settings_fname=None, fsave=False):
    """
    creates gmt plot
    :param str fname:
        name of file with results, usually includes Mag_Dist_Eps- ...
    :param str fout:
        root of csv file name with mean disagg values and to for plot
    :param str settings_fname:
        name of file with gmt settings; optional,
        not currently very useful
    :param bool _script:
        true if want to view the gmt script
    """
    cmds = []

    # Set plot defaults    
    stt_default = {'MAP_GRID_CROSS_SIZE_PRIMARY':'0.2i',
                 'MAP_FRAME_TYPE': 'PLAIN',
                 'PS_MEDIA': 'a4',
                 'FONT_TITLE': '18p',
                 'FONT_LABEL': '12p',
                 'FONT_ANNOT_PRIMARY': '12p'}

    stt_plot = {'VIEW': "-p140/50"}

    if settings_fname is not None:
        stt_plot, stt_default = get_plt_settings(settings_fname, stt_plot, stt_default)

    for key in stt_default.keys():
        cmds.append('gmt set {} = {}'.format(key, stt_default[key]))

    VIEW = stt_plot['VIEW']

    # Get header from disagg output
    with open(fname) as f:
        header = f.readline()
    coords = header.split('lon=')[1]
    lon = coords.split(',')[0]
    lat = coords.split('lat=')[1].strip()[:-1]

    # Create color palette
    lim = get_disagg_header_info(header, 'tectonic_region_types=[')
    LIM="0.5/{}/1".format(len(lim)+0.5)
    CPTT1="./tmp/rel.cpt"
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    cmds.append('gmt makecpt -Cpolar -T{} > {}'.format(LIM,CPTT1))

    # Get other limits           
    if 'EXT0' in stt_plot.keys():
        EXT0 = stt_plot['EXT0']
    else:
        cmd = 'gmt info {} -I0.1'.format(fout)
        EXT0 = subprocess.check_output(cmd, shell=True).decode("utf-8").strip()
        limf = [float(l) for l in EXT0[2:].split('/')]
        # Adjust if bad ratio
        xr = abs(limf[0]-limf[1]); yr = abs(limf[2]-limf[3])
        if xr/yr < 0.5:
            limf[0] -= 0.25*yr; limf[1] += 0.25*yr
        if yr/xr < 0.5:
            limf[2] -= 0.25*xr; limf[3] += 0.25*xr
        EXT0 = f"-R{'/'.join([str(l) for l in limf])}"

    if 'PRO' in stt_plot.keys():
        PRO = stt_plot['PRO']
    else:
        PRO = "-Jx3.2d/3.2d"

    # Create gmt script
    cmd = 'gmt info {} -T0.00001+c2'.format(fout)
    ZRA = subprocess.check_output(cmd, shell=True).decode("utf-8").strip()
    ZLIM = float(ZRA.split('/')[1])*1.2
    EXT = EXT0+f'/0.0/{ZLIM}'

    cmd = "awk 'FNR==NR{sum+=$3;next;}{print $1, $2, $3, $4, $5}'"
    cmd += f" {fout} {fout} > tmp.txt"
    subprocess.call(cmd, shell=True)

    cmds.append('gmt begin {} png'.format(fout.replace('.csv', '')))

    cmd = f'gmt coast {EXT} {PRO} -JZ5 {VIEW} -Sazure2 -Gwheat -Wfaint -A1000'
    cmds.append(cmd)

    cmd = f'echo {lon} {lat} 0 0.5 > site.txt'  
    cmds.append(cmd)
    cmd = f'gmt psxyz site.txt {PRO} {EXT} {VIEW} -JZ5 -Sa -Ggreen -W0.3 -Baf '
    cmds.append(cmd)

    cmd = 'gmt psxyz tmp.txt {} {} -JZ5 {} -C{} -t5 '.format(PRO, EXT, VIEW, CPTT1)
    cmd += '-SO0.35/0.35b'
    cmd += ' -Bz{}+l"Joint probability" -BWSneZ -Wthin,black'.format(ZLIM*0.25)
    cmds.append(cmd)

    cmd = 'gmt psscale {} {} -Dn1.15/0.2+w4/0.3 -Bx1+l"TRT"'.format(PRO, EXT)
    cmd += ' -JZ -Np -C{} {}'.format(CPTT1, VIEW)
    cmds.append(cmd)
    cmds.append('gmt end')
    
    for cmd in cmds:
        out = subprocess.call(cmd, shell=True) 

    if fsave==True:
        f = open("plot_llt.sh", "a")
        f.writelines(cmds)
        f.close()
        print('Script saved to plot_llt.sh')
    
    print('plotted to {}.png'.format(fout.replace('.csv','')))
    

def plot_disagg_LLT(fname, poe, imt, location, settings_fname, threshold,
                    fsave=False): 
    """
    plots the mean disaggregation by LLT
    
    :param str fname:      
        name of file with results, usually includes TRT_Lon_Lat- ... 
    :param float poe:        
        poe to be isolated/plotted, corresponding to investigation 
        time specified in job
    :param str imt:        
        imt to be isolated/plotted
    :param str location:   
        name to be used in output file names
    :param str settings_fname:   
        file with settings that override gmt and plot defaults
    :param float threshold:  
        contribution included in output if above this value
    """  
    # Format outputs to be plotted by GMT
    imt_short = re.sub('[\W_]+', '', imt)
    fout = '{}-{}-{}-llt.csv'.format(location, poe, imt_short)
    mean_llt_for_gmt(fname, fout, poe, imt, threshold)    

    # Make plot
    plot_gmt(fname, fout, settings_fname, fsave=fsave)


def main(fname,
         poe,
         imt,
         *,
         location='site', 
         settings_fname=None,
         threshold=1e-9,
         fsave=False):
    """
    plots mean disaggregation by magnitude-distance-epsilon
    
    Example to run:

    python plot_disagg_MDE.py ../tests/tools/case_8/expected/Mag_Dist_Eps-mean-0.csv 0.002105 SA\(0.1\)

    """ 
    plot_disagg_LLT(fname,
                    poe,
                    imt,
                    location=location,
                    settings_fname=settings_fname,
                    threshold=float(threshold),
                    fsave=bool(fsave))


main.fname = 'Name of the file with disaggregation results'
main.poe = 'Probability of exceedance'
main.imt = 'Intensity measure type'
main.location = 'String to be used in file output names'
main.settings_fname = 'filename that includes gmt settings'
main.threshold = 'contribution threshold above which to include'
main.fsave = 'True if want to save gmt script'

if __name__ == '__main__':
    sap.run(main)
