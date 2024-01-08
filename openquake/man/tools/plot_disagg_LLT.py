#!/usr/bin/env python

import os
import re
import toml
import pandas as pd
import numpy as np
import subprocess

from openquake.man.tools.csv_output import (make_llt_df, get_rlz_llt, 
                                            write_gmt_llt, get_disagg_header_info,
                                            make_gmt_file_llt)
from openquake.baselib import sap

"""
used to plot mean disaggregation by MDE results using GMT

example of usage: 

./plot_disagg_LLT.py <filename> <probability of exceedance> <IMT>

example of usage (filename is absolute path): 

./plot_disagg_LLT.py ../tests/tools/data/TRT_Lon_Lat-mean-1.csv 0.002105 SA\(0.1\)

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
    

def plot_gmt(fname, fout, settings_fname=None):
    """
    creates gmt plot
    :param str fname:      
        name of file with results, usually includes Mag_Dist_Eps- ... 
    :param str fout:   
        root of csv file name with mean disagg values and to for plot
    :param str settings_fname: 
        name of file with gmt settings; optional, 
        not currently very useful
    """ 
    
    cmds = []
    
    # set plot defaults    
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
    
    # get header from disagg output
    with open(fname) as f:
        header = f.readline()
    coords = header.split('lon=')[1]
    lon = coords.split(',')[0]
    lat = coords.split('lat=')[1].strip()[:-1]
        
    # create color palette
    lim = get_disagg_header_info(header, 'tectonic_region_types=[')
    LIM="0.5/{}/1".format(len(lim)+0.5)
    CPTT1="./tmp/rel.cpt"
    if not os.path.exists('tmp'):
        os.makedirs('tmp')   
    cmds.append('gmt makecpt -Cpolar -T{} > {}'.format(LIM,CPTT1))
                
    # get other limits           
    if 'EXT0' in stt_plot.keys():
        EXT0 = stt_plot['EXT0']
    else:
        cmd = 'gmt info {} -I0.1'.format(fout)
        EXT0 = subprocess.check_output(cmd, shell=True).decode("utf-8").strip()

    if 'PRO' in stt_plot.keys():
        PRO = stt_plot['PRO']
    else:
        PRO = "-Jx3.2d/3.2d"
                
    # create gmt script
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

    cmd = f'echo {lon} {lat} 0 0.3 > site.txt'  
    cmds.append(cmd)
    cmd = f'gmt psxyz site.txt {PRO} {EXT} {VIEW} -JZ5 -Sa -Ggreen -W0.2 -Baf '
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
    
    print('plotted to {}.png'.format(fout))
    

def plot_disagg_LLT(fname, poe, imt, location, settings_fname, threshold): 
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
    
    # format outputs to be plotted by GMT
    fout = make_gmt_file_llt(fname, poe, imt, location, threshold)    
    
    # make plot
    plot_gmt(fname, fout, settings_fname)


def main(fname, poe, imt, *, location='site', 
         settings_fname=None, threshold=1e-9):
    """
    plots mean disaggregation by magnitude-distance-epsilon
    
    Example to run:

    python plot_disagg_MDE.py ../tests/tools/case_8/expected/Mag_Dist_Eps-mean-0.csv 0.002105 SA\(0.1\)

    """ 
    
    plot_disagg_LLT(fname, poe, imt, location=location,
              settings_fname=settings_fname, threshold=float(threshold))


main.fname = 'Name of the file with disaggregation results'
main.poe = 'Probability of exceedance'
main.imt = 'Intensity measure type'
main.location = 'String to be used in file output names'
main.settings_fname = 'filename that includes gmt settings'
main.threshold = 'contribution threshold above which to include'

if __name__ == '__main__':
    sap.run(main)
