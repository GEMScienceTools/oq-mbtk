#!/usr/bin/env python

import os
import re
import toml
import pandas as pd
import numpy as np
import subprocess

from openquake.man.tools.csv_output import (get_disagg_header_info, 
                                            get_rlzs_mde, get_mean_mde, 
                                            mean_mde_for_gmt)
from openquake.baselib import sap

"""
used to plot mean disaggregation by MDE results using GMT

example of usage: 

./plot_disagg_MDE.py <filename> <probability of exceedance> <IMT>

example of usage (filename is absolute path): 

./plot_disagg_MDE.py ../tests/tools/data/Mag_Dist_Eps-1.csv 0.002105 SA\(0.1\)

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
    
    stt_main_keys = settings['main'].keys()
    for key in stt_main_keys:
        stt_default[key] = settings['main'][key]

    stt_plot_keys = settings['plot'].keys()
    for key in stt_plot_keys:
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
    
    stt_plot = {'VIEW': "-p120/50"}
    
    if settings_fname is not None:
        stt_plot, stt_default = get_plt_settings(settings_fname, stt_plot, stt_default)
        
    for key in stt_default.keys():
        cmds.append('gmt set {} = {}'.format(key, stt_default[key]))
    
    VIEW = stt_plot['VIEW'] 
    
    # get header from disagg output
    with open(fname) as f:
        header = f.readline()
        
    # create color palette
    lim = get_disagg_header_info(header, 'eps_bin_edges=[')[-1]
    LIM="-{}/{}/1".format(lim, lim)
    CPTT1="./tmp/rel.cpt"
    if not os.path.exists('tmp'):
        os.makedirs('tmp')   
    cmds.append('gmt makecpt -Cpolar -T{} > {}'.format(LIM,CPTT1))
                
    # get other limits           
    cmd = 'gmt info {}.csv -I0.1'.format(fout)
    extinfo = subprocess.check_output(cmd, shell=True).decode("utf-8").strip()
    maxmag = float(extinfo.split('/')[1])
    
    mags_full = get_disagg_header_info(header, 'mag_bin_edges=[', fl=True)
    dmag = mags_full[-1]-mags_full[-2]
    mags = [m for m in mags_full if m <= (maxmag + dmag)]
    dists = get_disagg_header_info(header, 'dist_bin_edges=[', fl=True)
    ddist = dists[-1]-dists[-2]

    size = 3.2
    xsep = size / len(mags)*0.85
    ysep = size / len(dists)*0.85
    
    if 'PRO' in stt_plot.keys():
        PRO = stt_plot['PRO']
    else:
        fac_x = size / (mags[-1] - mags[0])
        fac_y = size / (dists[-1] - dists[0])
        PRO = "-Jx{}i/{}i".format(fac_x, fac_y)
                
    cmd = 'gmt info {}.csv -T0.00001+c4'.format(fout)
    ZRA = subprocess.check_output(cmd, shell=True).decode("utf-8").strip()
    ZLIM = float(ZRA.split('/')[1])*1.2
    EXT = '-R{}/{}/{}/{}/0.0/{}'.format(mags[0], mags[-1], dists[0], dists[-1], ZLIM)
                
    # create gmt script
    cmds.append('gmt begin {} png'.format(fout))
    
    cmd = 'gmt psxyz {}.csv {} {} -JZ5 {} -C{} -t5 '.format(fout, PRO, EXT, VIEW, CPTT1)
    cmd += '-SO{}i/{}ib -Bx{}+l"Magnitude" -By{}+l"Distance (km)"'.format(xsep, ysep, dmag, ddist)
    cmd += ' -Bz{}+l"Joint probability" -BWSneZ -Wthin,black'.format(ZLIM/2)
    cmds.append(cmd)

    cmd = 'gmt psscale {} {} -Dn1.15/0.2+w4/0.3 -Bx1+lEpsilon'.format(PRO, EXT)
    cmd += ' -JZ -Np -C{} {}'.format(CPTT1, VIEW)
    cmds.append(cmd)
    cmds.append('gmt end')
    
    for cmd in cmds:
        out = subprocess.call(cmd, shell=True) 
    
    print('plotted to {}.png'.format(fout))
    

def plot_disagg_MDE(fname, poe, imt, location, settings_fname, threshold): 
    """
    plots the mean disaggregation by MDE
    
    :param str fname:      
        name of file with results, usually includes Mag_Dist_Eps- ... 
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
    imt_short = re.sub('[\W_]+', '', imt)
    fout = '{}-{}-{}-mde'.format(location, poe, imt_short)
    mean_mde_for_gmt(fname, fout+'.csv', poe, imt, threshold)    
    
    # make plot
    plot_gmt(fname, fout, settings_fname)


def main(fname, poe, imt, *, location='site', 
         settings_fname=None, threshold=1e-10):
    """
    plots disaggregation by magnitude-distance-epsilon
    """ 
    
    plot_disagg_MDE(fname, poe, imt, location=location,
              settings_fname=settings_fname, threshold=float(threshold))


main.fname = 'Name of the file with disaggregation results'
main.poe = 'Probability of exceedance'
main.imt = 'Intensity measure type'
main.location = 'String to be used in file output names'
main.settings_fname = 'filename that includes gmt settings'
main.threshold = 'contribution threshold above which to include'

if __name__ == '__main__':
    sap.run(main)
