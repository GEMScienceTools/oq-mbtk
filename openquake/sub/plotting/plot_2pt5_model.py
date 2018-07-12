#!/usr/bin/env python

import code
import os
import re
import sys
import glob
import numpy
import pickle
import configparser

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.basemap import Basemap


def plot_sub_profile(sid, dat, axes):
    axes.plot(dat[:,0], dat[:,1], dat[:,2])

def plot_edge(sid, dat, axes):
    axes.plot(dat[:,0], dat[:,1], dat[:,2], '--')

def plot_catalogue(filename, axes, lims, qual):
    cat = pickle.load(open(filename, 'rb'))
    lo = cat.data['longitude']
    if qual==1:
        lo = numpy.array([x+360 if x<0 else x for x in lo])
    la = cat.data['latitude']
    de = cat.data['depth']
    idx = numpy.nonzero((lo > lims[0]) & (lo < lims[1]) &
                        (la > lims[2]) & (la < lims[3]) &
                        (de > lims[4]) & (de < lims[5]))
    #axes.plot(lo[idx], la[idx], de[idx], '.', alpha=0.5)

def plot_sub_profiles(foldername):
    """
    """
    #
    minlo = +1e10
    minla = +1e10
    maxlo = -1e10
    maxla = -1e10
    minde = +1e10
    maxde = -1e10
    #
    mpp = Basemap()
    # Create figure
    fig = plt.figure()
    ax = Axes3D(fig)
    # checking whether straddles dateline
    qual = 0
    for filename in glob.glob(os.path.join(foldername, 'cs_*.csv')):
        dat = numpy.loadtxt(filename)
        sid = re.sub('^cs_', '', re.split('\.', os.path.basename(filename))[0])
        if ((min(dat[:,0])/max(dat[:,0])<0) & (max(dat[:,0]>150))):
            qual = 1

    # Plotting subduction profiles
    sps = {}
    for filename in glob.glob(os.path.join(foldername, 'cs_*.csv')):
        dat = numpy.loadtxt(filename)
        if qual==1:
            dat[:,0] = numpy.array([x+360 if x<0 else x for x in dat[:,0]])

        sid = re.sub('^cs_', '', re.split('\.', os.path.basename(filename))[0])
        if re.search('[a-zA-Z]', sid):
            sid = '%03d' % int(sid)
            print (sid)
        sps[sid] = numpy.loadtxt(filename)
        plot_sub_profile(sid, dat, ax)

        minlo = min(dat[:,0]) if minlo > min(dat[:,0]) else minlo
        maxlo = max(dat[:,0]) if maxlo < max(dat[:,0]) else maxlo
        minla = min(dat[:,1]) if minla > min(dat[:,1]) else minla
        maxla = max(dat[:,1]) if maxla < max(dat[:,1]) else maxla
        minde = min(dat[:,2]) if minde > min(dat[:,2]) else minde
        maxde = max(dat[:,2]) if maxde < max(dat[:,2]) else maxde

    for filename in sorted(glob.glob(os.path.join(foldername, 'edge_*.csv'))):
        print (filename)
        dat = numpy.loadtxt(filename)
        if qual==1:
            dat[:,0] = numpy.array([x+360 if x<0 else x for x in dat[:,0]])

        sid = re.sub('^edge_', '', re.split('\.', os.path.basename(filename))[0])
        sps[sid] = numpy.loadtxt(filename)
        plot_edge(sid, dat, ax)

        minlo = min(dat[:,0]) if minlo > min(dat[:,0]) else minlo
        maxlo = max(dat[:,0]) if maxlo < max(dat[:,0]) else maxlo
        minla = min(dat[:,1]) if minla > min(dat[:,1]) else minla
        maxla = max(dat[:,1]) if maxla < max(dat[:,1]) else maxla
        minde = min(dat[:,2]) if minde > min(dat[:,2]) else minde
        maxde = max(dat[:,2]) if maxde < max(dat[:,2]) else maxde

    ax.set_xlim([minlo, maxlo])
    ax.set_ylim([minla, maxla])

    ax.invert_zaxis()
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Depth [km]')

    return ax, (minlo, maxlo, minla, maxla, minde, maxde), qual


def main(argv):
    """
    argv[0] - Folder containing the cross-section profiles and the edges
    argv[1] - .ini file
    """
    foldername = argv[0]
    axes, lims, qual = plot_sub_profiles(foldername)
    print (lims)

    if len(argv) > 1:
        config = configparser.ConfigParser()
        config.read(argv[1])
        fname_eqk_cat = config['data']['catalogue_pickle_filename']
        #code.interact(local=locals())
        if re.search('[a-z]', fname_eqk_cat):
            plot_catalogue(fname_eqk_cat, axes, lims, qual)

    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
