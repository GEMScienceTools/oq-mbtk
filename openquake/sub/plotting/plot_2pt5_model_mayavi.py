#!/usr/bin/env python

import os
import re
import sys
import glob
import numpy
import pickle
import configparser

from pyproj import Proj

from mayavi import mlab

SCALING = -3.

def plt_catalogue(filename):
    """
    :parameter str filename:
    """
    # Set projection
    p1 = Proj(proj='aeqd')
    # Load catalogue
    cat = pickle.load(open(filename, 'rb'))
    #
    x, y = p1(cat.data['longitude'], cat.data['latitude'])
    mlab.points3d(x/1e3, y/1e3, SCALING*cat.data['depth'], color=(0, 0, 1),
                  scale_factor=4.)


def plot_sub_profiles(foldername):
    """
    """
    # Projection
    p1 = Proj(proj='aeqd')
    # Read files
    for filename in glob.glob(os.path.join(foldername, 'cs_*.csv')):
        dat = numpy.loadtxt(filename)
        sid = re.sub('^cs_', '', re.split('\.', os.path.basename(filename))[0])
        x, y = p1(dat[:,0], dat[:,1])
        mlab.plot3d(x/1e3, y/1e3, SCALING*dat[:,2], tube_radius=2,
                    color=(1,0,0))

def plot_edges(foldername):
    """
    """
    # Projection
    p1 = Proj(proj='aeqd')
    # Read files
    for filename in glob.glob(os.path.join(foldername, 'edge_*.csv')):
        dat = numpy.loadtxt(filename)
        sid = re.sub('^edge_', '', re.split('\.', os.path.basename(filename))[0])
        x, y = p1(dat[:,0], dat[:,1])
        mlab.plot3d(x/1e3, y/1e3, SCALING*dat[:,2], tube_radius=2,
                    color=(1,1,0))


def main(argv):
    """
    """
    foldername = argv[0]

    config = configparser.ConfigParser()
    config.read(argv[1])
    fname_eqk_cat = config['data']['catalogue_pickle_filename']

    # Create figure
    mlab.figure(1, size=(400, 400), bgcolor=(0.75, 0.75, 0.75))
    #
    plot_sub_profiles(foldername)
    plot_edges(foldername)
    # Plot catalogue
    #plt_catalogue(fname_eqk_cat)
    #
    mlab.show()

if __name__ == "__main__":
    main(sys.argv[1:])
