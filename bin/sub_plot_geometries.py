#!/usr/bin/env python
# coding: utf-8

import os
import glob
import numpy
import pickle
import configparser
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

from pyproj import Proj

from openquake.baselib import sap

SCALING = -3.


def polyline_from_points(points):
    poly = pv.PolyData()
    poly.points = points
    the_cell = np.arange(0, len(points), dtype=np.int_)
    the_cell = np.insert(the_cell, 0, len(points))
    poly.lines = the_cell
    return poly


def plt_catalogue(filename, plotter, max_hypo_depth=350):
    """
    :parameter str filename:
    """
    scaling = -1e2
    # Set projection
    p1 = Proj(proj='aeqd')
    # Load catalogue
    cat = pickle.load(open(filename, 'rb'))
    points = np.array([cat.data['longitude'], cat.data['latitude'],
                       cat.data['depth']]).T
    points = points[points[:, 2] < max_hypo_depth, :]
    points[:, 2] /= scaling
    mesh = pv.PolyData(points)
    mesh['scalars'] = points[:, 2] * scaling
    cmap = plt.cm.get_cmap("jet_r")
    _ = plotter.add_mesh(mesh=mesh, cmap=cmap,
                         render_points_as_spheres=True)

def plot_profiles(foldername, plotter, p1):
    """
    Plot the subduction profiles
    """
    # Read files
    for filename in glob.glob(os.path.join(foldername, 'cs_*.csv')):
        points = numpy.loadtxt(filename)
        points[:, 2] /= -1e2
        polyline = polyline_from_points(points)
        tube = polyline.tube(radius=0.01)
        _ = plotter.add_mesh(tube, smooth_shading=True, color='blue')


def plot_edges(foldername, plotter, p1, color):
    """
    Plot the subduction edges
    """
    # Read files
    for filename in glob.glob(os.path.join(foldername, 'edge_*.csv')):
        points = numpy.loadtxt(filename)
        points[:, 2] /= -1e2
        polyline = polyline_from_points(points)
        tube = polyline.tube(radius=0.01)
        _ = plotter.add_mesh(tube, smooth_shading=True, color=color)


def main(fname_config):

    config = configparser.ConfigParser()
    config.read(fname_config)

    plotter = pv.Plotter()
    plotter.set_background('grey')

    projection = Proj(proj='aeqd')
    catalogue_filename = config['general']['catalogue_filename']
    plt_catalogue(catalogue_filename, plotter)

    for section in config.sections():
        if section == 'general':
            pass
        elif section == 'crustal':
            continue
        else:
            foldername = config[section]['folder']
            if 'int' in foldername:
                color = 'red'
            else:
                color = 'purple'
            plot_edges(foldername, plotter, projection, color)
            plot_profiles(foldername, plotter, projection)

    _ = plotter.show(interactive=True)


msg = 'Name of a .ini configuration file used to classify a catalogue'
main.fname_config = msg

if __name__ == '__main__':
    sap.run(main)
