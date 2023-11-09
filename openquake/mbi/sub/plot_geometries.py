#!/usr/bin/env python
# coding: utf-8

import re
import os
import glob
import h5py
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


def plt_catalogue(filename, plotter, projection, max_hypo_depth=350,
                  mask=None, **kwargs):
    """
    :parameter str filename:
    """
    scaling = -1e2
    # Load catalogue
    cat = pickle.load(open(filename, 'rb'))
    # Create an array
    points = np.array([cat.data['longitude'], cat.data['latitude'],
                       cat.data['depth']]).T
    # Select
    if mask is not None:
        points = points[mask, :]
    points = points[points[:, 2] < max_hypo_depth, :]
    points[:, 2] /= scaling
    mesh = pv.PolyData(points)
    if 'color' in kwargs:
        _ = plotter.add_mesh(mesh=mesh, render_points_as_spheres=True, **kwargs)
    else:
        mesh['scalars'] = points[:, 2] * scaling
        cmap = plt.cm.get_cmap("jet_r")
        _ = plotter.add_mesh(mesh=mesh, cmap=cmap,
                             render_points_as_spheres=True, **kwargs)


def plot_profiles(foldername, plotter, p1):
    """
    Plot the subduction profiles
    """
    # Read files
    pattern = '_(\\d+).csv'
    points_lab = []
    labs = []
    for filename in sorted(glob.glob(os.path.join(foldername, 'cs_*.csv'))):
        points = numpy.loadtxt(filename)
        points[:, 2] /= -1e2
        polyline = polyline_from_points(points)
        tube = polyline.tube(radius=0.02)
        _ = plotter.add_mesh(tube, smooth_shading=True, color='blue')
        # Label
        mtch = re.search(pattern, filename)
        points_lab.append(list(points[0, :]))
        labs.append(f'{mtch.group(1):s}')
    pset = pv.PointSet(points_lab)
    plotter.add_point_labels(pset, labs)


def plot_edges(foldername, plotter, p1, color):
    """
    Plot the subduction edges
    """
    # Read files
    for filename in glob.glob(os.path.join(foldername, 'edge_*.csv')):
        points = numpy.loadtxt(filename)
        points[:, 2] /= -1e2
        polyline = polyline_from_points(points)
        tube = polyline.tube(radius=0.02)
        _ = plotter.add_mesh(tube, smooth_shading=True, color=color)


def main(fname_config, plot_catalogue, plot_classification):

    if plot_catalogue in ['true', 'True', 'TRUE']:
        plot_catalogue = True
    else:
        plot_catalogue = False

    if plot_classification in ['true', 'True', 'TRUE']:
        plot_classification = True
    else:
        plot_classification = False

    config = configparser.ConfigParser()
    config.read(fname_config)

    # Set the root folder
    rf = config['general']['root_folder']
    rf = os.path.normpath(rf)

    # Set the plotter
    plotter = pv.Plotter()
    plotter.set_background('grey')

    # Set the projection
    projection = Proj(proj='aeqd')

    # Plotting catalogue for display purposes
    catalogue_filename = config['general']['catalogue_filename']
    catalogue_filename = os.path.join(rf, catalogue_filename)
    if plot_catalogue:
        plt_catalogue(catalogue_filename, plotter, projection, point_size=5)

    # Plotting catalogue for classification
    if plot_classification:
        plt_catalogue(catalogue_filename, plotter, projection, point_size=5,
                      color='white')

    # find classification fname
    classification_fname = os.path.join(rf, config['general']['treg_filename'])

    for section in config.sections():

        if section == 'general':
            pass
        elif section == 'crustal':
            if plot_classification:
                classification = h5py.File(classification_fname, 'r')
                mask = classification['crustal'][:]
            # Plot classified earthquakes
                plt_catalogue(catalogue_filename, plotter, projection,
                              mask=mask, color='cyan')
                classification.close()
        else:
            foldername = config[section]['folder']
            foldername = os.path.join(rf, foldername)
            if 'int' in section:
                color = 'red'
                plot_edges(foldername, plotter, projection, color)
                plot_profiles(foldername, plotter, projection)
            else:
                color = 'purple'
                plot_edges(foldername, plotter, projection, color)
                plot_profiles(foldername, plotter, projection)
            # Plot classified earthquakes
            if plot_classification:
                classification = h5py.File(classification_fname, 'r')
                mask = classification[section][:]
                if sum(mask) > 0:
                    plt_catalogue(catalogue_filename, plotter, projection,
                                  mask=mask, color=color, point_size=15)
                classification.close()

    _ = plotter.show(interactive=True)


msg = 'Name of a .ini configuration file used to classify a catalogue'
main.fname_config = msg
msg = 'Flag controlling catalogue plotting'
main.plot_catalogue = msg
msg = 'Flag controlling classification plotting'
main.plot_classification = msg

if __name__ == '__main__':
    sap.run(main)
