#!/usr/bin/env python

import re
import sys
import numpy
import pickle
import configparser
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

from openquake.sub.cross_sections import Trench
from openquake.baselib import sap

def get_cs(trench, ini_filename, cs_len, cs_depth, interdistance, qual,
           fname_out_cs='cs_traces.cs'):
    """
    :parameter trench:
        An instance of the :class:`Trench` class
    :parameter ini_filename:
        The name of the .ini file
    :parameter cs_len:
        Length of the cross-section [km]
    :parameter interdistance:
        Separation distance between cross-sections [km]
    :parameter qual:
        Boolean when true fixes longitudes in proximity of the IDL
    :parameter fname_out_cs:
        Name of the file where we write the traces of the cross sections
    """

    # Plot the traces of cross-sections
    fou = open(fname_out_cs, 'w')

    cs_dict = {}
    for idx, (cs, out) in enumerate(
            trench.iterate_cross_sections(interdistance, cs_len)):

        if cs is not None:
            cs_dict['%s' % idx] = cs

            if qual == 1:
                cs.plo[:] = ([x-360. if x > 180. else x for x in cs.plo[:]])

            # Set the length
            tmp_len = numpy.min([cs_len, out]) if out is not None else cs_len
            tmps = '%f %f %f %f %f %d %s\n' % (cs.plo[0],
                                               cs.pla[0],
                                               cs_depth,
                                               tmp_len,
                                               cs.strike[0],
                                               idx,
                                               ini_filename)
            print(tmps.rstrip())
            fou.write(tmps)
    fou.close()

    return cs_dict


def plot(trench, cat, cs_dict, interdistance):
    """
    :parameter trench:
        An instance of the :class:`Trench` class
    :parameter cat:
        An instance of the :class:`` class
    :parameter cs_dict:
        A <key,value> dictionary where the key is a section ID and the value is
        an instance of the :class:`CrossSection` class
    :parameter interdistance:
        Separation distance between cross-sections [km]
    """
    from mpl_toolkits.basemap import Basemap

    minlo = min(trench.axis[:, 0]) - 5
    minla = min(trench.axis[:, 1]) - 5
    maxlo = max(trench.axis[:, 0]) + 5
    maxla = max(trench.axis[:, 1]) + 5
    midlo = (minlo+maxlo)/2
    midla = (minla+maxla)/2


    _ = plt.figure(figsize=(12, 9))

    # Plot the basemap
    m = Basemap(llcrnrlon=minlo, llcrnrlat=minla,
                urcrnrlon=maxlo, urcrnrlat=maxla,
                resolution='i', projection='tmerc',
                lon_0=midlo, lat_0=midla)

    #
    # Draw paralleles and meridians with labels
    # labels = [left,right,top,bottom]
#    m.drawcoastlines()
    m.drawmeridians(numpy.arange(numpy.floor(minlo/10.)*10,
                                 numpy.ceil(maxlo/10.)*10, 5.),
                    labels=[False, False, False, True])
    m.drawparallels(numpy.arange(numpy.floor(minla/10.)*10,
                                 numpy.ceil(maxla/10.)*10, 5.),
                    labels=[True, False, False, False])

    #
    # Plot the instrumental catalogue
    xa, ya = m(cat.data['longitude'], cat.data['latitude'])
    szea = (cat.data['magnitude']*100)**1.5
    patches = []
    for x, y, sze in zip(list(xa), list(ya), szea):
        circle = Circle((x, y), sze, ec='white')
        patches.append(circle)
    colors = cat.data['depth']
    p = PatchCollection(patches, zorder=6, edgecolors='white')
    p.set_alpha(0.5)
    p.set_clim([0, 200])
    p.set_array(numpy.array(colors))
    plt.gca().add_collection(p)
    plt.colorbar(p, fraction=0.02, pad=0.04, extend='max')

    x, y = m(trench.axis[:, 0], trench.axis[:, 1])
    plt.plot(x, y, '-g', linewidth=2, zorder=10)
    plt.plot(x, y, '--y', linewidth=4, zorder=20)

    # Plot the traces of cross-sections
    if interdistance != 0:
        ts = trench.resample(interdistance)
        x, y = m(ts.axis[:, 0], ts.axis[:, 1])

    else:
        for key in cs_dict:
            cs = cs_dict[key]
        #if cs is not None:
            x, y = m(cs.plo, cs.pla)
            plt.plot(x, y, ':r', linewidth=2, zorder=20)
            text = plt.text(x[0], y[0], '%s' % key, ha='center',
                            va='center', size=10, zorder=30)
            text = plt.text(x[-1], y[-1], '%s' % key, ha='center',
                            va='center', size=10, zorder=30)
            text.set_path_effects([PathEffects.withStroke(linewidth=3,
                                                          foreground="w")])
    m.drawcoastlines(zorder=26)
    plt.show()


def main(config_fname):
    """
    config_fname is the .ini file
    """

    # Parse .ini file
    config = configparser.ConfigParser()
    config.read(config_fname)
    fname_trench = config['data']['trench_axis_filename']
    fname_eqk_cat = config['data']['catalogue_pickle_filename']
    cs_length = float(config['section']['lenght'])
    cs_depth = float(config['section']['dep_max'])
    interdistance = float(config['section']['interdistance'])

    # Load trench axis
    fin = open(fname_trench, 'r')
    lotmp = []
    latmp = []
    for line in fin:
        aa = re.split('\\s+', re.sub('^\\s+', '', line))
        lotmp.append(float(aa[0]))
        latmp.append(float(aa[1]))
    fin.close()
    qual = 0
    if (min(lotmp)/max(lotmp) < 0.) & (min(lotmp) < -150.):
        qual = 1
        lotmp = (x+360. if x < 0. else x for x in lotmp)
    trench = list(zip(lotmp, latmp))
    trench = Trench(numpy.array(trench))

    # Load catalogue
    with open(fname_eqk_cat, 'rb') as fin:
        cat = pickle.load(fin)

    # Get cross-sections
    cs_dict = get_cs(trench, argv[0], cs_length, cs_depth, interdistance, qual)

    # Plotting
    if False:
        plot(trench, cat, cs_dict, interdistance)

main.config = 'config file for creating cross sections from trench axis'

if __name__ == "__main__":
    sap.run(main)
