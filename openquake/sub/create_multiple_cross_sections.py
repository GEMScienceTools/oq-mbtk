#!/usr/bin/env python

import re
import sys
import numpy
import pickle
import configparser
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

from openquake.sub.cross_sections import CrossSection, Trench
from openquake.hmtk.seismicity.selector import CatalogueSelector


def get_cs(trench, ini_filename, cs_len, cs_depth, interdistance,qual):
    """
    :parameter trench:
        An instance of the :class:`Trench` class
    :parameter ini_filename:
        The name of the .ini file
    :parameter cs_len:
        Length of the cross-section [km]
    :parameter interdistance:
        Separation distance between cross-sections [km]
    """
    #
    # Plot the traces of cross-sections
    fou = open('cs_traces.cs', 'w')

    cs_dict = {}
    for idx, cs in enumerate(trench.iterate_cross_sections(interdistance,
                                                           cs_len)):
        if cs is not None:
            cs_dict['%s' % idx] = cs
            if qual==1:
                cs.plo[:] = ([x-360. if x>180. else x for x in cs.plo[:]])
            tmps = '%f %f %f %f %f %d %s\n' % (cs.plo[0],
                                            cs.pla[0],
                                            cs_depth,
                                            cs_len,
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

    minlo = min(trench.axis[:, 0]) - 5
    minla = min(trench.axis[:, 1]) - 5
    maxlo = max(trench.axis[:, 0]) + 5
    maxla = max(trench.axis[:, 1]) + 5
    midlo = (minlo+maxlo)/2
    midla = (minla+maxla)/2

    fig = plt.figure(figsize=(12,9))

    #
    # Plot the basemap
    m = Basemap(llcrnrlon=minlo, llcrnrlat=minla,
                urcrnrlon=maxlo, urcrnrlat=maxla,
                resolution='i', projection='tmerc',
                lon_0=midlo, lat_0=midla)

    #
    # Draw paralleles and meridians with labels
    # labels = [left,right,top,bottom]
    m.drawcoastlines()
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
    plt.colorbar(p,fraction=0.02, pad=0.04, extend='max')

    #
    # Plot the traces of cross-sections
    ts = trench.resample(interdistance)

    x, y = m(trench.axis[:, 0], trench.axis[:, 1])
    plt.plot(x, y, '-g', linewidth=2, zorder=10)
    x, y = m(ts.axis[:, 0], ts.axis[:, 1])
    plt.plot(x, y, '--y', linewidth=4, zorder=20)

    for key in cs_dict:
        cs = cs_dict[key]
        if cs is not None:
            x, y = m(cs.plo, cs.pla)
            plt.plot(x, y, ':r', linewidth=2, zorder=20)
            text = plt.text(x[0], y[0], '%s' % key, ha='center',
                            va='center', size=10, zorder=30)
            text = plt.text(x[-1], y[-1], '%s' % key, ha='center',
                            va='center', size=10, zorder=30)
            text.set_path_effects([PathEffects.withStroke(linewidth=3,
                                                          foreground="w")])
    plt.show()


def main(argv):
    """
    argv[0] is the .ini file
    """

    # Parse .ini file
    config = configparser.ConfigParser()
    config.read(argv[0])
    fname_trench = config['data']['trench_axis_filename']
    fname_eqk_cat = config['data']['catalogue_pickle_filename']
    cs_length = float(config['section']['lenght'])
    cs_depth = float(config['section']['dep_max'])
    interdistance = float(config['section']['interdistance'])

    # Load trench axis
    fin = open(fname_trench, 'r')
    lotmp = []; latmp = []
    for line in fin:
        aa = re.split('\s+', re.sub('^\s+', '', line))
        lotmp.append(float(aa[0]))
        latmp.append(float(aa[1]))
    fin.close()
    qual = 0
    if (min(lotmp)/max(lotmp)<0.) & (min(lotmp)<-150.):
        qual = 1
        lotmp = (x+360. if x<0. else x for x in lotmp)
    trench = list(zip(lotmp,latmp))
    trench = Trench(numpy.array(trench))

    # Load catalogue
    cat = pickle.load(open(fname_eqk_cat, 'rb'))

    # Get cross-sections
    cs_dict = get_cs(trench, argv[0], cs_length, cs_depth, interdistance, qual)

    # Plotting
    plot(trench, cat, cs_dict, interdistance)

if __name__ == "__main__":
    main(sys.argv[1:])
