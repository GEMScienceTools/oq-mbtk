#!/usr/bin/env python

import re
import sys
import pickle
import numpy
import scipy
import matplotlib
import configparser
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from obspy.imaging.beachball import beach

from matplotlib.backend_bases import KeyEvent
from matplotlib.patches import (Circle, Rectangle)
from matplotlib.collections import PatchCollection

from openquake.sub.cross_sections import (CrossSection,
                                               CrossSectionData)
from openquake.sub.utils import plot_planes_at, mecclass

from openquake.hazardlib.geo.geodetic import geodetic_distance
from openquake.hazardlib.geo.geodetic import point_at

# basic settings
MAX_DEPTH = 350
YPAD = 10
MAX_DIST = 1000
fig_length = 10


KAVERINA = {'N': 'blue',
            'SS': 'green',
            'R': 'red',
            'N-SS': 'turquoise',
            'SS-N': 'palegreen',
            'R-SS': 'goldenrod',
            'SS-R': 'yellow'}

def onclick(event):
    print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          (event.button, event.x, event.y, event.xdata, event.ydata))

def _plot_h_eqk_histogram(axes, csda, dep_max=[], dis_max=[]):
    """
    """
    if csda.ecat is None:
        return

    plt.sca(axes)

    newcat = csda.ecat

    olo = csda.csec.olo
    ola = csda.csec.ola
    dsts = geodetic_distance(olo, ola,
                             newcat.data['longitude'],
                             newcat.data['latitude'])

    tmp_mag = newcat.data['magnitude'][:]
    tmp_dep = newcat.data['depth'][:]

    numpy.seterr(invalid='ignore')
    iii = numpy.nonzero((tmp_mag > 3.5) & (tmp_dep > 0.))

    if dep_max and dis_max:
        edges_dep = numpy.arange(0, dep_max, 5)
        edges_dist = numpy.arange(0, dis_max, 5)
    else:
        edges_dep = numpy.arange(0, MAX_DEPTH, 5)
        edges_dist = numpy.arange(0, MAX_DIST, 5)

    seism_depth_hist = scipy.histogram(tmp_dep[iii], edges_dep)
    seism_dist_hist = scipy.histogram(dsts[iii], edges_dist)


    plt.bar(edges_dist[:-1], height=seism_dist_hist[0],
            width=numpy.diff(edges_dist)[0], fc='none', ec='blue')

    if csda.gcmt is not None:
        cat_gcmt = csda.gcmt
        cmt_dst = geodetic_distance(olo,
                                    ola,
                                    cat_gcmt.data['longitude'],
                                    cat_gcmt.data['latitude'])
        gcmt_dist_hist = scipy.histogram(cmt_dst, edges_dist)
        plt.bar(edges_dist[:-1], height=gcmt_dist_hist[0],
                width=numpy.diff(edges_dist)[0], fc='red', alpha=0.4)

    xmax = numpy.ceil(max(seism_depth_hist[0])/10.)*10.

    plt.xlim([0, xmax+xmax*0.05])
    plt.xlabel('Distance from the trench [km]')

    # Moving ticks on top
    axes.xaxis.tick_top()
    axes.xaxis.set_label_position('top')
    axes.set_axisbelow(False)
    axes.grid(which='both', zorder=20)
    ymax = numpy.ceil(max(seism_dist_hist[0])/10.)*10.
    axes.set_ylim([0, ymax+ymax*0.05])

    # Limits no fixed
    if dis_max:
        axes.set_xlim([0, dis_max])
    else:
        axes.set_xlim([0, MAX_DIST])


def _plot_v_eqk_histogram(axes, csda, dep_max=[], dis_max=[]):

    if csda.ecat is None:
        return

    newcat = csda.ecat
    tmp_mag = newcat.data['magnitude'][:]
    tmp_dep = newcat.data['depth'][:]
    iii = numpy.nonzero((tmp_mag > 3.5) & (tmp_dep > 0.))


    if dep_max and dis_max:
        edges_dep = numpy.arange(0, dep_max, 5)
        edges_dist = numpy.arange(0, dis_max, 5)
    else:
        edges_dep = numpy.arange(0, MAX_DEPTH, 5)
        edges_dist = numpy.arange(0, MAX_DIST, 5)

    seism_depth_hist = scipy.histogram(tmp_dep[iii], edges_dep)

    plt.barh(edges_dep[:-1], seism_depth_hist[0],
             height=numpy.diff(edges_dep)[0], ec='blue')
    plt.ylabel('Depth [km]')

    if csda.gcmt is not None:
        cat_gcmt = csda.gcmt
        tmp_dep = cat_gcmt.data['depth'][:]
        gcmt_dep_hist = scipy.histogram(tmp_dep, edges_dep)

        plt.barh(edges_dep[:-1], gcmt_dep_hist[0]-1,
                height=numpy.diff(edges_dep)[0]-1,fc='red' )

    xmax = numpy.ceil(max(seism_depth_hist[0])/10.)*10.
    axes.grid(which='both', zorder=20)
    axes.set_xlim([0, xmax+xmax*0.05])
    if dep_max:
        axes.set_ylim([dep_max, -YPAD])
        axes.set_ybound(lower=dep_max, upper=-YPAD)
    else:
        axes.set_ylim([MAX_DEPTH, -YPAD])
        axes.set_ybound(lower=MAX_DEPTH, upper=-YPAD)

    axes.invert_xaxis()

def _plot_slab1pt0(axes, csda):
    """
    :parameter axes:
    :parameter csda:
    """
    if csda.slab1pt0 is None:
        return
    plt.sca(axes)
    olo = csda.csec.olo
    ola = csda.csec.ola
    slab1pt0 = csda.slab1pt0
    slb_dst = geodetic_distance(olo, ola, slab1pt0[:, 0], slab1pt0[:, 1])
    slb_dep = slab1pt0[:, 2]
    iii = numpy.argsort(slb_dst)
    if len(iii) > 2:
        plt.plot(slb_dst[iii], -1*slb_dep[iii], '-b', linewidth=3, zorder=30)
        plt.text(slb_dst[iii[-1]], -1*slb_dep[iii[-1]], 'Slab1.0', fontsize=8)

def _plot_np_intersection(axes, csda):
    """
    """

    if csda.gcmt is None:
        return

    olo = csda.csec.olo
    ola = csda.csec.ola

    cat_gcmt = csda.gcmt
    cmt_dst = geodetic_distance(olo, ola, cat_gcmt.data['longitude'],
                                cat_gcmt.data['latitude'])
    cmt_dep = cat_gcmt.data['depth']
    cmts = numpy.array(cat_gcmt.gcmts)

    idx = 0
    for ddd, dep, eve, mag, yea in zip(list(cmt_dst),
                                       list(cmt_dep),
                                       list(cmts),
                                       cat_gcmt.data['magnitude'],
                                       cat_gcmt.data['year']):

        if yea > 1000 and mag > 1.0:

            # Kaverina classification
            plungeb = cat_gcmt.data['plunge_b'][idx]
            plungep = cat_gcmt.data['plunge_p'][idx]
            plunget = cat_gcmt.data['plunge_t'][idx]
            mclass = mecclass(plunget, plungeb, plungep)

            plot_planes_at(ddd,
                           dep,
                           [eve.nodal_planes.nodal_plane_1['strike'],
                            eve.nodal_planes.nodal_plane_2['strike']],
                           [eve.nodal_planes.nodal_plane_1['dip'],
                            eve.nodal_planes.nodal_plane_2['dip']],
                           [mag, mag],
                           strike_cs=csda.csec.strike[0],
                           dip_cs=90.,
                           aratio=1.0,
                           color=KAVERINA[mclass],
                           linewidth=2.0,
                           axis=axes)
        idx += 1


def _plot_focal_mech(axes, csda):
    """
    :parameter axes:
    :parameter csda:
    """
    if csda.gcmt is None:
        return

    olo = csda.csec.olo
    ola = csda.csec.ola

    cat_gcmt = csda.gcmt
    cmt_dst = geodetic_distance(olo, ola, cat_gcmt.data['longitude'],
                                cat_gcmt.data['latitude'])
    cmt_dep = cat_gcmt.data['depth']
    cmts = numpy.array(cat_gcmt.gcmts)

    idx = 0
    for ddd, dep, eve, mag, yea in zip(list(cmt_dst),
                                       list(cmt_dep),
                                       list(cmts),
                                       cat_gcmt.data['magnitude'],
                                       cat_gcmt.data['year']):

        if yea > 1000 and mag > 1.0:

            # Kaverina classification
            plungeb = cat_gcmt.data['plunge_b'][idx]
            plungep = cat_gcmt.data['plunge_p'][idx]
            plunget = cat_gcmt.data['plunge_t'][idx]
            mclass = mecclass(plunget, plungeb, plungep)

            com = eve.moment_tensor._to_6component()
            # REMOVE
            try:
                bcc = beach(com, xy=(ddd, dep), width=eve.magnitude*2,
                            linewidth=1, zorder=20, size=mag,
                            facecolor=KAVERINA[mclass])
                bcc.set_alpha(0.5)
                axes.add_collection(bcc)
            except:
                pass
        idx += 1


def _plot_moho(axes, csda):
    """
    :parameter axes:
    :parameter csda:
    """
    if csda.moho is None:
        print("No CRUST1.0...")
        return
    plt.sca(axes)
    olo = csda.csec.olo
    ola = csda.csec.ola
    moho = csda.moho
    if moho.size==3:
       moho = numpy.concatenate((moho,moho),axis=0).reshape((2,3))
    mdsts = geodetic_distance(olo, ola, moho[:, 0], moho[:, 1])
    iii = numpy.argsort(mdsts)
    plt.plot(mdsts[iii], moho[iii, 2], '--p', zorder=100, linewidth=2)
    #plt.text(mdsts[iii[-1]], moho[iii[-1]], 'Crust1.0', fontsize=8)

def _plot_litho(axes, csda):
    """
    :parameter axes:
    :parameter csda:
    """
    if csda.litho is None:
        print("No LITHO1.0...")
        return
    plt.sca(axes)
    olo = csda.csec.olo
    ola = csda.csec.ola
    litho = csda.litho
    if litho.size==3:
       litho = numpy.concatenate((litho,litho),axis=0).reshape((2,3))
    lists = geodetic_distance(olo, ola, litho[:, 0], litho[:, 1])
    lll = numpy.argsort(lists)
    plt.plot(lists[lll], litho[lll, 2], '-.', zorder=100, linewidth=2)
    #plt.text(lists[lll[-1]], litho[lll[-1]], 'Litho1.0', fontsize=8)


def _plot_topo(axes, csda):
    """
    :parameter axes:
    :parameter csda:
    """
    if csda.topo is None:
        return
    plt.sca(axes)
    olo = csda.csec.olo
    ola = csda.csec.ola
    topo = csda.topo
    tbsts = geodetic_distance(olo, ola, topo[:, 0], topo[:, 1])
    jjj = numpy.argsort(tbsts)
    plt.plot(tbsts[jjj], ((-1*topo[jjj, 2])/1000.), '-g', zorder=100,
             linewidth=2)


def _plot_volc(axes, csda):
    """
    :parameter axes:
    :parameter csda:
    """
    if csda.volc is None:
        return

    olo = csda.csec.olo
    ola = csda.csec.ola
    patches = []

    if (len(csda.volc)-1) >= 1:
        vuls = geodetic_distance(olo, ola,
                                 csda.volc[:, 0],
                                 csda.volc[:, 1])
        for v in vuls:
            square = Rectangle((v, -10.0), 7, 12)
            patches.append(square)

    else:
        vuls = geodetic_distance(olo, ola,
                                 csda.volc[0],
                                 csda.volc[1])
        square = Rectangle((vuls, -10.0), 7, 12)
        patches.append(square)

    vv = PatchCollection(patches, zorder=6, color='red', edgecolors='red')
    vv.set_alpha(0.85)
    axes.add_collection(vv)




def _plot_eqks(axes, csda):
    """
    :parameter axes:
    :parameter csda:
    """

    if csda.ecat is None:
        return

    newcat = csda.ecat

    olo = csda.csec.olo
    ola = csda.csec.ola
    dsts = geodetic_distance(olo, ola,
                             newcat.data['longitude'],
                             newcat.data['latitude'])
    sze = (newcat.data['magnitude'])**0.5
    patches = []
    for dst, dep, mag in zip(dsts,
                             newcat.data['depth'],
                             newcat.data['magnitude']):
        circle = Circle((dst, dep), (mag*0.5)**1.5, ec='white')
        patches.append(circle)
    colors = newcat.data['magnitude']
    p = PatchCollection(patches, zorder=6, edgecolors='white')
    p.set_alpha(0.5)
    p.set_array(numpy.array(colors))
    axes.add_collection(p)
    #plt.colorbar(p, fraction=0.1)


def _print_legend(axes, depp, lnght):
    x = int(lnght / 2.)
    xstep = 40
    if depp:
        y = depp+27
    else:
        y = MAX_DEPTH+27

    patches = []
    note = 'Rupture mechanism classification (Kaverina et al. 1996)'
    if depp:
        axes.annotate(note, xy=(x, depp+20), xycoords='data',
                      annotation_clip=False, fontsize=8)
    else:
        axes.annotate(note, xy=(x, MAX_DEPTH+20), xycoords='data',
                      annotation_clip=False, fontsize=8)

    for key in sorted(KAVERINA):
        box = matplotlib.patches.Rectangle(xy=(x, y), width=10, height=10,
                                           color=KAVERINA[key], clip_on=False)
        axes.annotate(key, xy=(x+12,y+8), annotation_clip=False, fontsize=8)
        x += xstep
        axes.add_patch(box)


def _print_info(axes, csec, depp):
    """
    """
    plt.sca(axes)
    note = 'Cross-Section origin: %.2f %.2f' % (csec.olo, csec.ola)
    axes.annotate(note, xy=(0.0, depp+20), xycoords='data',
                  annotation_clip=False, fontsize=8)

    note = 'Cross-Section strike: %.1f [degree]' % (csec.strike[0])
    axes.annotate(note, xy=(0.0, depp+30), xycoords='data',
                  annotation_clip=False, fontsize=8)

    note = 'Cross-Section lenght: %.1f [km]' % (csec.length[0])
    plt.gca().annotate(note, xy=(0.0, depp+40), xycoords='data',
                       annotation_clip=False, fontsize=8)

def plot(csda, depp, lnght):
    """
    """
    # Computing figure width
    fig_width = fig_length * (depp+YPAD) / lnght

    # Creating the figure
    fig = plt.figure(figsize=(fig_length, fig_width))

    #fig = plt.figure(figsize=(15,9))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 5], height_ratios=[1, 5])
    #
    ax0 = plt.subplot(gs[0])
    plt.axis('off')

    ax3 = plt.subplot(gs[3])
    ax3.xaxis.tick_top()
    #ax3.set_xticklabels([])
    #ax3.set_yticklabels([])

    _print_info(plt.subplot(gs[3]), csda.csec, depp)
    _print_legend(plt.subplot(gs[3]), depp, lnght)

    # Plotting
    _plot_eqks(plt.subplot(gs[3]), csda)
    _plot_moho(plt.subplot(gs[3]), csda)
    _plot_litho(plt.subplot(gs[3]), csda)
    _plot_topo(plt.subplot(gs[3]), csda)
    _plot_volc(plt.subplot(gs[3]), csda)

    _plot_focal_mech(plt.subplot(gs[3]), csda)
    _plot_slab1pt0(plt.subplot(gs[3]), csda)
    _plot_np_intersection(plt.subplot(gs[3]), csda)
    _plot_h_eqk_histogram(plt.subplot(gs[1]), csda, depp, lnght)
    _plot_v_eqk_histogram(plt.subplot(gs[2]), csda, depp, lnght)

    # Main panel
    ax3 = plt.subplot(gs[3])
    ax3.autoscale(enable=False, tight=True)
    ax3.invert_yaxis()
    #plt.xlim([0, 500])

    plt.xlim([0, lnght])
    plt.ylim([depp, -YPAD])

    ax3.grid(which='both', zorder=20)

    # Showing results
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    line, = ax3.plot([], [], zorder=100)  # empty line
    point, = ax3.plot([], [], 'xr', zorder=100)
    linebuilder = LineBuilder(line, point, csda.csec)

    return fig



class LineBuilder:

    def __init__(self, line, point, csec):
        self.line = line
        self.point = point
        self.xp = list(point.get_xdata())
        self.yp = list(point.get_ydata())
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)
        self.pid = line.figure.canvas.mpl_connect('key_press_event', self)
        self.csec = csec
        self.data = []

    def __call__(self, event):

        if isinstance(event, KeyEvent):
            if event.key is 'd':
                print('----------------------')
                self.xs = []
                self.ys = []
                self.xp = []
                self.yp = []
                self.line.set_data(self.xp, self.yp)
                self.point.set_data(self.xs, self.ys)
                self.line.figure.canvas.draw()
                self.point.figure.canvas.draw()
                self.data = []
            elif event.key is 'f':
                dat = numpy.array(self.data)
                fname = './cs_%s.csv' % (self.csec.ids)
                numpy.savetxt(fname, dat)
                print('Section data saved to: %s' % (fname))
            else:
                pass

        else:
            olo = self.csec.olo
            ola = self.csec.ola
            assert len(self.csec.strike) == 1
            if event.xdata is not None:
                strike = self.csec.strike[0]
                nlo, nla = point_at(olo, ola, strike, event.xdata)

                cnt = len(self.xs)+1
                print('%03d, %+7.4f, %+7.4f, %6.2f' % (cnt, nlo, nla,
                                                       event.ydata))

                if event.inaxes != self.line.axes:
                    return

                self.xp.append(event.xdata)
                self.yp.append(event.ydata)
                self.xs.append(event.xdata)
                self.ys.append(event.ydata)
                self.data.append([nlo, nla, event.ydata])
                self.point.set_data(self.xs, self.ys)
                self.line.set_data(self.xs, self.ys)
                self.line.figure.canvas.draw()
                self.point.figure.canvas.draw()


def plt_cs(olo, ola, depp, lnght, strike, ids, ini_filename):
    """
    """
    csec = CrossSection(olo, ola, [lnght], [strike], ids)
    csda = CrossSectionData(csec)

    config = configparser.ConfigParser()
    config.read(ini_filename)
    fname_trench = config['data']['trench_axis_filename']
    fname_eqk_cat = config['data']['catalogue_pickle_filename']
    fname_slab = config['data']['slab1pt0_filename']
    fname_crust = config['data']['crust1pt0_filename']
    fname_gcmt = config['data']['gcmt_filename']
    fname_topo = config['data']['topo_filename']
    fname_litho = config['data']['litho_filename']
    fname_volc = config['data']['volc_filename']

    csda.set_trench_axis(fname_trench)
    cat = pickle.load(open(fname_eqk_cat, 'rb'))
    csda.set_catalogue(cat)
    if re.search('[a-z]', fname_slab):
        csda.set_slab1pt0(fname_slab)
    csda.set_crust1pt0_moho_depth(fname_crust)
    csda.set_gcmt(fname_gcmt)
    csda.set_topo(fname_topo)
    csda.set_litho_moho_depth(fname_litho)
    csda.set_volcano(fname_volc)

    fig = plot(csda, depp, lnght)

    return fig


def main(argv):

    olo = float(argv[0])
    ola = float(argv[1])
    depp = float(argv[2])
    lnght = float(argv[3])
    strike = float(argv[4])
    ids = argv[5]
    ini_filename = argv[6]

    print('Working on cross section: %s' % (ids))
    fig = plt_cs(olo, ola, depp, lnght, strike, ids, ini_filename)
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])
