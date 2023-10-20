#!/usr/bin/env python
# coding: utf-8

import os
import numpy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from glob import glob
from openquake.wkf.utils import create_folder, _get_src_id


KAVERINA = {'N': 'blue',
            'SS': 'green',
            'R': 'red',
            'N-SS': 'turquoise',
            'SS-N': 'palegreen',
            'R-SS': 'goldenrod',
            'SS-R': 'yellow'}


def get_simpler(dct):
    ndct = {'N': [], 'SS': [], 'R': []}
    for key in dct.keys():
        if key == 'SS-N' or key == 'SS-R':
            ndct['SS'] += dct[key]
        elif key == 'N-SS':
            ndct['N'] += dct[key]
        elif key == 'R-SS':
            ndct['R'] += dct[key]
        else:
            ndct[key] += dct[key]
    return ndct


def get_simplified_classification(histo, keys):
    simpl_class = {'N': 0, 'SS': 0, 'R': 0}
    for num, key in zip(histo, keys):
        if key == 'SS-N' or key == 'SS-R':
            simpl_class['SS'] += num
        elif key == 'N-SS':
            simpl_class['N'] += num
        elif key == 'R-SS':
            simpl_class['R'] += num
        else:
            simpl_class[key] += num
    return simpl_class


def mecclass(plungt, plungb, plungp):
    """
    This is taken from the FMC package.
    See https://josealvarezgomez.wordpress.com/

    It provides a classification of the rupture mechanism based on the
    Kaverina et al. (1996) methodology.

    :parameter plungt:
    :parameter plungb:
    :parameter plungp:

    """
    plunges = numpy.asarray((plungp, plungb, plungt))
    P = plunges[0]
    B = plunges[1]
    T = plunges[2]
    maxplung, axis = plunges.max(0), plunges.argmax(0)
    if maxplung >= 67.5:
        if axis == 0:  # P max
            clase = 'N'  # normal faulting
        elif axis == 1:  # B max
            clase = 'SS'  # strike-slip faulting
        elif axis == 2:  # T max
            clase = 'R'  # reverse faulting
    else:
        if axis == 0:  # P max
            if B > T:
                clase = 'N-SS'  # normal - strike-slip faulting
            else:
                clase = 'N'  # normal faulting
        if axis == 1:  # B max
            if P > T:
                clase = 'SS-N'  # strike-slip - normal faulting
            else:
                clase = 'SS-R'  # strike-slip - reverse faulting
        if axis == 2:  # T max
            if B > P:
                clase = 'R-SS'  # reverse - strike-slip faulting
            else:
                clase = 'R'  # reverse faulting
    return clase


def plot_histogram(gs0, fmclassification, title=""):

    classes = ['N', 'R', 'SS', 'N-SS', 'SS-N', 'SS-R', 'R-SS']

    bin_edges = numpy.array([0, 1, 2, 3, 4, 5, 6, 7])
    histo = []

    for key in classes:
        if key in fmclassification:
            histo.append(fmclassification[key])
        else:
            histo.append(0)

    simplified = get_simplified_classification(histo, classes)

    histosimple = []
    for key in classes:
        if key in simplified:
            histosimple.append(simplified[key])
        else:
            histosimple.append(0)

    ax = plt.subplot(gs0)
    ax.set_title(title)

    plt.bar(bin_edges[0:-1], histo,
            width=numpy.diff(bin_edges),
            edgecolor='red',
            facecolor='orange',
            linewidth=3,
            alpha=1.0,
            align='edge',
            label='Kaverina')

    plt.bar(bin_edges[0:-1], histosimple,
            width=numpy.diff(bin_edges),
            edgecolor='blue',
            facecolor='None',
            linewidth=3,
            alpha=1.0,
            align='edge',
            label='Simplified')

    plt.ylabel(r'Earthquake count', fontsize=14)
    plt.grid(which='major', axis='y', linestyle='--')

    be = numpy.array(bin_edges)
    mid = be[0:-1]+(be[1]-be[0])/2.
    plt.xticks(mid, classes)

    ylimdff = abs(numpy.diff(numpy.array(plt.gca().get_ylim()))[0])

    for i, h in enumerate(histosimple):
        prc = h/sum(histosimple)
        dlt = 0.025 * ylimdff
        if prc > 0.5:
            dlt = -0.04 * ylimdff
        if prc > 1e-1:
            plt.text(mid[i], h+dlt, "{:.2f}".format(prc))
    _ = plt.legend()


def plot_xx(gs0, dip_1, dip_2, strike_1, strike_2):

    classes = ['N', 'R', 'SS', 'N-SS', 'SS-N', 'SS-R', 'R-SS']

    KAVERINA = {'N': 'blue',
                'SS': 'green',
                'R': 'red',
                'N-SS': 'turquoise',
                'SS-N': 'palegreen',
                'R-SS': 'goldenrod',
                'SS-R': 'yellow'}

    fs = 14
    gs = gs0.subgridspec(4, 2, wspace=0.0, hspace=0.0)

    for key, igs in zip(classes, range(0, len(classes))):
        ax = plt.subplot(gs[igs])
        if key in strike_1:
            plt.plot(strike_1[key], dip_1[key], 'o', markersize=8,
                     color=KAVERINA[key])
            plt.plot(strike_2[key], dip_2[key], 'o', markersize=6,
                     color=KAVERINA[key], alpha=0.5, markeredgecolor='blue')
        plt.xlim([0, 360])
        plt.ylim([0, 90])
        plt.grid(which='major', )

        x = numpy.arange(30, 90, 30)
        ax.set_yticks(x)
        x = numpy.arange(30, 360, 30)
        ax.set_xticks(x)
        if igs in [0, 1, 2, 3, 4]:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('strike', fontsize=fs)
        if igs in [1, 3, 5]:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel('dip', fontsize=fs)

        plt.text(.05, .90, key,
                 horizontalalignment='left',
                 transform=ax.transAxes)


def plot_density_simple(gs0, dip1, dip2, stk1, stk2):

    fs = 14
    gs = gs0.subgridspec(3, 2, wspace=0.0, hspace=0.0)
    cmap = plt.get_cmap('Blues')

    total_solutions = sum(numpy.array([len(dip1[k]) for k in dip1.keys()]))

    for key, igs in zip(dip1.keys(), range(0, len(dip1.keys()))):

        xbins = numpy.arange(0, 361, 60)
        ybins = numpy.arange(0, 91, 30)
        X, Y = numpy.meshgrid(xbins, ybins)

        ax = plt.subplot(gs[igs, 0])
        plt.set_cmap(cmap)
        if key in dip1.keys():
            hist, xedges, yedges = numpy.histogram2d(stk1[key], dip1[key],
                                                     bins=[xbins, ybins])
            hist = hist.T / total_solutions
            ax.pcolormesh(X, Y, hist)

        plt.xlim([0, 360])
        plt.ylim([0, 90])
        plt.grid(which='major', )

        x = numpy.arange(30, 90, 30)
        ax.set_yticks(x)
        x = numpy.arange(0, 360, 60)
        ax.set_xticks(x)
        if igs in [0, 1]:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('strike', fontsize=fs)
        ax.set_ylabel('dip', fontsize=fs)
        plt.text(.05, .90, "{:s} I plane".format(key),
                 horizontalalignment='left',
                 transform=ax.transAxes)

    for key, igs in zip(dip1.keys(), range(0, len(dip1.keys()))):
        ax = plt.subplot(gs[igs, 1])
        plt.set_cmap(cmap)
        if key in dip1.keys():
            hist, xedges, yedges = numpy.histogram2d(stk2[key], dip2[key],
                                                     bins=[xbins, ybins])
            hist = hist.T / total_solutions
            ax.pcolormesh(X, Y, hist)

        plt.xlim([0, 360])
        plt.ylim([0, 90])
        plt.grid(which='major', )

        x = numpy.arange(30, 90, 30)
        ax.yaxis.set_ticklabels([])
        ax.set_yticks(x)
        x = numpy.arange(60, 360, 60)
        ax.set_xticks(x)
        if igs in [0, 1]:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('strike', fontsize=fs)
        plt.text(.05, .90, "{:s} II plane".format(key),
                 horizontalalignment='left',
                 transform=ax.transAxes)


def plot_yy(gs0, dip1, dip2, stk1, stk2):

    KAVERINA = {'N': 'blue',
                'SS': 'green',
                'R': 'red',
                'N-SS': 'turquoise',
                'SS-N': 'palegreen',
                'R-SS': 'goldenrod',
                'SS-R': 'yellow'}

    fs = 14
    gs = gs0.subgridspec(3, 1, wspace=0.0, hspace=0.0)
    for key, igs in zip(dip1.keys(), range(0, len(dip1.keys()))):
        ax = plt.subplot(gs[igs])
        if key in dip1.keys():
            plt.plot(stk1[key], dip1[key], 'o', markersize=8,
                     color=KAVERINA[key])
            plt.plot(stk2[key], dip2[key], 'o', markersize=6,
                     color=KAVERINA[key], alpha=0.5, markeredgecolor='blue')
        plt.xlim([0, 360])
        plt.ylim([0, 90])
        plt.grid(which='major', )

        x = numpy.arange(30, 90, 30)
        ax.set_yticks(x)
        x = numpy.arange(30, 360, 30)
        ax.set_xticks(x)
        if igs in [0, 1]:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('strike', fontsize=fs)
        ax.set_ylabel('dip', fontsize=fs)
        plt.text(.05, .90, key,
                 horizontalalignment='left',
                 transform=ax.transAxes)


def process_gcmt_datafames(fname_folder: str, folder_out: str):
    """
    :param fnames:
        A list containing the names of the files to be processed or a pattern
    :param folder_out:
        The name of the output folder
    """

    create_folder(folder_out)

    if isinstance(fname_folder, str):
        fnames = [f for f in glob(fname_folder)]
    else:
        fnames = fname_folder

    for fname in fnames:

        df = pd.read_csv(fname)
        if len(df.dip1) < 1:
            continue

        # See https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/gridspec_nested.html
        f = plt.figure(figsize=(15, 15))
        gs0 = gridspec.GridSpec(2, 2, figure=f)
        src_id = _get_src_id(fname)

        ext = "png"
        fmt = "zone_{:s}.{:s}"
        figure_name = os.path.join(folder_out, fmt.format(src_id, ext))
        cat_name = os.path.join(folder_out, fmt.format(src_id, "csv"))
	
        fmclassification = {}
        eventfm = {}
        dip_1 = {}
        dip_2 = {}
        strike_1 = {}
        strike_2 = {}
        for idx, row in df.iterrows():

            plungeb = row.loc['plunge_b']
            plungep = row['plunge_p']
            plunget = row['plunge_t']
            mclass = mecclass(plunget, plungeb, plungep)
            eventfm[idx] = mclass
            if mclass in fmclassification:
                fmclassification[mclass] += 1
                dip_1[mclass].append(row['dip1'])
                dip_2[mclass].append(row['dip2'])
                strike_1[mclass].append(row['strike1'])
                strike_2[mclass].append(row['strike2'])
            else:
                fmclassification[mclass] = 1
                dip_1[mclass] = [row['dip1']]
                dip_2[mclass] = [row['dip2']]
                strike_1[mclass] = [row['strike1']]
                strike_2[mclass] = [row['strike2']]

        title = "Source: {:s}".format(src_id)
        _ = plot_histogram(gs0[0, 0], fmclassification, title)
        plot_xx(gs0[0, 1], dip_1, dip_2, strike_1, strike_2)

        stk1 = get_simpler(strike_1)
        stk2 = get_simpler(strike_2)
        dip1 = get_simpler(dip_1)
        dip2 = get_simpler(dip_2)
        plot_yy(gs0[1, 0], dip1, dip2, stk1, stk2)

        plot_density_simple(gs0[1, 1], dip1, dip2, stk1, stk2)

        plt.savefig(figure_name, format=ext)
        plt.close()
        
        df['fm'] = eventfm
        df.to_csv(cat_name)

    return fmclassification
