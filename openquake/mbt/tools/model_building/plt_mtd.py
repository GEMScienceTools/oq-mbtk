#!/usr/bin/env python

import h5py
import numpy
import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from openquake.hmtk.seismicity.catalogue import Catalogue
from openquake.hmtk.seismicity.selector import CatalogueSelector
from openquake.mbt.tools.model_building.plt_tools import (_load_catalogue,
                                                          _get_extremes)


def plot_mtd(catalogue_fname, label, tr_fname, cumulative, store, mwid=0.1,
             twid=20., pmint=None, pmaxt=None, ylim=None):
    #
    #
    fig = create_mtd(catalogue_fname, label, tr_fname, cumulative, store, mwid,
                   twid, pmint, pmaxt, ylim)
    #
    # showing figure
    if store is not None:
        lbl = ''
        ext = 'png'
        if label is not None:
            lbl = label
        figure_fname = 'fig_mtd_{:s}.{:s}'.format(lbl, ext)
        plt.savefig(figure_fname, format=ext)
    else:
        plt.show()
    return fig


def create_mtd(catalogue_fname, label, tr_fname, cumulative, store, mwid=0.1,
               twid=20., pmint=None, pmaxt=None, ylim=None):
    """
    :param catalogue_fname:
    :param label:
    :param tr_fname:
    """
    mwid = float(mwid)
    twid = float(twid)
    if pmint is not None:
        pmint = int(pmint)
    if pmaxt is not None:
        pmaxt = int(pmaxt)
    #
    # loading catalogue
    if isinstance(catalogue_fname, str):
        cat = _load_catalogue(catalogue_fname)
    elif isinstance(catalogue_fname, Catalogue):
        cat = catalogue_fname
    else:
        raise ValueError('Unknown instance')

    # Check catalogue
    if cat is None or len(cat.data['magnitude']) < 1:
        return None

    # Select earthquakes belonging to a given TR
    idx = numpy.full(cat.data['magnitude'].shape, True, dtype=bool)
    if label is not None and tr_fname is not None:
        f = h5py.File(tr_fname, 'r')
        idx = f[label][:]
        f.close()
    #
    # select catalogue
    sel = CatalogueSelector(cat, create_copy=False)
    sel.select_catalogue(idx)
    start = datetime.datetime(pmint, 1, 1) if pmint is not None else None
    stop = datetime.datetime(pmaxt, 12, 31) if pmaxt is not None else None
    sel.within_time_period(start, stop)

    # Check if the catalogue contains earthquakes
    if len(cat.data['magnitude']) < 1:
        return None

    #
    # find rounded min and max magnitude
    mmin, mmax = _get_extremes(cat.data['magnitude'], mwid)
    tmin, tmax = _get_extremes(cat.data['year'], twid)
    if ylim is not None:
        mmin = ylim[0]
        mmax = ylim[1]
    #
    #
    if pmint is None:
        pmint = tmin
    if pmaxt is None:
        pmaxt = tmax
    #
    # histogram
    bins_ma = numpy.arange(mmin, mmax+mwid*0.01, mwid)
    bins_time = numpy.arange(tmin, tmax+twid*0.01, twid)
    his, _, _ = numpy.histogram2d(cat.data['year'], cat.data['magnitude'],
                                  bins=(bins_time, bins_ma))
    his = his.T
    #
    # complementary cumulative
    if cumulative:
        ccu = numpy.zeros_like(his)
        for i in range(his.shape[1]):
            cc = numpy.cumsum(his[::-1, i])
            ccu[:, i] = cc[::-1]
    #
    # plotting
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    #
    #
    X, Y = numpy.meshgrid(bins_time, bins_ma)
    if cumulative:
        his = ccu
    pcm = ax.pcolormesh(X, Y, his, norm=colors.LogNorm(vmin=1e-1,
                                                       vmax=his.max()),
                        cmap='BuGn')
    #
    # plotting number of occurrences
    for it, vt in enumerate(bins_time[:-1]):
        for im, vm in enumerate(bins_ma[:-1]):
            ax.text(vt+twid/2, vm+mwid/2, '{:.0f}'.format(his[im, it]),
                    fontsize=7, ha='center')

    #
    # plotting colorbar
    cb = fig.colorbar(pcm, ax=ax, extend='max')
    cb.set_label('Number of occurrences')
    #
    # finishing the plot
    plt.ylabel('Magnitude')
    plt.xlabel('Time')
    if label is not None:
        plt.title('label: {:s}'.format(label))
    plt.grid(linestyle='-')

    return fig
