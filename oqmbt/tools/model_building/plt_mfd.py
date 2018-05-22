import h5py
import numpy

import matplotlib.pyplot as plt

from openquake.hmtk.seismicity.selector import CatalogueSelector
from openquake.hmtk.seismicity.occurrence.weichert import Weichert

from oqmbt.tools.model_building.plt_tools import _load_catalogue, _get_extremes


def _get_compl_table(hdf5_fname, label):
    f = h5py.File(hdf5_fname, 'r')
    tab = f[label][:]
    f.close()
    return tab


def _compute_mfd(cat, compl_table, mwid):
    """
    """
    weichert_config = {'magnitude_interval': mwid,
                       'reference_magnitude': 0.0}
    weichert = Weichert()
    bval_wei, sigmab, aval_wei, sigmaa = weichert.calculate(cat,
                                                            weichert_config,
                                                            compl_table)
    #
    # info
    print('bval: %.6f (sigma=%.3f)' % (bval_wei, sigmab))
    print('aval: %.6f (sigma=%.3f)' % (aval_wei, sigmaa))
    return aval_wei, bval_wei, sigmaa, sigmab


def plot_mfd(catalogue_fname, grd, label, store, tr_fname,
             compl_table=None, mwid=0.1):
    """
    :param catalogue_fname:
    :param label:
    :param tr_fname:
    :param compl_fname:
    :param grd:
    """
    mwid = float(mwid)
    #
    # loading catalogue
    cat = _load_catalogue(catalogue_fname)
    #
    # select earthquakes belonging to a given TR
    idx = numpy.full(cat.data['magnitude'].shape, True, dtype=bool)
    if label is not None and tr_fname is not None:
        f = h5py.File(tr_fname, 'r')
        idx = f[label][:]
        f.close()
    #
    # select catalogue
    sel = CatalogueSelector(cat, create_copy=False)
    sel.select_catalogue(idx)
    #
    # find rounded min and max magnitude
    mmin, mmax = _get_extremes(cat.data['magnitude'], mwid)
    tmin, tmax = _get_extremes(cat.data['year'], 10)
    #
    # compute histogram
    bins = numpy.arange(mmin, mmax+mwid*0.01, mwid)
    his, _ = numpy.histogram(cat.data['magnitude'], bins)
    #
    # plotting
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    #
    # add cumulative plot
    cs = numpy.cumsum(his[::-1])
    plt.bar(bins[:-1], cs[::-1], mwid, align='edge', ec='cyan', fc='none')
    plt.plot(bins[:-1]+mwid/2, cs[::-1], '-r', label='cumulative')
    #
    # add incremental plot
    plt.bar(bins[:-1], his, mwid, align='edge', ec='orange', fc='none')
    plt.plot(bins[:-1]+mwid/2, his, '-b', label='incremental')
    #
    #
    if grd:
        if compl_table is None:
            compl_table = numpy.array([[tmin, mmin]])
        agr, bgr, asig, bsig = _compute_mfd(cat, compl_table, mwid)
    #
    # info
    num = len(cat.data['magnitude'])
    print('Number of earthquakes in the catalogue          : {:d}'.format(num))
    num = max(cs)
    print('Maximum value in the c. cumulative distribution : {:d}'.format(num))
    #
    # finish plot
    plt.legend()
    plt.yscale('log')
    plt.ylabel('Number of earthquakes')
    plt.xlabel('Magnitude')
    if label is not None:
        plt.title('label: {:s}'.format(label))
    plt.grid(linestyle='-')
    if grd:
        plt.text(0.65, 0.70, 'bval: %.3f (sigma=%.3f)' % (bgr, bsig),
                 horizontalalignment='left',
                 verticalalignment='center',
                 fontsize=8,
                 transform=ax.transAxes)
        plt.text(0.65, 0.75, 'aval: %.3f (sigma=%.3f)' % (agr, asig),
                 horizontalalignment='left',
                 verticalalignment='center',
                 fontsize=8,
                 transform=ax.transAxes)
        #
        #
        print(compl_table[-1, 0])
        ascaled = numpy.log10(10**agr*(tmax-compl_table[-1, 0]))
        v = 10.**(-bins*bgr+ascaled)
        plt.plot(bins, v, '--g', lw=2)

    if store is not None:
        lbl = ''
        ext = 'png'
        if label is not None:
            lbl = label
        figure_fname = 'fig_mfd_{:s}.{:s}'.format(lbl, ext)
        plt.savefig(figure_fname, format=ext)
    else:
        plt.show()
