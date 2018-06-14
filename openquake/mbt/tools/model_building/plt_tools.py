
import numpy
import pickle
import pathlib

from openquake.hmtk.parsers.catalogue import CsvCatalogueParser


def _get_extremes(dat, wid):
    vmin = min(dat)
    vmin = vmin - vmin % wid
    vmax = max(dat)
    vmax = vmax + (wid - vmax % wid)
    return vmin, vmax


def _load_catalogue(catalogue_fname):
    ext = pathlib.Path(catalogue_fname).suffix
    if ext == '.pkl' or ext == '.p':
        #
        # load pickle file
        cat = pickle.load(open(catalogue_fname, 'rb'))
    elif ext == '.csv' or ext == '.hmtk':
        #
        # load hmtk file
        parser = CsvCatalogueParser(catalogue_fname)
        cat = parser.read_file()
    return cat


def plot_edges(edges, ax):
    """
    :param edges:
        A list of :class:`openquake.hazardlib.geo.line.Line` instances
    :param ax:
        A
    """
    for e in edges:
        coo = [(p.longitude, p.latitude, p.depth) for p in e.points]
        coo = numpy.array(coo)
        ax.plot(coo[:, 0], coo[:, 1], coo[:, 2], '-')
        ax.plot([coo[0, 0]], [coo[0, 1]], [coo[0, 2]], 'o')


def plot_profiles(pfiles, ax, names=None):
    """
    :param pfiles:
        A
    :param ax:
        A
    """
    for i, pr in enumerate(pfiles):
        coo = [(p.longitude, p.latitude, p.depth) for p in pr.points]
        coo = numpy.array(coo)
        ax.plot(coo[:, 0], coo[:, 1], coo[:, 2], ':')
        if names is not None:
            ax.text(coo[0, 0], coo[0, 1], coo[0, 2], names[i])


def plot_profiles_names(pfiles, ax, names):
    """
    :param pfiles:
        A
    :param ax:
        A
    """
    for i, pr in enumerate(pfiles):
        coo = [(p.longitude, p.latitude, p.depth) for p in pr.points]
        coo = numpy.array(coo)
        ax.text(coo[0, 0], coo[0, 1], coo[0, 2], names[i])
