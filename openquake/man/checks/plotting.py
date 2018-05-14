
import numpy as np
import matplotlib.pyplot as plt

from openquake.hazardlib.source.point import PointSource


def plot_mfd_cumulative(mfd, fig=None, label='', color=None, linewidth=1,
                        title=''):
    aa = np.array(mfd.get_annual_occurrence_rates())
    cml = np.cumsum(aa[::-1, 1])
    if color is None:
        color = np.random.rand(3)
    plt.plot(aa[:, 0], cml[::-1], label=label, lw=linewidth, color=color)
    plt.title(title)


def plot_mfd(mfd, fig=None, label='', color=None, linewidth=1):
    bw = 0.1
    aa = np.array(mfd.get_annual_occurrence_rates())
    occs = []
    if color is None:
        color = np.random.rand(3)
    for mag, occ in mfd.get_annual_occurrence_rates():
        plt.plot([mag-bw/2, mag+bw/2], [occ, occ], lw=2, color='grey')
        occs.append(occ)
    plt.plot(aa[:, 0], aa[:, 1], label=label, lw=linewidth)


def plot_models(models):
    """
    :param models:
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for model in models:
        for src in model:
            plot_source(src)


def plot_source(src):
    """
    :param src:
    """
    if isinstance(src, PointSource):
        plt.plot(src.location.longitude, src.location.latitude, 'sg')
    else:
        print(type(src))
        raise ValueError('Unhandled exception')


def plot_polygon(poly):
    """
    :param src:
    """
    plt.plot(poly.lons, poly.lats, '-b', linewidth=3)


def plot_end():
    plt.show()
