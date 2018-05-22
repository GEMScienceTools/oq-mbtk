"""
"""

import pickle
import numpy as np


def plot_catalogue(ax, pickle_fname, vsc):
    cat = pickle.load(open(pickle_fname, 'rb'))
    ax.scatter(cat.data['longitude'], cat.data['latitude'],
               cat.data['depth']*vsc,
               s=cat.data['magnitude'], c=cat.data['depth'])


def plot_mesh(ax, msh, vsc, lw=1, color='green'):
    """
    :param numpy.ndarray msh:
    :param float vsc:
    """
    for i in range(0, msh.shape[0] - 1):
        for j in range(0, msh.shape[1] - 1):
            xt = [msh[i, j, 0], msh[i + 1, j, 0], msh[i + 1, j + 1, 0],
                  msh[i, j + 1, 0], msh[i, j, 0]]
            yt = [msh[i, j, 1], msh[i + 1, j, 1], msh[i + 1, j + 1, 1],
                  msh[i, j + 1, 1], msh[i, j, 1]]
            zt = [msh[i, j, 2] * vsc, msh[i + 1, j, 2] * vsc,
                  msh[i + 1, j + 1, 2] * vsc,
                  msh[i, j + 1, 2] * vsc, msh[i, j, 2] * vsc]
            if all(np.isfinite(xt)):
                ax.plot(xt, yt, zt, color=color, linewidth=lw)
