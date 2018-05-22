"""
"""

import h5py
import pickle
import numpy as np

from mayavi import mlab


def _get_border(x, y, z):
    """
    """
    x = x.squeeze()
    y = y.squeeze()
    z = z.squeeze()
    #
    #
    xb = list(x[0, :].flatten())
    xb += list(x[:, -1].flatten())
    xb += list(x[-1, ::-1].flatten())
    xb += list(x[::-1, 0].flatten())
    #
    yb = list(y[0, :].flatten())
    yb += list(y[:, -1].flatten())
    yb += list(y[-1, ::-1].flatten())
    yb += list(y[::-1, 0].flatten())
    #
    zb = list(z[0, :].flatten())
    zb += list(z[:, -1].flatten())
    zb += list(z[-1, ::-1].flatten())
    zb += list(z[::-1, 0].flatten())
    #
    #
    xb = np.array(xb)
    yb = np.array(yb)
    zb = np.array(zb)
    #
    idx = np.isfinite(xb)
    #
    return xb[idx], yb[idx], zb[idx]


def plot_ruptures(label, rupture_hdf5_fname, vsc):
    """
    :param label:
    :param rupture_hdf5_fname:
    :param vsc:
    """
    f = h5py.File(rupture_hdf5_fname, 'r')
    for i, key in enumerate(f['ruptures'][label]):
        #
        # geneate a random color
        col = tuple(np.random.rand(3))
        #
        # get nodes
        x = f['ruptures'][label][key]['lons'][:]
        y = f['ruptures'][label][key]['lats'][:]
        z = f['ruptures'][label][key]['deps'][:]
        #
        # plot mesh
        # mlab.points3d(x, y, z*vsc, mode='point', color=col)
        #
        # get and plot border
        xb, yb, zb = _get_border(x, y, z)
        mlab.plot3d(xb, yb, zb*vsc, line_width=0.25, color=col)
    f.close()


def plot_catalogue(pickle_fname, vsc):
    cat = pickle.load(open(pickle_fname, 'rb'))
    mlab.points3d(cat.data['longitude'], cat.data['latitude'],
                  cat.data['depth']*vsc, mode='point', color=(1, 0, 0))


def plot_mesh(msh, vsc, lw=1, color=(0, 1, 0)):
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
                  msh[i + 1, j + 1, 2] * vsc, msh[i, j + 1, 2] * vsc,
                  msh[i, j, 2] * vsc]
            if all(np.isfinite(xt)):
                mlab.plot3d(xt, yt, zt, color=color, line_width=lw)
