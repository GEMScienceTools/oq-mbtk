
import h5py
import numpy as np

from mayavi import mlab


def plot_whole_mesh_mayavi(hdf5_fname, vsc=1.0):
    """
    """
    f5 = h5py.File(hdf5_fname, 'r')
    #
    #
    msh = f5['slab/top'][:]
    #
    # plot upper mesh
    plot_mesh_mayavi(msh, vsc)
    #
    #
    for idx, key in enumerate(f5['inslab']):
        if idx > 0:
            continue
        plot_mesh_mayavi(f5['inslab'][key][:], vsc)
    mlab.show()
    #
    #
    f5.close()


def plot_mesh_mayavi(msh, vsc, lw=2, color=(1, 0, 0)):
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
                mlab.plot3d(xt, yt, zt, color=color, line_width=lw)
