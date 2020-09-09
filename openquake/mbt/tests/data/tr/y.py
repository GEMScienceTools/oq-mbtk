import sys
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d

from pyproj import Proj
from openquake.mbt.tools.tr.catalogue import get_catalogue
from openquake.sub.utils import _read_edges, plot_complex_surface
from openquake.hazardlib.geo.utils import plane_fit


def get_points_on_plane(edges, poi):
    """
    :param edges:
    :param points:
    """

    #
    # creating a matrix of points
    pnts = []
    for edge in edges:
        pnts += [[pnt.longitude, pnt.latitude, pnt.depth] for pnt in
                 edge.points]
    pnts = np.array(pnts)
    #
    # projecting the points
    p = Proj(proj='lcc', lon_0=np.mean(pnts[:, 0]), lat_2=45)
    x, y = p(pnts[:, 0], pnts[:, 1])
    x = x / 1e3  # m -> km
    y = y / 1e3  # m -> km
    #
    # fit the plane
    tmp = np.vstack((x.flatten(), y.flatten(), pnts[:, 2].flatten())).T
    pnt, ppar = plane_fit(tmp)
    # z = -(a * x + b * y - d) / c
    d = - np.sum(pnt*ppar)
    xp, yp = p(poi[:, 0], poi[:, 1])
    poi[:, 2] = -(ppar[0] * xp/1e3 + ppar[1] * yp/1e3 + d) / ppar[2]
    #
    return poi


def main(argv):
    catalogue_filename = 'catalogue.csv'
    c = get_catalogue(catalogue_filename)

    edges_folder = './profiles/int/'
    tedges = _read_edges(edges_folder)

    minlo = 9.0
    maxlo = 11.0
    minla = 44.0
    maxla = 46.0
    npo = 500

    poi = np.zeros((npo, 3))
    poi[:, 0] = minlo + np.random.rand((npo)) * (maxlo-minlo)
    poi[:, 1] = minla + np.random.rand((npo)) * (maxla-minla)
    poi = get_points_on_plane(tedges, poi)

    fig, ax = plot_complex_surface(tedges)
    ax.plot(c.data['longitude'], c.data['latitude'], c.data['depth'], 'og')
    ax.plot(poi[:, 0], poi[:, 1], poi[:, 2], '.r')

    circle = Circle((10, 45), .5, alpha=.8)
    ax.add_patch(circle)
    art3d.pathpatch_2d_to_3d(circle, z=30, zdir='z')

    for i, (lo, la, de) in enumerate(zip(c.data['longitude'],
                                         c.data['latitude'],
                                         c.data['depth'])):
        ts = '{:d}'.format(i)
        ax.text(lo, la, de, ts)
    ax.set_zlim([0, 70])
    ax.invert_zaxis()
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
