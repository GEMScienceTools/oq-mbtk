"""
Module :module:`openquake.mbt.tools.smooth3d` exports :class:`Smooth3d`
"""

import numpy as np
import scipy.constants as consts

from rtree import index
from pyproj import Proj
from scipy.stats import norm


def _generator(mesh, p):
    for cnt, (lon, lat, dep) in enumerate(zip(mesh.lons.flatten('F'),
                                              mesh.lats.flatten('F'),
                                              mesh.depths.flatten('F'))):
        x, y = tuple(t/1e3 for t in p(lon, lat))
        yield (cnt, (x, y, dep, x, y, dep), 1)


class Smoothing3D:
    """
    Class for performing the 3D smoothing of a catalogue
    """

    def __init__(self, catalogue, mesh, bin_h, bin_z):
        """
        :parameter catalogue:
            A catalogue in the hmtk format
        :parameter mesh:
            An instance of :class:`openquake.hazardlib.geo.mesh.Mesh`
        :parameter bin_h:
            The lenght of the cell composing the grid [km]
        :parameter bin_z:
            The lenght of the cell along the vertical axis [km]
        """
        self.catalogue = catalogue
        self.mesh = mesh
        self.bin_h = bin_h
        self.bin_z = bin_z
        self._create_spatial_index()

    def _create_spatial_index(self):
        """
        This creates the spatial index for the input mesh
        """

        # Setting rtree properties
        prop = index.Property()
        prop.dimension = 3

        # Set the geographic projection
        lons = self.mesh.lons.flatten('F')
        mean_lat = np.mean(self.mesh.lats)
        self.p = Proj(proj='lcc', lon_0=np.mean(lons), lat_1=mean_lat-10.,
                      lat_2=mean_lat+10.)

        # Create the spatial index for the grid mesh
        r = index.Index(_generator(self.mesh, self.p), properties=prop)

        # Set the rtree
        self.rtree = r

    def gaussian(self, bffer, sigmas):
        """
        Smooths the seismicity using a gaussian kernel.

        :parameter bffer:
            Maximum distance [km]
        :parameter sigma:
            Standard deviation of the gaussian distribution
        """

        # Initialise the array where we store the results of the smoothing
        values = np.zeros((len(self.mesh.lons.flatten('F'))))

        # Projected coordinates of the catalogue
        xs, ys = self.p(self.catalogue.data['longitude'],
                        self.catalogue.data['latitude'])
        xs /= 1e3
        ys /= 1e3

        # Projected coordinates of the grid
        xg, yg = self.p(self.mesh.lons.flatten('F'),
                        self.mesh.lats.flatten('F'))
        xg /= 1e3
        yg /= 1e3
        zg = self.mesh.depths.flatten('F')

        # Smoothing the catalogue
        for x, y, z in zip(xs, ys, self.catalogue.data['depth']):

            # find nodes within the bounding box
            idxs = list(self.rtree.intersection((x-bffer*1.05,
                                                 y-bffer*1.05,
                                                 max(0, z-bffer*1.05),
                                                 x+bffer*1.05,
                                                 y+bffer*1.05,
                                                 z+bffer*1.05)))
            if len(idxs):

                # Distances between earthquake and the selected nodes of the
                # 3D mesh
                dsts = ((x-xg[idxs])**2 + (y-yg[idxs])**2 +
                        (z-zg[idxs])**2)**.5

                # Find the indexes of the cells at a distance shorter than
                # 'bffer'
                jjj = np.ndarray.astype(np.nonzero(dsts < bffer)[0], int)
                idxs = np.array(idxs)
                iii = idxs[jjj]

                # `data` contains the coordinates of the points where we
                # calculate the values of the multivariate gaussian
                # MN: 'data' assigned but never used
                # data = np.vstack((xg[iii].flatten(), yg[jjj].flatten(),
                #                  zg[iii].flatten())).T
                # xxx = multivariate_gaussian([x, y, z], sigmas, data)

                xxx = 1./dsts[jjj]**0.01

                # update the array where we store the results of the smoothing
                values[iii] += xxx

        return values/np.sum(values)


def multivariate_gaussian(means, sigmas, data):
    """
    :parameter means:
    :parameter sigmas:
    :parameter data:
    """
    sq2pi = (2*consts.pi)**0.5
    out = np.ones((data.shape[0]))
    for i, (mu, sigma) in enumerate(zip(list(means), list(sigmas))):
        f1 = 1./(sigma*sq2pi)
        dst = (data[:, i]-mu)
        f2 = np.exp(-dst**2./(2*sigma**2))
        out *= (f1*f2)
    return out
