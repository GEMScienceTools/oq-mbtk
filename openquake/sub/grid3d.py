"""
:module:
"""

import rtree
import numpy as np

from pyproj import Proj
from scipy.interpolate import griddata
from shapely.geometry import Point

from openquake.sub.misc.alpha_shape import alpha_shape


def generator_function(mesh):
    """
    Generator function for quick loading of a 3D spatial index

    :param mesh:
        An instance of :class:`~openquake.hazardlib.geo.mesh.Mesh`
    """
    #
    lo = mesh.lons.flatten()
    la = mesh.lats.flatten()
    de = mesh.depths.flatten()
    #
    idxs = np.nonzero(np.isfinite(de))
    #
    for i in idxs[0]:
        if i:
            yield (i, (lo[i], la[i], lo[i], la[i]), None)


class Grid3d():
    """
    :param minlo:
    :param maxlo:
    :param minla:
    :param maxla:
    :param minde:
    :param maxde:
    :param hspa:
    :param vspa:
    """

    def __init__(self, minlo, minla, minde, maxlo, maxla, maxde, hspa, vspa):
        """
        """

        minlo = minlo+360 if minlo<0 else minlo
        maxlo = maxlo+360 if maxlo<0 else maxlo

        self.minlo = minlo
        self.minla = minla
        self.minde = minde
        self.maxlo = maxlo
        self.maxla = maxla
        self.maxde = maxde
        self.hspa = hspa
        self.vspa = vspa
        #
        # set projection
        clon = (self.minlo+self.maxlo)/2.
        self.p = Proj('+proj=lcc +lon_0={:f}'.format(clon))
        #
        # initialise the grid
        self._create_equally_spaced_grid()

    def _create_equally_spaced_grid(self):
        """
        """
        #
        # compute the projected coordinates of the limits of the grid
        minx, miny = self.p(self.minlo, self.minla)
        minx = np.floor(minx/self.hspa/1e3)*self.hspa
        miny = np.ceil(miny/self.hspa/1e3)*self.hspa
        #
        maxx, maxy = self.p(self.maxlo, self.maxla)
        maxx = np.floor(maxx/self.hspa/1e3)*self.hspa
        maxy = np.ceil(maxy/self.hspa/1e3)*self.hspa
        #
        minz = np.floor(self.minde/self.vspa)*self.vspa
        maxz = np.ceil(self.maxde/self.vspa)*self.vspa
        #
        xs = np.arange(minx, maxx, self.hspa)
        ys = np.arange(miny, maxy, self.hspa)
        zs = np.arange(minz, maxz, self.vspa)
        #
        #
        self.gridx, self.gridy, self.gridz = np.meshgrid(xs, ys, zs)
        shp = self.gridx.shape
        #
        #
        tlo, tla = self.p(self.gridx.flatten()*1e3, self.gridy.flatten()*1e3,
                          inverse=True)
        self.gridlo = np.reshape(tlo, shp)
        self.gridla = np.reshape(tla, shp)

    def get_coordinates_vectors(self):
        """
        This returns three vectors containing the coordinates for all the nodes
        of the 3D grid
        """
        return (self.gridlo.flatten(), self.gridla.flatten(),
                self.gridz.flatten())

    def select_nodes_within_two_meshesa(self, meshup, meshlo):
        """
        :param meshup:
        :param meshlo:
        """
        idxs = np.isfinite(meshup.depths)
        #
        # spatial index for top and bottom slabs
        siup = rtree.index.Index(generator_function(meshup))
        silo = rtree.index.Index(generator_function(meshlo))
        #
        # compute the concave hull for the top and bottom slab
        lonsup = meshup.lons[idxs].flatten()
        lonsup = ([x+360 if x<0 else x for x in lonsup])
        lonslo = meshlo.lons[idxs].flatten()
        lonslo = ([x+360 if x<0 else x for x in lonslo])
        ch_up, _ = alpha_shape(lonsup, meshup.lats[idxs].flatten(), 1.0)
        ch_lo, _ = alpha_shape(lonslo, meshlo.lats[idxs].flatten(), 1.0)
        #
        #
        mupde = meshup.depths.flatten()
        mlode = meshlo.depths.flatten()
        #
        # find the points within the top and bottom
        pin = []
        for idx, (lo, la, de) in enumerate(zip(self.gridlo.flatten(),
                                               self.gridla.flatten(),
                                               self.gridz.flatten())):
            if ch_up.contains(Point(lo, la)) and ch_lo.contains(Point(lo, la)):
                iup = list(siup.nearest((lo, la, lo, la), 1))
                ilo = list(silo.nearest((lo, la, lo, la), 2))
                if (de - mupde[iup[0]] > 0. and de - mlode[ilo[0]] < 0.):
                    pin.append(idx)

        return self.gridlo.flatten()[pin], self.gridla.flatten()[pin], \
            self.gridz.flatten()[pin]

    def select_nodes_within_two_meshes(self, meshup, meshlo):
        """
        This method selects the points within the slab

        :parameter :class:`openquake.hazardlib.geo.mesh.Mesh` meshup:
            The upper mesh
        :parameter :class:`openquake.hazardlib.geo.mesh.Mesh` meshlo:
            The lower mesh
        """
        #
        # mesh projected x and y
        i = np.isfinite(meshup.lons)
        mux, muy = self.p(meshup.lons[i].flatten(), meshup.lats[i].flatten())
        mlx, mly = self.p(meshlo.lons[i].flatten(), meshlo.lats[i].flatten())
        mux /= 1e3
        muy /= 1e3
        mlx /= 1e3
        mly /= 1e3
        #
        # upper depths for all the points
        coos = np.stack((mux, muy)).T
        upd = griddata(coos, meshup.depths[i].flatten(),
                       (self.gridx[:, :, :], self.gridy[:, :, :]),
                       method='linear')
        upd = np.squeeze(upd)
        #
        # lower depths for all the points
        coos = np.stack((mlx, mly)).T
        lod = griddata(coos, meshlo.depths[i].flatten(),
                       (self.gridx[:, :, :], self.gridy[:, :, :]),
                       method='linear')
        lod = np.squeeze(lod)
        #
        # creating the 3d grid with the upper depths and selecting nodes
        # below it
        # upd = np.expand_dims(upd, axis=2)
        # lod = np.expand_dims(lod, axis=2)
        ug = np.tile(upd, (1, 1, self.gridz.shape[2]))
        lg = np.tile(lod, (1, 1, self.gridz.shape[2]))
        ug = upd
        lg = lod
        #
        # select the nodes
        iii = np.nonzero((np.isfinite(ug)) & (np.isfinite(lg)) &
                         (self.gridz <= ug) & (self.gridz >= lg))
        iii = np.nonzero((self.gridz <= lg) & (self.gridz >= ug))
        #
        # back to geographic coordinates
        lo, la = self.p(self.gridx[iii[0], iii[1], iii[2]]*1e3,
                        self.gridy[iii[0], iii[1], iii[2]]*1e3, inverse=True)
        #
        return (lo, la, self.gridz[iii[0], iii[1], iii[2]])
