"""
"""

import numpy as np

from copy import deepcopy
from pyproj import Proj

from openquake.hazardlib.geo.utils import plane_fit


def create_lower_surface_mesh(msh, slab_thickness):
    """
    This method is used to build the bottom surface of the slab. It computes at
    each point the plane fitting a local portion of the top-surface and uses
    the perpendicular to find the corresponding node for the bottom surface.

    :parameter mesh:
        An instance of the :class:`openquake.hazardlib.geo.mesh.Mesh` that
        describes the top of the slab within which we place inslab seismicity
    :parameter slab_thickness:
        Thickness of the slab [km]
    :returns:
        An instance of :class:`openquake.hazardlib.geo.mesh.Mesh`
    """
    #
    # save original shape of the 2.5D mesh
    oshape = msh[:, :, 0].shape
    #
    # project the points using Lambert Conic Conformal - for the reference
    # meridian 'lon_0' we use the mean longitude of the grid
    reference_longitude = np.mean(msh[:, :, 0].flatten('C'))

    all_lons = msh[:, :, 0].flatten('C')
    all_lons = np.array(([x+360 if x<0 else x for x in all_lons]))
    real_lons = msh[:, :, 0][~np.isnan(msh[:,:,0])].flatten('C')
    reference_longitude = np.mean(real_lons)

    p = Proj('+proj=lcc +lon_0={:f}'.format(reference_longitude))
    x, y = p(msh[:, :, 0].flatten('C'), msh[:, :, 1].flatten('C'))
    x = x / 1e3  # m -> km
    y = y / 1e3  # m -> k
    z = msh[:, :, 2].flatten('C')
    #
    #
    ii = np.isfinite(z)
    pnt, ppar_default = plane_fit(np.vstack((x[ii], y[ii], z[ii])).T)
    #
    # reshaping
    x = np.reshape(x, oshape, order='C')
    y = np.reshape(y, oshape, order='C')
    #
    # initialize the lower mesh
    lowm = deepcopy(msh)
    #
    #
    dlt = 1
    for ir in range(0, x.shape[0]):
        for ic in range(0, x.shape[1]):
            #
            # initialise the indexes
            rlow = ir - dlt
            rupp = ir + dlt + 1
            clow = ic - dlt
            cupp = ic + dlt + 1
            #
            # fixing indexes at the borders of the mesh
            if rlow < 0:
                rlow = 0
                rupp = rlow + dlt*2 + 1
            if clow < 0:
                clow = 0
                cupp = clow + dlt*2 + 1
            if rupp >= x.shape[0]:
                rupp = x.shape[0] - 1
                rlow = rupp - (dlt*2 + 1)
            if cupp >= x.shape[1]:
                cupp = x.shape[1] - 1
                clow = cupp - (dlt*2 + 1)
            #
            # get the subset of nodes and compute equation of the interpolating
            # plane
            xx = np.vstack((x[rlow:rupp, clow:cupp].flatten(),
                            y[rlow:rupp, clow:cupp].flatten(),
                            msh[rlow:rupp, clow:cupp, 2].flatten())).T

            ii = np.isfinite(xx[:, 2])
            if np.sum(ii) > 4:
                try:
                    pnt, ppar = plane_fit(xx[ii, :])
                except:
                    raise ValueError('Plane interpolation failed')
            else:
                ppar = ppar_default
            #
            # compute the points composing the new surface. The new surface
            # is at a distance 'slab_tickness' below the original surface in a
            # direction perpendicular to the fitted planes
            corr = 1
            if np.sign(ppar[2]) == -1:
                corr = -1
            xls = x[ir, ic] + corr * slab_thickness * ppar[0]
            yls = y[ir, ic] + corr * slab_thickness * ppar[1]
            zls = msh[ir, ic, 2] + corr * slab_thickness * ppar[2]
            #
            # back-conversion to geographic coordinates
            llo, lla = p(xls*1e3, yls*1e3, inverse=True)
            #
            # updating the mesh
            lowm[ir, ic, 0] = llo
            lowm[ir, ic, 1] = lla
            lowm[ir, ic, 2] = zls
    #
    #
    return lowm
