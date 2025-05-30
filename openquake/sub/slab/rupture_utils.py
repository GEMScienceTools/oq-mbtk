"""
:module:`openquake.sub.slab.rupture_utils`
"""

import numpy as np


def get_number_ruptures(omsh, rup_s, rup_d, f_strike=1, f_dip=1, wei=None):
    """
    Given a :class:`~openquake.hazardlib.geo.mesh.Mesh` instance and the size
    of a rupture (in terms of the number of rows and cols) it provides the
    number of ruptures admitted and the sum of their weights.

    :param omsh:
        A :class:`~openquake.hazardlib.geo.mesh.Mesh` instance describing the
        fault surface
    :param rup_s:
        Number of cols composing the rupture
    :param rup_d:
        Number of rows composing the rupture
    :param wei:
        Weights for each cell composing the fault surface
    :param f_strike:
        Floating distance along strike (multiple of sampling distance)
    :param f_dip:
        Floating distance along dip (multiple of sampling distance)
    """
    num_rup = 0
    wei_rup = []
    for i in np.arange(0, omsh.lons.shape[1] - rup_s, f_strike):
        for j in np.arange(0, omsh.lons.shape[0] - rup_d, f_dip):
            if (np.all(np.isfinite(omsh.lons[j:j + rup_d, i:i + rup_s]))):
                if wei is not None:
                    wei_rup.append(np.sum(wei[j:j + rup_d - 1,
                                              i:i + rup_s - 1]))
                num_rup += 1
    return num_rup


def get_ruptures(omsh, rup_s, rup_d, f_strike=1, f_dip=1):
    """
    Given a :class:`~openquake.hazardlib.geo.mesh.Mesh` instance and the size
    of a rupture (in terms of the number of rows and cols) it yields all the
    possible ruptures admitted by the fault geometry.

    :param omsh:
        A :class:`~openquake.hazardlib.geo.mesh.Mesh` instance describing the
        fault surface
    :param rup_s:
        Number of cols composing the rupture
    :param rup_d:
        Number of rows composing the rupture
    :param f_strike:
        Floating distance along strike (multiple of sampling distance)
    :param f_dip:
        Floating distance along dip (multiple of sampling distance)
    :returns:
        A tuple with three elements. The first one is a tuple with three 2D
        arrays containing the coordinates of the nodes representing the
        ruptures. Rge second and third one are the indexes of the of the
        upper-left node of the grid (the indexes refer to the grid used to
        describe the virstual fault).
    """

    # When f_strike is negative, the floating distance is interpreted as
    # a fraction of the rupture length (i.e. a multiple of the sampling
    # distance)
    if f_strike < 0:
        f_strike = int(np.floor(rup_s * abs(f_strike) + 1e-5))
        if f_strike < 1:
            f_strike = 1

    # See f_strike comment above
    if f_dip < 0:
        f_dip = int(np.floor(rup_d * abs(f_dip) + 1e-5))
        if f_dip < 1:
            f_dip = 1

    # Float the rupture on the virtual fault
    for i in np.arange(0, omsh.lons.shape[1] - rup_s + 1, f_strike):
        for j in np.arange(0, omsh.lons.shape[0] - rup_d + 1, f_dip):
            #
            nel = np.size(omsh.lons[j:j + rup_d, i:i + rup_s])
            nna = np.sum(np.isfinite(omsh.lons[j:j + rup_d, i:i + rup_s]))
            prc = nna/nel*100.
            if prc > 95. and nna >= 4:
                yield ((omsh.lons[j:j + rup_d, i:i + rup_s],
                        omsh.lats[j:j + rup_d, i:i + rup_s],
                        omsh.depths[j:j + rup_d, i:i + rup_s]), j, i)


def get_weights(centroids, r, values, proj):
    """
    Assign a weight to each centroid of the grid representing the fault
    surface.

    :param centroids:
        A :class:`~numpy.ndarray` instance with cardinality j x k x 3 where
        j and k corresponds to the number of cells along strike and along dip
        forming the fault surface
    :param r:
        A :class:`~rtree.index.Index` instance with the location smoothing
        grid
    :param values:
        A :class:`~numpy.ndarray` instance with lenght equal to the number of
        elements in the `centroids` matrix
    :param proj:
        An instance of Proj
    :returns:
        An :class:`numpy.ndarray` instance
    """

    # Projected centroids - projection shouldn't be an issue here as long as
    # we can get the nearest neighbour correctly
    ccx, ccy = proj(centroids[:, :, 0].flatten(), centroids[:, :, 1].flatten())
    ccx *= 1e-3
    ccy *= 1e-3
    ccz = centroids[:, :, 2].flatten()

    # Assign a weight to each centroid
    weights = np.zeros_like(ccx)
    weights[:] = np.nan
    for i in range(0, len(ccx)):
        if np.isfinite(ccz[i]):
            idx = list(r.nearest((ccx[i], ccy[i], ccz[i],
                                  ccx[i], ccy[i], ccz[i]), 1,
                                 objects=False))
            weights[i] = values[idx[0]]

    # Reshape the weights
    weights = np.reshape(weights, (centroids.shape[0], centroids.shape[1]))
    return weights


def heron_formula(coords):
    """
    TODO
    """


def get_mesh_area(mesh):
    """
    :param mesh:
        A :class:`numpy.ndarray` instance.
    """
    for j in range(0, mesh.shape[0]-1):
        for k in range(0, mesh.shape[1]-1):
            if np.all(np.isfinite(mesh.depths[j:j+1, k:k+1])):
                pass
                # TODO
                # calculate the area


def get_discrete_dimensions(area, sampling, aspr):
    """
    Computes the discrete dimensions of a rupture given area, sampling
    distance and aspect ratio.

    :param area:
    :param sampling:
    :param aspr:
    """
    # computing possible length and width
    lng1 = np.ceil((area * aspr)**0.5/sampling)*sampling
    wdtA = np.ceil(lng1/aspr/sampling)*sampling
    wdtB = np.floor(lng1/aspr/sampling)*sampling
    # computing possible length and width
    lng2 = np.floor((area * aspr)**0.5/sampling)*sampling
    wdtC = np.ceil(lng2/aspr/sampling)*sampling
    wdtD = np.floor(lng2/aspr/sampling)*sampling
    #
    dff = 1e10
    lng = None
    wdt = None
    if abs(lng1*wdtA-area) < dff and lng1 > 0. and wdtA > 0.:
        lng = lng1
        wdt = wdtA
        dff = abs(lng1*wdtA-area)
    if abs(lng1*wdtB-area) < dff and lng1 > 0. and wdtB > 0.:
        lng = lng1
        wdt = wdtB
        dff = abs(lng1*wdtB-area)
    if abs(lng2*wdtC-area) < dff and lng2 > 0. and wdtC > 0.:
        lng = lng2
        wdt = wdtC
        dff = abs(lng2*wdtC-area)
    if abs(lng2*wdtD-area) < dff and lng2 > 0. and wdtD > 0.:
        lng = lng2
        wdt = wdtD
        dff = abs(lng2*wdtD-area)
    area_error = abs(lng*wdt-area)/area
    # This is a check that verifies if the rupture size is compatible with the
    # original value provided. If not we raise a Value Error
    if (abs(wdt-sampling) < 1e-10 or abs(lng-sampling) < 1e-10 and
            area_error > 0.3):
        wdt = None
        lng = None
    elif area_error > 0.25 and lng > 1e-10 and wdt > 1e-10:
        print('Area discrepancy: ', area, lng*wdt, lng, wdt, aspr)
        
        #raise ValueError('Area discrepancy: ', area, lng*wdt, lng, wdt, aspr)
    return lng, wdt
