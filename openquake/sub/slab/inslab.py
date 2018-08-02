"""
"""

import re
import os
import glob
import numpy as np

from copy import deepcopy
from pyproj import Proj

from openquake.hazardlib.geo.mesh import Mesh
from openquake.hazardlib.geo.utils import plane_fit
from openquake.hazardlib.geo import Point, Line
from openquake.hazardlib.geo.surface import ComplexFaultSurface
from openquake.hazardlib.geo.geodetic import (distance, azimuth,
                                              npoints_towards)

from openquake.sub.grid3d import Grid3d


def _read_profile(filename):
    """
    :parameter filename:
        The name of the folder file (usually with prefix 'cs_')
        specifing the geometry of the top of the slab

    :returns:
        An instance of :class:`openquake.hazardlib.geo.line.Line`
    """
    points = []
    for line in open(filename, 'r'):
        aa = re.split('\s+', line)
        points.append(Point(float(aa[0]),
                            float(aa[1]),
                            float(aa[2])))
    return Line(points)


def _resample_profile(line, sampling_dist):
    """
    :parameter line:
        An instance of :class:`openquake.hazardlib.geo.line.Line`
    :parameter sampling_dist:
        A scalar definining the distance used to sample the profile
    :returns:
        An instance of :class:`openquake.hazardlib.geo.line.Line`
    """
    lo = [pnt.longitude for pnt in line.points]
    la = [pnt.latitude for pnt in line.points]
    de = [pnt.depth for pnt in line.points]
    #
    # initialise the cumulated distance
    cdist = 0.
    #
    # get the azimuth of the profile
    azim = azimuth(lo[0], la[0], lo[-1], la[-1])
    #
    # initialise the list with the resampled nodes
    idx = 0
    resampled_cs = [Point(lo[idx], la[idx], de[idx])]
    #
    # set the starting point
    slo = lo[idx]
    sla = la[idx]
    sde = de[idx]
    #
    # resampling
    while 1:
        #
        # check loop exit condition
        if idx > len(lo)-2:
            break
        #
        # compute the distance between the starting point and the next point
        # on the profile
        segment_len = distance(slo, sla, sde, lo[idx+1], la[idx+1], de[idx+1])
        #
        # search for the point
        if cdist+segment_len > sampling_dist:
            #
            # this is the lenght of the last segment-fraction needed to
            # obtain the sampling distance
            delta = sampling_dist - cdist
            #
            # compute the slope of the last segment and its horizontal length.
            # We need to manage the case of a vertical segment TODO
            segment_hlen = distance(slo, sla, 0., lo[idx+1], la[idx+1], 0.)
            segment_slope = np.arctan((de[idx+1] - sde) / segment_hlen)
            #
            # horizontal and vertical lenght of delta
            delta_v = delta * np.sin(segment_slope)
            delta_h = delta * np.cos(segment_slope)
            #
            # add a new point to the cross section
            pnts = npoints_towards(slo, sla, sde, azim, delta_h, delta_v, 2)
            #
            # update the starting point
            slo = pnts[0][-1]
            sla = pnts[1][-1]
            sde = pnts[2][-1]
            resampled_cs.append(Point(slo, sla, sde))
            #
            # reset the cumulative distance
            cdist = 0.

        else:
            cdist += segment_len
            idx += 1
            slo = lo[idx]
            sla = la[idx]
            sde = de[idx]

    line = Line(resampled_cs)
    return line


def _read_edge(filename):
    """
    :parameter filename:
        The name of the file with prefix 'edge'
        specifing the geometry of the top of the slab

    :returns:
        An instance of :class:`openquake.hazardlib.geo.line.Line`
    """
    points = []
    for line in open(filename, 'r'):
        aa = re.split('\s+', line)
        points.append(Point(float(aa[0]),
                            float(aa[1]),
                            float(aa[2])))
    return Line(points)


def create_planar_mesh(orig, ppar, spacing, lenght, width):
    """
    TODO
    2018.05.18 - This is currenly not used. Consider removing.

    :parameter orig:
    :parameter ppar:
    :parameter spacing:
    :parameter lenght:
    :parameter width:
    """
    #
    # compute the vector on the plane defining the steepest direction
    # https://www.physicsforums.com/threads/projecting-a-vector-onto-a-plane.496184/
    steep = np.cross(ppar, np.cross([0, 0, -1], ppar))
    steep = steep / sum(steep**2.)**0.5
    #
    # we need to rotate the 'steep' vector of -90 deg around the normal vector
    # to the plane


def regularize(mesh, spacing):
    """
    TODO
    2018.05.18 - This is currenly not used. Consider removing.

    Fitting https://gist.github.com/amroamroamro/1db8d69b4b65e8bc66a6
    """
    #
    #
    dlt_x = 10
    dlt_z = 10
    #
    # create a 3xn array with the points composing the mesh
    lld = np.array([mesh.lons.flatten('F'), mesh.lats.flatten('F'),
                    mesh.depths.flatten('F')]).T
    #
    # project the points using Lambert Conic Conformal - for the reference
    # meridian 'lon_0' we use the mean longitude of the mesh
    p = Proj('+proj=lcc +lon_0={:f}'.format(np.mean(lld[:, 0])))
    x, y = p(lld[:, 0], lld[:, 1])
    x = x / 1e3  # m -> km
    y = y / 1e3  # m -> km
    #
    # compute the distances between all the points composing the mesh and
    # a reference point indicated with the index 'idx'
    idx = 0
    dx = np.sign(lld[idx, 0]-lld[:, 0]) * ((lld[idx, 0]-lld[:, 0])**2 +
                                           (lld[idx, 1]-lld[:, 1])**2)**.5
    dz = (lld[idx, 2]-lld[:, 2])
    #
    # find nodes within the delta X and delta Z distances
    idx = np.nonzero((dx <= dlt_x) & (dz <= dlt_z))
    #
    # compute the equation of the plane fitting the portion of the slab surface
    xx = np.vstack((x[idx].T, y[idx].T, lld[idx, 2])).T
    pnt, ppar = plane_fit(xx)
    #
    # vertical plane
    vertical_plane = [ppar[0], ppar[1], 0]
    vertical_plane = vertical_plane / (sum(vertical_plane)**2)**.5
    #
    # strike direction
    stk = np.cross(ppar, vertical_plane)
    stk = stk / sum(stk**2.)**0.5
    #
    # project the top left point on the plane surface. First we compute
    # the distance from the point to the plane then we find the coordinates
    # of the point. TODO
    t = -np.sum(ppar*lld[0, :])/np.sum(ppar**2)
    orig = np.array([ppar[0]*t+x[0], ppar[1]*t+y[0], ppar[2]*t+lld[0, 2]])
    #
    # compute the vector on the plane defining the steepest direction
    # https://www.physicsforums.com/threads/projecting-a-vector-onto-a-plane.496184/
    dip = np.cross(ppar, np.cross([0, 0, -1], ppar))
    dip = dip / sum(dip**2.)**0.5
    #
    # create the rectangle in 3D
    rects = []
    pnt0 = [x[0], y[0], lld[0, 2]]
    pnt1 = [x[0]+stk[0]*dlt_x, y[0]+stk[1]*dlt_x, lld[0, 2]+stk[2]*dlt_x]
    pnt2 = [pnt1[0]+dip[0]*dlt_z, pnt1[1]+dip[1]*dlt_z, pnt1[2]+dip[2]*dlt_z]
    pnt3 = [x[0]+dip[0]*dlt_z, y[0]+dip[1]*dlt_z, lld[0, 2]+dip[2]*dlt_z]

    rects.append([pnt0, pnt1, pnt2, pnt3])
    rects = np.array(rects)

    # lo, la = p(, lld[:,1])

    return rects


def create_lower_surface_mesh(mesh, slab_thickness):
    """
    This method used to build the bottom surface of the slab computes at each
    point the plane fitting a local portion of the top-surface and uses the
    perpendicular to find the corresponding node for the bottom surface.

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
    oshape = mesh.lons.shape
    #
    # project the points using Lambert Conic Conformal - for the reference
    # meridian 'lon_0' we use the mean longitude of the grid
    p = Proj('+proj=lcc +lon_0={:f}'.format(np.mean(mesh.lons.flatten('F'))))
    x, y = p(mesh.lons.flatten('F'), mesh.lats.flatten('F'))
    x = x / 1e3  # m -> km
    y = y / 1e3  # m -> k
    #
    # reshaping
    x = np.reshape(x, oshape, order='F')
    y = np.reshape(y, oshape, order='F')
    #
    # initialize the lower mesh
    lowm = deepcopy(mesh)
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
                            mesh.depths[rlow:rupp, clow:cupp].flatten())).T

            ii = np.isfinite(xx[:, 2])
            if np.sum(ii) > 4:
                try:
                    pnt, ppar = plane_fit(xx[ii, :])
                except:
                    raise ValueError('Plane interpolation failed')
            #
            # compute the points composing the new surface. The new surface
            # is at a distance 'slab_tickness' below the original surface in a
            # direction perpendicular to the fitted planes
            corr = 1
            if np.sign(ppar[2]) == -1:
                corr = -1
            xls = x[ir, ic] + corr * slab_thickness * ppar[0]
            yls = y[ir, ic] + corr * slab_thickness * ppar[1]
            zls = mesh.depths[ir, ic] + corr * slab_thickness * ppar[2]
            #
            # back-conversion to geographic coordinates
            llo, lla = p(xls*1e3, yls*1e3, inverse=True)
            #
            # updating the mesh
            lowm.lons[ir, ic] = llo
            lowm.lats[ir, ic] = lla
            lowm.depths[ir, ic] = zls
    #
    #
    return lowm


def create_lower_surface_mesh_old(mesh, slab_thickness):
    """
    This method for the construction of the boottom surface of the slab finds
    the plane fitting the surface and the projects the top surface toward a
    direction perpendicular to the plane.

    NB don't forget the surface_to_mesh method in openquake.hazardlib.geo.mesh

    :parameter mesh:
        An instance of the :class:`openquake.hazardlib.geo.mesh.Mesh`
        describing the top of the slab within which we admit inslab seismicity
    :parameter float slab_thickness:
        Thickness of the slab [km]
    :returns:
        An instance of :class:`openquake.hazardlib.geo.mesh.Mesh`
    """
    oshape = mesh.lons.shape
    #
    # create a 3xn array with the points composing the mesh
    lld = np.array([mesh.lons.flatten('F'), mesh.lats.flatten('F'),
                    mesh.depths.flatten('F')]).T
    #
    # project the points using Lambert Conic Conformal - for the reference
    # meridian 'lon_0' we use the mean longitude of the grid
    p = Proj('+proj=lcc +lon_0={:f}'.format(np.mean(lld[:, 0])))
    x, y = p(lld[:, 0], lld[:, 1])
    x = x / 1e3  # m -> km
    y = y / 1e3  # m -> km
    #
    # compute the equation of the plane fitting the slab surface
    xx = np.vstack((x.T, y.T, lld[:, 2])).T
    pnt, ppar = plane_fit(xx)
    #
    # compute the points on the new surface. The new surface is at a distance
    # 'slab_tickness' below the original surface in a direction perpendicular
    # to the fitted plane
    corr = 1
    if np.sign(ppar[2]) == -1:
        corr = -1
    xls = x + corr * slab_thickness * ppar[0]
    yls = y + corr * slab_thickness * ppar[1]
    zls = lld[:, 2] + corr * slab_thickness * ppar[2]
    #
    # back-projection of the points composing the lower surface
    llo, lla = p(xls*1e3, yls*1e3, inverse=True)
    #
    # reshape the arrays containing the geographic coordinates of the lower
    # surface
    rllo = np.reshape(llo, oshape, order='F')
    rlla = np.reshape(lla, oshape, order='F')
    rzls = np.reshape(zls, oshape, order='F')
    #
    #
    return Mesh(rllo, rlla, rzls)


def create_ruptures(folder, mesh_spacing, slab_thickness, h_grid_spacing,
                    v_grid_spacing):
    """
    :parameter path:
        Path to the folder containing a number of edges defining the top of
        the slab.
    :parameter mesh_spacing:
        Mesh spacing used to discretize the complex fault [km]
    :parameter slab_thickness:
        Thickness of the slab [km]
    :parameter h_grid_spacing:
        Horizontal spacing of the grid used to describe the slab
    :parameter v_grid_spacing:
        Vertical spacing of the grid used to describe the slab
    """
    #
    # read the edges from the text files in the user-provided folder
    path = os.path.join(folder, 'edge*.*')
    tedges = []
    for fle in glob.glob(path):
        tedges.append(_read_edge(fle))
    #
    # create the complex fault surface
    surface = ComplexFaultSurface.from_fault_data(tedges,
                                                  mesh_spacing=mesh_spacing)
    #
    # build the lower surface i.e. the surface describing the bottom of the
    # slab
    lower_mesh = create_lower_surface_mesh(surface.mesh, slab_thickness)
    #
    # computing the limits of the grid
    minlo = np.amin([np.amin(lower_mesh.lons), np.amin(surface.mesh.lons)])
    maxlo = np.amax([np.amax(lower_mesh.lons), np.amax(surface.mesh.lons)])
    minla = np.amin([np.amin(lower_mesh.lats), np.amin(surface.mesh.lats)])
    maxla = np.amax([np.amax(lower_mesh.lats), np.amax(surface.mesh.lats)])
    minde = np.amin([np.amin(lower_mesh.depths), np.amin(surface.mesh.depths)])
    maxde = np.amax([np.amax(lower_mesh.depths), np.amax(surface.mesh.depths)])
    #
    # creating the regular grid describing the slab
    grd = Grid3d(minlo, minla, minde, maxlo, maxla, maxde, h_grid_spacing,
                 v_grid_spacing)
    gx, gy, gz = grd.select_nodes_within_two_meshes(surface.mesh, lower_mesh)
    #
    #
    return surface.mesh, lower_mesh, gx, gy, gz, grd


def get_rup(mesh_top, mesh_bottom, mfd, node, dip):
    """
    :parameter mesh_top:
    :parameter mesh_bottom:
    :parameter mfd:
    :parameter note:
        A tuple with x, y, and z
    """
    pass
