# ------------------- The OpenQuake Model Building Toolkit --------------------
# Copyright (C) 2022 GEM Foundation
#           _______  _______        __   __  _______  _______  ___   _
#          |       ||       |      |  |_|  ||  _    ||       ||   | | |
#          |   _   ||   _   | ____ |       || |_|   ||_     _||   |_| |
#          |  | |  ||  | |  ||____||       ||       |  |   |  |      _|
#          |  |_|  ||  |_|  |      |       ||  _   |   |   |  |     |_
#          |       ||      |       | ||_|| || |_|   |  |   |  |    _  |
#          |_______||____||_|      |_|   |_||_______|  |___|  |___| |_|
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------
# vim: tabstop=4 shiftwidth=4 softtabstop=4
# coding: utf-8

"""
Set of moment tensor utility functions
"""

import numpy as np
from math import fabs, log10, sqrt, acos, atan2, pi, sin, cos, degrees, radians


def tensor_components_to_use(mrr, mtt, mpp, mrt, mrp, mtp):
    """
    Converts components to Up, South, East definition
    USE = [[mrr, mrt, mrp],
           [mtt, mtt, mtp],
           [mrp, mtp, mpp]]
    """
    return np.array([[mrr, mrt, mrp], [mrt, mtt, mtp], [mrp, mtp, mpp]])


def tensor_components_to_ned(mrr, mtt, mpp, mrt, mrp, mtp):
    """
    Converts components to North, East, Down definition
    NED = [[mtt, -mtp, mrt],
           [-mtp, mpp, -mrp],
           [mrt, -mtp, mrr]]
    """
    return np.array([[mtt, -mtp, mrt], [-mtp, mpp, -mrp], [mrt, -mtp, mrr]])


def get_azimuth_plunge(vect, degrees=True):
    """
    For a given vector in USE format, retrieve the azimuth and plunge
    """
    if vect[0] > 0:
        vect = -1. * np.copy(vect)

    vect_hor = sqrt(vect[1] ** 2. + vect[2] ** 2.)
    plunge = atan2(-vect[0], vect_hor)
    azimuth = atan2(vect[2], -vect[1])
    if degrees:
        icr = 180. / pi
        return icr * azimuth % 360., icr * plunge
    else:
        return azimuth % (2. * pi), plunge


COORD_SYSTEM = {'USE': tensor_components_to_use,
                'NED': tensor_components_to_ned}

ROT_NED_USE = np.matrix([[0., 0., -1.],
                        [-1., 0., 0.],
                        [0., 1., 0.]])


def use_to_ned(tensor):
    '''
    Converts a tensor in USE coordinate sytem to NED
    '''
    return np.array(ROT_NED_USE.T * np.matrix(tensor) * ROT_NED_USE)


def ned_to_use(tensor):
    '''
    Converts a tensor in NED coordinate sytem to USE
    '''
    return np.array(ROT_NED_USE * np.matrix(tensor) * ROT_NED_USE.T)


def tensor_to_6component(tensor, frame='USE'):
    '''
    Returns a tensor to six component vector [Mrr, Mtt, Mpp, Mrt, Mrp, Mtp]
    '''
    if 'NED' in frame:
        tensor = ned_to_use(tensor)

    return [tensor[0, 0], tensor[1, 1], tensor[2, 2], tensor[0, 1],
            tensor[0, 2], tensor[1, 2]]


def normalise_tensor(tensor):
    '''
    Normalise the tensor by dividing it by its norm, defined such that
    np.sqrt(X:X)
    '''
    tensor_norm = np.linalg.norm(tensor)
    return tensor / tensor_norm, tensor_norm


def eigendecompose(tensor, normalise=False):
    """
    Performs and eigendecomposition of the tensor and orders into
    descending eigenvalues
    """
    if normalise:
        tensor, tensor_norm = normalise_tensor(tensor)
    else:
        tensor_norm = 1.
    eigvals, eigvects = np.linalg.eigh(tensor, UPLO='U')

    isrt = np.argsort(eigvals)
    eigenvalues = eigvals[isrt] * tensor_norm
    eigenvectors = eigvects[:, isrt]
    return eigenvalues, eigenvectors


def matrix_to_euler(rotmat):
    '''Inverse of euler_to_matrix().'''
    if not isinstance(rotmat, np.matrixlib.defmatrix.matrix):
        # As this calculation relies on np.matrix algebra - convert array to
        # matrix
        rotmat = np.matrix(rotmat)
    cvec = lambda x, y, z: np.matrix([[x, y, z]]).T
    ex = cvec(1., 0., 0.)
    ez = cvec(0., 0., 1.)
    exs = rotmat.T * ex
    ezs = rotmat.T * ez
    enodes = np.cross(ez.T, ezs.T).T
    if np.linalg.norm(enodes) < 1e-10:
        enodes = exs
    enodess = rotmat * enodes
    cos_alpha = float((ez.T*ezs))
    if cos_alpha > 1.:
        cos_alpha = 1.
    if cos_alpha < -1.:
        cos_alpha = -1.
    alpha = acos(cos_alpha)
    beta = np.mod(atan2(enodes[1, 0], enodes[0, 0]), pi * 2.)
    gamma = np.mod(-atan2(enodess[1, 0], enodess[0, 0]), pi*2.)

    return unique_euler(alpha, beta, gamma)


def unique_euler(alpha, beta, gamma):
    """s
    Uniquify euler angle triplet.
    Put euler angles into ranges compatible with (dip,strike,-rake)
    in seismology:
    alpha (dip) : [0, pi/2]
    beta (strike) : [0, 2*pi)
    gamma (-rake) : [-pi, pi)
    If alpha is near to zero, beta is replaced by beta+gamma and gamma is set
    to zero, to prevent that additional ambiguity.

    If alpha is near to pi/2, beta is put into the range [0,pi).
    """

    alpha = np.mod(alpha, 2.0 * pi)

    if 0.5 * pi < alpha and alpha <= pi:
        alpha = pi - alpha
        beta = beta + pi
        gamma = 2.0 * pi - gamma
    elif pi < alpha and alpha <= 1.5 * pi:
        alpha = alpha - pi
        gamma = pi - gamma
    elif 1.5 * pi < alpha and alpha <= 2.0 * pi:
        alpha = 2.0 * pi - alpha
        beta = beta + pi
        gamma = pi + gamma

    alpha = np.mod(alpha, 2.0 * pi)
    beta = np.mod(beta, 2.0 * pi)
    gamma = np.mod(gamma + pi, 2.0 * pi) - pi

    # If dip is exactly 90 degrees, one is still
    # free to choose between looking at the plane from either side.
    # Choose to look at such that beta is in the range [0,180)

    # This should prevent some problems, when dip is close to 90 degrees:
    if fabs(alpha - 0.5 * pi) < 1e-10:
        alpha = 0.5 * pi
    if fabs(beta - pi) < 1e-10:
        beta = pi
    if fabs(beta - 2. * pi) < 1e-10:
        beta = 0.
    if fabs(beta) < 1e-10:
        beta = 0.

    if alpha == 0.5 * pi and beta >= pi:
        gamma = - gamma
        beta = np.mod(beta-pi, 2.0 * pi)
        gamma = np.mod(gamma + pi, 2.0 * pi) - pi
        assert 0. <= beta < pi
        assert -pi <= gamma < pi

    if alpha < 1e-7:
        beta = np.mod(beta + gamma, 2.0 * pi)
        gamma = 0.

    return (alpha, beta, gamma)


def moment_magnitude_scalar(moment):
    '''
    Uses Hanks & Kanamori formula for calculating moment magnitude from
    a scalar moment (Nm)
    '''
    if isinstance(moment, np.ndarray):
        return (2. / 3.) * (np.log10(moment) - 9.05)
    else:
        return (2. / 3.) * (log10(moment) - 9.05)


# functions to construct second nodal plane from the first
# transcribed to Python from GMT source code
def computed_strike(nodal_plane, tol=1.0E-7):
    """
    Nodal plane is the nodal plane dict from the GCMTNodalPlanes object
    {"strike": , "dip":, "rake":  }
    """
    strike, dip, rake = [radians(nodal_plane[val])
                         for val in ["strike", "dip", "rake"]]
    cd1 = cos(dip)
    if fabs(nodal_plane["rake"]) < tol:
        a_m = 1.
    else:
        a_m = nodal_plane["rake"] / fabs(nodal_plane["rake"])
    s_r, c_r = sin(rake), cos(rake)
    s_s, c_s = sin(strike), cos(strike)
    if (cd1 < tol) and (fabs(c_r) < tol):
        # 2nd plane is horizontal and strike undertermined
        strike2 = nodal_plane["strike"] + 180.0
        return (strike2 % 360.)

    sp2 = -a_m * (c_r * c_s + (s_r * s_s * cd1))
    cp2 = a_m * (s_s * c_r - (s_r * c_s * cd1))
    strike2 = degrees(atan2(sp2, cp2))
    return (strike2 % 360.)


def computed_dip(nodal_plane, tol=1.0E-7):
    """
    Returns the second nodal plane dip from the first nodal plane
    """
    if fabs(nodal_plane["rake"]) < tol:
        a_m = 1.0
    else:
        a_m = nodal_plane["rake"] / fabs(nodal_plane["rake"])
    dip2 = acos(a_m * sin(radians(nodal_plane["rake"])) *
                sin(radians(nodal_plane["dip"])))
    return degrees(dip2)


def computed_rake(nodal_plane, tol=1.0E-7):
    """
    Returns the second nodal plane rake from the first nodal plane
    """
    str2 = computed_strike(nodal_plane, tol)
    dip2 = computed_dip(nodal_plane, tol)
    strike, dip, rake = [radians(nodal_plane[val])
                         for val in ["strike", "dip", "rake"]]
    if fabs(nodal_plane["rake"]) < tol:
        a_m = 1.0
    else:
        a_m = nodal_plane["rake"] / fabs(nodal_plane["rake"])
    s_d, c_d = sin(dip), cos(dip)
    s_s, c_s = sin(strike - radians(str2)), cos(strike)
    if fabs(dip2 - 90.) < tol:
        sinrake2 = a_m * c_d
    else:
        sinrake2 = -a_m * s_d * (c_s / c_d)
    rake2 = atan2(sinrake2, -a_m * s_d * s_s)
    return degrees(rake2), str2, dip2


def compute_second_nodal_plane(nodal_plane, tol=1.0E-7):
    """
    Given a nodal plane of the form {'strike':, 'dip':, 'rake':} returns the
    complementary plane as a dictionary of the same form
    """
    nodal_plane_2 = {}
    rake, strike, dip = computed_rake(nodal_plane, tol)
    nodal_plane_2["strike"] = strike
    nodal_plane_2["dip"] = dip
    nodal_plane_2["rake"] = rake
    return nodal_plane_2
