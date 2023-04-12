#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2017 GEM Foundation and G. Weatherill
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.
"""
Strong motion utilities.
"""
# WARNING: this module is intended to collect functions used in various places
# throughout the code. Consequently, try to limit the amount of stuff here and in
# particular the amount of imports, which might slow down the code unnecessarily
import os
import sys
import re
import numpy as np
from scipy.integrate import cumtrapz
from scipy.constants import g
from math import sqrt, pi, sin, cos
import warnings

from openquake.hazardlib import valid
from openquake.hazardlib.geo import PlanarSurface
from openquake.hazardlib.gsim import get_available_gsims
from openquake.hazardlib.scalerel.peer import PeerMSR
from openquake.hazardlib.gsim.gmpe_table import GMPETable
from openquake.hazardlib.gsim.base import GMPE
from openquake.hazardlib.gsim.mgmpe import modifiable_gmpe as mgmpe
from openquake.hazardlib.contexts import ContextMaker
from openquake.hazardlib.scalerel.wc1994 import WC1994
from openquake.smt.comparison.utils_gmpes import get_rupture, get_sites_from_rupture


if sys.version_info[0] >= 3:
    import pickle
else:
    import cPickle as pickle  # pylint: disable=import-error


# Get a list of the available GSIMs
AVAILABLE_GSIMS = get_available_gsims()

# Regular expression to get a GMPETable from string:
_gmpetable_regex = re.compile(r'^GMPETable\(([^)]+?)\)$')

TO_RAD = pi / 180.

def check_gsim_list(gsim_list):
    """
    Check the GSIM models or strings in `gsim_list`, and return a dict of
    gsim names (str) mapped to their :class:`openquake.hazardlib.Gsim`.
    Raises error if any Gsim in the list is supported in OpenQuake.

    If a Gsim is passed as instance, its string representation is inferred
    from the class name and optional arguments. If a Gsim is passed as string,
    the associated class name is fetched from the OpenQuake available Gsims.

    :param gsim_list: list of GSIM names (str) or OpenQuake Gsims
    :return: a dict of GSIM names (str) mapped to the associated GSIM
    """
    output_gsims = {}
    for gs in gsim_list:
        if isinstance(gs, GMPE):
            output_gsims[_get_gmpe_name(gs)] = gs  # get name of GMPE instance
        elif gs in AVAILABLE_GSIMS:
            output_gsims[gs] = AVAILABLE_GSIMS[gs]()
        else:
            match = _gmpetable_regex.match(gs)  # GMPETable ?
            if match:
                filepath = match.group(1).split("=")[1]  # get table filename
                output_gsims[gs] = GMPETable(gmpe_table=filepath)
            else:
                raise ValueError('%s Not supported by OpenQuake' % gs)

    return output_gsims


def _get_gmpe_name(gsim):
    """
    Returns the name of the GMPE given an instance of the class
    """
    match = _gmpetable_regex.match(str(gsim))  # GMPETable ?
    if match:
        filepath = match.group(1).split("=")[1][1:-1]
        return 'GMPETable(gmpe_table=%s)' % filepath
    else:
        gsim_name = gsim.__class__.__name__
        additional_args = []
        # Build the GSIM string by showing name and arguments. Keep things
        # simple (no replacements, no case changes) as we might want to be able
        # to get back the GSIM from its string in the future.
        for key in gsim.__dict__:
            if key.startswith("kwargs"):
                continue
            val = str(gsim.__dict__[key])  # quoting strings with json maybe?
            additional_args.append("{:s}={:s}".format(key, val))
        if len(additional_args):
            gsim_name_str = "({:s})".format(", ".join(additional_args))
            return gsim_name + gsim_name_str
        else:
            return gsim_name


def get_time_vector(time_step, number_steps):
    """
    General SMT utils
    """
    return np.cumsum(time_step * np.ones(number_steps, dtype=float)) - time_step


def nextpow2(nval):
    m_f = np.log2(nval)
    m_i = np.ceil(m_f)
    return int(2.0 ** m_i)


def convert_accel_units(acceleration, from_, to_='cm/s/s'):  # noqa
    """
    Converts acceleration from/to different units

    :param acceleration: the acceleration (numeric or numpy array)
    :param from_: unit of `acceleration`: string in "g", "m/s/s", "m/s**2",
        "m/s^2", "cm/s/s", "cm/s**2" or "cm/s^2"
    :param to_: new unit of `acceleration`: string in "g", "m/s/s", "m/s**2",
        "m/s^2", "cm/s/s", "cm/s**2" or "cm/s^2". When missing, it defaults
        to "cm/s/s"

    :return: acceleration converted to the given units (by default, 'cm/s/s')
    """
    m_sec_square = ("m/s/s", "m/s**2", "m/s^2")
    cm_sec_square = ("cm/s/s", "cm/s**2", "cm/s^2")
    acceleration = np.asarray(acceleration)
    if from_ == 'g':
        if to_ == 'g':
            return acceleration
        if to_ in m_sec_square:
            return acceleration * g
        if to_ in cm_sec_square:
            return acceleration * (100 * g)
    elif from_ in m_sec_square:
        if to_ == 'g':
            return acceleration / g
        if to_ in m_sec_square:
            return acceleration
        if to_ in cm_sec_square:
            return acceleration * 100
    elif from_ in cm_sec_square:
        if to_ == 'g':
            return acceleration / (100 * g)
        if to_ in m_sec_square:
            return acceleration / 100
        if to_ in cm_sec_square:
            return acceleration

    raise ValueError("Unrecognised time history units. "
                     "Should take either ''g'', ''m/s/s'' or ''cm/s/s''")


def get_velocity_displacement(time_step, acceleration, units="cm/s/s",
                              velocity=None, displacement=None):
    """
    Returns the velocity and displacment time series using simple integration
    :param float time_step:
        Time-series time-step (s)
    :param numpy.ndarray acceleration:
        Acceleration time-history
    :returns:
        velocity - Velocity Time series (cm/s)
        displacement - Displacement Time series (cm)
    """
    acceleration = convert_accel_units(acceleration, units)
    if velocity is None:
        velocity = time_step * cumtrapz(acceleration, initial=0.)
    if displacement is None:
        displacement = time_step * cumtrapz(velocity, initial=0.)
    return velocity, displacement


def _save_image(filename, fig, format='png', dpi=300, **kwargs):  # noqa
    """
    Saves the matplotlib figure `fig` to `filename`. Wrapper around `fig.savefig`
    with `dpi=300` by default and `format` inferred from `filename` extension
    or, if no extension is found, set as "png".
    If filename is empty this function does nothing and return

    :param str filename: str, the file path
    :param figure: a :class:`matplotlib.figure.Figure` (e.g. via
        `matplotlib.pyplot.figure()`)
    :param format: string, the image format. Default: 'png'. This argument is
        ignored if `filename` has a file extension, as `format` will be set
        equal to the extension without leading dot.
    :param str kwargs: additional keyword arguments to pass to `fig.savefig`.
        For details, see:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html
    """
    if not filename:
        return

    name, ext = os.path.splitext(filename)
    if ext:
        format = ext[1:]  # noqa
    else:
        filename = name + '.' + format

    fig.savefig(filename, dpi=dpi, format=format, **kwargs)


def load_pickle(pickle_file):
    """
    Python 2 & 3 compatible way of loading a Python Pickle file
    """
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


# Moved from sm_database: Mechanism type to Rake conversion:
MECHANISM_TYPE = {
    "Normal": -90.0,
    "Strike-Slip": 0.0,
    "Reverse": 90.0,
    "Oblique": 0.0,
    "Unknown": 0.0,
    "N": -90.0,  # Flatfile conventions
    "S": 0.0,
    "R": 90.0,
    "U": 0.0,
    "NF": -90.,  # ESM flatfile conventions
    "SS": 0.,
    "TF": 90.,
    "NS": -45.,  # Normal with strike-slip component
    "TS": 45.,  # Reverse with strike-slip component
    "O": 0.0
}


DIP_TYPE = {
    "Normal": 60.0,
    "Strike-Slip": 90.0,
    "Reverse": 35.0,
    "Oblique": 60.0,
    "Unknown": 90.0,
    "N": 60.0,  # Flatfile conventions
    "S": 90.0,
    "R": 35.0,
    "U": 90.0,
    "NF": 60.,  # ESM flatfile conventions
    "SS": 90.,
    "TF": 35.,
    "NS": 70.,  # Normal with strike-slip component
    "TS": 45.,  # Reverse with strike-slip component
    "O": 90.0
}


# mean utilities (geometric, arithmetic, ...):
SCALAR_XY = {
    "Geometric": lambda x, y: np.sqrt(x * y),
    "Arithmetic": lambda x, y: (x + y) / 2.,
    "Larger": lambda x, y: np.max(np.array([x, y]), axis=0),
    "Vectorial": lambda x, y: np.sqrt(x ** 2. + y ** 2.)
}


DEFAULT_MSR = PeerMSR()


def get_interpolated_period(target_period, periods, values):
    """
    Returns the spectra interpolated in loglog space

    :param float target_period: Period required for interpolation
    :param np.ndarray periods: Spectral Periods
    :param np.ndarray values: Ground motion values
    """
    if (target_period < np.min(periods)) or (target_period > np.max(periods)):
        raise ValueError("Period not within calculated range: %s" %
                         str(target_period))
    lval = np.where(periods <= target_period)[0][-1]
    uval = np.where(periods >= target_period)[0][0]

    if (uval - lval) == 0:
        return values[lval]

    d_y = np.log10(values[uval]) - np.log10(values[lval])
    d_x = np.log10(periods[uval]) - np.log10(periods[lval])
    return 10.0 ** (
        np.log10(values[lval]) +
        (np.log10(target_period) - np.log10(periods[lval])) * d_y / d_x
        )


def create_planar_surface(top_centroid, strike, dip, area, aspect):
    """
    Given a central location, create a simple planar rupture
    :param top_centroid:
        Centroid of trace of the rupture, as instance of :class:
            openquake.hazardlib.geo.point.Point
    :param float strike:
        Strike of rupture(Degrees)
    :param float dip:
        Dip of rupture (degrees)
    :param float area:
        Area of rupture (km^2)
    :param float aspect:
        Aspect ratio of rupture

    :returns: Rupture as an instance of the :class:
        openquake.hazardlib.geo.surface.planar.PlanarSurface
    """
    rad_dip = dip * pi / 180.
    width = sqrt(area / aspect)
    length = aspect * width
    # Get end points by moving the top_centroid along strike
    top_right = top_centroid.point_at(length / 2., 0., strike)
    top_left = top_centroid.point_at(length / 2.,
                                     0.,
                                     (strike + 180.) % 360.)
    # Along surface width
    surface_width = width * cos(rad_dip)
    vertical_depth = width * sin(rad_dip)
    dip_direction = (strike + 90.) % 360.

    bottom_right = top_right.point_at(surface_width,
                                      vertical_depth,
                                      dip_direction)
    bottom_left = top_left.point_at(surface_width,
                                    vertical_depth,
                                    dip_direction)

    # Create the rupture
    return PlanarSurface(strike, dip, top_left, top_right,
                         bottom_right, bottom_left)


def get_hypocentre_on_planar_surface(plane, hypo_loc=None):
    """
    Determines the location of the hypocentre within the plane
    :param plane:
        Rupture plane as instance of :class:
        openquake.hazardlib.geo.surface.planar.PlanarSurface
    :param tuple hypo_loc:
        Hypocentre location as fraction of rupture plane, as a tuple of
        (Along Strike, Down Dip), e.g. a hypocentre located in the centroid of
        the rupture plane would be input as (0.5, 0.5), whereas a hypocentre
        located in a position 3/4 along the length, and 1/4 of the way down
        dip of the rupture plane would be entered as (0.75, 0.25)
    :returns:
        Hypocentre location as instance of :class:
        openquake.hazardlib.geo.point.Point
    """

    centroid = plane.get_middle_point()
    if hypo_loc is None:
        return centroid

    along_strike_dist = (hypo_loc[0] * plane.length) - (0.5 * plane.length)
    down_dip_dist = (hypo_loc[1] * plane.width) - (0.5 * plane.width)
    if along_strike_dist >= 0.:
        along_strike_azimuth = plane.strike
    else:
        along_strike_azimuth = (plane.strike + 180.) % 360.
        along_strike_dist = (0.5 - hypo_loc[0]) * plane.length
    # Translate along strike
    hypocentre = centroid.point_at(along_strike_dist,
                                   0.,
                                   along_strike_azimuth)
    # Translate down dip
    horizontal_dist = down_dip_dist * cos(TO_RAD * plane.dip)
    vertical_dist = down_dip_dist * sin(TO_RAD * plane.dip)
    if down_dip_dist >= 0.:
        down_dip_azimuth = (plane.strike + 90.) % 360.
    else:
        down_dip_azimuth = (plane.strike - 90.) % 360.
        down_dip_dist = (0.5 - hypo_loc[1]) * plane.width
        horizontal_dist = down_dip_dist * cos(TO_RAD * plane.dip)

    return hypocentre.point_at(horizontal_dist,
                               vertical_dist,
                               down_dip_azimuth)


def vs30_to_z1pt0_as08(vs30):
    """
    Extracts a depth to 1.0 km/s velocity layer using the relationship
    proposed in Abrahamson & Silva 2008
    :param float vs30:
        Input Vs30 (m/s)
    """
    if vs30 < 180.:
        return np.exp(6.745)
    elif vs30 > 500.:
        return np.exp(5.394 - 4.48 * np.log(vs30 / 500.))
    else:
        return np.exp(6.745 - 1.35 * np.log(vs30 / 180.))


def vs30_to_z1pt0_cy08(vs30):
    """
    Extracts a depth to 1.0 km/s velocity layer using the relationship
    proposed in Chiou & Youngs 2008
    :param float vs30:
        Input Vs30 (m/s)
    """
    return np.exp(28.5 - (3.82 / 8.) * np.log((vs30 ** 8.) + (378.7 ** 8.)))


def z1pt0_to_z2pt5(z1pt0):
    """
    Calculates the depth to 2.5 km/s layer (km /s) using the model presented
    in Campbell & Bozorgnia (2007)
    :param float z1pt0:
        Depth (m) to the 1.0 km/s layer
    :returns:
        Depth (km) to 2.5 km/s layer
    """
    return 0.519 + 3.595 * (z1pt0 / 1000.)


def vs30_to_z1pt0_cy14(vs30, japan=False):
    """
    Returns the estimate depth to the 1.0 km/s velocity layer based on Vs30
    from Chiou & Youngs (2014) California model

    :param numpy.ndarray vs30:
        Input Vs30 values in m/s
    :param bool japan:
        If true returns the Japan model, otherwise the California model
    :returns:
        Z1.0 in m
    """
    if japan:
        c1 = 412. ** 2.
        c2 = 1360.0 ** 2.
        return np.exp((-5.23 / 2.0) * np.log((np.power(vs30, 2.) + c1) / (
            c2 + c1)))
    else:
        c1 = 571 ** 4.
        c2 = 1360.0 ** 4.
        return np.exp((-7.15 / 4.0) * np.log((vs30 ** 4. + c1) / (c2 + c1)))


def vs30_to_z2pt5_cb14(vs30, japan=False):
    """
    Converts vs30 to depth to 2.5 km/s interface using model proposed by
    Campbell & Bozorgnia (2014)

    :param vs30:
        Vs30 values (numpy array or float)

    :param bool japan:
        Use Japan formula (True) or California formula (False)

    :returns:
        Z2.5 in km
    """
    if japan:
        return np.exp(5.359 - 1.102 * np.log(vs30))
    else:
        return np.exp(7.089 - 1.144 * np.log(vs30))
    
def al_atik_sigma_check(gmpe, imtx, task):
    """
    Check if sigma is provided for a given GMPE and implement Al-Atik (2015)
    sigma model if specified. Also provides a warning if GMPE sigma is not
    provided by the specified GMPE. 
    :param gmpe:
        GMPE to check model sigma is provided, and to check if Al-Atik (2015)
        sigma model should be implemented for
    :param imtx:
        Intensity measure to perform sigma checks for with the specified GMPE
    :param task:
        Specify whether for computation of residuals ('residual') or use within
        comparison module ('comparison')
    """
    # Construct a generic context (M = 5.5, D = 100km) to check if sigma provided 
    tmp_rup = get_rupture(20, 40, 15, WC1994(), 5.5, 2, 0, 90, 0, 'fake', None)

    if 'KothaEtAl2020ESHM20' in str(gmpe):
        sp = {'vs30': 800, 'z1pt0': 31.07, 'z2pt5': 0.57, 'backarc': False,
              'vs30measured': True, 'region': 0}  #Fix region to 0 for check
    else:
        sp = {'vs30': 800, 'z1pt0': 31.07, 'z2pt5': 0.57, 'backarc': False,
              'vs30measured': True}  
            
    tmp_site = get_sites_from_rupture(tmp_rup, 'TC', 90, 'positive', 100, 99, sp)
    
    oqp = {'imtls': {k: [] for k in [imtx]}, 'mags': [f'{5.5:.2f}']}
    if '_toml=' in str(gmpe):
        tmp_gmm = valid.gsim(str(gmpe).split('_toml=')[1].replace(')',''))
    else:
        tmp_gmm = valid.gsim(gmpe)
    ctxm = ContextMaker('fake', [tmp_gmm], oqp)
    ctxs = list(ctxm.get_ctx_iter([tmp_rup], tmp_site))
    
    # Get model sigma and if not provided implement Al-Atik (2015) if specified
    tmp_mean, tmp_std, tmp_tau, tmp_phi = ctxm.get_mean_stds(ctxs)
    tmp_gmpe = str(tmp_gmm).split(']')[0].replace('[','')
    kwargs = {'gmpe': {tmp_gmpe: {'sigma_model_alatik2015': {}}},
              'sigma_model_alatik2015': {}}
    
    msg1 = 'Al-Atik (2015) sigma model has been used within an implementation of %s by the user.' %tmp_gmpe
    msg2 = 'A sigma model is not provided by default for %s GMPE.' %tmp_gmpe
    msg3 = 'For residual analysis a sigma model must be specified for %s GMPE.' %tmp_gmpe
    
    if tmp_std.all() == 0:
        sigma_model_flag = True
        if task == 'residual' and 'toml=' in str(gmpe) or task == 'comparison':
            if 'al_atik_2015_sigma' in str(gmpe): # No sigma so add
                gmpe = mgmpe.ModifiableGMPE(**kwargs)
                warnings.warn(msg1, stacklevel = 100)
            elif task == 'residual': # A sigma model is required for residuals
                raise ValueError(msg3)
            elif task == 'comparison': # No sigma and not specified in toml
                warnings.warn(msg2, stacklevel = 100)
                gmpe = valid.gsim(gmpe.split('(')[0])
        elif task == 'residual': # Task = 'residual' but no toml used so sigma model not specifiable
            warnings.warn(msg3, stacklevel = 100)
            gmpe = valid.gsim(gmpe)
    else:
        sigma_model_flag = False
        if 'al_atik_2015_sigma' in str(gmpe): #GMPE has sigma but override
            gmpe = mgmpe.ModifiableGMPE(**kwargs)
            warnings.warn(msg1, stacklevel = 100)
        elif task == 'comparison': # GMPE has sigma so retain (comparison use)
            gmpe = valid.gsim(gmpe)
        else: # GMPE has sigma so retain (residuals use)
            gmpe = valid.gsim(gmpe.split('(')[0])
    return gmpe, sigma_model_flag