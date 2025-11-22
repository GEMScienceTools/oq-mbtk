# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2025 GEM Foundation and G. Weatherill
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
Utilities used throughout the SMT (both Comparison and Residuals Module)
"""
import re
import numpy as np
from scipy.constants import g
from scipy.integrate import cumulative_trapezoid

from openquake.hazardlib.geo import PlanarSurface, Point
from openquake.hazardlib.source.rupture import BaseRupture
from openquake.hazardlib.gsim import get_available_gsims
from openquake.hazardlib.gsim.gmpe_table import GMPETable
from openquake.hazardlib.gsim.base import GMPE
from openquake.hazardlib import valid


# Get a list of the available GSIMs
AVAILABLE_GSIMS = get_available_gsims()

# Regular expression to get a GMPETable from string:
_gmpetable_regex = re.compile(r'^GMPETable\(([^)]+?)\)$')


### General utils for value validation
def get_float(xval):
    """
    Returns a float value, or none
    """
    if xval.strip():
        try:
            return float(xval)
        except:
            return None
    else:
        return None


def get_int(xval):
    """
    Returns an int value or none
    """
    if xval.strip():
        try:
            return int(xval)
        except:
            return None
    else:
        return None


def positive_float(value, key, verbose=False):
    """
    Returns True if the value is positive or zero, false otherwise
    """
    value = value.strip()
    if value and float(value) >= 0.0:
        return float(value)
    if verbose:
        print("Positive float value (or 0.0) is needed for %s - %s is given"
              % (key, str(value)))
    return None


def vfloat(value, key):
    """
    Returns value or None if not possible to calculate
    """
    value = value.strip()
    if value:
        try:
            return float(value)
        except:
            print("Invalid float value %s for %s" % (value, key))
    return None


def vint(value, key):
    """
    Returns value or None if not possible to calculate
    """
    value = value.strip()
    if "." in value:
        value = value.split(".")[0]
    if value:
        try:
            return int(value)
        except:
            print("Invalid int value %s for %s" % (value, key))
    return None


def positive_int(value, key):
    """
    Returns True if the value is positive or zero, false otherwise
    """
    value = value.strip()
    if value and int(value) >= 0.0:
        return int(value)
    print("Positive float value (or 0.0) is needed for %s - %s is given"
          % (key, str(value)))
    return False


def longitude(value):
    """
    Returns True if the longitude is valid, False otherwise
    """
    lon = float(value.strip())
    if not lon:
        return False
    if (lon >= -180.0) and (lon <= 180.0):
        return lon
    print("Longitude %s is outside of range -180 <= lon <= 180" % str(lon))
    return False


def latitude(value):
    """
    Returns True if the latitude is valid, False otherwise
    """
    lat = float(value.strip())
    if not lat:
        print("Latitude is missing")
        return False
    if (lat >= -90.0) and (lat <= 90.0):
        return lat 
    print("Latitude %s is outside of range -90 <= lat <= 90" % str(lat))
    return False


def strike(value):
    """
    Returns a float value in range 0 - 360.0
    """
    strike = value.strip()
    if not strike:
        return None
    strike = float(strike)
    if strike and (strike >= 0.0) and (strike <= 360.0):
        return strike
    print("Strike %s is not in range 0 - 360" % value)
    return None


def dip(value):
    """
    Returns a float value in range 0 - 90.
    """
    dip = value.strip()
    if not dip:
        return None
    dip = float(dip)
    if dip and (dip > 0.0) and (dip <= 90.0):
        return dip
    print("Dip %s is not in range 0 - 90" % value)
    return None


def rake(value):
    """
    Returns a float value in range -180 - 180
    """
    rake = value.strip()
    if not rake:
        return None
    rake = float(rake)
    if rake and (rake >= -180.0) and (rake <= 180.0):
        return rake
    print("Rake %s is not in range -180 - 180" % value)
    return None


### General utils for ctx management ###
def make_rup(lon,
             lat,
             dep,
             msr,
             mag,
             aratio,
             strike,
             dip,
             rake,
             trt,
             ztor=None):
    """
    Creates an OQ planar rupture given the hypocenter position
    """
    hypoc = Point(lon, lat, dep)
    srf = PlanarSurface.from_hypocenter(hypoc,
                                        msr,
                                        mag,
                                        aratio,
                                        strike,
                                        dip,
                                        rake,
                                        ztor)
    rup = BaseRupture(mag, rake, trt, hypoc, srf)
    rup.hypocenter.depth = dep
    return rup


def full_dtype_gmm():
    """
    Instantiate a DummyGMPE with all distance types. This is useful
    for returning all distance metrics from a ctx (otherwise only
    the distance types used by the given GMM are returned).
    """
    core_r_types = [
        'repi', 'rrup', 'rjb', 'rhypo', 'rx', "ry0", "rvolc"]
    gmpe = valid.gsim("DummyGMPE")
    orig_r_types = list(gmpe.REQUIRES_DISTANCES)
    for core in core_r_types:
        if core not in orig_r_types:
            orig_r_types.append(core)
    gmpe.REQUIRES_DISTANCES = frozenset(orig_r_types)
    return gmpe


### General utils for time series
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


def get_time_vector(time_step, number_steps):
    """
    Returns a time vector
    """
    return np.cumsum(time_step * np.ones(number_steps, dtype=float)) - time_step


def get_velocity_displacement(time_step, acceleration, units="cm/s/s",
                              velocity=None, displacement=None):
    """
    Returns the velocity and displacement time series using simple integration
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
        velocity = time_step * cumulative_trapezoid(acceleration, initial=0.)
    if displacement is None:
        displacement = time_step * cumulative_trapezoid(velocity, initial=0.)
    return velocity, displacement


def equalise_series(series_x, series_y):
    """
    For two time series from the same record but of different length
    cuts both records down to the length of the shortest record
    N.B. This assumes that the start times and the time-steps of the record
    are the same - if not then this may introduce biases into the record
    :param numpy.ndarray series_x:
         X Time series
    :param numpy.ndarray series_y:
         Y Time series
    """
    n_x = len(series_x)
    n_y = len(series_y)
    if n_x > n_y:
        return series_x[:n_y], series_y
    elif n_y > n_x:
        return series_x, series_y[:n_x]
    else:
        return series_x, series_y


def nextpow2(nval):
    m_f = np.log2(nval)
    m_i = np.ceil(m_f)
    return int(2.0 ** m_i)


### Utils for managing GMMs in the Residuals Module
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
            output_gsims[_get_gmpe_name(gs)] = gs # Get name of GMPE instance
        elif gs in AVAILABLE_GSIMS:
            output_gsims[gs] = AVAILABLE_GSIMS[gs]()
        else:
            match = _gmpetable_regex.match(gs) # GMPETable ?
            if match:
                filepath = match.group(1).split("=")[1] # Get table filename
                output_gsims[gs] = GMPETable(gmpe_table=filepath)
            else:
                raise ValueError('%s Not supported by OpenQuake' % gs)

    return output_gsims


def _get_gmpe_name(gsim):
    """
    Returns the name of the GMPE given an instance of the class
    """
    match = _gmpetable_regex.match(str(gsim)) # GMPETable ?
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
            val = str(gsim.__dict__[key]) 
            additional_args.append("{:s}={:s}".format(key, val))
        if len(additional_args):
            gsim_name_str = "({:s})".format(", ".join(additional_args))
            return gsim_name + gsim_name_str
        else:
            return gsim_name


def clean_gmm_label(gmpe, drop_weight_info=False):
    """
    Return a string of GMM which contains no slashes or new line
    syntax for use in plots (generally this occurs from the use of
    ModifiableGMPE with a GMM containing additional input arguments).

    Also can remove LT weight information if required.
    """
    # Clean the gmpe
    gmm_label = re.sub(r'\\+n', ' ', gmpe)
    gmm_label = re.sub(r'\\', '', gmm_label)

    lines = gmm_label.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith('gmpe = '):
            prefix, value = line.split('=', 1)
            parts = value.strip().split()
            value_clean = ', '.join(parts)
            lines[i] = f"{prefix.strip()} = {value_clean}"
    gmm_label = '\n'.join(lines)

    # Might not want to retain the GMC LT weight info
    if drop_weight_info is True:
        parts = [part.strip() for part in gmm_label.split(
            '\n') if "lt_weight_gmc" not in part]
        gmm_label = ', '.join(parts)

    return gmm_label


### Vs30 to z1pt0 and z2pt5 relationships from GMMs
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


def z1pt0_to_z2pt5_cb07(z1pt0):
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
