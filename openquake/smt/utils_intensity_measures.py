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
Utils for handling intensity measures and spectra:
    1) General intensity measure utils e.g. computing peak measures
    1) Response spectra
    2) Fourier amplitude spectra (FAS)
    3) Horizontal-Vertical Spectral Ratio (HVSR)
    5) Duration-based ground-motion intensity measures (e.g. Arias intensity, CAV)
    6) Obtaining rotation-based (and rotation-independent) defintions of the horizontal component
"""
import numpy as np
from math import pi
from scipy.constants import g
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt

import openquake.smt.response_spectrum as rsp
from openquake.smt import response_spectrum_smoothing as rsps
from openquake.smt.utils import (
    equalise_series, get_time_vector, nextpow2, _save_image,)

RESP_METHOD = {
    'Newmark-Beta': rsp.NewmarkBeta, 'Nigam-Jennings': rsp.NigamJennings}

SMOOTHING = {"KonnoOhmachi": rsps.KonnoOhmachi}

### General intensity measure utils
SCALAR_XY = {
    "Geometric": lambda x, y: np.sqrt(x * y),
    "Arithmetic": lambda x, y: (x + y) / 2.,
    "Larger": lambda x, y: np.max(np.array([x, y]), axis=0),
    "Vectorial": lambda x, y: np.sqrt(x ** 2. + y ** 2.)}


def get_peak_measures(time_step, acceleration, get_vel=False, get_disp=False):
    """
    Returns the peak measures from acceleration, velocity and displacement
    time-series
    :param float time_step:
        Time step of acceleration time series in s
    :param numpy.ndarray acceleration:
        Acceleration time series
    :param bool get_vel:
        Choose to return (and therefore calculate) velocity (True) or otherwise
        (false)
    :returns:
        * pga - Peak Ground Acceleration
        * pgv - Peak Ground Velocity
        * pgd - Peak Ground Displacement
        * velocity - Velocity Time Series
        * displacement - Displacement Time series
    """
    pga = np.max(np.fabs(acceleration))
    velocity = None
    displacement = None
    # If displacement is not required then do not integrate to get
    # displacement time series
    if get_disp:
        get_vel = True
    if get_vel:
        velocity = time_step * cumulative_trapezoid(acceleration, initial=0.)
        pgv = np.max(np.fabs(velocity))
    else:
        pgv = None
    if get_disp:
        displacement = time_step * cumulative_trapezoid(velocity, initial=0.)
        pgd = np.max(np.fabs(displacement))
    else:
        pgd = None
    return pga, pgv, pgd, velocity, displacement


def get_quadratic_intensity(acc_x, acc_y, time_step):
    """
    Returns the quadratic intensity of a pair of records, define as:
    (1. / duration) * \int_0^{duration} a_1(t) a_2(t) dt
    This assumes the time-step of the two records is the same!
    """
    assert len(acc_x) == len(acc_y)
    dur = time_step * float(len(acc_x) - 1)
    return (1. / dur) * np.trapz(acc_x * acc_y, dx=time_step)


### Response Spectra
def get_response_spectrum(acceleration, time_step, periods, damping=0.05, 
                          units="cm/s/s", method="Nigam-Jennings"):
    """
    Returns the elastic response spectrum of the acceleration time series.
    :param numpy.ndarray acceleration:
        Acceleration time series
    :param float time_step:
        Time step of acceleration time series in s
    :param numpy.ndarray periods:
        List of periods for calculation of the response spectrum
    :param float damping:
        Fractional coefficient of damping
    :param str units:
        Units of the INPUT ground motion records
    :param str method:
        Choice of method for calculation of the response spectrum
        - "Newmark-Beta"
        - "Nigam-Jennings"
    :returns:
        Outputs from :class: openquake.smt.response_spectrum.BaseResponseSpectrum
    """
    response_spec = RESP_METHOD[method](acceleration,
                                        time_step,
                                        periods, 
                                        damping,
                                        units)
    spectrum, time_series, accel, vel, disp = response_spec()
    spectrum["PGA"] = time_series["PGA"]
    spectrum["PGV"] = time_series["PGV"]
    spectrum["PGD"] = time_series["PGD"]
    return spectrum, time_series, accel, vel, disp


def get_response_spectrum_pair(acceleration_x, time_step_x, acceleration_y,
                               time_step_y, periods, damping=0.05,
                               units="cm/s/s", method="Nigam-Jennings"):
    """
    Returns the response spectra of a record pair
    :param numpy.ndarray acceleration_x:
        Acceleration time-series of x-component of record
    :param float time_step_x:
        Time step of x-time series (s)
    :param numpy.ndarray acceleration_y:
        Acceleration time-series of y-component of record
    :param float time_step_y:
        Time step of y-time series (s)
    """

    sax = get_response_spectrum(acceleration_x,
                                time_step_x,
                                periods,
                                damping, 
                                units, 
                                method)[0]
    say = get_response_spectrum(acceleration_y,
                                time_step_y,
                                periods, 
                                damping, 
                                units, 
                                method)[0]
    return sax, say


def geometric_mean_spectrum(sax, say):
    """
    Returns the geometric mean of the response spectrum
    :param dict sax:
        Dictionary of response spectrum outputs from x-component
    :param dict say:
        Dictionary of response spectrum outputs from y-component
    """
    sa_gm = {}
    for key in sax:
        if key == "Period":
            sa_gm[key] = sax[key]
        else:
            sa_gm[key] = np.sqrt(sax[key] * say[key])
    return sa_gm


def arithmetic_mean_spectrum(sax, say):
    """
    Returns the arithmetic mean of the response spectrum
    """
    sa_am = {}
    for key in sax:
        if key == "Period":
            sa_am[key] = sax[key]
        else:
            sa_am[key] = (sax[key] + say[key]) / 2.0
    return sa_am


def envelope_spectrum(sax, say):
    """
    Returns the envelope of the response spectrum
    """
    sa_env = {}
    for key in sax:
        if key == "Period":
            sa_env[key] = sax[key]
        else:
            sa_env[key] = np.max(np.column_stack([sax[key], say[key]]),
                                 axis=1)
    return sa_env


def get_response_spectrum_intensity(spec):
    """
    Returns the response spectrum intensity (Housner intensity), defined
    as the integral of the pseudo-velocity spectrum between the periods of
    0.1 s and 2.5 s
    :param dict spec:
        Response spectrum of the record as output from :class:
        openquake.smt.response_spectrum.BaseResponseSpectrum
    """
    idx = np.where(np.logical_and(spec["Period"] >= 0.1,
                                  spec["Period"] <= 2.5))[0]
    return np.trapz(spec["Pseudo-Velocity"][idx],
                    spec["Period"][idx])


def get_acceleration_spectrum_intensity(spec):
    """
    Returns the acceleration spectrum intensity, defined as the integral
    of the psuedo-acceleration spectrum between the periods of 0.1 and 0.5 s
    """
    idx = np.where(np.logical_and(spec["Period"] >= 0.1,
                                  spec["Period"] <= 0.5))[0]
    return np.trapz(spec["Pseudo-Acceleration"][idx],
                    spec["Period"][idx])


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
        (np.log10(target_period) - np.log10(periods[lval])) * d_y / d_x)


def larger_pga(sax, say):
    """
    Returns the spectral acceleration from the component with the larger PGA
    """
    if sax["PGA"] >= say["PGA"]:
        return sax
    else:
        return say


### FAS Functions
def get_fourier_spectrum(time_series, time_step):
    """
    Returns the Fourier spectrum of the time series
    :param numpy.ndarray time_series:
        Array of values representing the time series
    :param float time_step:
        Time step of the time series
    :returns:
        Frequency (as numpy array)
        Fourier Amplitude (as numpy array)
    """
    n_val = nextpow2(len(time_series))
    # numpy.fft.fft will zero-pad records whose length is less than the
    # specified nval
    # Get Fourier spectrum
    fspec = np.fft.fft(time_series, n_val)
    # Get frequency axes
    d_f = 1. / (n_val * time_step)
    freq = d_f * np.arange(0., (n_val / 2.0), 1.0)
    return freq, time_step * np.absolute(fspec[:int(n_val / 2.0)])


def plot_fourier_spectrum(time_series, time_step, figure_size=(7, 5),
                          filename=None, filetype="png", dpi=300):
    """
    Plots the Fourier spectrum of a time series 
    """
    freq, amplitude = get_fourier_spectrum(time_series, time_step)
    plt.figure(figsize=figure_size)
    plt.loglog(freq, amplitude, 'b-')
    plt.xlabel("Frequency (Hz)", fontsize=14)
    plt.ylabel("Fourier Amplitude", fontsize=14)
    _save_image(filename, plt.gcf(), filetype, dpi)
    plt.show()


### HVRS Functions
def get_hvsr(x_component, x_time_step, y_component, y_time_step, vertical,
             vertical_time_step, smoothing_params):
    """
    :param x_component:
        Time series of the x-component of the data
    :param float x_time_step:
        Time-step (in seconds) of the x-component
    :param y_component:
        Time series of the y-component of the data
    :param float y_time_step:
        Time-step (in seconds) of the y-component
    :param vertical:
        Time series of the vertical of the data
    :param float vertical_time_step:
        Time-step (in seconds) of the vertical component
    :param dict smoothing_params:
        Parameters controlling the smoothing of the individual spectra
        Should contain:
        * 'Function' - Name of smoothing method (e.g. KonnoOhmachi)
        * Controlling parameters
    :returns:
        * horizontal-to-vertical spectral ratio
        * frequency
        * maximum H/V
        * Period of Maximum H/V
    """
    smoother = SMOOTHING[smoothing_params["Function"]](smoothing_params)
    # Get x-component Fourier spectrum
    xfreq, xspectrum = get_fourier_spectrum(x_component, x_time_step)
    # Smooth spectrum
    xsmooth = smoother.apply_smoothing(xspectrum, xfreq)
    # Get y-component Fourier spectrum
    yfreq, yspectrum = get_fourier_spectrum(y_component, y_time_step)
    # Smooth spectrum
    ysmooth = smoother.apply_smoothing(yspectrum, yfreq)
    # Take geometric mean of x- and y-components for horizontal spectrum
    hor_spec = np.sqrt(xsmooth * ysmooth)
    # Get vertical Fourier spectrum
    vfreq, vspectrum = get_fourier_spectrum(vertical, vertical_time_step)
    # Smooth spectrum
    vsmooth = smoother.apply_smoothing(vspectrum, vfreq)
    # Get HVSR
    hvsr = hor_spec / vsmooth
    max_loc = np.argmax(hvsr)
    return hvsr, xfreq, hvsr[max_loc], 1.0 / xfreq[max_loc]


### Utils for duration-based IMT functions
def get_husid(acceleration, time_step):
    """
    Returns the Husid vector, defined as \int{acceleration ** 2.}
    :param numpy.ndarray acceleration:
        Vector of acceleration values
    :param float time_step:
        Time-step of record (s)
    """
    time_vector = get_time_vector(time_step, len(acceleration))
    husid = np.hstack([0., cumulative_trapezoid(acceleration ** 2., time_vector)])
    return husid, time_vector


def get_arias_intensity(acceleration, time_step, start_level=0., end_level=1.):
    """
    Returns the Arias intensity of the record
    :param float start_level:
        Fraction of the total Arias intensity used as the start time
    :param float end_level:
        Fraction of the total Arias intensity used as the end time
    """
    assert end_level >= start_level
    arias_factor = pi / (2.0 * (g * 100.))
    husid, time_vector = get_husid(acceleration, time_step)
    husid_norm = husid / husid[-1]
    idx = np.where(np.logical_and(husid_norm >= start_level,
                                  husid_norm <= end_level))[0]
    if len(idx) < len(acceleration):
        husid, time_vector = get_husid(acceleration[idx], time_step)
    return arias_factor * husid[-1]


def plot_husid(acceleration, time_step, start_level=0., end_level=1.0,
               figure_size=(7, 5), filename=None, filetype="png", dpi=300):
    """
    Creates a Husid plot for the record
    :param tuple figure_size:
        Size of the output figure (Width, Height)
    :param str filename:
        Name of the file to export
    :param str filetype:
        Type of file for export
    :param int dpi:
        FIgure resolution in dots per inch.
    """
    plt.figure(figsize=figure_size)
    husid, time_vector = get_husid(acceleration, time_step)
    husid_norm = husid / husid[-1]
    idx = np.where(np.logical_and(husid_norm >= start_level,
                                  husid_norm <= end_level))[0]
    plt.plot(time_vector, husid_norm, "b-", linewidth=2.0,
             label="Original Record")
    plt.plot(time_vector[idx], husid_norm[idx], "r-", linewidth=2.0,
             label="%5.3f - %5.3f Arias" % (start_level, end_level))
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Fraction of Arias Intensity", fontsize=14)
    plt.title("Husid Plot")
    plt.legend(loc=4, fontsize=14)
    _save_image(filename, plt.gcf(), filetype, dpi)
    plt.show()


def get_bracketed_duration(acceleration, time_step, threshold):
    """
    Returns the bracketed duration, defined as the time between the first and
    last excursions above a particular level of acceleration
    :param float threshold:
        Threshold acceleration in units of the acceleration time series
    """
    idx = np.where(np.fabs(acceleration) >= threshold)[0]
    if len(idx) == 0:
        # Record does not exced threshold at any point
        return 0.
    else:
        time_vector = get_time_vector(time_step, len(acceleration))
        return time_vector[idx[-1]] - time_vector[idx[0]] + time_step


def get_uniform_duration(acceleration, time_step, threshold):
    """
    Returns the total duration for which the record exceeds the threshold
    """ 
    idx = np.where(np.fabs(acceleration) >= threshold)[0]
    return time_step * float(len(idx))


def get_significant_duration(acceleration, time_step, start_level=0.,
                             end_level=1.0):
    """
    Returns the significant duration of the record
    """
    assert end_level >= start_level
    husid, time_vector = get_husid(acceleration, time_step)
    idx = np.where(np.logical_and(husid >= (start_level * husid[-1]),
                                  husid <= (end_level * husid[-1])))[0]
    return time_vector[idx[-1]] - time_vector[idx[0]] + time_step


def get_cav(acceleration, time_step, threshold=0.0):
    """
    Returns the cumulative absolute velocity above a given threshold of
    acceleration
    """
    acceleration = np.fabs(acceleration)
    idx = np.where(acceleration >= threshold)
    if len(idx) > 0:
        return np.trapz(acceleration[idx], dx=time_step)
    else:
        return 0.0


def get_arms(acceleration, time_step):
    """
    Returns the root mean square acceleration, defined as
    sqrt{(1 / duration) * \int{acc ^ 2} dt}
    """
    dur = time_step * float(len(acceleration) - 1)
    return np.sqrt((1. / dur) * np.trapz(acceleration  ** 2., dx=time_step))


### Utils for computing rotation-based definitions of horizontal component
def rotate_horizontal(series_x, series_y, angle):
    """
    Rotates two time-series according to a specified angle
    :param nunmpy.ndarray series_x:
        Time series of x-component
    :param nunmpy.ndarray series_y:
        Time series of y-component
    :param float angle:
        Angle of rotation (decimal degrees)
    """
    angle = angle * (pi / 180.0)
    rot_hist_x = (np.cos(angle) * series_x) + (np.sin(angle) * series_y)
    rot_hist_y = (-np.sin(angle) * series_x) + (np.cos(angle) * series_y)
    return rot_hist_x, rot_hist_y


def gmrotdpp(acceleration_x, time_step_x, acceleration_y, time_step_y, periods,
             percentile, damping=0.05, units="cm/s/s", method="Nigam-Jennings"):
    """
    Returns the rotationally-dependent geometric mean

    :param float percentile:
        Percentile of angles (float)
    :returns:
        - Dictionary contaning
        * angles - Array of rotation angles
        * periods - Array of periods
        * GMRotDpp - The rotationally-dependent geometric mean at the specified
                     percentile
        * GeoMeanPerAngle - An array of [Number Angles, Number Periods]
          indicating the Geometric Mean of the record pair when rotated to
          each period

    e.g. to compute GMRotD50 use a percentile of 50
    """
    if (percentile > 100. + 1E-9) or (percentile < 0.):
        raise ValueError("Percentile for GMRotDpp must be between 0. and 100.")
    # Get the time-series corresponding to the SDOF
    sax, _, x_a, _, _ = get_response_spectrum(acceleration_x,
                                              time_step_x,
                                              periods, damping,
                                              units, method)
    say, _, y_a, _, _ = get_response_spectrum(acceleration_y,
                                              time_step_y,
                                              periods, damping,
                                              units, method)
    x_a, y_a = equalise_series(x_a, y_a)
    angles = np.arange(0., 90., 1.)
    max_a_theta = np.zeros([len(angles), len(periods)], dtype=float)
    max_a_theta[0, :] = np.sqrt(np.max(np.fabs(x_a), axis=0) *
                                np.max(np.fabs(y_a), axis=0))
    for iloc, theta in enumerate(angles):
        if iloc == 0:
            max_a_theta[iloc, :] = np.sqrt(np.max(np.fabs(x_a), axis=0) *
                                           np.max(np.fabs(y_a), axis=0))
        else:
            rot_x, rot_y = rotate_horizontal(x_a, y_a, theta)
            max_a_theta[iloc, :] = np.sqrt(np.max(np.fabs(rot_x), axis=0) *
                                           np.max(np.fabs(rot_y), axis=0))

    gmrotd = np.percentile(max_a_theta, percentile, axis=0)
    return {
        "angles": angles,
        "periods": periods,
        "GMRotDpp": gmrotd,
        "GeoMeanPerAngle": max_a_theta}


def gmrotdpp_slow(acceleration_x, time_step_x, acceleration_y, time_step_y,
                  periods, percentile, damping=0.05, units="cm/s/s",
                  method="Nigam-Jennings"):
    """
    Returns the rotationally-dependent geometric mean. This "slow" version
    will rotate the original time-series and calculate the response spectrum
    at each angle. This is a slower process, but it means that GMRotDpp values
    can be calculated for othe time-series parameters (i.e. PGA, PGV and PGD) 
    Inputs as for gmrotdpp
    """
    key_list = ["PGA",
                "PGV",
                "PGD",
                "Acceleration",
                "Velocity", 
                "Displacement",
                "Pseudo-Acceleration",
                "Pseudo-Velocity"]

    if (percentile > 100. + 1E-9) or (percentile < 0.):
        raise ValueError("Percentile for GMRotDpp must be between 0. and 100.")
    accel_x, accel_y = equalise_series(acceleration_x, acceleration_y)
    angles = np.arange(0., 90., 1.)

    gmrotdpp = {
        "Period": periods,
        "PGA": np.zeros(len(angles), dtype=float),
        "PGV": np.zeros(len(angles), dtype=float),
        "PGD": np.zeros(len(angles), dtype=float),
        "Acceleration": np.zeros([len(angles), len(periods)], dtype=float),
        "Velocity": np.zeros([len(angles), len(periods)], dtype=float),
        "Displacement": np.zeros([len(angles), len(periods)], dtype=float),
        "Pseudo-Acceleration": np.zeros([len(angles), len(periods)], 
                                        dtype=float),
        "Pseudo-Velocity": np.zeros([len(angles), len(periods)], dtype=float)}
    # Get the response spectra for each angle
    for iloc, theta in enumerate(angles):
        if np.fabs(theta) < 1E-9:
            rot_x, rot_y = (accel_x, accel_y)
        else:
            rot_x, rot_y = rotate_horizontal(accel_x, accel_y, theta)
        sax, say = get_response_spectrum_pair(rot_x, time_step_x,
                                              rot_y, time_step_y,
                                              periods, damping,
                                              units, method)

        sa_gm = geometric_mean_spectrum(sax, say)
        for key in key_list:
            if key in ["PGA", "PGV", "PGD"]:
                 gmrotdpp[key][iloc] = sa_gm[key]
            else:
                 gmrotdpp[key][iloc, :] = sa_gm[key]
              
    # Get the desired fractile
    for key in key_list:
        gmrotdpp[key] = np.percentile(gmrotdpp[key], percentile, axis=0)
    return gmrotdpp


def _get_gmrotd_penalty(gmrotd, gmtheta):
    """
    Calculates the penalty function of 4 of Boore, Watson-Lamprey and
    Abrahamson (2006), corresponding to the sum of squares difference between
    the geometric mean of the pair of records and that of the desired GMRotDpp
    """
    n_angles, n_per = np.shape(gmtheta)
    penalty = np.zeros(n_angles, dtype=float)
    coeff = 1. / float(n_per)
    for iloc in range(0, n_angles):
        penalty[iloc] = coeff * np.sum(
            ((gmtheta[iloc, :] / gmrotd) - 1.) ** 2.)

    locn = np.argmin(penalty)
    return locn, penalty


def gmrotipp(acceleration_x, time_step_x, acceleration_y, time_step_y, periods,
             percentile, damping=0.05, units="cm/s/s", method="Nigam-Jennings"):
    """
    Returns the rotationally-independent geometric mean (GMRotIpp)

    e.g. to compute GMRotI50 use a percentile of 50
    """
    acceleration_x, acceleration_y = equalise_series(acceleration_x,
                                                     acceleration_y)
    gmrot = gmrotdpp(acceleration_x, time_step_x, acceleration_y,
                     time_step_y, periods, percentile, damping, units, method)
   
    
    min_loc, penalty = _get_gmrotd_penalty(gmrot["GMRotDpp"],
                                           gmrot["GeoMeanPerAngle"])
    target_angle = gmrot["angles"][min_loc]

    rot_hist_x, rot_hist_y = rotate_horizontal(acceleration_x,
                                               acceleration_y,
                                               target_angle)
    sax, say = get_response_spectrum_pair(rot_hist_x, time_step_x,
                                          rot_hist_y, time_step_y,
                                          periods, damping, units, method)

    gmroti = geometric_mean_spectrum(sax, say)
    gmroti["GMRotD{:.2f}".format(percentile)] = gmrot["GMRotDpp"]
    return gmroti


def rotdpp(acceleration_x, time_step_x, acceleration_y, time_step_y, periods,
           percentile, damping=0.05, units="cm/s/s", method="Nigam-Jennings"):
    """
    Returns the rotationally dependent spectrum RotDpp as defined by Boore
    (2010)

    e.g. to compute RotD50 use a percentile of 50
    """
    if np.fabs(time_step_x - time_step_y) > 1E-10:
        raise ValueError("Record pair must have the same time-step!")
    acceleration_x, acceleration_y = equalise_series(acceleration_x,
                                                     acceleration_y)
    theta_set = np.arange(0., 180., 1.)
    max_a_theta = np.zeros([len(theta_set), len(periods) + 1])
    max_v_theta = np.zeros_like(max_a_theta)
    max_d_theta = np.zeros_like(max_a_theta)
    for iloc, theta in enumerate(theta_set):
        theta_rad = np.radians(theta)
        arot = acceleration_x * np.cos(theta_rad) +\
            acceleration_y * np.sin(theta_rad)
        saxy = get_response_spectrum(arot, time_step_x, periods, damping,
            units, method)[0]
        max_a_theta[iloc, 0] = saxy["PGA"]
        max_a_theta[iloc, 1:] = saxy["Pseudo-Acceleration"]
        max_v_theta[iloc, 0] = saxy["PGV"]
        max_v_theta[iloc, 1:] = saxy["Pseudo-Velocity"]
        max_d_theta[iloc, 0] = saxy["PGD"]
        max_d_theta[iloc, 1:] = saxy["Displacement"]
    rotadpp = np.percentile(max_a_theta, percentile, axis=0)
    rotvdpp = np.percentile(max_v_theta, percentile, axis=0)
    rotddpp = np.percentile(max_d_theta, percentile, axis=0)
    output = {"Pseudo-Acceleration": rotadpp[1:],
              "Pseudo-Velocity": rotvdpp[1:],
              "Displacement": rotddpp[1:],
              "PGA": rotadpp[0],
              "PGV": rotvdpp[0],
              "PGD": rotddpp[0]}
    return output, max_a_theta, max_v_theta, max_d_theta, theta_set


def rotipp(acceleration_x, time_step_x, acceleration_y, time_step_y, periods,
           percentile, damping=0.05, units="cm/s/s", method="Nigam-Jennings"):
    """
    Returns the rotationally independent spectrum RotIpp as defined by
    Boore (2010)
    """
    if np.fabs(time_step_x - time_step_y) > 1E-10:
        raise ValueError("Record pair must have the same time-step!")
    acceleration_x, acceleration_y = equalise_series(acceleration_x,
                                                     acceleration_y)
    target, rota, rotv, rotd, angles = rotdpp(acceleration_x, time_step_x,
                                              acceleration_y, time_step_y,
                                              periods, percentile, damping,
                                              units, method)
    locn, penalty = _get_gmrotd_penalty(
        np.hstack([target["PGA"],target["Pseudo-Acceleration"]]),
        rota)
    target_theta = np.radians(angles[locn])
    arotpp = acceleration_x * np.cos(target_theta) +\
        acceleration_y * np.sin(target_theta)
    spec = get_response_spectrum(arotpp, time_step_x, periods, damping, units,
        method)[0]
    spec["GMRot{:2.0f}".format(percentile)] = target
    return spec
