# ------------------- The OpenQuake Model Building Toolkit --------------------
# Copyright (C) 2022 GEM Foundation
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
Module :mod:`openquake.cat.utils` contains tools for working with catalogue
files
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt

MARKER_NORMAL = np.array([0, 31, 59, 90, 120, 151, 181,
                          212, 243, 273, 304, 334])

MARKER_LEAP = np.array([0, 31, 60, 91, 121, 152, 182,
                        213, 244, 274, 305, 335])

SECONDS_PER_DAY = 86400.0


def _prepare_coords(lons1, lats1, lons2, lats2):
    """
    Convert two pairs of spherical coordinates in decimal degrees
    to numpy arrays of radians. Makes sure that respective coordinates
    in pairs have the same shape.
    """
    lons1 = np.array(np.radians(lons1))
    lats1 = np.array(np.radians(lats1))
    assert lons1.shape == lats1.shape
    lons2 = np.array(np.radians(lons2))
    lats2 = np.array(np.radians(lats2))
    assert lons2.shape == lats2.shape
    return lons1, lats1, lons2, lats2


def decimal_year(year, month, day):
    """
    Allows to calculate the decimal year for a vector of dates
    (TODO this is legacy code kept to maintain comparability with previous
    declustering algorithms!)

    :param year: year column from catalogue matrix
    :type year: numpy.ndarray
    :param month: month column from catalogue matrix
    :type month: numpy.ndarray
    :param day: day column from catalogue matrix
    :type day: numpy.ndarray
    :returns: decimal year column
    :rtype: numpy.ndarray
    """
    marker = np.array([0., 31., 59., 90., 120., 151., 181.,
                       212., 243., 273., 304., 334.])
    tmonth = (month - 1).astype(int)
    day_count = marker[tmonth] + day - 1.
    dec_year = year + (day_count / 365.)

    return dec_year


def leap_check(year):
    """
    Returns logical array indicating if year is a leap year
    """
    return np.logical_and((year % 4) == 0,
                          np.logical_or((year % 100 != 0), (year % 400) == 0))


def decimal_time(year, month, day, hour, minute, second):
    """
    Returns the full time as a decimal value
    :param year:
        Year of events (integer numpy.ndarray)
    :param month:
        Month of events (integer numpy.ndarray)
    :param day:
        Days of event (integer numpy.ndarray)
    :param hour:
        Hour of event (integer numpy.ndarray)
    :param minute:
        Minute of event (integer numpy.ndarray)
    :param second:
        Second of event (float numpy.ndarray)
    :returns decimal_time:
        Decimal representation of the time (as numpy.ndarray)
    """
    tmonth = month - 1
    day_count = MARKER_NORMAL[tmonth] + day - 1
    id_leap = leap_check(year)
    leap_loc = np.where(id_leap)[0]
    day_count[leap_loc] = MARKER_LEAP[tmonth[leap_loc]] + day[leap_loc] - 1
    year_secs = (day_count.astype(float) * SECONDS_PER_DAY) + second + \
        (60. * minute.astype(float)) + (3600. * hour.astype(float))
    decimal_time = year.astype(float) + (year_secs / (365. * 24. * 3600.))
    decimal_time[leap_loc] = year[leap_loc].astype(float) + \
        (year_secs[leap_loc] / (366. * 24. * 3600.))
    return decimal_time


def haversine(lon1, lat1, lon2, lat2, radians=False, earth_rad=6371.227):
    """
    Allows to calculate geographical distance
    using the haversine formula.

    :param lon1: longitude of the first set of locations
    :type lon1: numpy.ndarray
    :param lat1: latitude of the frist set of locations
    :type lat1: numpy.ndarray
    :param lon2: longitude of the second set of locations
    :type lon2: numpy.float64
    :param lat2: latitude of the second set of locations
    :type lat2: numpy.float64
    :keyword radians: states if locations are given in terms of radians
    :type radians: bool
    :keyword earth_rad: radius of the earth in km
    :type earth_rad: float
    :returns: geographical distance in km
    :rtype: numpy.ndarray
    """
    if radians is False:
        cfact = np.pi / 180.
        lon1 = cfact * lon1
        lat1 = cfact * lat1
        lon2 = cfact * lon2
        lat2 = cfact * lat2

    # Number of locations in each set of points
    if not np.shape(lon1):
        nlocs1 = 1
        lon1 = np.array([lon1])
        lat1 = np.array([lat1])
    else:
        nlocs1 = np.max(np.shape(lon1))
    if not np.shape(lon2):
        nlocs2 = 1
        lon2 = np.array([lon2])
        lat2 = np.array([lat2])
    else:
        nlocs2 = np.max(np.shape(lon2))
    # Pre-allocate array
    distance = np.zeros((nlocs1, nlocs2))
    i = 0
    while i < nlocs2:
        # Perform distance calculation
        dlat = lat1 - lat2[i]
        dlon = lon1 - lon2[i]
        aval = ((np.sin(dlat / 2.) ** 2.) + (np.cos(lat1) * np.cos(lat2[i]) *
                (np.sin(dlon / 2.) ** 2.)))
        distance[:, i] = (2. * earth_rad * np.arctan2(np.sqrt(aval),
                                                      np.sqrt(1 - aval))).T
        i += 1
    return distance


def greg2julian(year, month, day, hour, minute, second):
    """
    Function to convert a date from Gregorian to Julian format
    :param year:
        Year of events (integer numpy.ndarray)
    :param month:
        Month of events (integer numpy.ndarray)
    :param day:
        Days of event (integer numpy.ndarray)
    :param hour:
        Hour of event (integer numpy.ndarray)
    :param minute:
        Minute of event (integer numpy.ndarray)
    :param second:
        Second of event (float numpy.ndarray)
    :returns julian_time:
        Julian representation of the time (as float numpy.ndarray)
    """
    year = year.astype(float)
    month = month.astype(float)
    day = day.astype(float)

    timeut = hour.astype(float) + (minute.astype(float) / 60.0) + \
        (second / 3600.0)

    julian_time = ((367.0 * year) - np.floor(7.0 * (year +
             np.floor((month + 9.0) / 12.0)) / 4.0) - np.floor(3.0 *
             (np.floor((year + (month - 9.0) / 7.0) / 100.0) + 1.0) /
             4.0) + np.floor((275.0 * month) / 9.0) + day +
             1721028.5 + (timeut / 24.0))
    return julian_time


def _set_string(value):
    """
    Turns a number into a string prepended with + or - depending on
    whether the number if positive or negative.
    """
    if value >= 0.0:
        return f"+ {value:.3f}"
    else:
        return f"- {value:.3f}"


def _to_latex(string):
    """
    For a string given in the form XX(YYYY) returns the LaTeX string to
    place bracketed contents as a subscript
    :param
    """
    lb = string.find("(")
    ub = string.find(")")
    return "$" + string[:lb] + ("_{%s}$" % string[lb+1:ub])


def piecewise_linear_scalar(params, xval):
    '''Piecewise linear function for a scalar variable xval (float).
    :param params:
        Piecewise linear parameters (numpy.ndarray) in the following form:
        [slope_i,... slope_n, turning_point_i, ..., turning_point_n, intercept]
        Length params === 2 * number_segments, e.g.
        [slope_1, slope_2, slope_3, turning_point1, turning_point_2, intercept]
    :param xval:
        Value for evaluation of function (float)
    :returns:
        Piecewise linear function evaluated at point xval (float)
    '''
    n_params = len(params)
    if math.fabs(float(n_params / 2) - float(n_params) / 2.) > 1E-7:
        raise ValueError(
            'Piecewise Function requires 2 * nsegments parameters')

    n_seg = n_params / 2

    if n_seg == 1:
        return params[1] + params[0] * xval

    gradients = params[0:n_seg]
    turning_points = params[n_seg: -1]
    c_val = np.array([params[-1]])

    for iloc in range(1, n_seg):
        c_val = np.hstack([c_val, (c_val[iloc - 1] + gradients[iloc - 1] *
                           turning_points[iloc - 1]) - (gradients[iloc] *
                           turning_points[iloc - 1])])

    if xval <= turning_points[0]:
        return gradients[0] * xval + c_val[0]
    elif xval > turning_points[-1]:
        return gradients[-1] * xval + c_val[-1]
    else:
        select = np.nonzero(turning_points <= xval)[0][-1] + 1
    return gradients[select] * xval + c_val[select]


def polynomial(params, xval):
    """
    Returns the polynomial f(xval) where the order is defined by the
    number of params, i.e.
    yval = \\SUM_{i=1}^{Num Params} params[i] * (xval ** i - 1)
    """
    yval = np.zeros_like(xval)
    for iloc, param in enumerate(params):
        yval += (param * (xval ** float(iloc)))
    return yval


def exponential(params, xval):
    """
    Returns an exponential function
    """
    assert len(params) == 3
    return np.exp(params[0] + params[1] * xval) + params[2]


def build_filename(filename, filetype='png', resolution=300):
    """
    Uses the input properties to create the string of the filename
    :param str filename:
        Name of the file
    :param str filetype:
        Type of file
    :param int resolution:
        DPI resolution of the output figure
    """
    filevals = os.path.splitext(filename)
    if filevals[1]:
        filetype = filevals[1][1:]
    if not filetype:
        filetype = 'png'

    filename = filevals[0] + '.' + filetype

    if not resolution:
        resolution = 300
    return filename, filetype, resolution


def _save_image(filename, filetype='png', resolution=300):
    """
    If filename is specified, saves the image
    :param str filename:
        Name of the file
    :param str filetype:
        Type of file
    :param int resolution:
        DPI resolution of the output figure
    """
    if filename:
        filename, filetype, resolution = build_filename(filename,
                                                        filetype,
                                                        resolution)
        plt.savefig(filename, dpi=resolution, format=filetype,
                    bbox_inches="tight")
    else:
        pass
    return


def _save_image_tight(fig, lgd, filename, filetype='png', resolution=300):
    """
    If filename is specified, saves the image

    :param str filename:
        Name of the file where to save the figure
    :param str filetype:
        Type of file
    :param int resolution:
        DPI resolution of the output figure
    """
    if filename:
        filename, filetype, resolution = build_filename(filename,
                                                        filetype,
                                                        resolution)
        fig.savefig(filename, bbox_extra_artists=(lgd,),
                    bbox_inches="tight", dpi=resolution, format=filetype)
    else:
        pass
    return


AGEN_CODES = {
     "USPetal": {"name": "Brazilian catalogue - version 2014 ",
             "alias": "BSB",
             "country": "Brasil"
             },
     "GSS": {"name": "Dewey and Spence (1979) -from CERESIS catalogue ",
             "alias": "",
             "country": "Peru"
             },
     "G-R": {"name": "Gutenberg and Richter-Seismicity of the Earth - from CERESIS catalogue",
             "alias": "",
             "country": "Worldwide"
             },
     "HYPO": {"name": "Instituto Geofisico de los Andes (Hypo71)-from CERESIS catalogue",
             "alias": "",
             "country": "Colombia",
             },
     "PSA": {"name": "Instituto Nacional de Prevencion Sismica, INPRES",
             "alias": "",
             "country": "Argentina"
             },
     "ANSS": {"name": "Advanced National Seismic System, NEIC",
             "alias": "NEIC",
             "country": "U.S.A."
             },
     "LSP": {"name": "W.Lescano, J.Shikiya, P. Huaco - from CERESIS catalogue",
             "alias": "",
             "country": "all events are near or within Peru"
             },
     "NEI": {"name": "National Earthquake Information Center",
             "alias": "NEIC",
             "country": "U.S.A."
             },
     "ESB": {"name": "Observatório Sismológico da Universidade de Brasília",
             "alias": "BDF",
             "country": "Brazil"
             },
     "BCI": {"name": "Bureau Central International de Sismologie",
             "alias": "BCIS",
             "country": "France"
             },
     "JLL": {"name": "Unknown - from CERESIS catalogue",
             "alias": "",
             "country": "all the reported events are near or within Trinidad-Tobago"
             },
     "DEW": {"name": "Dewey (1972) - from CERESIS catalogue",
             "alias": "",
             "country": "Venezuela"
             },
     "SYK": {"name": "Sykes and Edwing (1965) - Caribbean",
             "alias": "",
             "country": "Caribbean"
             },
     "IPT": {"name": "Unknown - from CERESIS catalogue",
             "alias": "",
             "country": "all the reported events are near of within Brazil"
             },
     "IGA": {"name": "Instituto Geofisico de los Andes - from CERESIS catalogue",
             "alias": "",
             "country": "all events are near of within Colombia"
             },
     "IGF": {"name": "Instituto Astronomico e Geofísico- from CERESIS catalogue",
             "alias": "USP",
             "country": "Brazil"
             },
     "NAT": {"name": "Natal seismological WWNSS station - from CERESIS catalogue",
             "alias": "",
             "country": "Brazil"
             },
     "IGP": {"name": "Instituto Geofísico- from CERESIS catalogue",
             "alias": "",
             "country": "Peru"
             },
     "IGO": {"name": "Instituto Geofisico (Ocola info)- from CERESIS catalogue",
             "alias": "",
             "country": "Peru"
             },
     "JAK": {"name": "Unknown - from CERESIS catalogue",
             "alias": "",
             "country": "all the reported events are near or within Ecuador"
             },
     "OAE": {"name": "Observatorio Astronomico de Quito",
             "alias": "",
             "country": "Ecuador"
             },
     "RAM": {"name": "Ramirez (1975) - from CERESIS catalogue",
             "alias": "",
             "country": "Colombia"
             },
     "ESP": {"name": "Unknown - from CERESIS catalogue",
             "alias": "",
             "country": "all the reported events are near or within Colombia"
             },
     "EPN": {"name": "Escuela Politecnica Nacional, Quito - from CERESIS catalogue",
             "alias": "",
             "country": "Ecuador"
             },
     "EGR": {"name": "Egred (1968) - from CERESIS catalogue",
             "alias": "",
             "country": "Ecuador"
             },
     "URN": {"name": "Unknown - from CERESIS catalogue",
             "alias": "",
             "country": "all the reported events are near or within Brazil"
             },
     "EDR": {"name": "National Earthquake Information Service - from CERESIS catalogue",
             "alias": "NEIC, NEIS",
             "country": "U.S.A."
             },
     "IGE": {"name": "Instituto Geofisico de los Andes",
             "alias": "",
             "country": "Colombia"
             },
     "JBG": {"name": "Unknown - from CERESIS catalogue",
             "alias": "",
             "country": "all the reported events are near or within Bolivia"
             },
     "FIE": {"name": "Fiedler (1961-1972) - from CERESIS catalogue",
             "alias": "",
             "country": "Venezuela"
             },
     "OSS": {"name": "Unknown - from CERESIS catalogue",
             "alias": "",
             "country": "all the reported events are near or within Ecuador"
             },
     "LCO": {"name": "Unknown - from CERESIS catalogue",
             "alias": "",
             "isc_cod": "",
             "country": "all the reported events are near or within Peru"
             },
     "OSO": {"name": "Unknown - from CERESIS catalogue",
             "alias": "",
             "country": "all the reported events are near or within Ecuador"
             },
     "ABE": {"name": "Unknown, may be ABE studies - from CERESIS catalogue",
             "alias": "",
             "country": "all the reported events are near or within Ecuador"
             },
     "LSG": {"name": "Unknown - from CERESIS catalogue",
             "alias": "",
             "country": "all the reported events are near or within Ecuador"
             },
     "GC": {"name": "Unknown - from CERESIS catalogue",
             "alias": "",
             "country": ""
             },
     "GRA": {"name": "Grases (1975) - from CERESIS catalogue",
             "alias": "",
             "country": "Venezuela"
             },
     "WCA": {"name": "Woodward-Clyde(1969) - from CERESIS catalogue",
             "alias": "",
             "country": "Worldwide"
             },
     "CEN": {"name": "Centeno-Grau (1969) - from CERESIS catalogue",
             "alias": "",
             "country": "Venezuela"
             },
     "IAC": {"name": "Unknown - from CERESIS catalogue",
             "alias": "",
             "country": ""
             },
     "IGH": {"name": "Instituto Geofisico de Peru - Huaco info - from CERESIS catalogue",
             "alias": "",
             "country": "Peru"
             },
     "DNA": {"name": "Unknown, may be DNAG (Engdahl&Rinehart,1991) - from CERESIS catalogue",
             "alias": "",
             "country": ""
             },
     "USE": {"name": "Unknown - from CERESIS catalogue",
             "alias": "",
             "country": ""
             },
     "USG": {"name" : "United States Geological Survey-from CERESIS catalogue",
             "alias": "USGS",
             "country": "U.S.A."
             },
# TODO - organize the previous CERESIS Agencies/Catalogues
     "AAA": {"name" : "Alma-ata",
             "alias": "",
             "country": "Kazakhstan"
             },
     "AAB": {"name" : "Alma-ata 2",
             "alias": "",
             "country": "Kazakhstan"
             },
     "AAC": {"name" : "Unknown Historical Agency",
             "alias": "",
             "country": ""
             },
     "AAE": {"name" : "University of Addis Ababa",
             "alias": "",
             "country": "Ethiopia"
             },
     "AAM": {"name" : "University of Michigan",
             "alias": "",
             "country": "U.S.A."
             },
     "ABA": {"name" : "Alger-bouzareah",
             "alias": "ALG",
             "country": "Algeria"
             },
     "ABE": {"name": "Abe (1981, 1984)-Abe&Noguchi (1983a,b)",
             "alias": "",
             "country": "Worldwide"
             },
     "ACI": {"name" : "Universita di Calabria",
             "alias": "",
             "country": "Italy"
             },
     "ADE": {"name" : "Primary Industries and Resources SA",
             "alias": "",
             "country": "Australia"
             },
     "ADH": {"name" : "Observatorio Afonso Chaves",
             "alias": "",
             "country": "Portugal"
             },
     "AEIC": {"name" : "Alaska Earthquake Information Center",
             "alias": "",
             "country": "U.S.A."
             },
     "AFAR": {"name" : "The Afar Depression: Interpretation of the 1960-2000 Earthquakes",
             "alias": "",
             "country": "Israel"
             },
     "AFI": {"name" : "Apia Observatory",
             "alias": "",
             "country": "Western Samoa"
             },
     "AGS": {"name" : "Alaska Seismic Project",
             "alias": "",
             "country": "U.S.A."
             },
     "ALG": {"name" : "Algiers University",
             "alias": "",
             "country": "Algeria"
             },
     "ALI": {"name" : "Observatorio Sismologico Vicente Inglada",
             "alias": "",
             "country": "Spain"
             },
     "ALM": {"name" : "Instituto Geografico y Catastral de Almeria",
             "alias": "",
             "country": "Spain"
             },
     "ANCORP": {"name" : "Andean Continental Research Project",
             "alias": "",
             "country": "Germany"
             },
     "ANF": {"name" : "USArray Array Network Facility",
             "alias": "",
             "country": "U.S.A."
             },
     "ANT": {"name" : "Antofagasta",
             "alias": "GUC",
             "country": "Chile"
             },
     "ANUBIS": {"name" : "Antarctic Network of Broadband Seismometers",
             "alias": "",
             "country": "U.S.A."
             },
     "APA": {"name" : "Apatity",
             "alias": "",
             "country": "Russia"
             },
     "API": {"name" : "Apia Observatory",
             "alias": "",
             "country": "Western Samoa"
             },
     "APT": {"name" : "University of Connecticut",
             "alias": "",
             "country": "U.S.A."
             },
     "AQU": {"name" : "L'Aquila",
             "alias": "",
             "country": "Italy"
             },
     "ARA0": {"name" : "Arcess Array",
             "alias": "",
             "country": "Norway"
             },
     "ARE": {"name" : "Instituto Geofisico del Peru",
             "alias": "",
             "country": "Peru"
             },
     "ARO": {"name" : "Observatoire Geophysique d'Arta",
             "alias": "",
             "country": "Djibouti"
             },
     "ASIES": {"name" : "Institute of Earth Sciences -  Academia Sinica",
             "alias": "",
             "country": "Chinese Taipei"
             },
     "ASL": {"name" : "Albuquerque Seismological Laboratory",
             "alias": "",
             "country": "U.S.A."
             },
     "ASM": {"name" : "University of Asmara",
             "alias": "",
             "country": "Eritrea"
             },
     "ASRS": {"name" : "Altai-Sayan Seismological Centre -  GS SB RAS",
             "alias": "",
             "country": "Russia"
             },
     "ATA": {"name" : "The Earthquake Research Center Ataturk University",
             "alias": "",
             "country": "Turkey"
             },
     "ATH": {"name" : "National Observatory of Athens",
             "alias": "",
             "country": "Greece"
             },
     "AUST": {"name" : "Geoscience Australia",
             "alias": "",
             "country": "Australia"
             },
     "AVE": {"name" : "Averroes",
             "alias": "",
             "country": "Morocco"
             },
     "AWI": {"name" : "Alfred Wegener Institute for Polar and Marine Research",
             "alias": "",
             "country": "Germany"
             },
     "AZER": {"name" : "Republic Center of Seismic Survey",
             "alias": "",
             "country": "Azerbaijan"
             },
     "AZO": {"name" : "Centro de Informao e Vigilancia Sismovulcanica dos Azores",
             "alias": "",
             "country": "Portugal"
             },
     "BAA": {"name" : "Servicio Meteorologico Nacional",
             "alias": "",
             "country": "Argentina"
             },
     "BAK": {"name" : "Unknown Historical Agency",
             "alias": "",
             "country": ""
             },
     "BANJO": {"name" : "Broadband ANdean JOint Experiment",
             "alias": "",
             "country": "U.S.A."
             },
     "BASV": {"name" : "British Antarctic Survey",
             "alias": "",
             "country": "United Kingdom"
             },
     "BCIS": {"name" : "Bureau Central International de Sismologie",
             "alias": "",
             "country": "France"
             },
     "BCI": {"name": "Bureau Central International de Sismologie - in CERESIS catalogue ",
             "alias": "BCIS",
             "country": "France"
             },
     "B&D": {"name" : "Bath and Duda (1979).",
             "alias": "",
             "country": "Worldwide"
             },
     "BDF": {"name" : "Observatorio Sismologico da Universidade de Brasilia",
             "alias": "",
             "country": "Brazil"
             },
     "BELR": {"name" : "Centre of Geophysical Monitoring",
             "alias": "",
             "country": "Belarus"
             },
     "BEO": {"name" : "Seismological Survey of Serbia",
             "alias": "",
             "country": "Serbia"
             },
     "BER": {"name" : "University of Bergen",
             "alias": "",
             "country": "Norway"
             },
     "BERK": {"name" : "Berkheimer H",
             "alias": "",
             "country": "Germany"
             },
     "BGLD": {"name" : "Geophysikalisches Observatorium der Ludwig-Maximilians Universitat",
             "alias": "",
             "country": "Germany"
             },
     "BGR": {"name" : "Bundesanstalt for Geowissenschaften und Rohstoffe",
             "alias": "",
             "country": "Germany"
             },
     "BGS": {"name" : "British Geological Survey",
             "alias": "",
             "country": "United Kingdom"
             },
     "BHP": {"name" : "Unknown Historical Agency",
             "alias": "",
             "country": ""
             },
     "BHUJ": {"name" : "Bhuj Aftershock Study",
             "alias": "",
             "country": "U.S.A."
             },
     "BHUJ2": {"name" : "Study of Aftershocks of the Bhuj Earthquake by Japanese Research Team",
             "alias": "",
             "country": "Japan"
             },
     "BIAK": {"name" : "Biak earthquake aftershocks (17-Feb-1996)",
             "alias": "",
             "country": "U.S.A."
             },
     "BJI": {"name" : "China Earthquake Networks Center",
             "alias": "",
             "country": "China"
             },
     "BJT": {"name" : "Baijiatuan",
             "alias": "",
             "country": "China"
             },
     "BKK": {"name" : "Thai Meteorological Department",
             "alias": "",
             "country": "Thailand"
             },
     "BLA": {"name" : "Virginia Tech",
             "alias": "",
             "country": "U.S.A."
             },
     "BNG": {"name" : "Observatoire ORSTOM de Bangui",
             "alias": "",
             "country": "Central African Republic"
             },
     "BNS": {"name" : "Erdbebenstation  Geologisches Institut der Universitat  Kol",
             "alias": "",
             "country": "Germany"
             },
     "BOG": {"name" : "Universidad Javeriana",
             "alias": "",
             "country": "Colombia"
             },
     "BOM": {"name" : "Unknown Historical Agency",
             "alias": "",
             "country": ""
             },
     "BOU": {"name" : "University of Colorado at Boulder",
             "alias": "",
             "country": "U.S.A."
             },
     "BOZ": {"name" : "Unknown Historical Agency",
             "alias": "",
             "country": ""
             },
     "BRA": {"name" : "Geophysical Institute -  Slovak Academy of Sciences",
             "alias": "",
             "country": "Slovakia"
             },
     "BRG": {"name" : "Seismological Observatory Berggieehebel - TU Bergakademie Freiberg",
             "alias": "",
             "country": "Germany"
             },
     "BRK": {"name" : "Berkeley Seismological Laboratory",
             "alias": "",
             "country": "U.S.A."
             },
     "BRS": {"name" : "Brisbane Seismograph Station",
             "alias": "",
             "country": "Australia"
             },
     "BSB": {"name" : "Brazilian Seismic Bulletin",
             "alias": "",
             "country": "Brazil"
             },
     "BSE": {"name" : "Boise State University",
             "alias": "",
             "country": "U.S.A."
             },
     "BUC": {"name" : "National Institute for Earth Physics",
             "alias": "",
             "country": "Romania"
             },
     "BUD": {"name" : "Geodetic and Geophysical Research Institute",
             "alias": "",
             "country": "Hungary"
             },
     "BUG": {"name" : "Institute of Geology -  Mineralogy & Geophysics",
             "alias": "",
             "country": "Germany"
             },
     "BUL": {"name" : "Goetz Observatory",
             "alias": "",
             "country": "Zimbabwe"
             },
     "BUT": {"name" : "Montana Bureau of Mines and Geology",
             "alias": "",
             "country": "U.S.A."
             },
     "BYKL": {"name" : "Baykal Regional Seismological Centre -  GS SB RAS",
             "alias": "",
             "country": "Russia"
             },
     "CADCG": {"name" : "Central America Data Centre",
             "alias": "CASC",
             "country": "Costa Rica"
             },
     "CAN": {"name" : "Australian National University",
             "alias": "",
             "country": "Australia"
             },
     "CANSK": {"name" : "Canadian and Scandinavian Networks",
             "alias": "HFS",
             "country": "Sweden"
             },
     "CAR": {"name" : "Instituto Sismologico de Caracas",
             "alias": "",
             "country": "Venezuela"
             },
     "CASC": {"name" : "Central American Seismic Center",
             "alias": "",
             "country": "Costa Rica"
             },
     "CDWR": {"name" : "California Department of Water Resources",
             "alias": "",
             "country": "U.S.A."
             },
     "CENT": {"name" : "Centennial Earthquake Catalog",
             "alias": "",
             "country": "U.S.A."
             },
     "CERI": {"name" : "Center for Earthquake Research and Information",
             "alias": "",
             "country": "U.S.A."
             },
     "CGS": {"name" : "Coast and Geodetic Survey of the United States",
             "alias": "NEIS",
             "country": "U.S.A."
             },
     "CHM": {"name" : "Unknown Historical Agency",
             "alias": "",
             "country": ""
             },
     "CHU": {"name" : "Unknown Historical Agency",
             "alias": "",
             "country": ""
             },
     "CIG": {"name" : "Servicio Geologico Nacional de El Salvador",
             "alias": "",
             "country": "El Salvador"
             },
     "CINCA": {"name" : "Crustal Investigations Off- and On-shore Nazca - Central Andes",
             "alias": "",
             "country": "Germany"
             },
     "CLL": {"name" : "Geophysikalisches Observatorium Collm",
             "alias": "",
             "country": "Germany"
             },
     "CMWS": {"name" : "Laboratory of Seismic Monitoring of Caucasus Mineral Water Region - GSRAS",
             "alias": "",
             "country": "Russia"
             },
     "CNG": {"name" : "Seismographic Station Changalane",
             "alias": "",
             "country": "Mozambique"
             },
     "CNH": {"name" : "Changchun",
             "alias": "BJI",
             "country": "China"
             },
     "CNRM": {"name" : "Centre National de Recherche",
             "alias": "",
             "country": "Morocco"
             },
     "COI": {"name" : "Unknown Historical Agency",
             "alias": "",
             "country": ""
             },
     "COM": {"name" : "Comitan",
             "alias": "TAC",
             "country": "Mexico"
             },
     "CON": {"name" : "Unknown Historical Agency",
             "alias": "",
             "country": ""
             },
     "COP": {"name" : "Unknown Historical Agency",
             "alias": "",
             "country": ""
             },
     "COR": {"name" : "COAS Physical Oceanography",
             "alias": "",
             "country": "U.S.A."
             },
     "COSMOS": {"name" : "Consortium of Organizations for Strong Motion Observations",
             "alias": "",
             "country": "U.S.A."
             },
     "CRA": {"name" : "Unknown Historical Agency",
             "alias": "",
             "country": ""
             },
     "CRAAG": {"name" : "Centre de Recherche en Astronomie -  Astrophysique et Geophysique",
             "alias": "",
             "country": "Algeria"
             },
     "CRT": {"name" : "Cartuja Seismological Station",
             "alias": "MDD",
             "country": "Spain"
             },
     "CSC": {"name" : "University of South Carolina",
             "alias": "",
             "country": "U.S.A."
             },
     "CSEM": {"name" : "Centre Sismologique Euro-Mediterranen (CSEM/EMSC)",
             "alias": "",
             "country": "France"
             },
     "DASA": {"name" : "Defense Atomic Support Agency",
             "alias": "DOE",
             "country": "U.S.A."
             },
     "DBN": {"name" : "Koninklijk Nederlands Meteorologisch Instituut",
             "alias": "",
             "country": "Netherlands"
             },
     "DDA": {"name" : "Disaster and Emergency Management Presidency",
             "alias": "",
             "country": "Turkey"
             },
     "DHMR": {"name" : "Yemen National Seismological Center",
             "alias": "",
             "country": "Yemen"
             },
     "DIAS": {"name" : "Dublin Institute for Advanced Studies",
             "alias": "",
             "country": "Ireland"
             },
     "DJA": {"name" : "Badan Meteorologi -  Klimatologi dan Geofisika",
             "alias": "",
             "country": "Indonesia"
             },
     "DMN": {"name" : "Department of Mines and Geology -  Ministry of Industry of Nepal",
             "alias": "",
             "country": "Nepal"
             },
     "DNK": {"name" : "Geological Survey of Denmark and Greenland",
             "alias": "",
             "country": "Denmark"
             },
     "DOE": {"name" : "Department of Energy",
             "alias": "",
             "country": "U.S.A."
             },
     "DRS": {"name" : "Dagestan Branch  Geophysical Survey -  Russian Academy of Sciences",
             "alias": "",
             "country": "Russia"
             },
     "DSN": {"name" : "Dubai Seismic Network",
             "alias": "",
             "country": "United Arab Emirates"
             },
     "DUSS": {"name" : "Damascus University -  Syria",
             "alias": "",
             "country": "Syria"
             },
     "EAF": {"name" : "East African Network",
             "alias": "",
             "country": ""
             },
     "EAGLE": {"name" : "Ethiopia-Afar Geoscientific Lithospheric Experiment",
             "alias": "",
             "country": ""
             },
     "EBM": {"name" : "Esen Boulak",
             "alias": "",
             "country": "Mongolia"
             },
     "EBR": {"name" : "Observatori de l'Ebre",
             "alias": "",
             "country": "Spain"
             },
     "EBSE": {"name" : "Ethiopian Broadband Seismic Experiment",
             "alias": "",
             "country": ""
             },
     "ECX": {"name" : "Red Sismica del Noroeste de Mexico (RESOM)",
             "alias": "",
             "country": "Mexico"
             },
     "EFATE": {"name" : "OBS Experiment near Efate - Vanuatu",
             "alias": "",
             "country": "U.S.A."
             },
     "EHB": {"name" : "Engdahl -  van der Hilst and Buland",
             "alias": "",
             "country": "U.S.A."
             },
     "EIDC": {"name" : "Experimental (GSETT3) International Data Center",
             "alias": "",
             "country": "U.S.A."
             },
     "EKA": {"name" : "Eskdalemuir Array Station",
             "alias": "",
             "country": "United Kingdom"
             },
     "ENT": {"name" : "Geological Survey and Mines Department",
             "alias": "",
             "country": "Uganda"
             },
     "EPSI": {"name" : "Reference events computed by the ISC for EPSI project",
             "alias": "",
             "country": "United Kingdom"
             },
     "ERDA": {"name" : "Energy Research and Development Administration",
             "alias": "DOE",
             "country": "U.S.A."
             },
     "ERI": {"name" : "Earthquake Research Institute -  University of Tokyo",
             "alias": "",
             "country": "Japan"
             },
     "ESB": {"name": "Observatório Sismológico da Universidade de Brasília",
             "alias": "BDF",
             "country": "Brazil"
             },
     "ESK": {"name" : "Unknown Historical Agency",
             "alias": "",
             "country": ""
             },
     "ESLA": {"name" : "Centro Sismologico de Sonseca",
             "alias": "",
             "country": "Spain"
             },
     "EST": {"name" : "Geological Survey of Estonia",
             "alias": "",
             "country": "Estonia"
             },
     "EUO": {"name" : "Department of Geological Sciences -  University of Oregon",
             "alias": "",
             "country": "U.S.A."
             },
     "FBR": {"name" : "Fabra Observatory",
             "alias": "",
             "country": "Spain"
             },
     "FDF": {"name" : "Fort de France",
             "alias": "",
             "country": "Martinique"
             },
     "FIA0": {"name" : "Finessa Array",
             "alias": "",
             "country": "Finland"
             },
     "FKK": {"name" : "Unknown Historical Agency",
             "alias": "",
             "country": ""
             },
     "FOR": {"name" : "Unknown Historical Agency",
             "alias": "",
             "country": ""
             },
     "FRU": {"name" : "Unknown Historical Agency",
             "alias": "",
             "country": ""
             },
     "FUNV": {"name" : "Fundacion Venezolana de Investigaciones Sismologicas",
             "alias": "",
             "country": "Venezuela"
             },
     "FUR": {"name" : "Geophysikalisches Observatorium der Universitat Munchen",
             "alias": "",
             "country": "Germany"
             },
     "GBA": {"name" : "Bhaba Atomic Research Centre",
             "alias": "",
             "country": "India"
             },
     "GBZT": {"name" : "Marmara Research Center",
             "alias": "",
             "country": "Turkey"
             },
     "GCG": {"name" : "INSIVUMEH",
             "alias": "",
             "country": "Guatemala"
             },
     "GCMT": {"name" : "The Global CMT Project",
             "alias": "",
             "country": "U.S.A."
             },
     "GDNRW": {"name" : "Geologischer Dienst Nordrhein-Westfalen",
             "alias": "",
             "country": "Germany"
             },
     "GEC2": {"name" : "Geress Array",
             "alias": "",
             "country": "Germany"
             },
     "GEN": {"name" : "Dipartimento per lo Studio del Territorio e delle sue Risorse (RSNI)",
             "alias": "",
             "country": "Italy"
             },
     "GFZ": {"name" : "Helmholtz Centre Potsdam GFZ German Research Centre For Geosciences",
             "alias": "",
             "country": "Germany"
             },
     "GII": {"name" : "The Geophysical Institute of Israel",
             "alias": "",
             "country": "Israel"
             },
     "GLD": {"name" : "Golden",
             "alias": "",
             "country": "U.S.A."
             },
     "GM": {"name" : "U.S. Geological Survey",
             "alias": "",
             "country": "U.S.A."
             },
     "GOL": {"name" : "Colorado School of Mines",
             "alias": "",
             "country": "U.S.A."
             },
     "GOM": {"name" : "Observatoire Volcanologique de Goma",
             "alias": "",
             "country": "Democratic Republic of the Congo"
             },
     "GRA": {"name" : "Unknown Historical Agency",
             "alias": "",
             "country": ""
             },
     "GRA1": {"name" : "Grafenberg Array",
             "alias": "",
             "country": "Germany"
             },
     "GRAL": {"name" : "National Council for Scientific Research",
             "alias": "",
             "country": "Lebanon"
             },
     "GRO": {"name" : "Unknown Historical Agency",
             "alias": "",
             "country": ""
             },
     "GS": {"name" : "U.S. Geological Survey",
             "alias": "USGS",
             "country": "U.S.A."
             },
     "GSAST": {"name" : "Geophysial Survey of the Academy of Sciences of the Republic of Tajikistan",
             "alias": "",
             "country": "Tajikistan"
             },
     "GSDM": {"name" : "Geological Survey Department Malawi",
             "alias": "",
             "country": "Malawi"
             },
     "GSET2": {"name" : "Group of Scientific Experts Second Technical Test 1991 -  April 22 - June 2",
             "alias": "",
             "country": ""
             },
    "GTFE": {"name" : "German Task Force for Earthquakes",
             "alias": "",
             "country": "Germany"
             },
     "GUC": {"name" : "Departamento de Geofisica -  Universidad de Chile",
             "alias": "",
             "country": "Chile"
             },
     "GUTE": {"name" : "Gutenberg and Richter -  Seismicity of the Earth",
             "alias": "",
             "country": "Worldwide"
             },
     "GUV": {"name" : "CVG Electrificacion del Caroni",
             "alias": "",
             "country": "Venezuela"
             },
     "HAM": {"name" : "Unknown Historical Agency",
             "alias": "",
             "country": ""
             },
     "HAN": {"name" : "Hannover",
             "alias": "SZGRF",
             "country": "Germany"
             },
     "HDC": {"name" : "Observatorio Vulcanologico y Sismologico de Costa Rica",
             "alias": "",
             "country": "Costa Rica"
             },
     "HEL": {"name" : "Institute of Seismology -  University of Helsinki",
             "alias": "",
             "country": "Finland"
             },
     "HFS": {"name" : "Hagfors Observatory",
             "alias": "",
             "country": "Sweden"
             },
     "HFS1": {"name" : "Hagfors Observatory",
             "alias": "",
             "country": "Sweden"
             },
     "HFS2": {"name" : "Hagfors Observatory",
             "alias": "",
             "country": "Sweden"
             },
     "HIMNT": {"name" : "Himalayan Nepal Tibet Experiment",
             "alias": "",
             "country": "U.S.A."
             },
     "HKC": {"name" : "Hong Kong Observatory",
             "alias": "",
             "country": "Hong Kong"
             },
     "HLUG": {"name" : "Hessisches Landesamt fur Umwelt und Geologie",
             "alias": "",
             "country": "Germany"
             },
     "HLW": {"name" : "National Research Institute of Astronomy and Geophysics",
             "alias": "",
             "country": "Egypt"
             },
     "HNR": {"name" : "Ministry of Mines -  Energy and Rural Electrification",
             "alias": "",
             "country": "Solomon Islands"
             },
     "HOKK_DSZ": {"name" : "Hokkaido Double Seismic Zone",
             "alias": "",
             "country": ""
             },
     "HON": {"name" : "Pacific Tsunami Warning Center - NOAA",
             "alias": "",
             "country": "U.S.A."
             },
     "HROE": {"name" : "Geophysikalisches Observatorium - Hohe Rhvn-Fladungen",
             "alias": "",
             "country": "Germany"
             },
     "HRV": {"name" : "Unknown Historical Agency",
             "alias": "",
             "country": ""
             },
     "HRVD": {"name" : "Harvard University",
             "alias": "",
             "country": "U.S.A."
             },
     "HRVD_LR": {"name" : "Department of Geological Sciences -  Harvard University",
             "alias": "",
             "country": "U.S.A."
             },
     "HVO": {"name" : "Hawaiian Volcano Observatory",
             "alias": "",
             "country": "U.S.A."
             },
     "HYB": {"name" : "National Geophysical Research Institute",
             "alias": "",
             "country": "India"
             },
     "HYD": {"name" : "National Geophysical Research Institute",
             "alias": "NDI",
             "country": "India"
             },
     "IAG": {"name": "Instituto de Astronomia Geofísica e Ciências Atmosféricas",
             "alias": "USP",
             "country": "Brazil"
             },
     "IAGE": {"name" : "Instituto Andaluz de Geofisica",
             "alias": "",
             "country": "Spain"
             },
     "IASBS": {"name" : "Institute for Advanced Studies in Basic Sciences",
             "alias": "",
             "country": "Iran"
             },
     "IASPEI": {"name" : "IASPEI Working Group on Reference Events",
             "alias": "",
             "country": "U.S.A."
             },
     "IBER": {"name" : "Institute of Earth Sciences Jaume Almera - CSIC",
             "alias": "",
             "country": "Spain"
             },
     "ICE": {"name" : "Instituto Costarricense de Electricidad",
             "alias": "",
             "country": "Costa Rica"
             },
     "IDC": {"name" : "International Data Centre -  CTBTO",
             "alias": "",
             "country": "Austria"
             },
     "IDG": {"name" : "Institute of Dynamics of Geosphere -  Russian Academy of Sciences",
             "alias": "",
             "country": "Russia"
             },
     "IEC": {"name" : "Institute of the Earth Crust -  SB RAS",
             "alias": "",
             "country": "Russia"
             },
     "IEPN": {"name" : "Institute of Environmental Problems of the North -  Russian Academy of Sciences",
             "alias": "",
             "country": "Russia"
             },
     "IFREE": {"name" : "Institute For Research on Earth Evolution",
             "alias": "",
             "country": "Japan"
             },
     "IGIL": {"name" : "Instituto Geofisico do Infante Dom Luiz",
             "alias": "",
             "country": "Portugal"
             },
     "IGQ": {"name" : "Servicio Nacional de Sismologia y Vulcanologia",
             "alias": "",
             "country": "Ecuador"
             },
     "IGS": {"name" : "Institute of Geological Sciences",
             "alias": "",
             "country": "United Kingdom"
             },
     "INC": {"name" : "Unknown Historical Agency",
             "alias": "",
             "country": ""
             },
     "INDEPTH3": {"name" : "International Deep Profiling of Tibet and the Himalayas",
             "alias": "",
             "country": "U.S.A."
             },
     "INDR": {"name" : "Inst. Nacional de Recursos Hidraulicos",
             "alias": "",
             "country": "Dominican Republic"
             },
     "INET": {"name" : "Instituto Nicaraguense de Estudios Territoriales",
             "alias": "",
             "country": "Nicaragua"
             },
     "INMG": {"name" : "Instituto Portugues do Mar e da Atmosfera -  I.P.",
             "alias": "",
             "country": "Portugal"
             },
     "INTV": {"name" : "Instituto de Tecnologia Venezolana para el Petroleo",
             "alias": "",
             "country": "Venezuela"
             },
     "INY": {"name" : "Cornell university (INSTOC)",
             "alias": "",
             "country": "U.S.A."
             },
     "IPEC": {"name" : "The Institute of Physics of the Earth (IPEC)",
             "alias": "",
             "country": "Czech Republic"
             },
     "IPER": {"name" : "Institute of Physics of the Earth -  Academy of Sciences  Moscow",
             "alias": "",
             "country": "Russia"
             },
     "IGEPN": {"name" : "The Geophysical Institute, at Escuela Politécnica Nacional",
             "alias": "",
             "country": "Ecuador"
             },
     "IPGP": {"name" : "Institut de Physique du Globe de Paris",
             "alias": "",
             "country": "France"
             },
     "IPRG": {"name" : "Institute for Petroleum Research and Geophysics",
             "alias": "",
             "country": "Israel"
             },
     "IRIS": {"name" : "IRIS Data Management Center",
             "alias": "",
             "country": "U.S.A."
             },
     "IRK": {"name" : "Unknown Historical Agency",
             "alias": "",
             "country": ""
             },
     "IRSM": {"name" : "Institute of Rock Structure and Mechanics",
             "alias": "",
             "country": "Czech Republic"
             },
     "ISC": {"name" : "International Seismological Centre",
             "alias": "",
             "country": "United Kingdom"
             },
     "ISC1": {"name" : "International Seismological Centre",
             "alias": "ISC",
             "country": ""
             },
     "ISCJB": {"name" : "International Seismological Centre",
             "alias": "",
             "country": "United Kingdom"
             },
     "ISC-GEM": {"name" : "ISC-GEM catalogue - ver.2014",
             "alias": "",
             "country": "Worldwide"
             },
     "ISK": {"name" : "Kandilli Observatory and Research Institute",
             "alias": "",
             "country": "Turkey"
             },
     "ISN": {"name" : "Iraqi Meteorological and Seismology Organisation",
             "alias": "",
             "country": "Iraq"
             },
     "ISS": {"name" : "International Seismological Summary",
             "alias": "",
             "country": "United Kingdom"
             },
     "IST": {"name" : "Institute of Physics of the Earth - Technical University of Istanbul",
             "alias": "",
             "country": "Turkey"
             },
     "JEN": {"name" : "Geodynamisches Observatorium Moxa",
             "alias": "",
             "country": "Germany"
             },
     "JER": {"name" : "Seismological Laboratory -  Geological Survey of Israel",
             "alias": "",
             "country": "Israel"
             },
     "JMA": {"name" : "Japan Meteorological Agency",
             "alias": "",
             "country": "Japan"
             },
     "JOH": {"name" : "Bernard Price Institute of Geophysics",
             "alias": "",
             "country": "South Africa"
             },
     "JSA": {"name" : "Jesuit Society of America",
             "alias": "SLM",
             "country": "U.S.A."
             },
     "JSN": {"name" : "Jamaica Seismic Network",
             "alias": "",
             "country": "Jamaica"
             },
     "JSO": {"name" : "Jordan Seismological Observatory",
             "alias": "",
             "country": "Jordan"
             },
     "KAAPVAAL": {"name" : "Kaapvaal Craton Seismic Experiment",
             "alias": "",
             "country": "U.S.A."
             },
     "KAF": {"name" : "Kangasniemi Station",
             "alias": "",
             "country": "Finland"
             },
     "KBC": {"name" : "Institut de Recherches Geologiques et Minires",
             "alias": "",
             "country": "Cameroon"
             },
     "KBL": {"name" : "Afghanistan Seismological Observatory",
             "alias": "",
             "country": "Afghanistan"
             },
     "KEW": {"name" : "Kew Observatory",
             "alias": "",
             "country": "United Kingdom"
             },
     "KHC": {"name" : "Geofysikalni Ustav -  Ceske Akademie Ved",
             "alias": "",
             "country": "Czech Republic"
             },
     "KHO": {"name" : "Khorog",
             "alias": "MOS",
             "country": "Tajikistan"
             },
     "KIR": {"name" : "Kiruna",
             "alias": "UPP",
             "country": "Sweden"
             },
     "KISR": {"name" : "Kuwait Institute for Scientific Research",
             "alias": "",
             "country": "Kuwait"
             },
     "KLM": {"name" : "Malaysian Meteorological Service",
             "alias": "",
             "country": "Malaysia"
             },
     "KMA": {"name" : "Korea Meteorological Administration",
             "alias": "",
             "country": "Republic of Korea"
             },
     "KNET": {"name" : "Kyrgyz Seismic Network",
             "alias": "",
             "country": "Kyrgyzstan"
             },
     "KOB": {"name" : "Unknown Historical Agency",
             "alias": "",
             "country": ""
             },
     "KOC": {"name" : "Unknown Historical Agency",
             "alias": "",
             "country": ""
             },
     "KOLA": {"name" : "Kola Regional Seismic Centre -  GS RAS",
             "alias": "",
             "country": "Russia"
             },
     "KRAR": {"name" : "Krasnoyarsk Scientific Research Inst. of Geology and Mineral Resources -  Russia",
             "alias": "",
             "country": "Russia"
             },
     "KRISP": {"name" : "Kenya Rift International Seismic Project",
             "alias": "",
             "country": "Germany"
             },
     "KRL": {"name" : "Geodetisches Institut der Universitat Karlsruhe",
             "alias": "",
             "country": "Germany"
             },
     "KRNET": {"name" : "Institute of Seismology -  Academy of Sciences of Kyrgyz Republic",
             "alias": "",
             "country": "Kyrgyzstan"
             },
     "KRSC": {"name" : "Kamchatkan Experimental and Methodical Seismological Department -  GS RAS",
             "alias": "",
             "country": "Russia"
             },
     "KSA": {"name" : "Observatoire de Ksara",
             "alias": "",
             "country": "Lebanon"
             },
     "KUK": {"name" : "Geological Survey Department of Ghana",
             "alias": "",
             "country": "Ghana"
             },
     "LAO": {"name" : "Large Aperture Seismic Array",
             "alias": "",
             "country": "U.S.A."
             },
     "LDG": {"name" : "Laboratoire de Detection et de Geophysique/CEA",
             "alias": "",
             "country": "France"
             },
     "LDN": {"name" : "University of Western Ontario",
             "alias": "",
             "country": "Canada"
             },
     "LDO": {"name" : "Lamont-Doherty Earth Observatory",
             "alias": "",
             "country": "U.S.A."
             },
     "LED": {"name" : "Landeserdbebendienst Baden-Warttemberg",
             "alias": "LEDBW",
             "country": "Germany"
             },
     "LEDBW": {"name" : "Landeserdbebendienst Baden-Warttemberg",
             "alias": "",
             "country": "Germany"
             },
     "LEM": {"name" : "Lembang Station",
             "alias": "DJA",
             "country": "Indonesia"
             },
     "LER": {"name" : "Besucherbergwerk Binweide Station",
             "alias": "",
             "country": "Germany"
             },
     "LIB": {"name" : "Tripoli",
             "alias": "",
             "country": "Libya"
             },
     "LIC": {"name" : "Station Geophysique de Lamto",
             "alias": "",
             "country": "Ivory Coast"
             },
     "LIM": {"name" : "Lima",
             "alias": "ARE",
             "country": "Peru"
             },
     "LIS": {"name" : "Instituto de Meteorologia",
             "alias": "",
             "country": "Portugal"
             },
     "LIT": {"name" : "Geological Survey of Lithuania",
             "alias": "",
             "country": "Lithuania"
             },
     "LJU": {"name" : "Environmental Agency of the Republic of Slovenia",
             "alias": "",
             "country": "Slovenia"
             },
     "LPA": {"name" : "Universidad Nacional de La Plata",
             "alias": "",
             "country": "Argentina"
             },
     "LPB": {"name" : "Unknown Historical Agency",
             "alias": "",
             "country": ""
             },
     "LPZ": {"name" : "Observatorio San Calixto",
             "alias": "",
             "country": "Bolivia"
             },
     "LRA": {"name" : "Unknown Historical Agency",
             "alias": "",
             "country": ""
             },
     "LRSM": {"name" : "Long Range Seismic Measurements Project",
             "alias": "",
             "country": ""
             },
     "LSZ": {"name" : "Geological Survey Department of Zambia",
             "alias": "",
             "country": "Zambia"
             },
     "LTX": {"name" : "Lajitas Seismic Array",
             "alias": "",
             "country": "U.S.A."
             },
     "LVSN": {"name" : "Latvian Seismic Network",
             "alias": "",
             "country": "Latvia"
             },
     "LVV": {"name" : "Department of Seismic Activity of Carpathian area (Lviv)",
             "alias": "MOS",
             "country": "Ukraine"
             },
     "LWI": {"name" : "Centre de Geophysique du Zaire",
             "alias": "",
             "country": "Democratic Republic of the Congo"
             },
     "MAL": {"name" : "Malaga",
             "alias": "MDD",
             "country": "Spain"
             },
     "MAN": {"name" : "Philippine Institute of Volcanology and Seismology",
             "alias": "",
             "country": "Philippines"
             },
     "MASS": {"name" : "Marcelo Assumpcao",
             "alias": "",
             "country": "Brazil"
             },
     "MAT": {"name" : "The Matsushiro Seismological Observatory",
             "alias": "",
             "country": "Japan"
             },
     "MCO": {"name" : "Macao Meteorological and Geophysical Bureau - Macao",
             "alias": "",
             "country": " China"
             },
     "MDD": {"name" : "Instituto Geografico Nacional",
             "alias": "",
             "country": "Spain"
             },
     "MED_RCMT": {"name" : "MedNet Regional Centroid - Moment Tensors",
             "alias": "",
             "country": "Italy"
             },
     "MELT": {"name" : "Mantle Electromagnetic and Tomography Experiment",
             "alias": "",
             "country": "U.S.A."
             },
     "MER": {"name" : "Merida",
             "alias": "TAC",
             "country": "Mexico"
             },
     "MERI": {"name" : "Maharashta Engineering Research Institute",
             "alias": "",
             "country": "India"
             },
     "MES": {"name" : "Messina Seismological Observatory",
             "alias": "",
             "country": "Italy"
             },
     "MEX": {"name" : "Instituto de Geofisica de la UNAM",
             "alias": "",
             "country": "Mexico"
             },
     "MIRAS": {"name" : "Mining Institute of the Ural Branch of the Russian Academy of Sciences",
             "alias": "",
             "country": "Russia"
             },
     "MKY": {"name" : "Unknown Historical Agency",
             "alias": "",
             "country": ""
             },
     "MLI": {"name": "NEIC Monthly listings",
             "alias": "NEIC",
             "country": "U.S.A."
             },
     "MLR": {"name" : "Muntele Rosu Station",
             "alias": "",
             "country": "Romania"
             },
     "MNH": {"name" : "Institut fur Angewandte Geophysik der Universitat Munchen",
             "alias": "",
             "country": "Germany"
             },
     "MOLD": {"name" : "Institute of Geophysics and Geology",
             "alias": "",
             "country": "Moldova"
             },
     "MOS": {"name" : "Geophysical Survey of Russian Academy of Sciences",
             "alias": "",
             "country": "Russia"
             },
     "MOZ": {"name" : "Direccao Nacional de Geologia",
             "alias": "",
             "country": "Mozambique"
             },
     "MRB": {"name" : "Institut Cartografic de Catalunya",
             "alias": "",
             "country": "Spain"
             },
     "MROB": {"name" : "Geophysikalisches Observatorium - Rosenbuhl",
             "alias": "",
             "country": "Germany"
             },
     "MSI": {"name" : "Messina Seismological Observatory",
             "alias": "",
             "country": "Italy"
             },
     "MSSP": {"name" : "Micro Seismic Studies Programme -  PINSTECH",
             "alias": "",
             "country": "Pakistan"
             },
     "MSUGS": {"name" : "Michigan State University -  Department of Geological Sciences",
             "alias": "",
             "country": "U.S.A."
             },
     "MUM": {"name" : "Manipur University",
             "alias": "",
             "country": "India"
             },
     "MUN": {"name" : "Mundaring Observatory",
             "alias": "",
             "country": "Australia"
             },
     "MVOV": {"name" : "Montserrat Volcano Observatory",
             "alias": "",
             "country": "Montserrat"
             },
     "MZEK": {"name" : "Geophysikalisches Observatorium - Zeckenberg",
             "alias": "",
             "country": "Germany"
             },
     "NAG": {"name" : "Unknown Historical Agency",
             "alias": "",
             "country": ""
             },
     "NAI": {"name" : "University of Nairobi",
             "alias": "",
             "country": "Kenya"
             },
     "NAM": {"name" : "The Geological Survey of Namibia",
             "alias": "",
             "country": "Namibia"
             },
     "NAN": {"name" : "Nanking Station",
             "alias": "BJI",
             "country": "China"
             },
     "NANGA": {"name" : "Nanga Parbat Seismic Experiment",
             "alias": "",
             "country": "U.S.A."
             },
     "NAO": {"name" : "Stiftelsen NORSAR",
             "alias": "",
             "country": "Norway"
             },
     "NCEDC": {"name" : "Northern California Earthquake Data Center",
             "alias": "",
             "country": "U.S.A."
             },
     "NDI": {"name" : "India Meteorological Department",
             "alias": "",
             "country": "India"
             },
     "NEIC": {"name" : "National Earthquake Information Center",
             "alias": "",
             "country": "U.S.A."
             },
     "NEI": {"name": "National Earthquake Information Center - CERESIS catalogue",
             "alias": "NEIC",
             "country": "U.S.A."
             },
     "NEIS": {"name" : "National Earthquake Information Service",
             "alias": "NEIC",
             "country": "U.S.A."
             },
     "NERS": {"name" : "North Eastern Regional Seismological Centre -  GS RAS",
             "alias": "",
             "country": "Russia"
             },
     "NEU": {"name" : "Neuchatel Station",
             "alias": "ZUR",
             "country": "Switzerland"
             },
     "NIC": {"name" : "Cyprus Geological Survey Department",
             "alias": "",
             "country": "Cyprus"
             },
     "NIED": {"name" : "National Research Institute for Earth Science and Disaster Prevention",
             "alias": "",
             "country": "Japan"
             },
     "NNC": {"name" : "National Nuclear Center",
             "alias": "",
             "country": "Kazakhstan"
             },
     "NORI": {"name" : "Geophysikalisches Observatorium - Nordlinger Ries",
             "alias": "",
             "country": "Germany"
             },
     "NORS": {"name" : "North Ossetia (Alania) Branch -  Geophysical Survey  Russian Academy of Sciences",
             "alias": "",
             "country": "Russia"
             },
     "NOU": {"name" : "IRD Centre de Noumea",
             "alias": "",
             "country": "New Caledonia"
             },
     "NPO": {"name" : "North Pole Environmental Observatory",
             "alias": "",
             "country": "U.S.A."
             },
     "NRA0": {"name" : "Noress Array",
             "alias": "",
             "country": "Norway"
             },
     "NSSC": {"name" : "National Syrian Seismological Center",
             "alias": "",
             "country": "Syria"
             },
     "NSSP": {"name" : "National Survey of Seismic Protection",
             "alias": "",
             "country": "Armenia"
             },
     "NUR": {"name" : "Nurmijarvi Station",
             "alias": "HEL",
             "country": "Finland"
             },
     "OAX": {"name" : "Oaxaca",
             "alias": "TAC",
             "country": "Mexico"
             },
     "OBER": {"name" : "Geophysikalisches Observatorium - Oberstdorf",
             "alias": "",
             "country": "Germany"
             },
     "OBM": {"name" : "Research Centre of Astronomy and Geophysics",
             "alias": "",
             "country": "Mongolia"
             },
     "OGA": {"name" : "Geophysikalisches Observatorium - Obergurgl/A",
             "alias": "",
             "country": "Germany"
             },
     "OGSO": {"name" : "Ohio Geological Survey",
             "alias": "",
             "country": "U.S.A."
             },
     "OMAN": {"name" : "Sultan Qaboos University",
             "alias": "",
             "country": "Oman"
             },
     "ORF": {"name" : "Orfeus Data Center",
             "alias": "",
             "country": "Netherlands"
             },
     "OSA": {"name" : "Osa Peninsula Project -  Costa Rica",
             "alias": "",
             "country": "U.S.A."
             },
     "OSC": {"name" : "Observatorio San Calixto",
             "alias": "LPZ",
             "country": "Bolivia"
             },
     "OSS_ISC": {"name" : "Ova Spin",
             "alias": "",
             "country": "Switzerland"
             },
     "OSUB": {"name" : "Osservatorio Sismologico Universita di Bari",
             "alias": "",
             "country": "Italy"
             },
     "OTT": {"name" : "Canadian Hazards Information Service -  Natural Resources Canada",
             "alias": "",
             "country": "Canada"
             },
     "OXD": {"name" : "Unknown Historical Agency",
             "alias": "",
             "country": ""
             },
     "PAL": {"name" : "Palisades",
             "alias": "",
             "country": "U.S.A."
             },
     "PANAMA97": {"name" : "Panama Canal Seismicity Study",
             "alias": "",
             "country": "U.S.A."
             },
     "PAS": {"name" : "California Institute of Technology",
             "alias": "",
             "country": "U.S.A."
             },
     "PAV": {"name" : "Pavia",
             "alias": "ROM",
             "country": "Italy"
             },
     "PDA": {"name" : "Universidade dos Azores",
             "alias": "",
             "country": "Portugal"
             },
     "PDE": {"name" : "Preliminary Determination of Epicentres",
             "alias": "NEIC",
             "country": "U.S.A."
             },
     "PDEW": {"name": "Preliminary Determination of Epicentres - Weekly",
             "alias": "PDE",
             "country": "U.S.A."
             },
     "PDEQ": {"name": "Preliminary Determination of Epicentres - Quick",
             "alias": "PDE",
             "country": "U.S.A."
             },
     "PDG": {"name" : "Seismological Institute of Montenegro",
             "alias": "",
             "country": "Montenegro"
             },
     "PEK": {"name" : "Peking",
             "alias": "BJI",
             "country": "China"
             },
     "PFO": {"name" : "Pinyon Flat Observatory",
             "alias": "",
             "country": "U.S.A."
             },
     "PGC": {"name" : "Pacific Geoscience Centre",
             "alias": "",
             "country": "Canada"
             },
     "PIN": {"name" : "Pinedale Seismic Array",
             "alias": "",
             "country": "U.S.A."
             },
     "PISCO": {"name" : "Proyecto de Investigacion Sismologica de la Cordillera Occidental",
             "alias": "",
             "country": "Germany"
             },
     "PIST": {"name" : "P. Stahl",
             "alias": "",
             "country": "France"
             },
     "PIT": {"name" : "Unknown Historical Agency",
             "alias": "",
             "country": ""
             },
     "PLV": {"name" : "National Center for Scientific Research",
             "alias": "",
             "country": "Vietnam"
             },
     "PMEL": {"name" : "Pacific seismicity from hydrophones",
             "alias": "",
             "country": "U.S.A."
             },
     "PMG": {"name" : "Port Moresby Geophysical Observatory",
             "alias": "",
             "country": "Papua New Guinea"
             },
     "PMR": {"name" : "Alaska Tsunami Warning Center",
             "alias": "",
             "country": "U.S.A."
             },
     "PNG": {"name" : "Unknown Historical Agency",
             "alias": "",
             "country": ""
             },
     "PNNL": {"name" : "Pacific Northwest National Laboratory",
             "alias": "",
             "country": "U.S.A."
             },
     "PNSN": {"name" : "Pacific Northwest Seismic Network",
             "alias": "",
             "country": "U.S.A."
             },
     "POO": {"name" : "Poona Observatory",
             "alias": "NDI",
             "country": "India"
             },
     "PPT": {"name" : "Laboratoire de Geophysique/CEA",
             "alias": "",
             "country": "French Polynesia"
             },
     "PRA": {"name" : "Academy of Sciences of the Czech Republic",
             "alias": "",
             "country": "Czech Republic"
             },
     "PRE": {"name" : "Council for Geoscience",
             "alias": "",
             "country": "South Africa"
             },
     "PRT": {"name" : "Observatorio San Domenico",
             "alias": "",
             "country": "Italy"
             },
     "PRU": {"name" : "Geophysical Institute -  Academy of Sciences of the Czech Republic",
             "alias": "",
             "country": "Czech Republic"
             },
     "P&S": {"name" : "Pacheco and Sykes (1992)",
             "alias": "",
             "country": "Worldwide"
             },
     "PTO": {"name" : "Instituto Geofisico da Universidade do Porto",
             "alias": "",
             "country": "Portugal"
             },
     "PTWC": {"name" : "Pacific Tsunami Warning Center",
             "alias": "",
             "country": "U.S.A."
             },
     "PUL": {"name" : "Unknown Historical Agency",
             "alias": "",
             "country": ""
             },
     "PUNA": {"name" : "Puna Plateau -  Argentina and Northern Chile Experiment",
             "alias": "",
             "country": "Germany"
             },
     "PUS": {"name" : "Unknown Historical Agency",
             "alias": "",
             "country": ""
             },
     "PYA": {"name" : "Unknown Historical Agency",
             "alias": "",
             "country": ""
             },
     "QCP": {"name" : "Manila Observatory",
             "alias": "",
             "country": "Philippines"
             },
     "QDM": {"name" : "Queensland Department of Mines",
             "alias": "",
             "country": "Australia"
             },
     "QUE": {"name" : "Pakistan Meteorological Department",
             "alias": "",
             "country": "Pakistan"
             },
     "QUI": {"name" : "Escuela Politecnica Nacional",
             "alias": "",
             "country": "Ecuador"
             },
     "RAB": {"name" : "Rabaul Volcanological Observatory",
             "alias": "",
             "country": "Papua New Guinea"
             },
     "RBA": {"name" : "Universita Mohammed V",
             "alias": "",
             "country": "Morocco"
             },
     "RDP": {"name" : "Unknown Historical Agency",
             "alias": "",
             "country": ""
             },
     "REN": {"name" : "MacKay School of Mines",
             "alias": "",
             "country": "U.S.A."
             },
     "REB": {"name": "Unknown Agency - from GCMT catalogue",
             "alias": "",
             "country": "from GCMT catalogue"
             },
     "REY": {"name" : "Icelandic Meteorological Office",
             "alias": "",
             "country": "Iceland"
             },
     "RIPT": {"name" : "Research Inst. of Pulse Technique",
             "alias": "",
             "country": "Russia"
             },
     "RISSC": {"name" : "Laboratory of Research on Experimental and Computational Seimology",
             "alias": "",
             "country": "Italy"
             },
     "RIV": {"name" : "Riverview Observatory",
             "alias": "",
             "country": "Australia"
             },
     "RJOB": {"name" : "Geophysikalisches Observatorium - Jochberg",
             "alias": "",
             "country": "Germany"
             },
     "RMIT": {"name" : "Royal Melbourne Institute of Technology",
             "alias": "",
             "country": "Australia"
             },
     "RNON": {"name" : "Geophysikalisches Observatorium - Staufen-Nonn",
             "alias": "",
             "country": "Germany"
             },
     "ROC": {"name" : "Odenbach Seismic Observatory",
             "alias": "",
             "country": "U.S.A."
             },
     "ROM": {"name" : "Istituto Nazionale di Geofisica e Vulcanologia",
             "alias": "",
             "country": "Italy"
             },
     "ROTZ": {"name" : "Geophysikalisches Observatorium - Rotzenmuhle",
             "alias": "",
             "country": "Germany"
             },
     "RRLJ": {"name" : "Regional Research Laboratory Jorhat",
             "alias": "",
             "country": "India"
             },
     "RSMAC": {"name" : "Red Sismica Mexicana de Apertura Continental",
             "alias": "",
             "country": "Mexico"
             },
     "RSNC": {"name" : "Red Sismologica Nacional de Colombia",
             "alias": "",
             "country": "Colombia"
             },
     "RSPR": {"name" : "Red Sismica de Puerto Rico",
             "alias": "",
             "country": "U.S.A."
             },
     "RYD": {"name" : "King Saud University",
             "alias": "",
             "country": "Saudi Arabia"
             },
     "SAN": {"name" : "Santiago",
             "alias": "GUC",
             "country": "Chile"
             },
     "SAPSE": {"name" : "Southern Alps Passive Seismic Experiment",
             "alias": "",
             "country": "New Zealand"
             },
     "SAR": {"name" : "Sarajevo Seismological Station",
             "alias": "",
             "country": "Bosnia and Herzegovina"
             },
     "SCB": {"name" : "Observatorio San Calixto",
             "alias": "",
             "country": "Bolivia"
             },
     "SCE": {"name" : "Geophysikalisches Observatorium - Schlegeis/Austria",
             "alias": "",
             "country": "Germany"
             },
     "SCEDC": {"name" : "Southern California Earthquake Data Center",
             "alias": "",
             "country": "U.S.A."
             },
     "SDD": {"name" : "Universidad Autonoma de Santo Domingo",
             "alias": "",
             "country": "Dominican Republic"
             },
     "SEA": {"name" : "Geophysics Program AK-50",
             "alias": "",
             "country": "U.S.A."
             },
     "SEDA": {"name" : "Seismic Exploration of the Deep Altiplano",
             "alias": "",
             "country": "U.S.A."
             },
     "SEPA": {"name" : "Seismic Experiment in Patagonia and Antarctica",
             "alias": "",
             "country": "U.S.A."
             },
     "SET": {"name" : "Setif Observatory",
             "alias": "",
             "country": "Algeria"
             },
     "SFS": {"name" : "Real Instituto y Observatorio de la Armada",
             "alias": "",
             "country": "Spain"
             },
     "SGS": {"name" : "Saudi Geological Survey",
             "alias": "",
             "country": "Saudi Arabia"
             },
     "SHI": {"name" : "Shiraz Observatory",
             "alias": "",
             "country": "Iran"
             },
     "SHL": {"name" : "Central Seismological Observatory",
             "alias": "",
             "country": "India"
             },
     "SIGU": {"name" : "Subbotin Institute of Geophysics -  National Academy of Sciences",
             "alias": "",
             "country": "Ukraine"
             },
     "SIK": {"name" : "Seismic Institute of Kosovo",
             "alias": "",
             "country": ""
             },
     "SIO": {"name" : "Scripps Institution of Oceanography",
             "alias": "",
             "country": "U.S.A."
             },
     "SJA": {"name" : "Instituto Nacional de Prevencion Sismica",
             "alias": "",
             "country": "Argentina"
             },
     "SJP": {"name" : "Unknown Historical Agency",
             "alias": "",
             "country": ""
             },
     "SJS": {"name" : "Instituto Costarricense de Electricidad",
             "alias": "",
             "country": "Costa Rica"
             },
     "SKHL": {"name" : "Sakhalin Experimental and Methodological Seismological Expedition -  GS RAS",
             "alias": "",
             "country": "Russia"
             },
     "SKL": {"name" : "Sakhalin Complex Scientific Research Institute",
             "alias": "",
             "country": "Russia"
             },
     "SKO": {"name" : "Seismological Observatory Skopje",
             "alias": "",
             "country": "FYR Macedonia"
             },
     "SLC": {"name" : "Salt Lake City",
             "alias": "",
             "country": "U.S.A."
             },
     "SLM": {"name" : "Saint Louis University",
             "alias": "",
             "country": "U.S.A."
             },
     "SMI": {"name" : "Smithsonian Institution",
             "alias": "",
             "country": "U.S.A."
             },
     "SNET": {"name" : "Servicio Nacional de Estudios Territoriales",
             "alias": "",
             "country": "El Salvador"
             },
     "SNM": {"name" : "New Mexico Institute of Mining and Technology",
             "alias": "",
             "country": "U.S.A."
             },
     "SNSN": {"name" : "Saudi National Seismic Network",
             "alias": "",
             "country": "Saudi Arabia"
             },
     "SOC": {"name" : "Unknown Historical Agency",
             "alias": "",
             "country": ""
             },
     "SOD": {"name" : "Sodankyla Seismological Station",
             "alias": "HEL",
             "country": "Finland"
             },
     "SOF": {"name" : "Geophysical Institute -  Bulgarian Academy of Sciences",
             "alias": "",
             "country": "Bulgaria"
             },
     "SOME": {"name" : "Seismological Experimental Methodological Expedition",
             "alias": "",
             "country": "Kazakhstan"
             },
     "SPA": {"name" : "USGS - South Pole",
             "alias": "",
             "country": "Antarctica"
             },
     "SPASE": {"name" : "Southwest Pacific Seismic Experiment",
             "alias": "",
             "country": "U.S.A."
             },
     "SPC": {"name" : "Skalnate-Pleso Seismological Station",
             "alias": "BRA",
             "country": "Slovakia"
             },
     "SPGM": {"name" : "Service de Physique du Globe",
             "alias": "RBA",
             "country": "Morocco"
             },
     "SPITAK": {"name" : "",
             "alias": "",
             "country": "Armenia"
             },
     "SRI": {"name" : "Stanford Research Institute",
             "alias": "",
             "country": "U.S.A."
             },
     "SSN": {"name" : "Sudan Seismic Network",
             "alias": "",
             "country": "Sudan"
             },
     "SSNC": {"name" : "Servicio Sismologico Nacional Cubano",
             "alias": "",
             "country": "Cuba"
             },
     "SSS": {"name" : "Centro de Estudios y Investigaciones Geotecnicas del San Salvador",
             "alias": "",
             "country": "El Salvador"
             },
     "STK": {"name" : "Stockholm Seismological Station",
             "alias": "HFS",
             "country": "Sweden"
             },
     "STL": {"name" : "Santa Lucia Seismological Station - CERESIS",
             "alias": "GUC",
             "country": "Chile"
             },
     "STR": {"name" : "Institut de Physique du Globe",
             "alias": "",
             "country": "France"
             },
     "STU": {"name" : "Stuttgart Seismological Station",
             "alias": "LEDBW",
             "country": "Germany"
             },
     "SUC": {"name" : "Unknown Historical Agency",
             "alias": "",
             "country": ""
             },
     "SUS": {"name" : "Unknown Historical Agency",
             "alias": "",
             "country": ""
             },
     "SVA": {"name" : "Department of Mineral Resources",
             "alias": "",
             "country": "Fiji"
             },
     "SVE": {"name" : "Unknown Historical Agency",
             "alias": "",
             "country": ""
             },
     "SVSA": {"name" : "Sistema de Vigilancia Sismologica dos Azores",
             "alias": "",
             "country": "Portugal"
             },
     "SWE": {"name" : "",
             "alias": "",
             "country": ""
             },
     "SWEQ": {"name": "Ekstrom (2006)-SurfaceWaveLocation",
             "alias": "",
             "country": "U.S.A."
             },
     "SYKES": {"name" : "Sykes Catalogue of earthquakes 1950 onwards",
             "alias": "",
             "country": ""
             },
     "SYO": {"name" : "National Institute of Polar Research",
             "alias": "",
             "country": "Japan"
             },
     "SZGRF": {"name" : "Seismologisches Zentralobservatorium Grofenberg",
             "alias": "",
             "country": "Germany"
             },
     "TAB": {"name" : "Tabriz Seismological Observatory",
             "alias": "",
             "country": "Iran"
             },
     "TAC": {"name" : "Estacion Central de Tacubaya",
             "alias": "",
             "country": "Mexico"
             },
     "TAN": {"name" : "Antananarivo",
             "alias": "",
             "country": "Madagascar"
             },
     "TANZANIA": {"name" : "Tanzania Broadband Seismic Experiment",
             "alias": "",
             "country": "U.S.A."
             },
     "TAP": {"name" : "CWB",
             "alias": "",
             "country": ""
             },
     "TAS": {"name" : "Unknown Historical Agency",
             "alias": "",
             "country": ""
             },
     "TAU": {"name" : "University of Tasmania",
             "alias": "",
             "country": "Australia"
             },
     "TEH": {"name" : "Tehran University",
             "alias": "",
             "country": "Iran"
             },
     "TEIC": {"name" : "Center for Earthquake Research and Information",
             "alias": "",
             "country": "U.S.A."
             },
     "THE": {"name" : "Department of Geophysics -  Aristotle University of Thessaloniki",
             "alias": "",
             "country": "Greece"
             },
     "THR": {"name" : "International Institute of Earthquake Engineering and Seismology (IIEES)",
             "alias": "",
             "country": "Iran"
             },
     "TIENSHAN": {"name" : "Tien Shan Continental Dynamics",
             "alias": "",
             "country": "U.S.A."
             },
     "TIF": {"name" : "Seismic Monitoring Centre of Georgia",
             "alias": "",
             "country": "Georgia"
             },
     "TIR": {"name" : "The Institute of Seismology -  Academy of Sciences of Albania",
             "alias": "",
             "country": "Albania"
             },
     "TOK": {"name" : "Tokyo Observatory",
             "alias": "JMA",
             "country": "Japan"
             },
     "TOL": {"name" : "Toledo Observatory",
             "alias": "MDD",
             "country": "Spain"
             },
     "TRI": {"name" : "Istituto Nazionale di Oceanografia e di Geofisica Sperimentale (OGS)",
             "alias": "",
             "country": "Italy"
             },
     "TRN": {"name" : "University of the West Indies",
             "alias": "",
             "country": "Trinidad and Tobago"
             },
     "TTG": {"name" : "Titograd Seismological Station",
             "alias": "PDG",
             "country": "Montenegro"
             },
     "TUC": {"name" : "Unknown Historical Agency",
             "alias": "",
             "country": ""
             },
     "TUL": {"name" : "Oklahoma Geological Survey",
             "alias": "",
             "country": "U.S.A."
             },
     "TUN": {"name" : "Institut National de la Metorologie",
             "alias": "",
             "country": "Tunisia"
             },
     "TVA": {"name" : "Tennessee Valley Authority",
             "alias": "",
             "country": "U.S.A."
             },
     "TZN": {"name" : "University of Dar Es Salaam",
             "alias": "",
             "country": "Tanzania"
             },
     "UAV": {"name" : "Red Sismologica de Los Andes Venezolanos",
             "alias": "",
             "country": "Venezuela"
             },
     "UCC": {"name" : "Royal Observatory of Belgium",
             "alias": "",
             "country": "Belgium"
             },
     "UCR": {"name" : "Universidad de Costa Rica",
             "alias": "",
             "country": "Costa Rica"
             },
     "UGN": {"name" : "Institute of Geonics AS CR",
             "alias": "",
             "country": "Czech Republic"
             },
     "ULE": {"name" : "University of Leeds",
             "alias": "",
             "country": "United Kingdom"
             },
     "UNAH": {"name" : "Universidad Nacional Autonoma de Honduras",
             "alias": "",
             "country": "Honduras"
             },
     "UNK": {"name" : "Unknown source",
             "alias": "",
             "country": ""
             },
     "UNKOWN": {"name" : "Unknown source",
             "alias": "",
             "country": ""
             },
     "UPA": {"name" : "Universidad de Panama",
             "alias": "",
             "country": "Panama"
             },
     "UPP": {"name" : "University of Uppsala",
             "alias": "",
             "country": "Sweden"
             },
     "UPSL": {"name" : "University of Patras -  Department of Geology",
             "alias": "",
             "country": "Greece"
             },
     "USAEC": {"name" : "United States Atomic Energy Commission",
             "alias": "DOE",
             "country": "U.S.A."
             },
     "USAF": {"name" : "US Air Force Technical Applications Center",
             "alias": "",
             "country": "U.S.A."
             },
     "USBR": {"name" : "US Bureau of Reclamation",
             "alias": "",
             "country": "U.S.A."
             },
     "USCGS": {"name" : "United States Coast and Geodetic Survey",
             "alias": "NEIC",
             "country": "U.S.A."
             },
     "USGS": {"name" : "United States Geological Survey",
             "alias": "",
             "country": "U.S.A."
             },
     "USP": {"name" : "University of Sao Paulo",
             "alias": "",
             "country": "Brazil"
             },
     "USPetal": {"name" : "IAG-USP,UnB,UFRN,ON,UNESP,IPT and others",
             "alias": "USP",
             "country": "Brazil"
             },
     "UUSS": {"name" : "The University of Utah Seismograph Stations",
             "alias": "",
             "country": "U.S.A."
             },
     "UVC": {"name" : "Universidad del Valle",
             "alias": "",
             "country": "Colombia"
             },
     "VAO": {"name" : "Instituto Astronomico e Geofisico (IAG-USP)",
             "alias": "USP",
             "country": "Brazil"
             },
     "VIE": {"name" : "Asterreichischer Geophysikalischer Dienst",
             "alias": "",
             "country": "Austria"
             },
     "VKMS": {"name" : "Lab. of Seismic Monitoring -  Voronezh region  GSRAS & Voronezh State University",
             "alias": "",
             "country": "Russia"
             },
     "VLA": {"name" : "Vladivostok Seismological Station",
             "alias": "MOS",
             "country": "Russia"
             },
     "VRAC": {"name" : "Vranov Seismological Station",
             "alias": "",
             "country": "Czech Republic"
             },
     "VSI": {"name" : "University of Athens",
             "alias": "",
             "country": "Greece"
             },
     "WAR": {"name" : "Institute of Geophysics -  Polish Academy of Sciences",
             "alias": "",
             "country": "Poland"
             },
     "WBNET": {"name" : "West Bohemia Seismic Network",
             "alias": "",
             "country": "Czech Republic"
             },
     "WEL": {"name" : "Institute of Geological and Nuclear Sciences",
             "alias": "",
             "country": "New Zealand"
             },
     "WES": {"name" : "Weston Observatory",
             "alias": "",
             "country": "U.S.A."
             },
     "WET": {"name" : "Geophysikalisches Observatorium - Wettzell",
             "alias": "",
             "country": "Germany"
             },
     "WMO": {"name" : "Wichita Mountains Observatory",
             "alias": "NEIS",
             "country": "U.S.A."
             },
     "WOODLARK": {"name" : "Woodlark-D - Entrecasteaux Rift  Papua New Guinea",
             "alias": "",
             "country": "U.S.A."
             },
     "X": {"name" : "Unknown Historical Agency",
             "alias": "",
             "country": ""
             },
     "YAL": {"name" : "Unknown Historical Agency",
             "alias": "",
             "country": ""
             },
     "YARS": {"name" : "Yakutiya Regional Seismological Center -  GS SB RAS",
             "alias": "",
             "country": "Russia"
             },
     "ZAG": {"name" : "Seismological Survey of the Republic of Croatia",
             "alias": "",
             "country": "Croatia"
             },
     "ZKW": {"name" : "Unknown Historical Agency",
             "alias": "",
             "country": ""
             },
     "ZON": {"name" : "Universidad Nacional de San Juan",
             "alias": "",
             "country": "Argentina"
             },
     "ZSC": {"name" : "Zose Seismological Station",
             "alias": "BJI",
             "country": "China"
             },
     "ZUR": {"name" : "Swiss Seismological Sevice (SED)",
             "alias": "",
             "country": "Switzerland"
             },
     "ZUR_RMT": {"name" : "Zurich Moment Tensors",
             "alias": "",
             "country": "Switzerland"
             },
     "----": {"name" : "Unknown Agency from CENTENNIAL catalogue",
             "alias": "",
             "country": ""
             }
 }
