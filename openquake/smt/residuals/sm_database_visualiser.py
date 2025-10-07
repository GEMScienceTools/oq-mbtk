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
Tool for creating visualisation of database information
"""
import numpy as np
import matplotlib.pyplot as plt

from openquake.calculators.postproc.plots import add_borders
from openquake.smt.utils_intensity_measures import _save_image
from openquake.smt.residuals.sm_database_selector import SMRecordSelector 


DISTANCES = {
    "repi": lambda rec: rec.distance.repi,
    "rhypo": lambda rec: rec.distance.rhypo,
    "rjb": lambda rec: rec.distance.rjb,
    "rrup": lambda rec: rec.distance.rrup,
    "rx": lambda rec: rec.distance.r_x,
}

DISTANCE_LABEL = {
    "repi": "Epicentral Distance (km)",
    "rhypo": "Hypocentral Distance (km)",
    "rjb": "Joyner-Boore Distance (km)",
    "rrup": "Rupture Distance (km)",
    "r_x": "R-x Distance (km)"
}

NEHRP_BOUNDS = {
    "A": (1500.0, np.inf),
    "B": (760.0, 1500.0),
    "C": (360.0, 760.),
    "D": (180., 360.),
    "E": (0., 180.)
}

EC8_BOUNDS = {
    "A": (800., np.inf),
    "B": (360.0, 800.),
    "C": (180.0, 360.),
    "D": (0., 360)
}


def get_eq_and_st_coordinates(db1):
    """
    From the strong motion database, returns lists of latitudes and longitudes
    of the events and stations
    """
    eq_coos, st_coos = [], []
    for record in db1.records:
        eq_coo = (record.event.longitude, record.event.latitude)
        st_coo = (record.site.longitude, record.site.latitude)
        if eq_coo not in eq_coos:
            eq_coos.append(eq_coo)
        if st_coo not in st_coos:
            st_coos.append(st_coo)
    e_lon, e_lat = [], []
    for eq in eq_coos:
        e_lon.append(eq[0])
        e_lat.append(eq[1])
    s_lon, s_lat = [], []
    for st in st_coos:
        s_lon.append(st[0])
        s_lat.append(st[1])

    return np.array(e_lon), np.array(e_lat), np.array(s_lon), np.array(s_lat)


def get_magnitude_distances(db1, dist_type):
    """
    From the strong motion database, returns lists of magnitude and distance
    pairs
    """
    mags = []
    dists = []
    for record in db1.records:
        mags.append(record.event.magnitude.value)
        if dist_type == "rjb":
            rjb = DISTANCES[dist_type](record)
            if rjb:
                dists.append(rjb)
            else:
                dists.append(DISTANCES["repi"](record))
        elif dist_type == "rrup":
            rrup = DISTANCES[dist_type](record)
            if rrup:
                dists.append(rrup)
            else:
                dists.append(DISTANCES["rhypo"](record))
        else:
            dists.append(DISTANCES[dist_type](record))
    return mags, dists


def db_magnitude_distance(db1,
                          dist_type,
                          figure_size=(7, 5),
                          figure_title=None,
                          filename=None,
                          filetype="png",
                          dpi=300):
    """
    Creates a plot of magnitude verses distance for a strong motion database
    """
    plt.figure(figsize=figure_size)
    mags, dists = get_magnitude_distances(db1, dist_type)
    plt.semilogx(np.array(dists), np.array(mags), "o", mec='k', mew=0.5)
    plt.xlabel(DISTANCE_LABEL[dist_type], fontsize=14)
    plt.ylabel("Magnitude", fontsize=14)
    plt.grid()
    if figure_title:
        plt.title(figure_title, fontsize=18)
    _save_image(filename, plt.gcf(), filetype, dpi)


def db_geographical_coverage(db1,
                             figure_size=(7, 5),
                             filename=None,
                             filetype='png',
                             dpi=300):
    """
    Creates a plot of the locations of event hypocenters and station locations
    for a strong motion database
    """
    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot(111)
    eq_lons, eq_lats, st_lons, st_lats = get_eq_and_st_coordinates(db1)
    ax.scatter(st_lons, st_lats, marker='^', color='g',
               label='Station locations')
    ax.scatter(eq_lons, eq_lats, marker='*', color='r',
               label='Event hypocenters')
    add_borders(ax)
    lons = np.concatenate([eq_lons, st_lons])
    lats = np.concatenate([eq_lats, st_lats])
    ax.set_xlim(np.floor(np.min(lons)-0.25), np.ceil(np.max(lons))+0.25)
    ax.set_ylim(np.floor(np.min(lats)-0.25), np.ceil(np.max(lats))+0.25)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()
    _save_image(filename, plt.gcf(), filetype, dpi)


def _site_selection(db1, site_class, classifier):
    """
    Select records within a particular site class and/or vs30 range
    """
    idx = []
    for iloc, rec in enumerate(db1.records):
        if classifier == "NEHRP":
        
            if rec.site.nehrp and (rec.site.nerhp == site_class):
                idx.append(iloc)
                continue

            if (rec.site.vs30 >= NEHRP_BOUNDS[site_class][0]) and\
                (rec.site.vs30 < NEHRP_BOUNDS[site_class][1]):
                idx.append(iloc)
        
        elif classifier == "EC8":
            if rec.site.ec8 and (rec.site.ec8 == site_class):
                idx.append(iloc)
                continue
              
            if rec.site.vs30:
                if (rec.site.vs30 >= EC8_BOUNDS[site_class][0]) and\
                   (rec.site.vs30 < EC8_BOUNDS[site_class][1]):
                    idx.append(iloc)
        else:
            raise ValueError("Unrecognised Site Classifier!")
            
    return idx


def db_magnitude_distance_by_site(db1,
                                  dist_type,
                                  classification="NEHRP",
                                  figure_size=(7, 5),
                                  filename=None,
                                  filetype="png",
                                  dpi=300):
    """
    Plot magnitude-distance comparison by site NEHRP or Eurocode 8 Site class   
    """ 
    if classification == "NEHRP":
        site_bounds = NEHRP_BOUNDS
    elif classification == "EC8":
        site_bounds = EC8_BOUNDS
    else:
        raise ValueError("Unrecognised Site Classifier!")
    selector = SMRecordSelector(db1)
    plt.figure(figsize=figure_size)
    total_idx = []
    for site_class in site_bounds.keys():
        site_idx = _site_selection(db1, site_class, classification)
        if site_idx:
            site_db = selector.select_records(site_idx, as_db=True)
            mags, dists = get_magnitude_distances(site_db, dist_type)
            plt.plot(np.array(dists), np.array(mags), "o", mec='k',
                     mew=0.5, label="Site Class %s" % site_class)
            total_idx.extend(site_idx)
    mag, dists = get_magnitude_distances(site_db, dist_type)
    plt.semilogx(np.array(dists), np.array(mags), "o", mfc="None", mec='k',
                 mew=0.5, label="Unclassified", zorder=0)
    plt.xlabel(DISTANCE_LABEL[dist_type], fontsize=14)
    plt.ylabel("Magnitude", fontsize=14)
    plt.grid()
    plt.legend(ncol=2,loc="lower right", numpoints=1)
    plt.title("Magnitude vs Distance (by %s Site Class)" % classification,
              fontsize=18)
    _save_image(filename, plt.gcf(), filetype, dpi)
    plt.show()


def db_magnitude_distance_by_trt(db1,
                                 dist_type,
                                 figure_size=(7, 5),
                                 filename=None,
                                 filetype="png",
                                 dpi=300):
    """
    Plot magnitude-distance comparison by tectonic region
    """
    trts=[]
    for i in db1.records:
        trts.append(i.event.tectonic_region)
    trt_types=list(set(trts))
    selector = SMRecordSelector(db1)
    plt.figure(figsize=figure_size)
    for trt in trt_types:
        subdb = selector.select_trt_type(trt, as_db=True)
        mag, dists = get_magnitude_distances(subdb, dist_type)
        plt.semilogx(dists, mag, "o", mec='k', mew=0.5, label=trt)
    plt.xlabel(DISTANCE_LABEL[dist_type], fontsize=14)
    plt.ylabel("Magnitude", fontsize=14)
    plt.title("Magnitude vs Distance by Tectonic Region", fontsize=18)
    plt.legend(loc='lower right', numpoints=1)
    plt.grid()
    _save_image(filename,  plt.gcf(), filetype, dpi)
    plt.show()
