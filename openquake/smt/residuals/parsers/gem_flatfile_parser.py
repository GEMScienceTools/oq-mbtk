# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2025 GEM Foundation
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
Parse the GEM globally homogenised flatfile into SMT metadata.
"""
import os
import csv
import pandas as pd
import copy
import pickle
from math import sqrt
from linecache import getline

from openquake.smt.residuals.sm_database import (GroundMotionDatabase,
                                                 GroundMotionRecord,
                                                 Earthquake,
                                                 Magnitude,
                                                 Rupture,
                                                 FocalMechanism,
                                                 GCMTNodalPlanes,
                                                 RecordSite,
                                                 RecordDistance)
from openquake.smt.residuals.parsers import valid
from openquake.smt.residuals.parsers.esm_flatfile_parser import (parse_ground_motion,
                                                                 parse_waveform_data)
from openquake.smt.residuals.parsers.base_database_parser import SMDatabaseReader
from openquake.smt.utils import MECHANISM_TYPE, DIP_TYPE

# Import the ESM dictionaries
from .esm_dictionaries import *

HDEFS = ["Geometric", "rotD00", "rotD50", "rotD100"]

HEADERS = ["event_id",
           "event_time",
           "ISC_ev_id",
           "ev_latitude",
           "ev_longitude",
           "ev_depth_km",
           "fm_type_code",
           "ML",
           "Mw",
           "Ms",
           "es_strike",
           "es_dip",
           "es_rake",
           "es_z_top",
           "es_length",
           "es_width",
           "network_code",
           "station_code",
           "st_latitude",
           "st_longitude",
           "st_elevation",
           "st_backarc",
           "vs30_m_sec",
           "vs30_meas_type",
           "z1pt0 (m)",
           "z2pt5 (km)",
           "epi_dist",
           "JB_dist",
           "rup_dist",
           "Rx_dist",
           "Ry0_dist",
           "U_hp",
           "V_hp",
           "W_hp",
           "U_lp",
           "V_lp",
           "W_lp",
           "shortest_usable_period",
           "longest_usable_period"
           ""
           ]

M_PRECEDENCE = ["Mw", "Ms", "ML"]


class GEMFlatfileParser(SMDatabaseReader):
    """
    Parses the data from the flatfile to a set of metadata objects
    """
    def parse(self, location='./'):
        """
        Parse the dataset
        """
        assert os.path.isfile(self.filename)
        headers = getline(self.filename, 1).rstrip("\n").split(",")
        for hdr in HEADERS:
            if hdr not in headers:
                raise ValueError("Required header %s is missing in file" % hdr)

        # Read in csv
        reader = csv.DictReader(open(self.filename, "r"), delimiter=",")
        self.database = GroundMotionDatabase(self.id, self.name)
        counter = 0
        for row in reader:
            # Build the metadata
            record = self._parse_record(row)
            if record:
                # Parse the strong motion
                record = parse_ground_motion(
                    os.path.join(location, "records"), row, record, headers)
                self.database.records.append(record)
            else:
                print("Record with sequence number %s is null/invalid"
                      % "{:s}-{:s}".format(row["event_id"], row["station_code"]))
            if (counter % 100) == 0:
                print("Processed record %s - %s" % (str(counter), record.id))
                
            counter += 1

    @classmethod
    def autobuild(cls, dbid, dbname, output_location, flatfile_directory):
        """
        Quick and dirty full database builder!
        """
        if os.path.exists(output_location):
            raise IOError("Target database directory %s already exists!"
                          % output_location)
        os.mkdir(output_location)

        # Add on the records folder
        os.mkdir(os.path.join(output_location, "records"))

        # Create an instance of the parser class
        database = cls(dbid, dbname, flatfile_directory)

        # Parse the records
        print("Parsing Records ...")
        database.parse(location=output_location)

        # Save itself to file
        metadata_file = os.path.join(output_location, "metadatafile.pkl")
        print("Storing metadata to file %s" % metadata_file)
        with open(metadata_file, "wb+") as f:
            pickle.dump(database.database, f)
            
        return database

    def _parse_record(self, metadata):
        """
        Parse a record
        """
        # Waveform ID not provided in file so concatenate Event and Station ID
        wfid = "_".join(
            [metadata["event_id"], metadata["network_code"], metadata["station_code"]])
        wfid = wfid.replace("-", "_")

        # Parse the event metadata
        event = self._parse_event_data(metadata)

        # Parse the distance metadata
        distances = self._parse_distances(metadata, event.depth)

        # Parse the station metadata
        site = self._parse_site_data(metadata)

        # Parse waveform data
        xcomp, ycomp, vertical = parse_waveform_data(metadata, wfid)
        
        # Shortest and longest usable periods
        sp = valid.vfloat(metadata['shortest_usable_period'], 'shortest_usable_period')
        lp = valid.vfloat(metadata['longest_usable_period'], 'longest_usable_period')

        return GroundMotionRecord(wfid,
                                  [None, None, None],
                                  event, distances, site,
                                  xcomp, ycomp,
                                  vertical=vertical,
                                  longest_period=lp,
                                  shortest_period=sp)


    def _parse_event_data(self, metadata):
        """
        Parses the event metadata
        """
        # ID and Name (name not in file so use ID again)
        eq_id = metadata["event_id"]
        eq_name = metadata["event_id"]

        # Date and time
        eq_datetime = pd.to_datetime(metadata["event_time"])
        
        # Latitude, longitude and depth
        eq_lat = valid.latitude(metadata["ev_latitude"])
        eq_lon = valid.longitude(metadata["ev_longitude"])
        eq_depth = valid.positive_float(metadata["ev_depth_km"], "ev_depth_km")
        if not eq_depth:
            raise ValueError(f'Depth missing for {eq_id} in admitted flatfile')

        # Make SMT EQ object
        eqk = Earthquake(eq_id, eq_name, eq_datetime, eq_lon, eq_lat, eq_depth,
                         None, # Magnitude not defined yet)
                         tectonic_region=metadata['event_trt_from_classifier']
                         )
        
        # Get preferred magnitude and list
        pref_mag, magnitude_list = self._parse_magnitudes(metadata)
        eqk.magnitude = pref_mag
        eqk.magnitude_list = magnitude_list
        eqk.rupture, eqk.mechanism = self._parse_rupture_mechanism(metadata,
                                                                   eq_id,
                                                                   eq_name,
                                                                   pref_mag,
                                                                   eq_depth)

        return eqk

    def _parse_magnitudes(self, metadata):
        """
        An order of precedence is required and the preferred magnitude will be
        the highest found
        """
        pref_mag = None
        mag_list = []
        for key in M_PRECEDENCE:
            mvalue = metadata[key].strip()
            if mvalue:
                mtype = key
                mag = Magnitude(float(mvalue), mtype)
                if not pref_mag:
                    pref_mag = copy.deepcopy(mag)
                mag_list.append(mag)
                
        return pref_mag, mag_list

    def _parse_rupture_mechanism(self, metadata, eq_id, eq_name, mag, depth):
        """
        Parse rupture mechanism
        """
        # Get the SoF
        sof = metadata["fm_type_code"]
        if pd.isnull(sof):
            sof = "U"

        # Initial rupture
        rupture = Rupture(eq_id, eq_name, mag, None, None, depth)

        # Mechanism
        mechanism = FocalMechanism(
            eq_id,
            eq_name,
            GCMTNodalPlanes(),
            None,
            mechanism_type=sof)
        
        # See if focal mechanism exists and get it if so
        fm_set = []
        for key in ["es_strike", "es_dip", "es_rake"]:
            if key in metadata:
                fm_param = valid.vfloat(metadata[key], key)
                if fm_param is not None:
                    fm_set.append(fm_param)

        if len(fm_set) == 3:
            # Has a valid focal mechanism (only the preferred nodal plane
            # solution is provided in the GEM flatfile like in ESM URL format)
            mechanism.nodal_planes.nodal_plane_1 = {
                "strike": fm_set[0], "dip": fm_set[1], "rake": fm_set[2]}

        if not mechanism.nodal_planes.nodal_plane_1:
            # Absolutely no information - base on style-of-faulting
            mechanism.nodal_planes.nodal_plane_1 = {
                "strike": 0.0, "dip": DIP_TYPE[sof], "rake": MECHANISM_TYPE[sof]
                }
            
        return rupture, mechanism
        
    def _parse_distances(self, metadata, hypo_depth):
        """
        Parse the distances provided in the flatfile. If not provided
        then we can calculate by constructing a finite rupture within
        the engine (this occurs within )
        """
        repi = valid.positive_float(metadata["epi_dist"], "epi_dist")
        if pd.isnull(repi):
            repi, rhypo = None, None
        else:
            rhypo = sqrt(repi ** 2. + hypo_depth ** 2.)
        rjb = valid.positive_float(metadata["JB_dist"], "JB_dist")
        if pd.isnull(rjb):
            rjb = None
        rrup = valid.positive_float(metadata["rup_dist"], "rup_dist")
        if pd.isnull(rrup):
            rrup = None
        r_x = valid.vfloat(metadata["Rx_dist"], "Rx_dist")
        if pd.isnull(r_x):
            r_x = None
        ry0 = valid.positive_float(metadata["Ry0_dist"], "Ry0_dist")
        if pd.isnull(ry0):
            ry0 = None
        return RecordDistance(repi, rhypo, rjb, rrup, r_x, ry0)

    def _parse_site_data(self, metadata):
        """
        Parses the site information
        """
        # Basic site/station information
        network_code = metadata["network_code"].strip()
        station_code = metadata["station_code"].strip()
        site_id = "{:s}-{:s}".format(network_code, station_code)
        site_lon = valid.longitude(metadata["st_longitude"])
        site_lat = valid.latitude(metadata["st_latitude"])
        elevation = valid.vfloat(metadata["st_elevation"], "st_elevation")

        # Vs30
        vs30 = valid.vfloat(metadata["vs30_m_sec"], "vs30_m_sec")
        if pd.isnull(vs30):
            # Need a station vs30 value for residuals (not really, given
            # some GMMs lack site terms, but good way to prevent confusing
            # nans in the expected values which appear when computing stats)
            raise ValueError(f"A vs30 value is missing for {site_id}")
        if  metadata["vs30_meas_type"] == "measured":
            vs30_measured = 1
        else:
            vs30_measured = 0 # Inferred

        # Get station backarc flag
        ba = metadata["st_backarc"]
        if ba == "no info provided":
            st_backarc = False
        elif int(ba) == 0:
            st_backarc = False
        else:
            try:
                assert int(ba) == 1
                st_backarc = True
            except:
                raise ValueError(
                    "Invalid option for station backarc in GEM Flatfile "
                    "(can be a value of 0, 1 or 'no info provided').")

        # Make the site object
        site = RecordSite(site_id,
                          station_code,
                          station_code,
                          site_lon,
                          site_lat,
                          elevation,
                          vs30,
                          vs30_measured,
                          network_code=network_code,
                          backarc=st_backarc)

        # Add basin params
        site.z1pt0 = valid.vfloat(metadata["z1pt0 (m)"], "z1pt0 (m)")
        site.z2pt5 = valid.vfloat(metadata["z2pt5 (km)"], "z2pt5 (km)")

        return site
    