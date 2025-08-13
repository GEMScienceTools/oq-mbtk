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
Parser for the NGAWest2 flatfile
"""
import pandas as pd
import os
import tempfile
import csv
import numpy as np
import copy
import h5py
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
                                                 Component,
                                                 RecordSite,
                                                 RecordDistance)
from openquake.smt.residuals.parsers.base_database_parser import SMDatabaseReader
from openquake.smt.residuals.parsers import valid
from openquake.smt.utils import MECHANISM_TYPE, DIP_TYPE


# Import the ESM dictionaries
from .esm_dictionaries import *

BASE = os.path.abspath("")

CONV_TO_CMS2 = 981

HCOMPS = ["rotD50"]

HEADERS = ["event_id",
           "event_time",
           "ev_latitude",
           "ev_longitude",
           "ev_depth_km",
           "fm_type_code",
           "Mw",
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
           "vs30_m_sec",
           "vs30_meas_type",
           "z1pt0 (m)",
           "z2pt5 (km)",
           "epi_dist",
           "epi_az",
           "JB_dist",
           "rup_dist",
           "Rx_dist",
           "U_hp",
           "V_hp",
           "W_hp",
           "U_lp",
           "V_lp",
           "W_lp"]


class NGAWest2FlatfileParser(SMDatabaseReader):
    """
    Parses the data from flatfile to a set of metadata objects
    """
    M_PRECEDENCE = ["Mw"]
    BUILD_FINITE_DISTANCES = False

    def parse(self, location='./'):
        """
        Parse the metadata
        """
        assert os.path.isfile(self.filename)
        headers = getline(self.filename, 1).rstrip("\n").split(";")
        for hdr in HEADERS:
            if hdr not in headers:
                raise ValueError("Required header %s is missing in file" % hdr)
            
        # Read in csv
        reader = csv.DictReader(open(self.filename, "r"), delimiter=";")
        self.database = GroundMotionDatabase(self.id, self.name)
        counter = 0
        for row in reader:
            # Build the metadata
            record = self._parse_record(row)
            if record:
                # Parse the strong motion
                record = self._parse_ground_motion(
                    os.path.join(location, "records"), row, record, headers)
                self.database.records.append(record)
            else:
                print("Record with sequence number %s is null/invalid"
                      % "{:s}-{:s}".format(
                          row["event_id"], row["station_code"]))
            if (counter % 100) == 0:
                print("Processed record %s - %s" % (str(counter), record.id))

            counter += 1

    @classmethod
    def autobuild(cls, dbid, dbname, output_location, ngaw2_horz,  ngaw2_vert):
        """
        Quick and dirty full database builder!
        """
        # Import ngawest2 format strong-motion flatfiles
        ngawest2 = pd.read_csv(ngaw2_horz)
        ngawest2_vert = pd.read_csv(ngaw2_vert)
        
        # Check RotD50 and vert records match
        assert all(ngawest2['Record Sequence Number'] == ngawest2_vert['Record Sequence Number'])
        
        # Count initial size for printing number records removed during checks
        initial_ngaw2_size = len(ngawest2)

        # Remove potential duplicate records in NGA-West2 flatfile
        ngawest2 = ngawest2.drop_duplicates(
            subset = ['Earthquake Name','Station Name'], keep='last')
        ngawest2_vert = ngawest2_vert.drop_duplicates(
            subset = ['Earthquake Name', 'Station Name'], keep='last')
        ngawest2 = ngawest2.reset_index(drop=True)
        ngawest2_vert = ngawest2_vert.reset_index(drop=True)

        # Remove records if earthquake not identifiable using lat/lon metadata
        idx_m = ngawest2.loc[ngawest2['Hypocenter Latitude (deg)']==-999].index
        ngawest2 = ngawest2.drop(idx_m).reset_index(drop=True)
        ngawest2_vert = ngawest2_vert.drop(idx_m).reset_index(drop=True)
        idx_m = ngawest2.loc[ngawest2['Hypocenter Longitude (deg)']==-999].index
        ngawest2 = ngawest2.drop(idx_m).reset_index(drop=True)
        ngawest2_vert=ngawest2_vert.drop(idx_m).reset_index(drop=True)
        idx_m = ngawest2.loc[ngawest2['Hypocenter Depth (km)']==-999].index
        ngawest2 = ngawest2.drop(idx_m).reset_index(drop=True)
        ngawest2_vert=ngawest2_vert.drop(idx_m).reset_index(drop=True)
        
        # If year not provided assign '0000' to work with datetime
        idx_m = ngawest2.loc[ngawest2['YEAR']=='-999'].index
        ngawest2.loc[idx_m, 'YEAR'] = '0000'
                
        # If month and day not provided assign '1010' to work with datetime
        idx_m = ngawest2.loc[ngawest2['MODY']=='-999'].index
        ngawest2.loc[idx_m,'MODY'] = '000'
        
        # If hours and minutes not provided assign '000' to work with datetime
        ngawest2 = ngawest2.reset_index(drop=True)
        ngawest2_vert = ngawest2_vert.reset_index(drop=True)
        for rec in range(0,len(ngawest2)):
            if ngawest2.loc[rec, 'HRMN']==-999: ngawest2.loc[rec, 'HRMN']='000'
        
        # Remove records with no acceleration values
        idx_m = ngawest2.loc[ngawest2['PGA (g)']==-999].index
        ngawest2 = ngawest2.drop(idx_m).reset_index(drop=True)
        ngawest2_vert = ngawest2_vert.drop(idx_m).reset_index(drop=True)

        # Remove records with no seismic moment to compute moment magnitude from
        idx_m = ngawest2.loc[ngawest2['Mo (dyne.cm)']==-999].index
        ngawest2 = ngawest2.drop(idx_m).reset_index(drop=True)
        ngawest2_vert=ngawest2_vert.drop(idx_m).reset_index(drop=True)
        
        # Remove records with no valid station name
        idx_m = ngawest2.loc[ngawest2['Station Name']==-999].index
        ngawest2 = ngawest2.drop(idx_m)
        ngawest2_vert=ngawest2_vert.drop(idx_m)
        
        # Remove records with no strike, dip or rake angle
        idx_m = ngawest2.loc[ngawest2['Strike (deg)']==-999].index
        ngawest2 = ngawest2.drop(idx_m).reset_index(drop=True)
        ngawest2_vert = ngawest2_vert.drop(idx_m).reset_index(drop=True)
        
        idx_m = ngawest2.loc[ngawest2['Dip (deg)']==-999].index
        ngawest2 = ngawest2.drop(idx_m).reset_index(drop=True)
        ngawest2_vert = ngawest2_vert.drop(idx_m).reset_index(drop=True)
        
        idx_m = ngawest2.loc[ngawest2['Rake Angle (deg)']==-999].index
        ngawest2 = ngawest2.drop(idx_m).reset_index(drop=True)
        ngawest2_vert = ngawest2_vert.drop(idx_m).reset_index(drop=True)
        
        # Replace -999 ztor with empty
        ngawest2['Depth to Top Of Fault Rupture Model'].replace(-999, None)
        
        # Remove records with no epicentral distance
        idx_m = ngawest2.loc[ngawest2['EpiD (km)']==-999].index
        ngawest2 = ngawest2.drop(idx_m).reset_index(drop=True)
        ngawest2_vert = ngawest2_vert.drop(idx_m).reset_index(drop=True)
        
        # If Joyner-Boore, rupture distance, Rx or Ry = -999 reassign as empty
        idx_m = ngawest2.loc[ngawest2['Joyner-Boore Dist. (km)']==-999].index
        ngawest2.loc[idx_m, 'Joyner-Boore Dist. (km)'] = None
    
        idx_m = ngawest2.loc[ngawest2['Campbell R Dist. (km)']==-999].index
        ngawest2.loc[idx_m, 'Campbell R Dist. (km)'] = None

        idx_m = ngawest2.loc[ngawest2['Rx']==-999].index
        ngawest2.loc[idx_m, 'Rx'] = None

        idx_m = ngawest2.loc[ngawest2['Ry 2']==-999].index
        ngawest2.loc[idx_m, 'Ry 2'] = None
        
        # Remove records with no vs30)
        idx_m = ngawest2.loc[ngawest2['Vs30 (m/s) selected for analysis']==-999].index
        ngawest2 = ngawest2.drop(idx_m).reset_index(drop=True)
        ngawest2_vert=ngawest2_vert.drop(idx_m).reset_index(drop=True)
    
        # Replace -999 in 'Owner' with unknown network code
        idx_m = ngawest2.loc[ngawest2['Owner']=='-999'].index
        ngawest2.loc[idx_m, 'Owner'] ='NoNetworkCode'
        ngawest2['Owner'] = 'NetworkCode-' + ngawest2['Owner'] 
        
        # Interpolate between SA(T=4.4s) and SA(T=4.6s) for SA(T=4.5)
        ngawest2['T4.500S'] = (ngawest2['T4.400S']+ngawest2['T4.600S'])/2
        ngawest2_vert['T4.500S'] = (ngawest2_vert['T4.400S'] + ngawest2_vert['T4.600S'])/2        
        
        # Get path to tmp csv containing reformatted dataframe
        tmp = _parse_ngawest2(ngawest2, ngawest2_vert, initial_ngaw2_size)        
        if os.path.exists(output_location):
            raise IOError("Target database directory %s already exists!"
                          % output_location)
        os.mkdir(output_location)
        
        # Add on the records folder
        os.mkdir(os.path.join(output_location, "records"))
        
        # Create an instance of the parser class
        database = cls(dbid, dbname, tmp)
        
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
        wfid = "_".join([metadata["event_id"], metadata["network_code"],
                         metadata["station_code"]])
        wfid = wfid.replace("-", "_")

        # Parse the event metadata
        event = self._parse_event_data(metadata)
        
        # Parse the distance metadata
        distances = self._parse_distances(metadata, event.depth)
        
        # Parse the station metadata
        site = self._parse_site_data(metadata)
        
        # Parse waveform data
        xcomp, ycomp, vert = self._parse_waveform_data(metadata, wfid)
        return GroundMotionRecord(wfid,
                                  [None, None, None],
                                  event, distances, site,
                                  xcomp, ycomp,
                                  vertical=vert)

    def _parse_event_data(self, metadata):
        """
        Parses the event metadata
        """
        # ID and Name (name not in file so use ID again)
        eq_id = metadata["event_id"]
        eq_name = metadata["event_id"]
            
        # Date
        eq_datetime = pd.to_datetime(metadata["event_time"])

        # Latitude, longitude and depth
        eq_lat = valid.latitude(metadata["ev_latitude"])
        eq_lon = valid.longitude(metadata["ev_longitude"])
        eq_depth = valid.positive_float(metadata["ev_depth_km"], "ev_depth_km")
        if not eq_depth:
            raise ValueError('Depth missing an events in admitted flatfile')
        
        eqk = Earthquake(eq_id, eq_name, eq_datetime, eq_lon, eq_lat, eq_depth,
                         magnitude=None, eq_country=None)
        
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
        NGAWest2 only provides Mw so no mag type precedence required
        """
        # Make Magnitude object just for Mw
        mag = Magnitude(float(metadata["Mw"].strip()), "Mw", source=None)

        # Preferred magnitude inherently must be the Mw value
        pref_mag = copy.deepcopy(mag)

        return pref_mag, [mag]

    def _parse_rupture_mechanism(self, metadata, eq_id, eq_name, mag, depth):
        """
        If rupture data is available - parse it, otherwise return None
        """
        # Get the SoF
        sof = metadata["fm_type_code"]

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
            # Has a valid focal mechanism (NGAWest2 flatfile only provides
            # one nodal plane)
            mechanism.nodal_planes.nodal_plane_1 = {
                "strike": fm_set[0], "dip": fm_set[1], "rake": fm_set[2]}

        if not mechanism.nodal_planes.nodal_plane_1:
            # Absolutely no information - base on stye-of-faulting
            mechanism.nodal_planes.nodal_plane_1 = {
                "strike": 0.0, "dip": DIP_TYPE[sof], "rake": MECHANISM_TYPE[sof]
                }
            
        return rupture, mechanism

    def _parse_distances(self, metadata, hypo_depth):
        """
        Parse the distances
        """
        repi = valid.positive_float(metadata["epi_dist"], "epi_dist")
        razim = valid.positive_float(metadata["epi_az"], "epi_az")
        rjb = valid.positive_float(metadata["JB_dist"], "JB_dist")
        rrup = valid.positive_float(metadata["rup_dist"], "rup_dist")
        r_x = valid.vfloat(metadata["Rx_dist"], "Rx_dist")
        rhypo = sqrt(repi ** 2. + hypo_depth ** 2.)
        
        if not isinstance(rjb, float):
            # In the first case Rjb == Repi
            rjb = copy.copy(repi)

        if not isinstance(rrup, float):
            # In the first case Rrup == Rhypo
            rrup = copy.copy(rhypo)

        if not isinstance(r_x, float):
            # In the first case Rx == -Repi (collapse to point and turn off
            # any hanging wall effect)
            r_x = copy.copy(-repi)

        # ngawest2 lacks Ry0 (only Ry) so proxy the first case of Ry0 == Repi
        ry0 = copy.copy(repi)
        
        distances = RecordDistance(repi, rhypo, rjb, rrup, r_x, ry0)
        distances.azimuth = razim

        return distances

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
        elevation = None

        # Vs30
        vs30 = valid.vfloat(metadata["vs30_m_sec"], "vs30_m_sec")
        vs30_measured_flag = metadata["vs30_meas_type"]
        if vs30_measured_flag == "measured":
            vs30_measured = 1
        else:
            vs30_measured = 0 # Either inferred or unknown

        # Make the site object
        site = RecordSite(site_id,
                          station_code,
                          station_code,
                          site_lon,
                          site_lat,
                          elevation,
                          vs30,
                          vs30_measured,
                          network_code=network_code, country=None)

        # Add basin params
        site.z1pt0 = valid.vfloat(metadata["z1pt0 (m)"], "z1pt0 (m)")
        site.z2pt5 = valid.vfloat(metadata["z2pt5 (km)"], "z2pt5 (km)")

        return site

    def _parse_waveform_data(self, metadata, wfid):
        """
        Parse the waveform data
        """
        # U channel - usually east
        xorientation = metadata["U_channel_code"].strip()
        xazimuth = valid.vfloat(metadata["U_azimuth_deg"], "U_azimuth_deg")
        xfilter = {"Low-Cut": valid.vfloat(metadata["U_hp"], "U_hp"),
                   "High-Cut": valid.vfloat(metadata["U_lp"], "U_lp")}
        xcomp = Component(wfid, xazimuth, waveform_filter=xfilter,
                          units="cm/s/s")
        
        # V channel - usually North
        vorientation = metadata["V_channel_code"].strip()
        vazimuth = valid.vfloat(metadata["V_azimuth_deg"], "V_azimuth_deg")
        vfilter = {"Low-Cut": valid.vfloat(metadata["V_hp"], "V_hp"),
                   "High-Cut": valid.vfloat(metadata["V_lp"], "V_lp")}
        vcomp = Component(wfid, vazimuth, waveform_filter=vfilter,
                          units="cm/s/s")
        zorientation = metadata["W_channel_code"].strip()
        if zorientation:
            zfilter = {"Low-Cut": valid.vfloat(metadata["W_hp"], "W_hp"),
                       "High-Cut": valid.vfloat(metadata["W_lp"], "W_lp")}
            zcomp = Component(wfid, None, waveform_filter=zfilter,
                              units="cm/s/s")
        else:
            zcomp = None
        
        return xcomp, vcomp, zcomp

    def _parse_ground_motion(self, location, row, record, headers):
        """
        Parse the ground-motion data
        """
        # Get the data
        scalars, spectra = self._retreive_ground_motion_from_row(row, headers)

        # Build the hdf5 files
        filename = os.path.join(location, "{:s}.hdf5".format(record.id))
        fle = h5py.File(filename, "w-")
        ims_grp = fle.create_group("IMS")
        for comp, key in [("X", "U"), ("Y", "V"), ("V", "W")]:
            comp_grp = ims_grp.create_group(comp)

            # Add on the scalars
            scalar_grp = comp_grp.create_group("Scalar")
            for imt in scalars[key]:
                if imt in ["ia"]:
                    # In the smt convention it is "Ia" for Arias Intensity
                    ikey = imt[0].upper() + imt[1:]
                else:
                    # Everything else to upper case (PGA, PGV, PGD, CAV)
                    ikey = imt.upper()
                dset = scalar_grp.create_dataset(ikey, (1,), dtype="f")
                dset[:] = scalars[key][imt]
            
            # Add on the spectra
            spectra_grp = comp_grp.create_group("Spectra")
            response = spectra_grp.create_group("Response")
            accel = response.create_group("Acceleration")
            accel.attrs["Units"] = "cm/s/s"
            
            # Add on the periods
            pers = spectra[key]["Periods"]
            periods = response.create_dataset("Periods", pers.shape, dtype="f")
            periods[:] = pers
            periods.attrs["Low Period"] = np.min(pers)
            periods.attrs["High Period"] = np.max(pers)
            periods.attrs["Number Periods"] = len(pers)

            # Add on the values
            values = spectra[key]["Values"]
            spectra_dset = accel.create_dataset("damping_05", values.shape, dtype="f")
            spectra_dset[:] = np.copy(values)
            spectra_dset.attrs["Damping"] = 5.0

        # Add on the horizontal values
        hcomp = ims_grp.create_group("H")
        
        # Scalars
        hscalar = hcomp.create_group("Scalar")
        for htype in HCOMPS:
            hcomp_scalars = hscalar.create_group(htype)
            for imt in scalars[htype]:
                if imt in ["ia"]:
                    # In the smt convention it is "Ia" for Arias Intensity
                    key = imt[0].upper() + imt[1:]
                else:
                    # Everything else to upper case (PGA, PGV, PGD, CAV)
                    key = imt.upper()          
                dset = hcomp_scalars.create_dataset(key, (1,), dtype="f")
                dset[:] = scalars[htype][imt]
        
        # Spectra
        hspectra = hcomp.create_group("Spectra")
        hresponse = hspectra.create_group("Response")
        pers = spectra["rotD50"]["Periods"]
        hpers_dset = hresponse.create_dataset("Periods", pers.shape, dtype="f")
        hpers_dset[:] = np.copy(pers)
        hpers_dset.attrs["Low Period"] = np.min(pers)
        hpers_dset.attrs["High Period"] = np.max(pers)
        hpers_dset.attrs["Number Periods"] = len(pers)
        haccel = hresponse.create_group("Acceleration")
        htype_grp = haccel.create_group("rotD50")
        hvals = spectra["rotD50"]["Values"]
        hspec_dset = htype_grp.create_dataset("damping_05", hvals.shape, dtype="f")
        hspec_dset[:] = hvals
        hspec_dset.attrs["Units"] = "cm/s/s"
        record.datafile = filename
        
        return record

    def _retreive_ground_motion_from_row(self, row, header_list):
        """
        Get the ground-motion data from a row (record) in the database
        """
        imts = ["U", "V", "W", "rotD50"] # NOTE: H1 and H2 not used (RotD50 in ngawest2)
        spectra = []
        scalar_imts = ["pga", "pgv", "pgd"]
        scalars = []
        for imt in imts:
            periods = []
            values = []
            key = "{:s}_T".format(imt)
            scalar_dict = {}
            for header in header_list:
                # Deal with the scalar case
                for scalar in scalar_imts:
                    if header == "{:s}_{:s}".format(imt, scalar):
                        # The value is a scalar
                        value = row[header].strip()
                        if value:
                            scalar_dict[scalar] = np.fabs(float(value))
                        else:
                            scalar_dict[scalar] = None
            scalars.append((imt, scalar_dict))
            for header in header_list:
                if key in header:
                    iky = header.replace(key, "").replace("_", ".")
                    periods.append(float(iky))
                    value = row[header].strip()
                    if value:
                        values.append(np.fabs(float(value)))
                    else:
                        values.append(np.nan)
            periods = np.array(periods)
            values = np.array(values)
            idx = np.argsort(periods)
            spectra.append((imt, {"Periods": periods[idx], "Values": values[idx]}))
         
        return dict(scalars), dict(spectra)


def _parse_ngawest2(ngawest2, ngawest2_vert, Initial_ngawest2_size):    
    """
    Convert NGAWest2 flatfile into an ESM format flatfile which can be
    readily parsed into SMT metadata.
    """
    # Reformat/map some of the metadata
    ngawest2['event_time'] = pd.Series()
    ngawest2['event_id'] = pd.Series()
    ngawest2['fm_type'] = pd.Series()
    ngawest2['station_id'] = pd.Series()
    ngawest2['vs30_meas'] = pd.Series()

    for idx, rec in ngawest2.iterrows():
        
        # Event time
        event_time_year = str(rec.YEAR)
        event_time_month_and_day = str(rec.MODY)
        
        if len(event_time_month_and_day) == 3:
            month = str('0') + str(event_time_month_and_day[0])
            day = event_time_month_and_day[1:3]     
        
        if len(event_time_month_and_day) == 4:
            month = str(event_time_month_and_day[:2])
            day = event_time_month_and_day[2:4]
        
        yyyy_mm_dd = str(event_time_year) + '-' + month + '-' + day
        
        event_time_hr_and_min = str(rec.HRMN)
        
        if len(event_time_hr_and_min) == 3:
            hour = str('0') + str(event_time_hr_and_min[0])
            minute = event_time_hr_and_min[1:3]
        
        if len(event_time_hr_and_min) == 4:
            hour = str(event_time_hr_and_min[:2])
            minute = event_time_hr_and_min[2:4]
            
        hh_mm_ss = str(hour) + ':' + str(minute) + ':' + '00'
        
        ngawest2.loc[idx, 'event_time'] = yyyy_mm_dd + ' ' + hh_mm_ss
    
        # Reformat event id
        delimited_event_id = str(rec['Earthquake Name'])
        delimited_event_id = delimited_event_id.replace(',','')
        delimited_event_id = delimited_event_id.replace(' ','')
        delimited_event_id = delimited_event_id.replace('/','')
        delimited_event_id = delimited_event_id.replace('.','')
        delimited_event_id = delimited_event_id.replace(':','')
        delimited_event_id = delimited_event_id.replace(';','')
        ngawest2.loc[idx, 'event_id'] = 'Earthquake-' + delimited_event_id 
        
        # Assign ESM18 fault_code based on code in NGA-West2
        if (rec['Mechanism Based on Rake Angle']==0
            or
            rec['Mechanism Based on Rake Angle']==-999):
            ngawest2.loc[idx, 'fm_type'] = 'SS'
        if (rec['Mechanism Based on Rake Angle']==1
            or
            rec['Mechanism Based on Rake Angle']==4):
            ngawest2.loc[idx, 'fm_type'] = 'NF'
        if (rec['Mechanism Based on Rake Angle']==2
            or
            rec['Mechanism Based on Rake Angle']==3):
            ngawest2.loc[idx, 'fm_type'] = 'TF'
        
        # Vs30 meas flag (Appendix C, pp. 116 of NGAWest2 report)
        ngawest2.loc[idx, 'vs30_meas'] = 'measured' if rec['Measured/Inferred Class'] == 0 else 'inferred'

        # Station id
        delimited_station_id = str(rec['Station Name'])
        delimited_station_id = delimited_station_id.replace(',','')
        delimited_station_id = delimited_station_id.replace(' ','')
        delimited_station_id = delimited_station_id.replace('/','')
        delimited_station_id = delimited_station_id.replace('.','')
        delimited_station_id = delimited_station_id.replace(':','')
        delimited_station_id = delimited_station_id.replace(';','')
        ngawest2.loc[idx, 'station_id'] = 'StationName-' + delimited_station_id
    
    # Construct dataframe with ESM18 format columns
    rfmt = pd.DataFrame(
    {
    # Non-GMIM headers   
    "event_id":ngawest2['event_id'],                                       
    "event_time":ngawest2['event_time'],
    "ev_latitude":ngawest2['Hypocenter Latitude (deg)'],    
    "ev_longitude":ngawest2['Hypocenter Longitude (deg)'],   
    "ev_depth_km":ngawest2['Hypocenter Depth (km)'],
    "fm_type_code":ngawest2['fm_type'],
    "Mw":ngawest2['Earthquake Magnitude'],

    "es_strike":ngawest2['Strike (deg)'],
    "es_dip":ngawest2['Dip (deg)'],
    "es_rake":ngawest2['Rake Angle (deg)'],
    "es_z_top":ngawest2['Depth to Top Of Fault Rupture Model'],
    "es_length":ngawest2['Fault Rupture Length for Calculation of Ry (km)'],   
    "es_width":ngawest2['Fault Rupture Width (km)'],
 
    "network_code": ngawest2['Owner'],
    "station_code":ngawest2['station_id'],
    "st_latitude":ngawest2['Station Latitude'],
    "st_longitude":ngawest2['Station Longitude'],   
    "vs30_m_sec":ngawest2['Vs30 (m/s) selected for analysis'],
    "vs30_meas_type":ngawest2['vs30_meas'],
    "z1pt0 (m)":ngawest2["Northern CA/Southern CA - H11 Z1 (m)"], # No preference is given between the H11 and S4 CVM models but the H11 model has covers more of the Southern California stations
    "z2pt5 (km)":ngawest2["Northern CA/Southern CA - H11 Z2.5 (m)"]/1000, # Provided in metres
 
    "epi_dist":ngawest2['EpiD (km)'],
    'epi_az':ngawest2['Source to Site Azimuth (deg)'],
    "JB_dist":ngawest2['Joyner-Boore Dist. (km)'],
    "rup_dist":ngawest2['Campbell R Dist. (km)'],
    "Rx_dist":ngawest2['Rx'],
 
    "U_channel_code":"H1",
    "U_azimuth_deg":ngawest2['H1 azimth (degrees)'],
    "V_channel_code":"H2",
    "V_azimuth_deg":ngawest2['H2 azimith (degrees)'],
    "W_channel_code":"V",

    "U_hp":ngawest2['HP-H1 (Hz)'],
    "V_hp":ngawest2['HP-H2 (Hz)'],
    "W_hp":ngawest2_vert['HP-V (Hz)'],  
    "U_lp":ngawest2['LP-H1 (Hz)'],
    "V_lp":ngawest2['LP-H2 (Hz)'],
    "W_lp":ngawest2_vert['LP-V (Hz)'], 

    "U_pga":None,
    "V_pga":None,
    "W_pga":ngawest2_vert['PGA (g)'] * CONV_TO_CMS2,
    "rotD50_pga":ngawest2['PGA (g)'] * CONV_TO_CMS2,
    "U_pgv":None,
    "V_pgv":None,
    "W_pgv":ngawest2_vert['PGV (cm/sec)'],
    "rotD50_pgv":ngawest2['PGV (cm/sec)'],
    "U_pgd":None,
    "V_pgd":None,
    "W_pgd":ngawest2_vert['PGD (cm)'],
    "rotD50_pgd":ngawest2['PGD (cm)'],
        
    "U_T0_010":None,
    "U_T0_025":None,
    "U_T0_040":None,
    "U_T0_050":None,
    "U_T0_070":None,
    "U_T0_100":None,
    "U_T0_150":None,
    "U_T0_200":None,
    "U_T0_250":None,
    "U_T0_300":None,
    "U_T0_350":None,
    "U_T0_400":None,
    "U_T0_450":None,
    "U_T0_500":None,
    "U_T0_600":None,
    "U_T0_700":None,
    "U_T0_750":None,
    "U_T0_800":None,
    "U_T0_900":None,
    "U_T1_000":None,
    "U_T1_200":None,
    "U_T1_400":None,
    "U_T1_600":None,
    "U_T1_800":None,
    "U_T2_000":None,
    "U_T2_500":None,
    "U_T3_000":None,
    "U_T3_500":None,
    "U_T4_000":None,
    "U_T4_500":None,
    "U_T5_000":None,
    "U_T6_000":None,
    "U_T7_000":None,
    "U_T8_000":None,
    "U_T9_000":None,
    "U_T10_000":None,
    
    "V_T0_010":None,
    "V_T0_025":None,
    "V_T0_040":None,
    "V_T0_050":None,
    "V_T0_070":None,
    "V_T0_100":None,
    "V_T0_150":None,
    "V_T0_200":None,
    "V_T0_250":None,
    "V_T0_300":None,
    "V_T0_350":None,
    "V_T0_400":None,
    "V_T0_450":None,
    "V_T0_500":None,
    "V_T0_600":None,
    "V_T0_700":None,
    "V_T0_750":None,
    "V_T0_800":None,
    "V_T0_900":None,
    "V_T1_000":None,
    "V_T1_200":None,
    "V_T1_400":None,
    "V_T1_600":None,
    "V_T1_800":None,
    "V_T2_000":None,
    "V_T2_500":None,
    "V_T3_000":None,
    "V_T3_500":None,
    "V_T4_000":None,
    "V_T4_500":None,
    "V_T5_000":None,
    "V_T6_000":None,
    "V_T7_000":None,
    "V_T8_000":None,
    "V_T9_000":None,
    "V_T10_000":None,

    "rotD50_T0_010":ngawest2['T0.010S'] * CONV_TO_CMS2,
    "rotD50_T0_025":ngawest2['T0.025S'] * CONV_TO_CMS2,
    "rotD50_T0_040":ngawest2['T0.040S'] * CONV_TO_CMS2,
    "rotD50_T0_050":ngawest2['T0.050S'] * CONV_TO_CMS2,
    "rotD50_T0_070":ngawest2['T0.070S'] * CONV_TO_CMS2,
    "rotD50_T0_100":ngawest2['T0.100S'] * CONV_TO_CMS2,
    "rotD50_T0_150":ngawest2['T0.150S'] * CONV_TO_CMS2,
    "rotD50_T0_200":ngawest2['T0.200S'] * CONV_TO_CMS2,
    "rotD50_T0_250":ngawest2['T0.250S'] * CONV_TO_CMS2,
    "rotD50_T0_300":ngawest2['T0.300S'] * CONV_TO_CMS2,
    "rotD50_T0_350":ngawest2['T0.350S'] * CONV_TO_CMS2,
    "rotD50_T0_400":ngawest2['T0.400S'] * CONV_TO_CMS2,
    "rotD50_T0_450":ngawest2['T0.450S'] * CONV_TO_CMS2,
    "rotD50_T0_500":ngawest2['T0.500S'] * CONV_TO_CMS2,
    "rotD50_T0_600":ngawest2['T0.600S'] * CONV_TO_CMS2,
    "rotD50_T0_700":ngawest2['T0.700S'] * CONV_TO_CMS2,
    "rotD50_T0_750":ngawest2['T0.750S'] * CONV_TO_CMS2,
    "rotD50_T0_800":ngawest2['T0.800S'] * CONV_TO_CMS2,
    "rotD50_T0_900":ngawest2['T0.900S'] * CONV_TO_CMS2,
    "rotD50_T1_000":ngawest2['T1.000S'] * CONV_TO_CMS2,
    "rotD50_T1_200":ngawest2['T1.200S'] * CONV_TO_CMS2,
    "rotD50_T1_400":ngawest2['T1.400S'] * CONV_TO_CMS2,
    "rotD50_T1_600":ngawest2['T1.600S'] * CONV_TO_CMS2,
    "rotD50_T1_800":ngawest2['T1.800S'] * CONV_TO_CMS2,
    "rotD50_T2_000":ngawest2['T2.000S'] * CONV_TO_CMS2,
    "rotD50_T2_500":ngawest2['T2.500S'] * CONV_TO_CMS2,
    "rotD50_T3_000":ngawest2['T3.000S'] * CONV_TO_CMS2,
    "rotD50_T3_500":ngawest2['T3.500S'] * CONV_TO_CMS2,
    "rotD50_T4_000":ngawest2['T4.000S'] * CONV_TO_CMS2,
    "rotD50_T4_500":ngawest2['T4.500S'] * CONV_TO_CMS2,
    "rotD50_T5_000":ngawest2['T5.000S'] * CONV_TO_CMS2,
    "rotD50_T6_000":ngawest2['T6.000S'] * CONV_TO_CMS2,
    "rotD50_T7_000":ngawest2['T7.000S'] * CONV_TO_CMS2,
    "rotD50_T8_000":ngawest2['T8.000S'] * CONV_TO_CMS2,
    "rotD50_T9_000":ngawest2['T9.000S'] * CONV_TO_CMS2,
    "rotD50_T10_000":ngawest2['T10.000S'] * CONV_TO_CMS2,
        
    "W_T0_010":ngawest2_vert['T0.010S'] * CONV_TO_CMS2,
    "W_T0_025":ngawest2_vert['T0.025S'] * CONV_TO_CMS2,
    "W_T0_040":ngawest2_vert['T0.040S'] * CONV_TO_CMS2,
    "W_T0_050":ngawest2_vert['T0.050S'] * CONV_TO_CMS2,
    "W_T0_070":ngawest2_vert['T0.070S'] * CONV_TO_CMS2,
    "W_T0_100":ngawest2_vert['T0.100S'] * CONV_TO_CMS2,
    "W_T0_150":ngawest2_vert['T0.150S'] * CONV_TO_CMS2,
    "W_T0_200":ngawest2_vert['T0.200S'] * CONV_TO_CMS2,
    "W_T0_250":ngawest2_vert['T0.250S'] * CONV_TO_CMS2,
    "W_T0_300":ngawest2_vert['T0.300S'] * CONV_TO_CMS2,
    "W_T0_350":ngawest2_vert['T0.350S'] * CONV_TO_CMS2,
    "W_T0_400":ngawest2_vert['T0.400S'] * CONV_TO_CMS2,
    "W_T0_450":ngawest2_vert['T0.450S'] * CONV_TO_CMS2,
    "W_T0_500":ngawest2_vert['T0.500S'] * CONV_TO_CMS2,
    "W_T0_600":ngawest2_vert['T0.600S'] * CONV_TO_CMS2,
    "W_T0_700":ngawest2_vert['T0.700S'] * CONV_TO_CMS2,
    "W_T0_750":ngawest2_vert['T0.750S'] * CONV_TO_CMS2,
    "W_T0_800":ngawest2_vert['T0.800S'] * CONV_TO_CMS2,
    "W_T0_900":ngawest2_vert['T0.900S'] * CONV_TO_CMS2,
    "W_T1_000":ngawest2_vert['T1.000S'] * CONV_TO_CMS2,
    "W_T1_200":ngawest2_vert['T1.200S'] * CONV_TO_CMS2,
    "W_T1_400":ngawest2_vert['T1.400S'] * CONV_TO_CMS2,
    "W_T1_600":ngawest2_vert['T1.600S'] * CONV_TO_CMS2,
    "W_T1_800":ngawest2_vert['T1.800S'] * CONV_TO_CMS2,
    "W_T2_000":ngawest2_vert['T2.000S'] * CONV_TO_CMS2,
    "W_T2_500":ngawest2_vert['T2.500S'] * CONV_TO_CMS2,
    "W_T3_000":ngawest2_vert['T3.000S'] * CONV_TO_CMS2,
    "W_T3_500":ngawest2_vert['T3.500S'] * CONV_TO_CMS2,
    "W_T4_000":ngawest2_vert['T4.000S'] * CONV_TO_CMS2,
    "W_T4_500":ngawest2_vert['T4.500S'] * CONV_TO_CMS2,
    "W_T5_000":ngawest2_vert['T5.000S'] * CONV_TO_CMS2,
    "W_T6_000":ngawest2_vert['T6.000S'] * CONV_TO_CMS2,
    "W_T7_000":ngawest2_vert['T7.000S'] * CONV_TO_CMS2,
    "W_T8_000":ngawest2_vert['T8.000S'] * CONV_TO_CMS2,
    "W_T9_000":ngawest2_vert['T9.000S'] * CONV_TO_CMS2,
    "W_T10_000":ngawest2_vert['T10.000S'] * CONV_TO_CMS2})
    
    # Make tmp file 
    tmp = os.path.join(BASE, tempfile.mkdtemp(), 'tmp.csv')
    
    # Export to tmp
    rfmt.to_csv(tmp, sep=';')

    # Inform user of number of discarded records (insufficient for SMT residual analysis)
    print(Initial_ngawest2_size - len(ngawest2),
          'records removed from imported NGA-West-2 flatfile during data quality checks.')

    return tmp