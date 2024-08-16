# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2024 GEM Foundation and G. Weatherill
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
Parser for the NGAWest2 flatfile format
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
from collections import OrderedDict

from openquake.smt.residuals.sm_database import (
    GroundMotionDatabase, GroundMotionRecord, Earthquake, Magnitude, Rupture,
    FocalMechanism, GCMTNodalPlanes, Component, RecordSite, RecordDistance)
from openquake.smt.utils_strong_motion import (
    MECHANISM_TYPE, DIP_TYPE, vs30_to_z1pt0_cy14, vs30_to_z2pt5_cb14)
from openquake.smt.residuals.parsers.base_database_parser import \
    SMDatabaseReader
from openquake.smt.residuals.parsers import valid

# Import the ESM dictionaries
from .esm_dictionaries import *

SCALAR_LIST = ["PGA", "PGV", "PGD"]

HEADER_STR = "event_id;event_time;ev_nation_code;ev_latitude;ev_longitude;"\
             "ev_depth_km;fm_type_code;ML;ML_ref;Mw;Mw_ref;Ms;Ms_ref;EMEC_Mw;"\
             "EMEC_Mw_type;EMEC_Mw_ref;event_source_id;es_strike;es_dip;"\
             "es_rake;es_strike_dip_rake_ref;es_z_top;es_length;es_width;"\
             "network_code;station_code;location_code;instrument_code;"\
             "sensor_depth_m;proximity_code;housing_code;st_nation_code;"\
             "st_latitude;st_longitude;st_elevation;vs30_m_sec;slope_deg;"\
             "vs30_m_sec_WA;epi_dist;epi_az;JB_dist;rup_dist;Rx_dist;Ry0_dist;"\
             "instrument_type_code;late_triggered_flag_01;U_channel_code;"\
             "U_azimuth_deg;V_channel_code;V_azimuth_deg;W_channel_code;"\
             "U_hp;V_hp;W_hp;U_lp;V_lp;W_lp"

HEADERS = set(HEADER_STR.split(";"))


class NGAWest2FlatfileParser(SMDatabaseReader):
    """
    Parses the data from flatfile to a set of metadata objects
    """
    M_PRECEDENCE = ["EMEC_Mw", "Mw", "Ms", "ML"]
    BUILD_FINITE_DISTANCES = False

    def parse(self, location='./'):
        """
        Parse the metadata
        """
        assert os.path.isfile(self.filename)
        headers = getline(self.filename, 1).rstrip("\n").split(";")
        for hdr in HEADERS:
            if hdr not in headers:
                raise ValueError("Required header %s is missing in file"
                                 % hdr)
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
                    os.path.join(location, "records"),
                    row, record, headers)
                self.database.records.append(record)

            else:
                print("Record with sequence number %s is null/invalid"
                      % "{:s}-{:s}".format(row["event_id"],
                                           row["station_code"]))
            if (counter % 100) == 0:
                print("Processed record %s - %s" % (str(counter),
                                                    record.id))

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
        for rec in range(0,len(ngawest2)):
            if ngawest2['Record Sequence Number'].iloc[rec]!=\
                ngawest2_vert['Record Sequence Number'].iloc[rec]:
                raise ValueError(
                    "Records within horz. and vert. do not match.")
        
        # Count initial size for printing number records removed during checks
        initial_ngaw2_size = len(ngawest2)

        # Remove potential duplicate records in NGA-West2 flatfile
        ngawest2 = ngawest2.drop_duplicates(
            subset = ['Earthquake Name','Station Name'], keep='last')
        ngawest2_vert = ngawest2_vert.drop_duplicates(
            subset = ['Earthquake Name', 'Station Name'], keep='last')
        ngawest2 = ngawest2.reset_index().drop(columns='index')
        ngawest2_vert = ngawest2_vert.reset_index().drop(columns='index')

        # Remove records if earthquake not identifiable using lat/lon metadata
        idx_m = ngawest2.loc[ngawest2['Hypocenter Latitude (deg)']==-999].index
        ngawest2 = ngawest2.drop(idx_m)
        ngawest2_vert = ngawest2_vert.drop(idx_m)
        ngawest2 = ngawest2.reset_index().drop(columns='index')
        ngawest2_vert = ngawest2_vert.reset_index().drop(columns='index')
        idx_m = ngawest2.loc[ngawest2['Hypocenter Longitude (deg)']==-999].index
        ngawest2 = ngawest2.drop(idx_m)
        ngawest2_vert=ngawest2_vert.drop(idx_m)
        ngawest2 = ngawest2.reset_index().drop(columns='index')
        ngawest2_vert = ngawest2_vert.reset_index().drop(columns='index')
        idx_m = ngawest2.loc[ngawest2['Hypocenter Depth (km)']==-999].index
        ngawest2 = ngawest2.drop(idx_m)
        ngawest2_vert=ngawest2_vert.drop(idx_m)
        ngawest2 = ngawest2.reset_index().drop(columns='index')
        ngawest2_vert = ngawest2_vert.reset_index().drop(columns='index')
        
        # If year not provided assign '0000' to work with datetime
        idx_m = ngawest2.loc[ngawest2['YEAR']=='-999'].index
        ngawest2['YEAR'].iloc[idx_m] = '0000'
                
        # If month and day not provided assign '1010' to work with datetime
        idx_m = ngawest2.loc[ngawest2['MODY']=='-999'].index
        ngawest2['MODY'].iloc[idx_m] = '000'
        
        # If hours and minutes not provided assign '000' to work with datetime
        ngawest2 = ngawest2.reset_index().drop(columns='index')
        ngawest2_vert = ngawest2_vert.reset_index().drop(columns='index')
        for rec in range(0,len(ngawest2)):
            if ngawest2['HRMN'][rec]==-999: ngawest2['HRMN'][rec]='000'
        
        # Remove records with no acceleration values
        idx_m = ngawest2.loc[ngawest2['PGA (g)']==-999].index
        ngawest2 = ngawest2.drop(idx_m)
        ngawest2_vert = ngawest2_vert.drop(idx_m)
        ngawest2 = ngawest2.reset_index().drop(columns='index')
        ngawest2_vert = ngawest2_vert.reset_index().drop(columns='index')

        # Remove records with no seismic moment to compute moment magnitude from
        idx_m = ngawest2.loc[ngawest2['Mo (dyne.cm)']==-999].index
        ngawest2 = ngawest2.drop(idx_m)
        ngawest2_vert=ngawest2_vert.drop(idx_m)
        ngawest2 = ngawest2.reset_index().drop(columns='index')
        ngawest2_vert = ngawest2_vert.reset_index().drop(columns='index')
        
        # Remove records with no valid station name
        idx_m = ngawest2.loc[ngawest2['Station Name']==-999].index
        ngawest2 = ngawest2.drop(idx_m)
        ngawest2_vert=ngawest2_vert.drop(idx_m)
        
        # Remove records with no strike, dip or rake angle
        idx_m = ngawest2.loc[ngawest2['Strike (deg)']==-999].index
        ngawest2 = ngawest2.drop(idx_m)
        ngawest2_vert = ngawest2_vert.drop(idx_m)
        ngawest2 = ngawest2.reset_index().drop(columns='index')
        ngawest2_vert = ngawest2_vert.reset_index().drop(columns='index')
        
        idx_m = ngawest2.loc[ngawest2['Dip (deg)']==-999].index
        ngawest2 = ngawest2.drop(idx_m)
        ngawest2_vert = ngawest2_vert.drop(idx_m)
        ngawest2 = ngawest2.reset_index().drop(columns='index')
        ngawest2_vert = ngawest2_vert.reset_index().drop(columns='index')
        
        idx_m = ngawest2.loc[ngawest2['Rake Angle (deg)']==-999].index
        ngawest2 = ngawest2.drop(idx_m)
        ngawest2_vert = ngawest2_vert.drop(idx_m)
        ngawest2 = ngawest2.reset_index().drop(columns='index')
        ngawest2_vert = ngawest2_vert.reset_index().drop(columns='index')
        
        # Replace -999 ztor with empty
        ngawest2['Depth to Top Of Fault Rupture Model'].replace(-999, None)
        
        # Remove records with no epicentral distance
        idx_m = ngawest2.loc[ngawest2['EpiD (km)']==-999].index
        ngawest2 = ngawest2.drop(idx_m)
        ngawest2_vert = ngawest2_vert.drop(idx_m)
        ngawest2 = ngawest2.reset_index().drop(columns='index')
        ngawest2_vert = ngawest2_vert.reset_index().drop(columns='index')
        
        # If Joyner-Boore, rupture distance, Rx or Ry = -999 reassign as empty
        idx_m = ngawest2.loc[ngawest2['Joyner-Boore Dist. (km)']==-999].index
        ngawest2['Joyner-Boore Dist. (km)'][idx_m] = None
    
        idx_m = ngawest2.loc[ngawest2['Campbell R Dist. (km)']==-999].index
        ngawest2['Campbell R Dist. (km)'][idx_m] = None

        idx_m = ngawest2.loc[ngawest2['Rx']==-999].index
        ngawest2['Rx'][idx_m] = None

        idx_m = ngawest2.loc[ngawest2['Ry 2']==-999].index
        ngawest2['Ry 2'][idx_m] = None
        
        # Remove records with no vs30)
        idx_m = ngawest2.loc[ngawest2['Vs30 (m/s) selected for analysis']==-999].index
        ngawest2 = ngawest2.drop(idx_m)
        ngawest2_vert=ngawest2_vert.drop(idx_m)
        ngawest2 = ngawest2.reset_index().drop(columns='index')
        ngawest2_vert = ngawest2_vert.reset_index().drop(columns='index')
        
        # Compute Mw from seismic moment using Hanks and Kamori
        ngawest2['Earthquake Magnitude'] = (np.log10(ngawest2[
            'Mo (dyne.cm)']) - 16.05)/1.5 # From ngawest2 database rep. pp.122
        
        ngawest2 = ngawest2.reset_index().drop(columns='index')
        ngawest2_vert = ngawest2_vert.reset_index().drop(
            columns='index')
      
        # Replace -999 in 'Owner' with unknown network code
        idx_m = ngawest2.loc[ngawest2['Owner']=='-999'].index
        ngawest2['Owner'].iloc[idx_m]='NoNetworkCode'
        ngawest2['Owner'] = 'NetworkCode-'+ngawest2['Owner'] 
        
        # Interpolate between SA(T=4.4s) and SA(T=4.6s) for SA(T=4.5)
        ngawest2['T4.500S'] = (ngawest2['T4.400S']+ngawest2['T4.600S'])/2
        ngawest2_vert['T4.500S'] = (ngawest2_vert['T4.400S'] + ngawest2_vert['T4.600S'])/2        
        
        # Get path to tmp csv once modified dataframe
        converted_base_data_path=_get_ESM18_headers(
            ngawest2, ngawest2_vert, initial_ngaw2_size)
                
        if os.path.exists(output_location):
            raise IOError("Target database directory %s already exists!"
                          % output_location)
        os.mkdir(output_location)
        # Add on the records folder
        os.mkdir(os.path.join(output_location, "records"))
        # Create an instance of the parser class
        database = cls(dbid, dbname, converted_base_data_path)
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
                         metadata["station_code"], metadata["location_code"]])
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
                                  vert=vert)

    def _parse_event_data(self, metadata):
        """
        Parses the event metadata
        """
        # ID and Name (name not in file so use ID again)
        eq_id = metadata["event_id"]
        eq_name = metadata["event_id"]
            
        # Date and time
        eq_datetime = valid.date_time(metadata["event_time"],
                                     "%Y-%m-%d %H:%M:%S")
        # Latitude, longitude and depth
        eq_lat = valid.latitude(metadata["ev_latitude"])
        eq_lon = valid.longitude(metadata["ev_longitude"])
        eq_depth = valid.positive_float(metadata["ev_depth_km"], "ev_depth_km")
        if not eq_depth:
            raise ValueError('Depth missing for one or more events in flatfile')
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
        So, here things get tricky. Up to four magnitudes are defined in the
        flatfile (EMEC Mw, MW, Ms and ML). An order of precedence is required
        and the preferred magnitude will be the highest found
        """
        pref_mag = None
        mag_list = []
        for key in self.M_PRECEDENCE:
            mvalue = metadata[key].strip()
            if mvalue:
                if key == "EMEC_Mw":
                    mtype = "Mw"
                    msource = "EMEC({:s}|{:s})".format(
                        metadata["EMEC_Mw_type"],
                        metadata["EMEC_Mw_ref"])
                else:
                    mtype = key
                    msource = metadata[key + "_ref"].strip()
                mag = Magnitude(float(mvalue),
                                mtype,
                                source=msource)
                if not pref_mag:
                    pref_mag = copy.deepcopy(mag)
                mag_list.append(mag)
        return pref_mag, mag_list

    def _parse_rupture_mechanism(self, metadata, eq_id, eq_name, mag, depth):
        """
        If rupture data is available - parse it, otherwise return None
        """

        sof = metadata["fm_type_code"]
        if not metadata["event_source_id"].strip():
            # No rupture model available. Mechanism is limited to a style
            # of faulting only
            rupture = Rupture(eq_id, eq_name, mag, None, None, depth)
            mechanism = FocalMechanism(
                eq_id, eq_name, GCMTNodalPlanes(), None,
                mechanism_type=sof)
            # See if focal mechanism exists
            fm_set = []
            for key in ["strike_1", "dip_1", "rake_1"]:
                if key in metadata:
                    fm_param = valid.vfloat(metadata[key], key)
                    if fm_param is not None:
                        fm_set.append(fm_param)
            if len(fm_set) == 3:
                # Have one valid focal mechanism
                mechanism.nodal_planes.nodal_plane_1 = {"strike": fm_set[0],
                                                        "dip": fm_set[1],
                                                        "rake": fm_set[2]}
            fm_set = []
            for key in ["strike_2", "dip_2", "rake_2"]:
                if key in metadata:
                    fm_param = valid.vfloat(metadata[key], key)
                    if fm_param is not None:
                        fm_set.append(fm_param)
            if len(fm_set) == 3:
                # Have one valid focal mechanism
                mechanism.nodal_planes.nodal_plane_2 = {"strike": fm_set[0],
                                                        "dip": fm_set[1],
                                                        "rake": fm_set[2]}

            if not mechanism.nodal_planes.nodal_plane_1 and not\
                mechanism.nodal_planes.nodal_plane_2:
                # Absolutely no information - base on stye-of-faulting
                mechanism.nodal_planes.nodal_plane_1 = {
                    "strike": 0.0,  # Basically unused
                    "dip": DIP_TYPE[sof],
                    "rake": MECHANISM_TYPE[sof]
                    }
            return rupture, mechanism

        strike = valid.strike(metadata["es_strike"])
        dip = valid.dip(metadata["es_dip"])
        rake = valid.rake(metadata["es_rake"])
        ztor = valid.positive_float(metadata["es_z_top"], "es_z_top")
        length = valid.positive_float(metadata["es_length"], "es_length")
        width = valid.positive_float(metadata["es_width"], "es_width")
        rupture = Rupture(eq_id, eq_name, mag, length, width, ztor)

        # Get mechanism type and focal mechanism
        # No nodal planes, eigenvalues moment tensor initially
        mechanism = FocalMechanism(
            eq_id, eq_name, GCMTNodalPlanes(), None,
            mechanism_type=metadata["fm_type_code"])
        if strike is None:
            strike = 0.0
        if dip is None:
            dip = DIP_TYPE[sof]
        if rake is None:
            rake = MECHANISM_TYPE[sof]
        # if strike is not None and dip is not None and rake is not None:
        mechanism.nodal_planes.nodal_plane_1 = {"strike": strike,
                                                "dip": dip,
                                                "rake": rake}
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
        ry0 = valid.positive_float(metadata["Ry0_dist"], "Ry0_dist")
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

        if not isinstance(ry0, float):
            # In the first case Ry0 == Repi
            ry0 = copy.copy(repi)
        distances = RecordDistance(repi, rhypo, rjb, rrup, r_x, ry0)
        distances.azimuth = razim
        return distances

    def _parse_site_data(self, metadata):
        """
        Parses the site information
        """
        network_code = metadata["network_code"].strip()
        station_code = metadata["station_code"].strip()
        site_id = "{:s}-{:s}".format(network_code, station_code)
        site_lon = valid.longitude(metadata["st_longitude"])
        site_lat = valid.latitude(metadata["st_latitude"])
        elevation = valid.vfloat(metadata["st_elevation"], "st_elevation")

        vs30 = valid.vfloat(metadata["vs30_m_sec"], "vs30_m_sec")
        vs30_topo = valid.vfloat(metadata["vs30_m_sec_WA"], "vs30_m_sec_WA")
        if vs30:
            vs30_measured = True
        elif vs30_topo:
            vs30 = vs30_topo
            vs30_measured = False
        else:
            vs30_measured = False
        site = RecordSite(site_id, station_code, station_code, site_lon,
                          site_lat, elevation, vs30, vs30_measured,
                          network_code=network_code, country=None)
        site.slope = valid.vfloat(metadata["slope_deg"], "slope_deg")
        site.sensor_depth = valid.vfloat(metadata["sensor_depth_m"],
                                         "sensor_depth_m")
        site.instrument_type = metadata["instrument_code"].strip()
        if site.vs30:
            site.z1pt0 = vs30_to_z1pt0_cy14(vs30)
            site.z2pt5 = vs30_to_z2pt5_cb14(vs30)
        housing_code = metadata["housing_code"].strip()
        if housing_code and (housing_code in HOUSING):
            site.building_structure = HOUSING[housing_code]
        return site

    def _parse_waveform_data(self, metadata, wfid):
        """
        Parse the waveform data
        """
        late_trigger = valid.vint(metadata["late_triggered_flag_01"],
                                  "late_triggered_flag_01")
        # U channel - usually east
        xorientation = metadata["U_channel_code"].strip()
        xazimuth = valid.vfloat(metadata["U_azimuth_deg"], "U_azimuth_deg")
        xfilter = {"Low-Cut": valid.vfloat(metadata["U_hp"], "U_hp"),
                   "High-Cut": valid.vfloat(metadata["U_lp"], "U_lp")}
        xcomp = Component(wfid, xazimuth, waveform_filter=xfilter,
                          units="cm/s/s")
        xcomp.late_trigger = late_trigger
        # V channel - usually North
        vorientation = metadata["V_channel_code"].strip()
        vazimuth = valid.vfloat(metadata["V_azimuth_deg"], "V_azimuth_deg")
        vfilter = {"Low-Cut": valid.vfloat(metadata["V_hp"], "V_hp"),
                   "High-Cut": valid.vfloat(metadata["V_lp"], "V_lp")}
        vcomp = Component(wfid, vazimuth, waveform_filter=vfilter,
                          units="cm/s/s")
        vcomp.late_trigger = late_trigger
        zorientation = metadata["W_channel_code"].strip()
        if zorientation:
            zfilter = {"Low-Cut": valid.vfloat(metadata["W_hp"], "W_hp"),
                       "High-Cut": valid.vfloat(metadata["W_lp"], "W_lp")}
            zcomp = Component(wfid, None, waveform_filter=zfilter,
                              units="cm/s/s")
            zcomp.late_trigger = late_trigger
        else:
            zcomp = None
        
        return xcomp, vcomp, zcomp

    def _parse_ground_motion(self, location, row, record, headers):
        """
        In this case we parse the information from the flatfile directly
        to hdf5 at the metadata stage
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
                if imt in ["ia", "housner"]:
                    # In the smt convention it is "Ia" and "Housner"
                    ikey = imt[0].upper() + imt[1:]
                else:
                    # Everything else to upper case (PGA, PGV, PGD, T90, CAV)
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
            spectra_dset = accel.create_dataset("damping_05", values.shape,
                                                dtype="f")
            spectra_dset[:] = np.copy(values)
            spectra_dset.attrs["Damping"] = 5.0
        # Add on the horz values
        hcomp = ims_grp.create_group("H")
        # Scalars - just geometric mean for now
        hscalar = hcomp.create_group("Scalar")
        for imt in scalars["Geometric"]:
            if imt in ["ia", "housner"]:
                # In the smt convention it is "Ia" and "Housner"
                key = imt[0].upper() + imt[1:]
            else:
                # Everything else to upper case (PGA, PGV, PGD, T90, CAV)
                key = imt.upper()
            dset = hscalar.create_dataset(key, (1,), dtype="f")
            dset[:] = scalars["Geometric"][imt]
        # For Spectra - can support multiple components
        hspectra = hcomp.create_group("Spectra")
        hresponse = hspectra.create_group("Response")
        pers = spectra["Geometric"]["Periods"]
        hpers_dset = hresponse.create_dataset("Periods", pers.shape, dtype="f")
        hpers_dset[:] = np.copy(pers)
        hpers_dset.attrs["Low Period"] = np.min(pers)
        hpers_dset.attrs["High Period"] = np.max(pers)
        hpers_dset.attrs["Number Periods"] = len(pers)
        haccel = hresponse.create_group("Acceleration")
        for htype in ["Geometric", "rotD00", "rotD50", "rotD100"]:
            if np.all(np.isnan(spectra[htype]["Values"])):
                # Component not determined
                continue
            if not (htype == "Geometric"):
                key = htype[0].upper() + htype[1:]
            else:
                key = copy.deepcopy(htype)
            htype_grp = haccel.create_group(htype)
            hvals = spectra[htype]["Values"]
            hspec_dset = htype_grp.create_dataset("damping_05", hvals.shape,
                                                  dtype="f")
            hspec_dset[:] = hvals
            hspec_dset.attrs["Units"] = "cm/s/s"
        record.datafile = filename
        return record


    def _retreive_ground_motion_from_row(self, row, header_list):
        """
        Get the ground-motion data from a row (record) in the database
        """
        imts = ["U", "V", "W", "rotD00", "rotD100", "rotD50"]
        spectra = []
        scalar_imts = ["pga", "pgv", "pgd", "T90", "housner", "ia", "CAV"]
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
                    if header == "{:s}90".format(key):
                        # Not a spectral period but T90
                        continue
                    iky = header.replace(key, "").replace("_", ".")
                    #print imt, key, header, iky
                    periods.append(float(iky))
                    value = row[header].strip()
                    if value:
                        values.append(np.fabs(float(value)))
                    else:
                        values.append(np.nan)
            periods = np.array(periods)
            values = np.array(values)
            idx = np.argsort(periods)
            spectra.append((imt, {"Periods": periods[idx],
                                   "Values": values[idx]}))
        # Add on the as-recorded geometric mean
        spectra = OrderedDict(spectra)
        scalars = OrderedDict(scalars)
        spectra["Geometric"] = {
            "Values": np.sqrt(spectra["U"]["Values"] *
                              spectra["V"]["Values"]),
            "Periods": np.copy(spectra["U"]["Periods"])
            }
        scalars["Geometric"] = dict([(key, None) for key in scalars["U"]])
        for key in scalars["U"]:
            if scalars["U"][key] and scalars["V"][key]:
                scalars["Geometric"][key] = np.sqrt(
                    scalars["U"][key] * scalars["V"][key])
        return scalars, spectra


def _get_ESM18_headers(ngawest2, ngawest2_vert, Initial_ngawest2_size):
    
    """
    Convert from NGAWest2 format flatfile to ESM18 format flatfile 
    """
    # Reformat to ESM18 format
    event_time = {}
    final_event_id = {}
    fm_type_code_converted = {}
    final_station_id = {}
    for rec in range(0,len(ngawest2)):
        
        # Event time
        event_time_year=ngawest2.YEAR.iloc[rec]
        event_time_month_and_day=str(ngawest2.MODY.iloc[rec])
        
        if len(event_time_month_and_day)==3:
            month=str('0')+str(event_time_month_and_day[0])
            day=event_time_month_and_day[1:3]     
        
        if len(event_time_month_and_day)==4:
            month=str(event_time_month_and_day[:2])
            day=event_time_month_and_day[2:4]
        
        yyyy_mm_dd = str(event_time_year) + '-' + month + '-' + day
        
        event_time_hr_and_min=str(ngawest2.HRMN.iloc[rec])
        
        if len(event_time_hr_and_min)==3:
            hour=str('0')+str(event_time_hr_and_min[0])
            minute=event_time_hr_and_min[1:3]
        
        if len(event_time_hr_and_min)==4:
            hour=str(event_time_hr_and_min[:2])
            minute=event_time_hr_and_min[2:4]
            
        hh_mm_ss = str(hour) + ':' + str(minute) + ':' + '00'
        
        event_time[rec] = yyyy_mm_dd + ' ' + hh_mm_ss
    
        # Reformat event id
        delimited_event_id = str(ngawest2['Earthquake Name'][rec])
        delimited_event_id = delimited_event_id.replace(',','')
        delimited_event_id = delimited_event_id.replace(' ','')
        delimited_event_id = delimited_event_id.replace('/','')
        delimited_event_id = delimited_event_id.replace('.','')
        delimited_event_id = delimited_event_id.replace(':','')
        delimited_event_id = delimited_event_id.replace(';','')
        final_event_id[rec] = 'Earthquake-' + delimited_event_id 
        
        # Assign ESM18 fault_code based on code in NGA-West2
        if ngawest2['Mechanism Based on Rake Angle'][rec]==0 or ngawest2[
                'Mechanism Based on Rake Angle'][rec]==-999:
            ESM18_equivalent_fm_type_code='SS'
        if ngawest2[
                'Mechanism Based on Rake Angle'][
                    rec]==1 or ngawest2['Mechanism Based on Rake Angle'][
                        rec]==4:
            ESM18_equivalent_fm_type_code='NF'
        if ngawest2['Mechanism Based on Rake Angle'][
                rec]==2 or ngawest2['Mechanism Based on Rake Angle'][rec]==3:
            ESM18_equivalent_fm_type_code='TF'
        fm_type_code_converted[rec] = ESM18_equivalent_fm_type_code
        
        # Station id
        delimited_station_id = str(ngawest2['Station Name'][rec])
        delimited_station_id = delimited_station_id.replace(',','')
        delimited_station_id = delimited_station_id.replace(' ','')
        delimited_station_id = delimited_station_id.replace('/','')
        delimited_station_id = delimited_station_id.replace('.','')
        delimited_station_id = delimited_station_id.replace(':','')
        delimited_station_id = delimited_station_id.replace(';','')
        final_station_id[rec] = 'StationName-'+delimited_station_id
    
    # Into df
    ngawest2['fm_type'] = pd.Series(fm_type_code_converted)
    ngawest2['station_id'] = pd.Series(final_station_id)
    ngawest2['event_time_reformatted'] = pd.Series(event_time)
    ngawest2['event_id_reformatted'] = pd.Series(final_event_id)

    # Create nation code (not provided in NGA-West-2 so assign flag)
    nation_code_default_string = np.full(len(ngawest2['Station Name']),
        "ngawest2_does_not_provide_nation_codes") 
    default_nation_code = pd.Series(nation_code_default_string)
    
    # Create channel codes for horz components as within ngawest2 format
    H1_string = np.full(len(ngawest2['Station Name']), "H1")
    default_H1_string = pd.Series(H1_string)
    H2_string = np.full(len(ngawest2['Station Name']), "H2")
    default_H2_string = pd.Series(H2_string)
    V_string = np.full(len(ngawest2['Station Name']), "V")
    default_V_string = pd.Series(V_string)
    
    # Create default value of 0 for location code string (arbitrary)
    location_string = np.full(len(ngawest2['Station Name']), "0.0")
    location_code_string = pd.Series(location_string)  
    
    # Create default values for headers not readily available or required
    r_string = np.full(len(ngawest2['Station Name']), "")
    default_string = pd.Series(r_string)    
    
    # Construct dataframe with original ESM 2018 format 
    ESM_original_headers = pd.DataFrame(
    {
    # Non-GMIM headers   
    "event_id":ngawest2['event_id_reformatted'],                                       
    "event_time":ngawest2['event_time_reformatted'],
    "ev_nation_code":default_nation_code,
    "ev_latitude":ngawest2['Hypocenter Latitude (deg)'],    
    "ev_longitude":ngawest2['Hypocenter Longitude (deg)'],   
    "ev_depth_km":ngawest2['Hypocenter Depth (km)'],
    "fm_type_code":ngawest2['fm_type'],
    "ML":default_string,
    "ML_ref":default_string,
    "Mw":ngawest2['Earthquake Magnitude'],
    "Mw_ref":default_string,
    "Ms":default_string,
    "Ms_ref":default_string,
    "EMEC_Mw":default_string,
    "EMEC_Mw_type":default_string,
    "EMEC_Mw_ref":default_string,
    "event_source_id":default_string,
 
    "es_strike":ngawest2['Strike (deg)'],
    "es_dip":ngawest2['Dip (deg)'],
    "es_rake":ngawest2['Rake Angle (deg)'],
    "es_strike_dip_rake_ref":default_string, 
    "es_z_top":ngawest2['Depth to Top Of Fault Rupture Model'],
    "es_length":ngawest2['Fault Rupture Length for Calculation of Ry (km)'],   
    "es_width":ngawest2['Fault Rupture Width (km)'],
 
    "network_code": ngawest2['Owner'],
    "station_code":ngawest2['station_id'],
    "location_code":location_code_string,
    "instrument_code":default_string,     
    "sensor_depth_m":default_string,
    "proximity_code":default_string,
    "housing_code":default_string,
    "st_nation_code":default_string,
    "st_latitude":ngawest2['Station Latitude'],
    "st_longitude":ngawest2['Station Longitude'],
    "st_elevation":default_string,
    
    "vs30_m_sec":ngawest2['Vs30 (m/s) selected for analysis'],
    "slope_deg":default_string,
    "vs30_m_sec_WA":default_string,
 
    "epi_dist":ngawest2['EpiD (km)'],
    "epi_az":default_string,
    "JB_dist":ngawest2['Joyner-Boore Dist. (km)'],
    "rup_dist":ngawest2['Campbell R Dist. (km)'],
    "Rx_dist":ngawest2['Rx'],
    "Ry0_dist":default_string,
 
    "instrument_type_code":default_string,      
    "late_triggered_flag_01":default_string,
    "U_channel_code":default_H1_string,
    "U_azimuth_deg":ngawest2['H1 azimth (degrees)'],
    "V_channel_code":default_H2_string,
    "V_azimuth_deg":ngawest2['H2 azimith (degrees)'],
    "W_channel_code":default_V_string, 
        
    "U_hp":ngawest2['HP-H1 (Hz)'],
    "V_hp":ngawest2['HP-H2 (Hz)'],
    "W_hp":ngawest2_vert['HP-V (Hz)'],  
    "U_lp":ngawest2['LP-H1 (Hz)'],
    "V_lp":ngawest2['LP-H2 (Hz)'],
    "W_lp":ngawest2_vert['LP-V (Hz)'], 
     
    # SMT uses GM of two horz components so place RotD50 here
    "U_pga":ngawest2['PGA (g)']*981,
    "V_pga":ngawest2['PGA (g)']*981,
    "W_pga":ngawest2_vert['PGA (g)']*981,
    "U_pgv":ngawest2['PGV (cm/sec)'],
    "V_pgv":ngawest2['PGV (cm/sec)'],
    "W_pgv":ngawest2_vert['PGV (cm/sec)'],
    "U_pgd":ngawest2['PGD (cm)'],
    "V_pgd":ngawest2['PGD (cm)'],
    "W_pgd":ngawest2_vert['PGD (cm)'],
        
    "U_T0_010":ngawest2['T0.010S']*981,
    "U_T0_025":ngawest2['T0.025S']*981,
    "U_T0_040":ngawest2['T0.040S']*981,
    "U_T0_050":ngawest2['T0.050S']*981,
    "U_T0_070":ngawest2['T0.070S']*981,
    "U_T0_100":ngawest2['T0.100S']*981,
    "U_T0_150":ngawest2['T0.150S']*981,
    "U_T0_200":ngawest2['T0.200S']*981,
    "U_T0_250":ngawest2['T0.250S']*981,
    "U_T0_300":ngawest2['T0.300S']*981,
    "U_T0_350":ngawest2['T0.350S']*981,
    "U_T0_400":ngawest2['T0.400S']*981,
    "U_T0_450":ngawest2['T0.450S']*981,
    "U_T0_500":ngawest2['T0.500S']*981,
    "U_T0_600":ngawest2['T0.600S']*981,
    "U_T0_700":ngawest2['T0.700S']*981,
    "U_T0_750":ngawest2['T0.750S']*981,
    "U_T0_800":ngawest2['T0.800S']*981,
    "U_T0_900":ngawest2['T0.900S']*981,
    "U_T1_000":ngawest2['T1.000S']*981,
    "U_T1_200":ngawest2['T1.200S']*981,
    "U_T1_400":ngawest2['T1.400S']*981,
    "U_T1_600":ngawest2['T1.600S']*981,
    "U_T1_800":ngawest2['T1.800S']*981,
    "U_T2_000":ngawest2['T2.000S']*981,
    "U_T2_500":ngawest2['T2.500S']*981,
    "U_T3_000":ngawest2['T3.000S']*981,
    "U_T3_500":ngawest2['T3.500S']*981,
    "U_T4_000":ngawest2['T4.000S']*981,
    "U_T4_500":ngawest2['T4.500S']*981,
    "U_T5_000":ngawest2['T5.000S']*981,
    "U_T6_000":ngawest2['T6.000S']*981,
    "U_T7_000":ngawest2['T7.000S']*981,
    "U_T8_000":ngawest2['T8.000S']*981,
    "U_T9_000":ngawest2['T9.000S']*981,
    "U_T10_000":ngawest2['T10.000S']*981,
    
    "V_T0_010":ngawest2['T0.010S']*981,
    "V_T0_025":ngawest2['T0.025S']*981,
    "V_T0_040":ngawest2['T0.040S']*981,
    "V_T0_050":ngawest2['T0.050S']*981,
    "V_T0_070":ngawest2['T0.070S']*981,
    "V_T0_100":ngawest2['T0.100S']*981,
    "V_T0_150":ngawest2['T0.150S']*981,
    "V_T0_200":ngawest2['T0.200S']*981,
    "V_T0_250":ngawest2['T0.250S']*981,
    "V_T0_300":ngawest2['T0.300S']*981,
    "V_T0_350":ngawest2['T0.350S']*981,
    "V_T0_400":ngawest2['T0.400S']*981,
    "V_T0_450":ngawest2['T0.450S']*981,
    "V_T0_500":ngawest2['T0.500S']*981,
    "V_T0_600":ngawest2['T0.600S']*981,
    "V_T0_700":ngawest2['T0.700S']*981,
    "V_T0_750":ngawest2['T0.750S']*981,
    "V_T0_800":ngawest2['T0.800S']*981,
    "V_T0_900":ngawest2['T0.900S']*981,
    "V_T1_000":ngawest2['T1.000S']*981,
    "V_T1_200":ngawest2['T1.200S']*981,
    "V_T1_400":ngawest2['T1.400S']*981,
    "V_T1_600":ngawest2['T1.600S']*981,
    "V_T1_800":ngawest2['T1.800S']*981,
    "V_T2_000":ngawest2['T2.000S']*981,
    "V_T2_500":ngawest2['T2.500S']*981,
    "V_T3_000":ngawest2['T3.000S']*981,
    "V_T3_500":ngawest2['T3.500S']*981,
    "V_T4_000":ngawest2['T4.000S']*981,
    "V_T4_500":ngawest2['T4.500S']*981,
    "V_T5_000":ngawest2['T5.000S']*981,
    "V_T6_000":ngawest2['T6.000S']*981,
    "V_T7_000":ngawest2['T7.000S']*981,
    "V_T8_000":ngawest2['T8.000S']*981,
    "V_T9_000":ngawest2['T9.000S']*981,
    "V_T10_000":ngawest2['T10.000S']*981,
        
    "W_T0_010":ngawest2_vert['T0.010S']*981,
    "W_T0_025":ngawest2_vert['T0.025S']*981,
    "W_T0_040":ngawest2_vert['T0.040S']*981,
    "W_T0_050":ngawest2_vert['T0.050S']*981,
    "W_T0_070":ngawest2_vert['T0.070S']*981,
    "W_T0_100":ngawest2_vert['T0.100S']*981,
    "W_T0_150":ngawest2_vert['T0.150S']*981,
    "W_T0_200":ngawest2_vert['T0.200S']*981,
    "W_T0_250":ngawest2_vert['T0.250S']*981,
    "W_T0_300":ngawest2_vert['T0.300S']*981,
    "W_T0_350":ngawest2_vert['T0.350S']*981,
    "W_T0_400":ngawest2_vert['T0.400S']*981,
    "W_T0_450":ngawest2_vert['T0.450S']*981,
    "W_T0_500":ngawest2_vert['T0.500S']*981,
    "W_T0_600":ngawest2_vert['T0.600S']*981,
    "W_T0_700":ngawest2_vert['T0.700S']*981,
    "W_T0_750":ngawest2_vert['T0.750S']*981,
    "W_T0_800":ngawest2_vert['T0.800S']*981,
    "W_T0_900":ngawest2_vert['T0.900S']*981,
    "W_T1_000":ngawest2_vert['T1.000S']*981,
    "W_T1_200":ngawest2_vert['T1.200S']*981,
    "W_T1_400":ngawest2_vert['T1.400S']*981,
    "W_T1_600":ngawest2_vert['T1.600S']*981,
    "W_T1_800":ngawest2_vert['T1.800S']*981,
    "W_T2_000":ngawest2_vert['T2.000S']*981,
    "W_T2_500":ngawest2_vert['T2.500S']*981,
    "W_T3_000":ngawest2_vert['T3.000S']*981,
    "W_T3_500":ngawest2_vert['T3.500S']*981,
    "W_T4_000":ngawest2_vert['T4.000S']*981,
    "W_T4_500":ngawest2_vert['T4.500S']*981,
    "W_T5_000":ngawest2_vert['T5.000S']*981,
    "W_T6_000":ngawest2_vert['T6.000S']*981,
    "W_T7_000":ngawest2_vert['T7.000S']*981,
    "W_T8_000":ngawest2_vert['T8.000S']*981,
    "W_T9_000":ngawest2_vert['T9.000S']*981,
    "W_T10_000":ngawest2_vert['T10.000S']*981})
    
    # Output to folder where converted flatfile read into parser   
    DATA = os.path.abspath('')
    tmp=tempfile.mkdtemp()
    converted_base_data_path = os.path.join(DATA, tmp, 'converted_flatfile.csv')
    ESM_original_headers.to_csv(converted_base_data_path,sep=';')

    print(Initial_ngawest2_size - len(ngawest2),
          'records removed from imported NGA-West-2 flatfile during data quality checks.')
    
    return converted_base_data_path