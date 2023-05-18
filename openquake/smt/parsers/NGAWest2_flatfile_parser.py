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
Parser for the NGAWest2 flatfile format
"""
import pandas as pd
import os, sys
import tempfile
import csv
import numpy as np
import copy
import h5py
from math import sqrt
from linecache import getline
from collections import OrderedDict

from openquake.smt.sm_database import GroundMotionDatabase, GroundMotionRecord,\
    Earthquake, Magnitude, Rupture, FocalMechanism, GCMTNodalPlanes,\
    Component, RecordSite, RecordDistance
from openquake.smt.sm_utils import  MECHANISM_TYPE, DIP_TYPE, vs30_to_z1pt0_cy14,\
    vs30_to_z2pt5_cb14
from openquake.smt.parsers.base_database_parser import SMDatabaseReader
from openquake.smt.parsers import valid

if sys.version_info[0] >= 3:
    import pickle
else:
    import cPickle as pickle

# Import the ESM dictionaries
from .esm_dictionaries import *

SCALAR_LIST = ["PGA", "PGV", "PGD", "CAV", "CAV5", "Ia", "D5-95"]

HEADER_STR = "event_id;event_time;ISC_ev_id;USGS_ev_id;INGV_ev_id;"\
             "EMSC_ev_id;ev_nation_code;ev_latitude;ev_longitude;"\
             "ev_depth_km;ev_hyp_ref;fm_type_code;ML;ML_ref;Mw;Mw_ref;Ms;"\
             "Ms_ref;EMEC_Mw;EMEC_Mw_type;EMEC_Mw_ref;event_source_id;"\
             "es_strike;es_dip;es_rake;es_strike_dip_rake_ref;es_z_top;"\
             "es_z_top_ref;es_length;es_width;es_geometry_ref;network_code;"\
             "station_code;location_code;instrument_code;sensor_depth_m;"\
             "proximity_code;housing_code;installation_code;st_nation_code;"\
             "st_latitude;st_longitude;st_elevation;ec8_code;"\
             "ec8_code_method;ec8_code_ref;vs30_m_sec;vs30_ref;"\
             "vs30_calc_method;vs30_meas_type;slope_deg;vs30_m_sec_WA;"\
             "epi_dist;epi_az;JB_dist;rup_dist;Rx_dist;Ry0_dist;"\
             "instrument_type_code;late_triggered_flag_01;U_channel_code;"\
             "U_azimuth_deg;V_channel_code;V_azimuth_deg;W_channel_code;"\
             "U_hp;V_hp;W_hp;U_lp;V_lp;W_lp"

HEADERS = set(HEADER_STR.split(";"))

COUNTRY_CODES = {"AL": "Albania", "AM": "Armenia", "AT": "Austria",
                 "AZ": "Azerbaijan", "BA": "Bosnia and Herzegowina",
                 "BG": "Bulgaria", "CH": "Switzerland", "CY": "Cyprus",
                 "CZ": "Czech Republic", "DE": "Germany",  "DZ": "Algeria",
                 "ES": "Spain", "FR": "France", "GE": "Georgia",
                 "GR": "Greece", "HR": "Croatia", "HU": "Hungary",
                 "IL": "Israel", "IR": "Iran", "IS": "Iceland", "IT": "Italy",
                 "JO": "Jordan",  "LI": "Lichtenstein", "MA": "Morocco",
                 "MC": "Monaco", "MD": "Moldova", "ME": "Montenegro",
                 "MK": "Macedonia", "MT": "Malta", "PL": "Poland",
                 "PT": "Portugal", "RO": "Romania", "RS": "Serbia",
                 "RU": "Russia", "SI": "Slovenia", "SM": "San Marino",
                 "SY": "Syria", "TM": "Turkmenistan", "TR": "Turkey",
                 "UA": "Ukraine", "UZ": "Uzbekistan", "XK": "Kosovo"}


class NGAWest2FlatfileParser(SMDatabaseReader):
    
    """
    Parses the metadata from the flatfile to a set of metadata objects
    """
    
    M_PRECEDENCE = ["EMEC_Mw", "Mw", "Ms", "ML"]
    BUILD_FINITE_DISTANCES = False

    def parse(self, location='./'):
        """
        """
        assert os.path.isfile(self.filename)
        headers = getline(self.filename, 1).rstrip("\n").split(";")
        for hdr in HEADERS:
            if hdr not in headers:
                raise ValueError("Required header %s is missing in file"
                                 % hdr)
        # Read in csv
        reader = csv.DictReader(open(self.filename, "r"), delimiter=";")
        metadata = []
        self.database = GroundMotionDatabase(self.id, self.name)
        counter = 0
        for row in reader:
            if self._sanitise(row, reader):
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
    def autobuild(cls, dbid, dbname, output_location, 
                  NGAWest2_flatfile_directory, 
                  NGAWest2_vertical_flatfile_directory):
        """
        Quick and dirty full database builder!
        """
        
        # Import NGAWest2 format strong-motion flatfiles
        NGAWest2 = pd.read_csv(NGAWest2_flatfile_directory)
        NGAWest2_vertical = pd.read_csv(NGAWest2_vertical_flatfile_directory)
        
        # Check RotD50 and vertical records match
        for rec in range(0,len(NGAWest2)):
            if NGAWest2['Record Sequence Number'
                        ].iloc[rec]!=NGAWest2_vertical[
                            'Record Sequence Number'].iloc[rec]:
                raise ValueError("Records within horizontal and vertical flatfiles do not match.")
        
        # Count initial size for printing number records removed during checks
        Initial_NGAWest2_size = len(NGAWest2)

        # Remove potential duplicate records in NGA-West2 flatfile
        NGAWest2 = NGAWest2.drop_duplicates(subset=
                                            ['Earthquake Name',
                                                    'Station Name'],
                                            keep='last')
        NGAWest2_vertical = NGAWest2_vertical.drop_duplicates(
            subset=['Earthquake Name', 'Station Name'], keep='last')
        NGAWest2 = NGAWest2.reset_index().drop(columns='index')
        NGAWest2_vertical = NGAWest2_vertical.reset_index().drop(
            columns='index')

        # Remove records if earthquake not identifiable using lat/lon metadata
        Index_to_drop=np.array(NGAWest2.loc[
            NGAWest2['Hypocenter Latitude (deg)']==-999][
                'Hypocenter Latitude (deg)'].index)
        NGAWest2=NGAWest2.drop(Index_to_drop)
        NGAWest2_vertical=NGAWest2_vertical.drop(Index_to_drop)
        Index_to_drop=np.array(NGAWest2.loc[NGAWest2[
            'Hypocenter Longitude (deg)']==-999][
                'Hypocenter Longitude (deg)'].index)
        NGAWest2=NGAWest2.drop(Index_to_drop)
        NGAWest2_vertical=NGAWest2_vertical.drop(Index_to_drop)
        Index_to_drop=np.array(NGAWest2.loc[NGAWest2[
            'Hypocenter Depth (km)']==-999]['Hypocenter Depth (km)'].index)
        NGAWest2=NGAWest2.drop(Index_to_drop)
        NGAWest2_vertical=NGAWest2_vertical.drop(Index_to_drop)
        
        # If year not provided assign '0000' to work with datetime
        Index_to_drop=np.array(NGAWest2.loc[NGAWest2['YEAR']=='-999'][
            'YEAR'].index)
        NGAWest2['YEAR'].iloc[Index_to_drop]='0000'
                
        # If month and day not provided assign '1010' to work with datetime
        Index_to_drop=np.array(NGAWest2.loc[NGAWest2['MODY']=='-999'][
            'MODY'].index)
        NGAWest2['MODY'].iloc[Index_to_drop]='000'
        
        # If hours and minutes not provided assign '000' to work with datetime
        NGAWest2 = NGAWest2.reset_index().drop(columns='index')
        NGAWest2_vertical=NGAWest2_vertical.reset_index().drop(columns='index')
        for rec in range(0,len(NGAWest2)):
            if NGAWest2['HRMN'][rec]==-999:
                NGAWest2['HRMN'][rec]='000'
        
        # Remove records with no acceleration values
        Index_to_drop=np.array(NGAWest2.loc[NGAWest2['PGA (g)']==-999][
            'PGA (g)'].index)
        NGAWest2=NGAWest2.drop(Index_to_drop)
        NGAWest2_vertical=NGAWest2_vertical.drop(Index_to_drop)
        NGAWest2 = NGAWest2.reset_index().drop(columns='index')
        NGAWest2_vertical = NGAWest2_vertical.reset_index().drop(
            columns='index')

        #Remove records with no seismic moment to compute moment magnitude from
        Index_to_drop=np.array(NGAWest2.loc[NGAWest2['Mo (dyne.cm)']==-999][
            'Mo (dyne.cm)'].index)
        NGAWest2=NGAWest2.drop(Index_to_drop)
        NGAWest2_vertical=NGAWest2_vertical.drop(Index_to_drop)
        NGAWest2 = NGAWest2.reset_index().drop(columns='index')
        NGAWest2_vertical = NGAWest2_vertical.reset_index().drop(
            columns='index')
        
        # Remove records with no valid station name
        Index_to_drop=np.array(NGAWest2.loc[NGAWest2['Station Name']==-999][
            'Station Name'].index)
        NGAWest2=NGAWest2.drop(Index_to_drop)
        NGAWest2_vertical=NGAWest2_vertical.drop(Index_to_drop)
        
        # Remove records with no strike, dip or rake angle
        Index_to_drop=np.array(NGAWest2.loc[NGAWest2['Strike (deg)']==-999][
            'Strike (deg)'].index)
        NGAWest2=NGAWest2.drop(Index_to_drop)
        NGAWest2_vertical=NGAWest2_vertical.drop(Index_to_drop)
        
        Index_to_drop=np.array(NGAWest2.loc[NGAWest2['Dip (deg)']==-999][
            'Dip (deg)'].index)
        NGAWest2=NGAWest2.drop(Index_to_drop)
        NGAWest2_vertical=NGAWest2_vertical.drop(Index_to_drop)
        
        Index_to_drop=np.array(NGAWest2.loc[NGAWest2['Rake Angle (deg)']==-999][
            'Rake Angle (deg)'].index)
        NGAWest2=NGAWest2.drop(Index_to_drop)
        NGAWest2_vertical=NGAWest2_vertical.drop(Index_to_drop)
        
        # If no ztor set to 0 km (workaround used to get ztor vals as oddly can't
        # be found as column using conventional pandas methods)
        ztor_column = NGAWest2.columns[32] # Column header 32 in NGAWest2 df
        ztor_vals = NGAWest2[ztor_column]
        ztor_vals[ztor_vals == -999] = 0
        NGAWest2['ztor'] = ztor_vals
        
        # Remove records with no epicentral distance
        Index_to_drop=np.array(NGAWest2.loc[NGAWest2['EpiD (km)']==-999][
            'EpiD (km)'].index)
        NGAWest2=NGAWest2.drop(Index_to_drop)
        NGAWest2_vertical=NGAWest2_vertical.drop(Index_to_drop)
        
        # If Joyner-Boore, rupture distance, Rx or Ry = -999 reassign as empty
        Index_to_drop=np.array(NGAWest2.loc[
            NGAWest2['Joyner-Boore Dist. (km)']==-999][
                'Joyner-Boore Dist. (km)'].index)
        NGAWest2['Joyner-Boore Dist. (km)'][Index_to_drop] = ''
    
        Index_to_drop=np.array(NGAWest2.loc[NGAWest2[
            'Campbell R Dist. (km)']==-999]['Campbell R Dist. (km)'].index)
        NGAWest2['Campbell R Dist. (km)'][Index_to_drop] = ''

        Index_to_drop=np.array(NGAWest2.loc[NGAWest2['Rx']==-999]['Rx'].index)
        NGAWest2['Rx'][Index_to_drop] = ''

        Index_to_drop=np.array(NGAWest2.loc[NGAWest2['Ry 2']==-999][
            'Ry 2'].index)
        NGAWest2['Ry 2'][Index_to_drop] = ''
        
        # Remove records with no vs30)
        Index_to_drop=np.array(NGAWest2.loc[
            NGAWest2['Vs30 (m/s) selected for analysis']==-999][
                'Vs30 (m/s) selected for analysis'].index)
        NGAWest2=NGAWest2.drop(Index_to_drop)
        NGAWest2_vertical=NGAWest2_vertical.drop(Index_to_drop)
        
        # Compute Mw from seismic moment using Hanks and Kamori
        NGAWest2['Earthquake Magnitude'] = (np.log10(NGAWest2[
            'Mo (dyne.cm)']) - 16.05)/1.5 # From NGAWest2 database rep. pp.122
        
        NGAWest2 = NGAWest2.reset_index().drop(columns='index')
        NGAWest2_vertical = NGAWest2_vertical.reset_index().drop(
            columns='index')
      
        # Replace -999 in 'Owner' with unknown network code
        Index_to_drop=np.array(NGAWest2.loc[NGAWest2['Owner']=='-999'][
            'Owner'].index)
        NGAWest2['Owner'].iloc[Index_to_drop]='NoNetworkCode'
        NGAWest2['Owner'] = 'NetworkCode-'+NGAWest2['Owner'] 
        
        # Interpolate between SA(T=4.4s) and SA(T=4.6s) for SA(T=4.5)
        NGAWest2['T4.500S']=(NGAWest2['T4.400S']+NGAWest2['T4.600S'])/2
        NGAWest2_vertical['T4.500S']=(NGAWest2_vertical[
            'T4.400S']+NGAWest2_vertical['T4.600S'])/2        
        
        converted_base_data_path=_get_ESM18_headers(
            NGAWest2,NGAWest2_vertical,Initial_NGAWest2_size)
                
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

    def _sanitise(self, row, reader):
        """
        TODO - Not implemented yet!
        """
        return True

    def _parse_record(self, metadata):
        # Conc. NGAWest2 record info to identify each record in flatfile:
        # --> event_id = NGAWest2['Earthquake Name'] 
        # --> station_id = NGAWest2['Station Name'] 
        # --> network_code = NGAWest2['Owner']
        # --> location_code = NGAWest2['Station ID No.']
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
        xcomp, ycomp, vertical = self._parse_waveform_data(metadata, wfid)
        return GroundMotionRecord(wfid,
                                  [None, None, None],
                                  event, distances, site,
                                  xcomp, ycomp,
                                  vertical=vertical)


    def _parse_event_data(self, metadata):
        """
        Parses the event metadata
        """
        # ID and Name (name not in file so use ID again)
        eq_id = metadata["event_id"]
        eq_name = metadata["event_id"]
        # Country
        cntry_code = metadata["ev_nation_code"].strip()
        if cntry_code and cntry_code in COUNTRY_CODES:
            eq_country = COUNTRY_CODES[cntry_code]
        else:
            eq_country = None
        # Date and time
        eq_datetime = valid.date_time(metadata["event_time"],
                                     "%Y-%m-%d %H:%M:%S")
        # Latitude, longitude and depth
        eq_lat = valid.latitude(metadata["ev_latitude"])
        eq_lon = valid.longitude(metadata["ev_longitude"])
        eq_depth = valid.positive_float(metadata["ev_depth_km"], "ev_depth_km")
        if not eq_depth:
            eq_depth = 0.0
        eqk = Earthquake(eq_id, eq_name, eq_datetime, eq_lon, eq_lat, eq_depth,
                         None, # Magnitude not defined yet
                         eq_country=eq_country)
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
        location_code = metadata["location_code"].strip()
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
        st_nation_code = metadata["st_nation_code"].strip()
        if st_nation_code:
            st_country = COUNTRY_CODES[st_nation_code]
        else:
            st_country = None
        site = RecordSite(site_id, station_code, station_code, site_lon,
                          site_lat, elevation, vs30, vs30_measured,
                          network_code=network_code,
                          country=st_country)
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
        # Add on the horizontal values
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
                    #values.append(np.fabs(float(row[header].strip())))
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

def _get_ESM18_headers(NGAWest2,NGAWest2_vertical,Initial_NGAWest2_size):
    
    """
    Convert first from NGAWest2 format flatfile to ESM18 format flatfile 
    """
    
    # Construct event_time in yyyy-mm-dd hh:mm:ss format
    event_time={}
    for rec in range(0,len(NGAWest2)):
        event_time_year=NGAWest2.YEAR.iloc[rec]
        event_time_month_and_day=str(NGAWest2.MODY.iloc[rec])
        
        if len(event_time_month_and_day)==3:
            month=str('0')+str(event_time_month_and_day[0])
            day=event_time_month_and_day[1:3]     
        
        if len(event_time_month_and_day)==4:
            month=str(event_time_month_and_day[:2])
            day=event_time_month_and_day[2:4]
        
        yyyy_mm_dd = str(event_time_year) + '-' + month + '-' + day
        
        event_time_hr_and_min=str(NGAWest2.HRMN.iloc[rec])
        
        if len(event_time_hr_and_min)==3:
            hour=str('0')+str(event_time_hr_and_min[0])
            minute=event_time_hr_and_min[1:3]
        
        if len(event_time_hr_and_min)==4:
            hour=str(event_time_hr_and_min[:2])
            minute=event_time_hr_and_min[2:4]
            
        hh_mm_ss = str(hour) + ':' + str(minute) + ':' + '00'
        
        event_time[rec] = yyyy_mm_dd + ' ' + hh_mm_ss
    
    event_time_reformatted = pd.Series(event_time)
    
    # generate event_id without delimiters
    final_event_id={}
    for rec in range(0,len(NGAWest2)):
        delimited_event_id=str(NGAWest2['Earthquake Name'][rec])
        delimited_event_id=delimited_event_id.replace(',','')
        delimited_event_id=delimited_event_id.replace(' ','')
        delimited_event_id=delimited_event_id.replace('/','')
        delimited_event_id=delimited_event_id.replace('.','')
        delimited_event_id=delimited_event_id.replace(':','')
        delimited_event_id=delimited_event_id.replace(';','')
        final_event_id[rec] = 'Earthquake-' + delimited_event_id 
    event_id_reformatted = pd.Series(final_event_id)
    
    # Assign ESM18 fault_code based on code in NGA-West2
    """
    Strike-Slip = 00 (SS)
    Normal = 01 (NF)
    Reverse = 02 (RF)
    Reverse - Oblique = 03 (RF)
    Normal - Oblique = 04 (NF)
    
    Assign strike-slip if -999 provided (i.e. fm_type_code unknown in NGAWest2)
    """
    
    fm_type_code_converted={}
    for rec in range(0,len(NGAWest2)):
        if NGAWest2['Mechanism Based on Rake Angle'][rec]==0 or NGAWest2[
                'Mechanism Based on Rake Angle'][rec]==-999:
            ESM18_equivalent_fm_type_code='SS'
        if NGAWest2[
                'Mechanism Based on Rake Angle'][
                    rec]==1 or NGAWest2['Mechanism Based on Rake Angle'][
                        rec]==4:
            ESM18_equivalent_fm_type_code='NF'
        if NGAWest2['Mechanism Based on Rake Angle'][
                rec]==2 or NGAWest2['Mechanism Based on Rake Angle'][rec]==3:
            ESM18_equivalent_fm_type_code='TF'
        fm_type_code_converted[rec] = ESM18_equivalent_fm_type_code
    reformatted_fm_type_code = pd.Series(fm_type_code_converted)
    
    # Station code without delimiters
    final_station_id={}
    for rec in range(0,len(NGAWest2)):
        delimited_station_id=str(NGAWest2['Station Name'][rec])
        delimited_station_id=delimited_station_id.replace(',','')
        delimited_station_id=delimited_station_id.replace(' ','')
        delimited_station_id=delimited_station_id.replace('/','')
        delimited_station_id=delimited_station_id.replace('.','')
        delimited_station_id=delimited_station_id.replace(':','')
        delimited_station_id=delimited_station_id.replace(';','')
        final_station_id[rec] = 'StationName-'+delimited_station_id
    station_id_reformatted = pd.Series(final_station_id)
    
    # Create nation code (not provided in NGA-West-2 so assign flag)
    nation_code_default_string = np.full(len(NGAWest2),str(
        "NGAWest2_does_not_provide_nation_codes")) 
    default_nation_code = pd.Series(nation_code_default_string)
    
    # Create channel codes for horizontal components as within NGAWest2 format
    H1_string = np.full(len(NGAWest2),str("H1"))
    default_H1_string = pd.Series(H1_string)
    H2_string = np.full(len(NGAWest2),str("H2"))
    default_H2_string = pd.Series(H2_string)
    V_string = np.full(len(NGAWest2),str("V"))
    default_V_string = pd.Series(V_string)
    
    # Create default value of 0 for location code string (arbitrary)
    location_string = np.full(len(NGAWest2),str("0.0"))
    location_code_string = pd.Series(location_string)  
    
    # Create default values for headers not readily available or required
    r_string = np.full(len(NGAWest2),str(""))
    default_string = pd.Series(r_string)    
    
    # Construct dataframe with original ESM 2018 format 
    ESM_original_headers = pd.DataFrame(
    {
    # Non-GMIM headers   
    "event_id":event_id_reformatted,                                       
    "event_time":event_time_reformatted,
    "ISC_ev_id":default_string,
    "USGS_ev_id":default_string,
    "INGV_ev_id":default_string,
    "EMSC_ev_id":default_string,
    "ev_nation_code":default_nation_code,
    "ev_latitude":NGAWest2['Hypocenter Latitude (deg)'],    
    "ev_longitude":NGAWest2['Hypocenter Longitude (deg)'],   
    "ev_depth_km":NGAWest2['Hypocenter Depth (km)'],
    "ev_hyp_ref":default_string,
    "fm_type_code":reformatted_fm_type_code,
    "ML":default_string,
    "ML_ref":default_string,
    "Mw":NGAWest2['Earthquake Magnitude'],
    "Mw_ref":default_string,
    "Ms":default_string,
    "Ms_ref":default_string,
    "EMEC_Mw":default_string,
    "EMEC_Mw_type":default_string,
    "EMEC_Mw_ref":default_string,
    "event_source_id":default_string,
 
    "es_strike":NGAWest2['Strike (deg)'],
    "es_dip":NGAWest2['Dip (deg)'],
    "es_rake":NGAWest2['Rake Angle (deg)'],
    "es_strike_dip_rake_ref":default_string, 
    "es_z_top":NGAWest2['ztor'],
    "es_z_top_ref":default_string,
    "es_length":default_string,   
    "es_width":default_string,
    "es_geometry_ref":default_string,
 
    "network_code": NGAWest2['Owner'],
    "station_code":station_id_reformatted,
    "location_code":location_code_string,
    "instrument_code":default_string,     
    "sensor_depth_m":default_string,
    "proximity_code":default_string,
    "housing_code":default_string,
    "installation_code":default_string,
    "st_nation_code":default_string,
    "st_latitude":NGAWest2['Station Latitude'],
    "st_longitude":NGAWest2['Station Longitude'],
    "st_elevation":default_string,
    
    "ec8_code":default_string,
    "ec8_code_method":default_string,
    "ec8_code_ref":default_string,
    "vs30_m_sec":NGAWest2['Vs30 (m/s) selected for analysis'],
    "vs30_ref":default_string,
    "vs30_calc_method":default_string, 
    "vs30_meas_type":default_string,
    "slope_deg":default_string,
    "vs30_m_sec_WA":default_string,
 
    "epi_dist":NGAWest2['EpiD (km)'],
    "epi_az":default_string,
    "JB_dist":NGAWest2['Joyner-Boore Dist. (km)'],
    "rup_dist":NGAWest2['Campbell R Dist. (km)'],
    "Rx_dist":NGAWest2['Rx'],
    "Ry0_dist":default_string,
 
    "instrument_type_code":default_string,      
    "late_triggered_flag_01":default_string,
    "U_channel_code":default_H1_string,
    "U_azimuth_deg":NGAWest2['H1 azimth (degrees)'],
    "V_channel_code":default_H2_string,
    "V_azimuth_deg":NGAWest2['H2 azimith (degrees)'],
    "W_channel_code":default_V_string, 
        
    "U_hp":default_string,
    "V_hp":default_string,
    "W_hp":default_string,  
    "U_lp":default_string,
    "V_lp":default_string,
    "W_lp":default_string,
     
    "U_pga":NGAWest2['PGA (g)']*981,
    "V_pga":NGAWest2['PGA (g)']*981,
    "W_pga":NGAWest2_vertical['PGA (g)']*981,
    "rotD50_pga":NGAWest2['PGA (g)']*981,
    "rotD100_pga":default_string,
    "rotD00_pga":default_string,
    "U_pgv":NGAWest2['PGV (cm/sec)'],
    "V_pgv":NGAWest2['PGV (cm/sec)'],
    "W_pgv":NGAWest2_vertical['PGV (cm/sec)'],
    "rotD50_pgv":NGAWest2['PGV (cm/sec)'],
    "rotD100_pgv":default_string,
    "rotD00_pgv":default_string,
    "U_pgd":NGAWest2['PGD (cm)'],
    "V_pgd":NGAWest2['PGD (cm)'],
    "W_pgd":NGAWest2_vertical['PGD (cm)'],
    "rotD50_pgd":NGAWest2['PGD (cm)'],
    "rotD100_pgd":default_string,
    "rotD00_pgd":default_string,
        
    "U_T90":default_string,
    "V_T90":default_string,
    "W_T90":default_string,
    "rotD50_T90":default_string,
    "rotD100_T90":default_string,
    "rotD00_T90":default_string, 
    "U_housner":default_string,
    "V_housner":default_string,
    "W_housner":default_string,
    "rotD50_housner":default_string,
    "rotD100_housner":default_string,
    "rotD00_housner":default_string,
    "U_CAV":default_string,
    "V_CAV":default_string,
    "W_CAV":default_string,
    "rotD50_CAV":default_string,
    "rotD100_CAV":default_string,
    "rotD00_CAV":default_string,
    "U_ia":default_string,
    "V_ia":default_string,
    "W_ia":default_string,
    "rotD50_ia":default_string,
    "rotD100_ia":default_string,
    "rotD00_ia":default_string,
    
    # Compute each horiz. comp. from RotD50 (assuming geo mean = RotD50) 
    "U_T0_010":NGAWest2['T0.010S']*981,
    "U_T0_025":NGAWest2['T0.025S']*981,
    "U_T0_040":NGAWest2['T0.040S']*981,
    "U_T0_050":NGAWest2['T0.050S']*981,
    "U_T0_070":NGAWest2['T0.070S']*981,
    "U_T0_100":NGAWest2['T0.100S']*981,
    "U_T0_150":NGAWest2['T0.150S']*981,
    "U_T0_200":NGAWest2['T0.200S']*981,
    "U_T0_250":NGAWest2['T0.250S']*981,
    "U_T0_300":NGAWest2['T0.300S']*981,
    "U_T0_350":NGAWest2['T0.350S']*981,
    "U_T0_400":NGAWest2['T0.400S']*981,
    "U_T0_450":NGAWest2['T0.450S']*981,
    "U_T0_500":NGAWest2['T0.500S']*981,
    "U_T0_600":NGAWest2['T0.600S']*981,
    "U_T0_700":NGAWest2['T0.700S']*981,
    "U_T0_750":NGAWest2['T0.750S']*981,
    "U_T0_800":NGAWest2['T0.800S']*981,
    "U_T0_900":NGAWest2['T0.900S']*981,
    "U_T1_000":NGAWest2['T1.000S']*981,
    "U_T1_200":NGAWest2['T1.200S']*981,
    "U_T1_400":NGAWest2['T1.400S']*981,
    "U_T1_600":NGAWest2['T1.600S']*981,
    "U_T1_800":NGAWest2['T1.800S']*981,
    "U_T2_000":NGAWest2['T2.000S']*981,
    "U_T2_500":NGAWest2['T2.500S']*981,
    "U_T3_000":NGAWest2['T3.000S']*981,
    "U_T3_500":NGAWest2['T3.500S']*981,
    "U_T4_000":NGAWest2['T4.000S']*981,
    "U_T4_500":NGAWest2['T4.500S']*981,
    "U_T5_000":NGAWest2['T5.000S']*981,
    "U_T6_000":NGAWest2['T6.000S']*981,
    "U_T7_000":NGAWest2['T7.000S']*981,
    "U_T8_000":NGAWest2['T8.000S']*981,
    "U_T9_000":NGAWest2['T9.000S']*981,
    "U_T10_000":NGAWest2['T10.000S']*981,
    
    "V_T0_010":NGAWest2['T0.010S']*981,
    "V_T0_025":NGAWest2['T0.025S']*981,
    "V_T0_040":NGAWest2['T0.040S']*981,
    "V_T0_050":NGAWest2['T0.050S']*981,
    "V_T0_070":NGAWest2['T0.070S']*981,
    "V_T0_100":NGAWest2['T0.100S']*981,
    "V_T0_150":NGAWest2['T0.150S']*981,
    "V_T0_200":NGAWest2['T0.200S']*981,
    "V_T0_250":NGAWest2['T0.250S']*981,
    "V_T0_300":NGAWest2['T0.300S']*981,
    "V_T0_350":NGAWest2['T0.350S']*981,
    "V_T0_400":NGAWest2['T0.400S']*981,
    "V_T0_450":NGAWest2['T0.450S']*981,
    "V_T0_500":NGAWest2['T0.500S']*981,
    "V_T0_600":NGAWest2['T0.600S']*981,
    "V_T0_700":NGAWest2['T0.700S']*981,
    "V_T0_750":NGAWest2['T0.750S']*981,
    "V_T0_800":NGAWest2['T0.800S']*981,
    "V_T0_900":NGAWest2['T0.900S']*981,
    "V_T1_000":NGAWest2['T1.000S']*981,
    "V_T1_200":NGAWest2['T1.200S']*981,
    "V_T1_400":NGAWest2['T1.400S']*981,
    "V_T1_600":NGAWest2['T1.600S']*981,
    "V_T1_800":NGAWest2['T1.800S']*981,
    "V_T2_000":NGAWest2['T2.000S']*981,
    "V_T2_500":NGAWest2['T2.500S']*981,
    "V_T3_000":NGAWest2['T3.000S']*981,
    "V_T3_500":NGAWest2['T3.500S']*981,
    "V_T4_000":NGAWest2['T4.000S']*981,
    "V_T4_500":NGAWest2['T4.500S']*981,
    "V_T5_000":NGAWest2['T5.000S']*981,
    "V_T6_000":NGAWest2['T6.000S']*981,
    "V_T7_000":NGAWest2['T7.000S']*981,
    "V_T8_000":NGAWest2['T8.000S']*981,
    "V_T9_000":NGAWest2['T9.000S']*981,
    "V_T10_000":NGAWest2['T10.000S']*981,   
        
    "W_T0_010":NGAWest2_vertical['T0.010S']*981,
    "W_T0_025":NGAWest2_vertical['T0.025S']*981,
    "W_T0_040":NGAWest2_vertical['T0.040S']*981,
    "W_T0_050":NGAWest2_vertical['T0.050S']*981,
    "W_T0_070":NGAWest2_vertical['T0.070S']*981,
    "W_T0_100":NGAWest2_vertical['T0.100S']*981,
    "W_T0_150":NGAWest2_vertical['T0.150S']*981,
    "W_T0_200":NGAWest2_vertical['T0.200S']*981,
    "W_T0_250":NGAWest2_vertical['T0.250S']*981,
    "W_T0_300":NGAWest2_vertical['T0.300S']*981,
    "W_T0_350":NGAWest2_vertical['T0.350S']*981,
    "W_T0_400":NGAWest2_vertical['T0.400S']*981,
    "W_T0_450":NGAWest2_vertical['T0.450S']*981,
    "W_T0_500":NGAWest2_vertical['T0.500S']*981,
    "W_T0_600":NGAWest2_vertical['T0.600S']*981,
    "W_T0_700":NGAWest2_vertical['T0.700S']*981,
    "W_T0_750":NGAWest2_vertical['T0.750S']*981,
    "W_T0_800":NGAWest2_vertical['T0.800S']*981,
    "W_T0_900":NGAWest2_vertical['T0.900S']*981,
    "W_T1_000":NGAWest2_vertical['T1.000S']*981,
    "W_T1_200":NGAWest2_vertical['T1.200S']*981,
    "W_T1_400":NGAWest2_vertical['T1.400S']*981,
    "W_T1_600":NGAWest2_vertical['T1.600S']*981,
    "W_T1_800":NGAWest2_vertical['T1.800S']*981,
    "W_T2_000":NGAWest2_vertical['T2.000S']*981,
    "W_T2_500":NGAWest2_vertical['T2.500S']*981,
    "W_T3_000":NGAWest2_vertical['T3.000S']*981,
    "W_T3_500":NGAWest2_vertical['T3.500S']*981,
    "W_T4_000":NGAWest2_vertical['T4.000S']*981,
    "W_T4_500":NGAWest2_vertical['T4.500S']*981,
    "W_T5_000":NGAWest2_vertical['T5.000S']*981,
    "W_T6_000":NGAWest2_vertical['T6.000S']*981,
    "W_T7_000":NGAWest2_vertical['T7.000S']*981,
    "W_T8_000":NGAWest2_vertical['T8.000S']*981,
    "W_T9_000":NGAWest2_vertical['T9.000S']*981,
    "W_T10_000":NGAWest2_vertical['T10.000S']*981,
        
    "rotD50_T0_010":NGAWest2['T0.010S']*981,
    "rotD50_T0_025":NGAWest2['T0.025S']*981,
    "rotD50_T0_040":NGAWest2['T0.040S']*981,
    "rotD50_T0_050":NGAWest2['T0.050S']*981,
    "rotD50_T0_070":NGAWest2['T0.070S']*981,
    "rotD50_T0_100":NGAWest2['T0.100S']*981,
    "rotD50_T0_150":NGAWest2['T0.150S']*981,
    "rotD50_T0_200":NGAWest2['T0.200S']*981,
    "rotD50_T0_250":NGAWest2['T0.250S']*981,
    "rotD50_T0_300":NGAWest2['T0.300S']*981,
    "rotD50_T0_350":NGAWest2['T0.350S']*981,
    "rotD50_T0_400":NGAWest2['T0.400S']*981,
    "rotD50_T0_450":NGAWest2['T0.450S']*981,
    "rotD50_T0_500":NGAWest2['T0.500S']*981,
    "rotD50_T0_600":NGAWest2['T0.600S']*981,
    "rotD50_T0_700":NGAWest2['T0.700S']*981,
    "rotD50_T0_750":NGAWest2['T0.750S']*981,
    "rotD50_T0_800":NGAWest2['T0.800S']*981,
    "rotD50_T0_900":NGAWest2['T0.900S']*981,
    "rotD50_T1_000":NGAWest2['T1.000S']*981,
    "rotD50_T1_200":NGAWest2['T1.200S']*981,
    "rotD50_T1_400":NGAWest2['T1.400S']*981,
    "rotD50_T1_600":NGAWest2['T1.600S']*981,
    "rotD50_T1_800":NGAWest2['T1.800S']*981,
    "rotD50_T2_000":NGAWest2['T2.000S']*981,
    "rotD50_T2_500":NGAWest2['T2.500S']*981,
    "rotD50_T3_000":NGAWest2['T3.000S']*981,
    "rotD50_T3_500":NGAWest2['T3.500S']*981,
    "rotD50_T4_000":NGAWest2['T4.000S']*981,
    "rotD50_T4_500":NGAWest2['T4.500S']*981,
    "rotD50_T5_000":NGAWest2['T5.000S']*981,
    "rotD50_T6_000":NGAWest2['T6.000S']*981,
    "rotD50_T7_000":NGAWest2['T7.000S']*981,
    "rotD50_T8_000":NGAWest2['T8.000S']*981,
    "rotD50_T9_000":NGAWest2['T9.000S']*981,
    "rotD50_T10_000":NGAWest2['T10.000S']*981,
     
    "rotD00_T0_010":default_string,
    "rotD00_T0_025":default_string,
    "rotD00_T0_040":default_string,
    "rotD00_T0_050":default_string,
    "rotD00_T0_070":default_string,
    "rotD00_T0_100":default_string,
    "rotD00_T0_150":default_string,
    "rotD00_T0_200":default_string,
    "rotD00_T0_250":default_string,
    "rotD00_T0_300":default_string,
    "rotD00_T0_350":default_string,
    "rotD00_T0_400":default_string,
    "rotD00_T0_450":default_string,
    "rotD00_T0_500":default_string,
    "rotD00_T0_600":default_string,
    "rotD00_T0_700":default_string,
    "rotD00_T0_750":default_string,
    "rotD00_T0_800":default_string,
    "rotD00_T0_900":default_string,
    "rotD00_T1_000":default_string,
    "rotD00_T1_200":default_string,
    "rotD00_T1_400":default_string,
    "rotD00_T1_600":default_string,
    "rotD00_T1_800":default_string,
    "rotD00_T2_000":default_string,
    "rotD00_T2_500":default_string,
    "rotD00_T3_000":default_string,
    "rotD00_T3_500":default_string,
    "rotD00_T4_000":default_string,
    "rotD00_T4_500":default_string,
    "rotD00_T5_000":default_string,
    "rotD00_T6_000":default_string,
    "rotD00_T7_000":default_string,
    "rotD00_T8_000":default_string,
    "rotD00_T9_000":default_string,
    "rotD00_T10_000":default_string,
    
    "rotD100_T0_010":default_string,
    "rotD100_T0_025":default_string,
    "rotD100_T0_040":default_string,
    "rotD100_T0_050":default_string,
    "rotD100_T0_070":default_string,
    "rotD100_T0_100":default_string,
    "rotD100_T0_150":default_string,
    "rotD100_T0_200":default_string,
    "rotD100_T0_250":default_string,
    "rotD100_T0_300":default_string,
    "rotD100_T0_350":default_string,
    "rotD100_T0_400":default_string,
    "rotD100_T0_450":default_string,
    "rotD100_T0_500":default_string,
    "rotD100_T0_600":default_string,
    "rotD100_T0_700":default_string,
    "rotD100_T0_750":default_string,
    "rotD100_T0_800":default_string,
    "rotD100_T0_900":default_string,
    "rotD100_T1_000":default_string,
    "rotD100_T1_200":default_string,
    "rotD100_T1_400":default_string,
    "rotD100_T1_600":default_string,
    "rotD100_T1_800":default_string,
    "rotD100_T2_000":default_string,
    "rotD100_T2_500":default_string,
    "rotD100_T3_000":default_string,
    "rotD100_T3_500":default_string,
    "rotD100_T4_000":default_string,
    "rotD100_T4_500":default_string,
    "rotD100_T5_000":default_string,
    "rotD100_T6_000":default_string,
    "rotD100_T7_000":default_string,
    "rotD100_T8_000":default_string,
    "rotD100_T9_000":default_string,
    "rotD100_T10_000":default_string})
    
    # Output to folder where converted flatfile read into parser   
    DATA = os.path.abspath('')
    temp_folder=tempfile.mkdtemp()
    converted_base_data_path = os.path.join(DATA,temp_folder,
                                            'converted_flatfile.csv')
    ESM_original_headers.to_csv(converted_base_data_path,sep=';')

    print(Initial_NGAWest2_size - len(NGAWest2),
          'records removed from imported NGA-West-2 flatfile during data quality checks.')
    
    return converted_base_data_path