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
Parser for a flatfile downloaded from the esm custom url database 
--> (https://esm-db.eu/esmws/flatfile/1/)

This parser assumes you have selected all available headers in your URL search
when downloading the flatfile
"""
import os
import tempfile
import csv
import pickle
import pandas as pd
from linecache import getline

from openquake.smt.residuals.sm_database import (GroundMotionDatabase,
                                                 GroundMotionRecord)
from openquake.smt.residuals.parsers.esm_flatfile_parser import (parse_event_data,
                                                                 parse_distances,
                                                                 parse_site_data,
                                                                 parse_waveform_data,
                                                                 parse_ground_motion)
from openquake.smt.residuals.sm_database import (GroundMotionDatabase,
                                                 GroundMotionRecord,
                                                 Rupture,
                                                 FocalMechanism, 
                                                 GCMTNodalPlanes)
from openquake.smt.residuals.parsers import valid
from openquake.smt.residuals.parsers.base_database_parser import SMDatabaseReader
from openquake.smt.utils import MECHANISM_TYPE, DIP_TYPE


# Import the esm dictionaries
from .esm_dictionaries import *

BASE = os.path.abspath("")

HDEFS = ["Geometric", "rotD00", "rotD50", "rotD100"]

SCALAR_LIST = ["PGA", "PGV", "PGD", "CAV", "CAV5", "Ia", "D5-95"]

HEADERS = ["event_id",
           "event_time",
           "ISC_ev_id",
           "USGS_ev_id",
           "INGV_ev_id",
           "EMSC_ev_id",
           "ev_nation_code",
           "ev_latitude",
           "ev_longitude",
           "ev_depth_km",
           "ev_hyp_ref",
           "fm_type_code",
           "ML",
           "ML_ref",
           "Mw",
           "Mw_ref",
           "Ms",
           "Ms_ref",
           "EMEC_Mw",
           "EMEC_Mw_type",
           "EMEC_Mw_ref",
           "es_strike",
           "es_dip",
           "es_rake",
           "es_strike_dip_rake_ref",
           "es_z_top",
           "es_z_top_ref",
           "es_length",
           "es_width",
           "es_geometry_ref",
           "network_code",
           "station_code",
           "location_code",
           "instrument_code",
           "sensor_depth_m",
           "proximity_code",
           "housing_code",
           "installation_code",
           "st_nation_code",
           "st_latitude",
           "st_longitude",
           "st_elevation",
           "ec8_code",
           "ec8_code_method",
           "ec8_code_ref",
           "vs30_m_sec",
           "vs30_ref",
           "vs30_calc_method",
           "vs30_meas_type",
           "slope_deg",
           "vs30_m_sec_WA",
           "epi_dist",
           "epi_az",
           "JB_dist",
           "rup_dist",
           "Rx_dist",
           "Ry0_dist",
           "instrument_type_code",
           "late_triggered_flag_01",
           "U_channel_code",
           "U_azimuth_deg",
           "V_channel_code",
           "V_azimuth_deg",
           "W_channel_code",
           "U_hp",
           "V_hp",
           "W_hp",
           "U_lp",
           "V_lp",
           "W_lp"]


def parse_rupture_mechanism(metadata, eq_id, eq_name, mag, depth):
    """
    Parse rupture mechanism. 

    NOTE: The esm URL format flatfile does not contain necessarily provide
    an ``event_source_id`` value for each record providing finite rupture
    information, and therefore this function differs from the same one 
    within the esm18 parser (within which if there is an ``event_source_id``
    there is also complete finite rupture information e.g. length, width).
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
        # Has a valid focal mechanism (esm URL format flatfile only
        # provides one nodal plane
        mechanism.nodal_planes.nodal_plane_1 = {
            "strike": fm_set[0], "dip": fm_set[1], "rake": fm_set[2]}

    if not mechanism.nodal_planes.nodal_plane_1:
        # Absolutely no information - base on style-of-faulting
        mechanism.nodal_planes.nodal_plane_1 = {
            "strike": 0.0, "dip": DIP_TYPE[sof], "rake": MECHANISM_TYPE[sof]
            }
        
    return rupture, mechanism


class ESMFlatfileParserURL(SMDatabaseReader):
    """
    Parses the data from the flatfile to a set of metadata objects
    """
    def parse(self, location='./'):
        """
        Parse the flatfile
        """
        assert os.path.isfile(self.filename)
        headers = getline(self.filename, 1).rstrip("\n").split(";")
        for hdr in HEADERS:
            if hdr not in headers:
                raise ValueError(
                    "Required header %s is missing in file" % hdr)
        # Read in csv
        with open(self.filename, "r", encoding="utf-8", newline='') as f:
            reader = csv.DictReader(f, delimiter=";")
            self.database = GroundMotionDatabase(self.id, self.name)
            counter = 0
            for row in reader:
                # Build the metadata
                record = self._parse_record(row)
                if record:
                    # Parse the strong motion
                    record = parse_ground_motion(os.path.join(
                        location, "records"), row, record, headers)
                    self.database.records.append(record)

                else:
                    print("Record with sequence number %s is null/invalid"
                          % "{:s}-{:s}".format(
                              row["event_id"], row["station_code"]))
                    
                if (counter % 100) == 0:
                    print(f"Processed record {counter} - {record.id}")
                    
                counter += 1

    @classmethod
    def autobuild(cls, dbid, dbname, output_location, flatfile_location):
        """
        Quick and dirty full database builder!
        """
        # Import esm URL format strong-motion flatfile
        esm = pd.read_csv(flatfile_location)

        # Get path to tmp csv containing reformatted dataframe
        tmp = _parse_esm_url(esm)
        
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
        # Waveform ID not provided in file so concatenate Event and Station ID
        wfid = "_".join([metadata["event_id"], metadata["network_code"],
                         metadata["station_code"], metadata["location_code"]])
        wfid = wfid.replace("-", "_")
        
        # Parse the event metadata
        event = parse_event_data(metadata, parse_rupture_mechanism)
        
        # Parse the distance metadata
        distances = parse_distances(metadata, event.depth)
        
        # Parse the station metadata
        site = parse_site_data(metadata)
        
        # Parse waveform data
        xcomp, ycomp, vertical = parse_waveform_data(metadata, wfid)
        return GroundMotionRecord(wfid,
                                  [None, None, None],
                                  event, distances, site,
                                  xcomp, ycomp,
                                  vertical=vertical)

def _parse_esm_url(esm):
    """
    Convert from esm URL format flatfile to esm18 format flatfile
    """
    # Handle empty fm type values
    esm['fm_type_code'] = [fm if pd.notnull(fm) else "U" for fm in esm.fm_type_code]

    # Construct dataframe with original esm format 
    esm_original_headers = pd.DataFrame(
    {
    # Non-GMIM headers   
    "event_id":esm.esm_event_id,                                       
    "event_time":esm.event_time.str.replace('T',' '),
    "ISC_ev_id":esm.isc_event_id,
    "USGS_ev_id":esm.usgs_event_id,
    "INGV_ev_id":esm.ingv_event_id,
    "EMSC_ev_id":esm.emsc_event_id,
    "ev_nation_code":esm.ev_nation_code,
    "ev_latitude":esm.ev_latitude,    
    "ev_longitude":esm.ev_longitude,   
    "ev_depth_km":esm.ev_depth_km,
    "ev_hyp_ref":None,
    "fm_type_code":esm.fm_type_code,
    "ML":esm.ml,
    "ML_ref":esm.ml_ref,
    "Mw":esm.mw,
    "Mw_ref":esm.mw_ref,
    "Ms":esm.ms,
    "Ms_ref":esm.ms_ref,
    "EMEC_Mw":esm.emec_mw,
    "EMEC_Mw_type":esm.emec_mw_type,
    "EMEC_Mw_ref":esm.emec_mw_ref,

    "es_strike":esm.es_strike,
    "es_dip":esm.es_dip,
    "es_rake":esm.es_rake,
    "es_strike_dip_rake_ref":None, 
    "es_z_top":esm.z_top,
    "es_z_top_ref":esm.es_z_top_ref,
    "es_length":esm.es_length,   
    "es_width":esm.es_width,
    "es_geometry_ref":esm.es_geometry_ref,
 
    "network_code":esm.network_code,
    "station_code":esm.station_code,
    "location_code":esm.location_code,
    "instrument_code":esm.instrument_type_code,     
    "sensor_depth_m":esm.sensor_depth_m,
    "proximity_code":esm.proximity,
    "housing_code":esm.hounsing,    # Currently typo in their database header
    "installation_code":esm.installation,
    "st_nation_code":esm.st_nation_code,
    "st_latitude":esm.st_latitude,
    "st_longitude":esm.st_longitude,
    "st_elevation":esm.st_elevation,
    
    "ec8_code":esm.ec8_code,
    "ec8_code_method":None,
    "ec8_code_ref":None,
    "vs30_m_sec":esm.vs30_m_s,
    "vs30_ref":None,
    "vs30_calc_method":None, 
    "vs30_meas_type":esm.vs30_meas_type,
    "slope_deg":esm.slope_deg,
    "vs30_m_sec_WA":esm.vs30_m_s_wa,
 
    "epi_dist":esm.epi_dist,
    "epi_az":esm.epi_az,  
    "JB_dist":esm.jb_dist,
    "rup_dist":esm.rup_dist, 
    "Rx_dist":esm.rx_dist, 
    "Ry0_dist":esm.ry0_dist,
 
    "instrument_type_code":esm.instrument_type_code,      
    "late_triggered_flag_01":esm.late_triggered_event_01,
    "U_channel_code":esm.u_channel_code,
    "U_azimuth_deg":esm.u_azimuth_deg,
    "V_channel_code":esm.v_channel_code,
    "V_azimuth_deg":esm.v_azimuth_deg,
    "W_channel_code":esm.w_channel_code,
    
    "U_hp":esm.u_hp,
    "V_hp":esm.v_hp,
    "W_hp":esm.w_hp,  
    "U_lp":esm.u_lp,
    "V_lp":esm.v_lp,
    "W_lp":esm.w_lp,
     
    "U_pga":esm.u_pga,
    "V_pga":esm.v_pga,
    "W_pga":esm.w_pga,
    "rotD50_pga":esm.rotd50_pga,
    "rotD100_pga":esm.rotd100_pga,
    "rotD00_pga":esm.rotd00_pga,
    "U_pgv":esm.u_pgv,
    "V_pgv":esm.v_pgv,
    "W_pgv":esm.w_pgv,
    "rotD50_pgv":esm.rotd50_pgv,
    "rotD100_pgv":esm.rotd100_pgv,
    "rotD00_pgv":esm.rotd00_pgv,
    "U_pgd":esm.u_pgd,
    "V_pgd":esm.v_pgd,
    "W_pgd":esm.w_pgd,
    "rotD50_pgd":esm.rotd50_pgd,
    "rotD100_pgd":esm.rotd100_pgd,
    "rotD00_pgd":esm.rotd00_pgv,
    "U_T90":esm.u_t90,
    "V_T90":esm.v_t90,
    "W_T90":esm.w_t90,
    "rotD50_T90":esm.rotd50_t90,
    "rotD100_T90":esm.rotd100_t90,
    "rotD00_T90":esm.rot_d00_t90, # This header has typo in current db version 
    "U_housner":esm.u_housner,
    "V_housner":esm.v_housner,
    "W_housner":esm.w_housner,
    "rotD50_housner":esm.rotd50_housner,
    "rotD100_housner":esm.rotd100_housner,
    "rotD00_housner":esm.rotd00_housner,
    "U_CAV":esm.u_cav,
    "V_CAV":esm.v_cav,
    "W_CAV":esm.w_cav,
    "rotD50_CAV":esm.rotd50_cav,
    "rotD100_CAV":esm.rotd100_cav,
    "rotD00_CAV":esm.rotd00_cav,
    "U_ia":esm.u_ia,
    "V_ia":esm.v_ia,
    "W_ia":esm.w_ia,
    "rotD50_ia":esm.rotd50_ia,
    "rotD100_ia":esm.rotd100_ia,
    "rotD00_ia":esm.rotd00_ia,
    
    "U_T0_010":esm.u_t0_010,
    "U_T0_025":esm.u_t0_025,
    "U_T0_040":esm.u_t0_040,
    "U_T0_050":esm.u_t0_050,
    "U_T0_070":esm.u_t0_070,
    "U_T0_100":esm.u_t0_100,
    "U_T0_150":esm.u_t0_150,
    "U_T0_200":esm.u_t0_200,
    "U_T0_250":esm.u_t0_250,
    "U_T0_300":esm.u_t0_300,
    "U_T0_350":esm.u_t0_350,
    "U_T0_400":esm.u_t0_400,
    "U_T0_450":esm.u_t0_450,
    "U_T0_500":esm.u_t0_500,
    "U_T0_600":esm.u_t0_600,
    "U_T0_700":esm.u_t0_700,
    "U_T0_750":esm.u_t0_750,
    "U_T0_800":esm.u_t0_800,
    "U_T0_900":esm.u_t0_900,
    "U_T1_000":esm.u_t1_000,
    "U_T1_200":esm.u_t1_200,
    "U_T1_400":esm.u_t1_400,
    "U_T1_600":esm.u_t1_600,
    "U_T1_800":esm.u_t1_800,
    "U_T2_000":esm.u_t2_000,
    "U_T2_500":esm.u_t2_500,
    "U_T3_000":esm.u_t3_000,
    "U_T3_500":esm.u_t3_500,
    "U_T4_000":esm.u_t4_000,
    "U_T4_500":esm.u_t4_500,
    "U_T5_000":esm.u_t5_000,
    "U_T6_000":esm.u_t6_000,
    "U_T7_000":esm.u_t7_000,
    "U_T8_000":esm.u_t8_000,
    "U_T9_000":esm.u_t9_000,
    "U_T10_000":esm.u_t10_000,
       
    "V_T0_010":esm.v_t0_010,
    "V_T0_025":esm.v_t0_025,
    "V_T0_040":esm.v_t0_040,
    "V_T0_050":esm.v_t0_050,
    "V_T0_070":esm.v_t0_070,
    "V_T0_100":esm.v_t0_100,
    "V_T0_150":esm.v_t0_150,
    "V_T0_200":esm.v_t0_200,
    "V_T0_250":esm.v_t0_250,
    "V_T0_300":esm.v_t0_300,
    "V_T0_350":esm.v_t0_350,
    "V_T0_400":esm.v_t0_400,
    "V_T0_450":esm.v_t0_450,
    "V_T0_500":esm.v_t0_500,
    "V_T0_600":esm.v_t0_600,
    "V_T0_700":esm.v_t0_700,
    "V_T0_750":esm.v_t0_750,
    "V_T0_800":esm.v_t0_800,
    "V_T0_900":esm.v_t0_900,
    "V_T1_000":esm.v_t1_000,
    "V_T1_200":esm.v_t1_200,
    "V_T1_400":esm.v_t1_400,
    "V_T1_600":esm.v_t1_600,
    "V_T1_800":esm.v_t1_800,
    "V_T2_000":esm.v_t2_000,
    "V_T2_500":esm.v_t2_500,
    "V_T3_000":esm.v_t3_000,
    "V_T3_500":esm.v_t3_500,
    "V_T4_000":esm.v_t4_000,
    "V_T4_500":esm.v_t4_500,
    "V_T5_000":esm.v_t5_000,
    "V_T6_000":esm.v_t6_000,
    "V_T7_000":esm.v_t7_000,
    "V_T8_000":esm.v_t8_000,
    "V_T9_000":esm.v_t9_000,
    "V_T10_000":esm.v_t10_000,
    
    "W_T0_010":esm.w_t0_010,
    "W_T0_025":esm.w_t0_025,
    "W_T0_040":esm.w_t0_040,
    "W_T0_050":esm.w_t0_050,
    "W_T0_070":esm.w_t0_070,
    "W_T0_100":esm.w_t0_100,
    "W_T0_150":esm.w_t0_150,
    "W_T0_200":esm.w_t0_200,
    "W_T0_250":esm.w_t0_250,
    "W_T0_300":esm.w_t0_300,
    "W_T0_350":esm.w_t0_350,
    "W_T0_400":esm.w_t0_400,
    "W_T0_450":esm.w_t0_450,
    "W_T0_500":esm.w_t0_500,
    "W_T0_600":esm.w_t0_600,
    "W_T0_700":esm.w_t0_700,
    "W_T0_750":esm.w_t0_750,
    "W_T0_800":esm.w_t0_800,
    "W_T0_900":esm.w_t0_900,
    "W_T1_000":esm.w_t1_000,
    "W_T1_200":esm.w_t1_200,
    "W_T1_400":esm.w_t1_400,
    "W_T1_600":esm.w_t1_600,
    "W_T1_800":esm.w_t1_800,
    "W_T2_000":esm.w_t2_000,
    "W_T2_500":esm.w_t2_500,
    "W_T3_000":esm.w_t3_000,
    "W_T3_500":esm.w_t3_500,
    "W_T4_000":esm.w_t4_000,
    "W_T4_500":esm.w_t4_500,
    "W_T5_000":esm.w_t5_000,
    "W_T6_000":esm.w_t6_000,
    "W_T7_000":esm.w_t7_000,
    "W_T8_000":esm.w_t8_000,
    "W_T9_000":esm.w_t9_000,
    "W_T10_000":esm.w_t10_000,
    
    "rotD50_T0_010":esm.rotd50_t0_010,
    "rotD50_T0_025":esm.rotd50_t0_025,
    "rotD50_T0_040":esm.rotd50_t0_040,
    "rotD50_T0_050":esm.rotd50_t0_050,
    "rotD50_T0_070":esm.rotd50_t0_070,
    "rotD50_T0_100":esm.rotd50_t0_100,
    "rotD50_T0_150":esm.rotd50_t0_150,
    "rotD50_T0_200":esm.rotd50_t0_200,
    "rotD50_T0_250":esm.rotd50_t0_250,
    "rotD50_T0_300":esm.rotd50_t0_300,
    "rotD50_T0_350":esm.rotd50_t0_350,
    "rotD50_T0_400":esm.rotd50_t0_400,
    "rotD50_T0_450":esm.rotd50_t0_450,
    "rotD50_T0_500":esm.rotd50_t0_500,
    "rotD50_T0_600":esm.rotd50_t0_600,
    "rotD50_T0_700":esm.rotd50_t0_700,
    "rotD50_T0_750":esm.rotd50_t0_750,
    "rotD50_T0_800":esm.rotd50_t0_800,
    "rotD50_T0_900":esm.rotd50_t0_900,
    "rotD50_T1_000":esm.rotd50_t1_000,
    "rotD50_T1_200":esm.rotd50_t1_200,
    "rotD50_T1_400":esm.rotd50_t1_400,
    "rotD50_T1_600":esm.rotd50_t1_600,
    "rotD50_T1_800":esm.rotd50_t1_800,
    "rotD50_T2_000":esm.rotd50_t2_000,
    "rotD50_T2_500":esm.rotd50_t2_500,
    "rotD50_T3_000":esm.rotd50_t3_000,
    "rotD50_T3_500":esm.rotd50_t3_500,
    "rotD50_T4_000":esm.rotd50_t4_000,
    "rotD50_T4_500":esm.rotd50_t4_500,
    "rotD50_T5_000":esm.rotd50_t5_000,
    "rotD50_T6_000":esm.rotd50_t6_000,
    "rotD50_T7_000":esm.rotd50_t7_000,
    "rotD50_T8_000":esm.rotd50_t8_000,
    "rotD50_T9_000":esm.rotd50_t9_000,
    "rotD50_T10_000":esm.rotd50_t10_000,
       
    
    "rotD100_T0_010":esm.rotd100_t0_010,
    "rotD100_T0_025":esm.rotd100_t0_025,
    "rotD100_T0_040":esm.rotd100_t0_040,
    "rotD100_T0_050":esm.rotd100_t0_050,
    "rotD100_T0_070":esm.rotd100_t0_070,
    "rotD100_T0_100":esm.rotd100_t0_100,
    "rotD100_T0_150":esm.rotd100_t0_150,
    "rotD100_T0_200":esm.rotd100_t0_200,
    "rotD100_T0_250":esm.rotd100_t0_250,
    "rotD100_T0_300":esm.rotd100_t0_300,
    "rotD100_T0_350":esm.rotd100_t0_350,
    "rotD100_T0_400":esm.rotd100_t0_400,
    "rotD100_T0_450":esm.rotd100_t0_450,
    "rotD100_T0_500":esm.rotd100_t0_500,
    "rotD100_T0_600":esm.rotd100_t0_600,
    "rotD100_T0_700":esm.rotd100_t0_700,
    "rotD100_T0_750":esm.rotd100_t0_750,
    "rotD100_T0_800":esm.rotd100_t0_800,
    "rotD100_T0_900":esm.rotd100_t0_900,
    "rotD100_T1_000":esm.rotd100_t1_000,
    "rotD100_T1_200":esm.rotd100_t1_200,
    "rotD100_T1_400":esm.rotd100_t1_400,
    "rotD100_T1_600":esm.rotd100_t1_600,
    "rotD100_T1_800":esm.rotd100_t1_800,
    "rotD100_T2_000":esm.rotd100_t2_000,
    "rotD100_T2_500":esm.rotd100_t2_500,
    "rotD100_T3_000":esm.rotd100_t3_000,
    "rotD100_T3_500":esm.rotd100_t3_500,
    "rotD100_T4_000":esm.rotd100_t4_000,
    "rotD100_T4_500":esm.rotd100_t4_500,
    "rotD100_T5_000":esm.rotd100_t5_000,
    "rotD100_T6_000":esm.rotd100_t6_000,
    "rotD100_T7_000":esm.rotd100_t7_000,
    "rotD100_T8_000":esm.rotd100_t8_000,
    "rotD100_T9_000":esm.rotd100_t9_000,
    "rotD100_T10_000":esm.rotd100_t10_000,      
 
    "rotD00_T0_010":esm.rotd00_t0_010,
    "rotD00_T0_025":esm.rotd00_t0_025,
    "rotD00_T0_040":esm.rotd00_t0_040,
    "rotD00_T0_050":esm.rotd00_t0_050,
    "rotD00_T0_070":esm.rotd00_t0_070,
    "rotD00_T0_100":esm.rotd00_t0_100,
    "rotD00_T0_150":esm.rotd00_t0_150,
    "rotD00_T0_200":esm.rotd00_t0_200,
    "rotD00_T0_250":esm.rotd00_t0_250,
    "rotD00_T0_300":esm.rotd00_t0_300,
    "rotD00_T0_350":esm.rotd00_t0_350,
    "rotD00_T0_400":esm.rotd00_t0_400,
    "rotD00_T0_450":esm.rotd00_t0_450,
    "rotD00_T0_500":esm.rotd00_t0_500,
    "rotD00_T0_600":esm.rotd00_t0_600,
    "rotD00_T0_700":esm.rotd00_t0_700,
    "rotD00_T0_750":esm.rotd00_t0_750,
    "rotD00_T0_800":esm.rotd00_t0_800,
    "rotD00_T0_900":esm.rotd00_t0_900,
    "rotD00_T1_000":esm.rotd00_t1_000,
    "rotD00_T1_200":esm.rotd00_t1_200,
    "rotD00_T1_400":esm.rotd00_t1_400,
    "rotD00_T1_600":esm.rotd00_t1_600,
    "rotD00_T1_800":esm.rotd00_t1_800,
    "rotD00_T2_000":esm.rotd00_t2_000,
    "rotD00_T2_500":esm.rotd00_t2_500,
    "rotD00_T3_000":esm.rotd00_t3_000,
    "rotD00_T3_500":esm.rotd00_t3_500,
    "rotD00_T4_000":esm.rotd00_t4_000,
    "rotD00_T4_500":esm.rotd00_t4_500,
    "rotD00_T5_000":esm.rotd00_t5_000,
    "rotD00_T6_000":esm.rotd00_t6_000,
    "rotD00_T7_000":esm.rotd00_t7_000,
    "rotD00_T8_000":esm.rotd00_t8_000,
    "rotD00_T9_000":esm.rotd00_t9_000,
    "rotD00_T10_000":esm.rotd00_t10_000})
    
    # Export to tmp
    tmp = os.path.join(BASE, tempfile.mkdtemp(), 'tmp.csv')
    esm_original_headers.to_csv(tmp, sep=';')

    return tmp