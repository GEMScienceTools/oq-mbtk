# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2024 GEM Foundation
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
from openquake.smt.residuals.parsers import valid
from openquake.smt.residuals.parsers.base_database_parser import SMDatabaseReader
from openquake.smt.utils_strong_motion import (
    MECHANISM_TYPE, DIP_TYPE, vs30_to_z1pt0_cy14, vs30_to_z2pt5_cb14)

# Import the ESM dictionaries
from .esm_dictionaries import *

SCALAR_LIST = ["PGA", "PGV", "PGD", "CAV", "CAV5", "Ia", "D5-95"]

HEADER_STR = "event_id;event_time;ISC_ev_id;ev_latitude;ev_longitude;"\
             "ev_depth_km;fm_type_code;ML;Mw;Ms;event_source_id;"\
             "es_strike;es_dip;es_rake;es_z_top;es_length;es_width;"\
             "network_code;station_code;location_code;"\
             "sensor_depth_m;"\
             "st_latitude;st_longitude;st_elevation;vs30_m_sec;slope_deg;"\
             "vs30_meas_type;epi_dist;epi_az;JB_dist;rup_dist;Rx_dist;"\
             "Ry0_dist;late_triggered_flag_01;U_channel_code;U_azimuth_deg;"\
             "V_channel_code;V_azimuth_deg;W_channel_code;U_hp;V_hp;W_hp;U_lp;"\
             "V_lp;W_lp"

HEADERS = set(HEADER_STR.split(";"))


class GEMFlatfileParser(SMDatabaseReader):
    """
    Parses the data from the flatfile to a set of metadata objects
    """
    M_PRECEDENCE = ["Mw", "Ms", "ML"]
    BUILD_FINITE_DISTANCES = False

    def parse(self, location='./'):
        """
        Parse the dataset
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
                print("Processed record %s - %s" % (str(counter), record.id))
                
            counter += 1

    @classmethod
    def autobuild(cls, dbid, dbname, output_location, flatfile_directory,
                  proxy=None, removal=None):
        """
        Quick and dirty full database builder!
        """
        # Import GEM strong-motion flatfile
        GEM = pd.read_csv(flatfile_directory)
    
        # Get path to tmp csv once modified dataframe
        conv_pth= prioritise_rotd50(GEM, proxy, removal)
        
        if os.path.exists(output_location):
            raise IOError("Target database directory %s already exists!"
                          % output_location)
        os.mkdir(output_location)
        # Add on the records folder
        os.mkdir(os.path.join(output_location, "records"))
        # Create an instance of the parser class
        database = cls(dbid, dbname, conv_pth)
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
                         eq_country=None)
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
        for key in self.M_PRECEDENCE:
            mvalue = metadata[key].strip()
            if mvalue:
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
        vs30_measured = None # User should check selected records (many diff. 
                             # flatfiles --> see vs30_meas_type column for if
                             # vs30 if available was measured of inferred)
        
        site = RecordSite(site_id, station_code, station_code, site_lon,
                          site_lat, elevation, vs30, vs30_measured,
                          network_code=network_code, country=None)
        site.slope = valid.vfloat(metadata["slope_deg"], "slope_deg")
        site.sensor_depth = valid.vfloat(metadata["sensor_depth_m"],
                                         "sensor_depth_m")
        if site.vs30:
            site.z1pt0 = vs30_to_z1pt0_cy14(vs30)
            site.z2pt5 = vs30_to_z2pt5_cb14(vs30)
        return site

    def _parse_waveform_data(self, metadata, wfid):
        """
        Parse the waveform data
        """
        late_trigger = valid.vint(metadata["late_triggered_flag_01"],
                                  "late_triggered_flag_01")
        # U channel - usually east
        xazimuth = valid.vfloat(metadata["U_azimuth_deg"], "U_azimuth_deg")
        xfilter = {"Low-Cut": valid.vfloat(metadata["U_hp"], "U_hp"),
                   "High-Cut": valid.vfloat(metadata["U_lp"], "U_lp")}
        xcomp = Component(wfid, xazimuth, waveform_filter=xfilter,
                          units="cm/s/s")
        xcomp.late_trigger = late_trigger
        
        # V channel - usually North
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


def prioritise_rotd50(df, proxy=None, removal=None):
    """
    Assign RotD50 values to horizontal acceleration columns for computation of
    residuals. If no RotD50 use the geometric mean if available (if specified
    in parser arguments) as a proxy for RotD50.
    
    RotD50 is available for the vast majority of the records in the GEM
    flatfile for PGA to 10 s.
    
    Records lacking acceleration values for any of the required spectral periods
    can also be removed (this information can alternatively just be printed)
    
    :param  proxy:
        If set to true, if a record is missing RotD50 try and use the geometric
        mean of the horizontal components (geometric mean is computed when
        calculating the residuals, here we just parse the two horizontals)
    
    :param  removal:
        If set to true records without complete RotD50 for any of the required
        spectral periods are removed. In instances that proxy is True, records
        without complete RotD50 even with use of geometric mean as a proxy are
        dropped
    """
    # TODO this approach is a bit hacky given we can use 'component' argument
    # within residuals.get_residuals() to specify if we want RotD50 or geometric
    # mean, but this function allows maximum number of records to be used in an
    # analysis by taking RotD50 if available, and then computing geometric mean
    # from the horizontal components if not. For spectral accelerations, RotD50
    # and geometric mean have been found to have a ratio close to 1 (e.g. Beyes
    # and Bommer, 2006)
    h_cols = ['U_T', 'V_T', 'U_pga', 'V_pga', 'U_pgv', 'V_pgv', 'U_pgd', 'V_pgd',
              'U_housner', 'V_housner', 'U_CAV', 'V_CAV', 'U_ia', 'V_ia']
    
    # Manage RotD50 vs horizontal components
    log, cols = [], []
    for idx, rec in df.iterrows():
        for col in rec.index:
            if 'rotD' in col:
                cols.append(col)
            for h_col in h_cols:
                if h_col in col:
                    if 'T90' not in col:
                        if 'U_' in col:    
                            rotd50_col = col.replace('U_', 'rotD50_')
                        if 'V_' in col:
                            rotd50_col = col.replace('V_', 'rotD50_')
                            
                        # If RotD50...
                        if not pd.isnull(rec[rotd50_col]): # Assign to h1, h2
                            df[col].iloc[idx] = rec[rotd50_col]
                            
                        # Otherwise...
                        else:
                            if proxy is True: # Use geo. mean as proxy
                                if not pd.isnull(rec[col]):
                                    pass # Use geo. mean from h1, h2 as proxy 
                            else:
                                log.append(idx) # Log as incomplete RotD50 vals
                            
    # Tidy dataframe
    cols = pd.Series(cols).unique()
    df = df.drop(columns=cols)

    # Drop if req. or else just inform number of recs missing acc. values
    no_vals = len(pd.Series(log).unique())
    if removal is True and log!= []:
        df = df.drop(log).reset_index()
        msg = 'Records without RotD50 acc. values for all periods between 0.01'
        msg += ' s and 10 s have been removed from database (%s records)' %no_vals
        print(msg)
        if len(df) == 0:
            raise ValueError('All records have been removed from the flatfile')        
    elif log != []:
        print('%s records lack RotD50 for all periods between 0.01 s and 10 s'
              % no_vals)
    
    # Output to temp folder where converted flatfile read into parser   
    conv_pth = os.path.join(tempfile.mkdtemp(), 'conv_flatfile_tmp.csv')
    df.to_csv(conv_pth, sep=';')
    
    return conv_pth