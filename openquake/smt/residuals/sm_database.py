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
Basic Pseudo-database built on top of hdf5 for a set of processed strong
motion records
"""
import os
import pickle
import numpy as np
import h5py
from datetime import datetime

from openquake.hazardlib import imt
from openquake.hazardlib import scalerel 
from openquake.hazardlib.site import Site, SiteCollection
from openquake.hazardlib.geo.point import Point
from openquake.hazardlib.geo import geodetic
from openquake.hazardlib.contexts import ContextMaker
from openquake.hazardlib.const import TRT

from openquake.smt import utils
from openquake.smt.residuals.context_db import ContextDB
import openquake.smt.utils_intensity_measures as utils_imts


class Magnitude(object):
    """
    Class to hold magnitude attributes
    :param float value:
        The magnitude value
    :param str mtype:
        Magnitude Type
    :param float sigma:
        The magnitude uncertainty (standard deviation)
    """
    def __init__(self, value, mtype, sigma=None, source=""):
        self.value = value
        self.mtype = mtype
        self.sigma = sigma
        self.source = source

    def __eq__(self, m):
        same_value = np.fabs(self.value - m.value) < 1.0E-3
        if self.sigma and m.sigma:
            same_sigma = np.fabs(self.sigma - m.sigma) < 1.0E-7
        else:
            same_sigma = self.sigma == m.sigma
        return same_value and same_sigma and (self.mtype == m.mtype) and\
            (self.source == m.source)


class Rupture(object):
    """
    Class to hold rupture attributes
    :param str id:
        Rupture (earthquake) ID
    :param str name:
        Event Name
    :param magnitude:
        Earthquake magnitude as instance of Magnitude class
    :param float length:
        Total rupture length (km)
    :param float width:
        Total rupture width (km)
    :param float depth:
        Depth to the top of rupture (km)
    :param float area:
        Rupture area in km^2
    :param surface:
        Rupture surface as instance of :class:
        openquake.hazardlib.geo.surface.base.BaseRuptureSurface
    :param tuple hypo_loc:
        Hypocentral location within rupture surface as a fraction of
        (along-strike length, down-dip width)
    """
    def __init__(self,
                 eq_id,
                 eq_name,
                 magnitude,
                 length,
                 width,
                 depth,
                 hypocentre=None,
                 area=None,
                 surface=None,
                 hypo_loc=None):
        self.id = eq_id
        self.name = eq_name
        self.magnitude = magnitude
        self.length = length
        self.width = width
        self.area = area
        self.area = self.get_area()
        self.depth = depth
        self.surface = surface
        self.hypocentre = hypocentre
        self.hypo_loc = hypo_loc
        self.aspect = None
        self.aspect = self.get_aspect()

    def get_area(self):
        """
        Returns the area of the rupture
        """
        if self.area:
            return self.area
        if self.length and self.width:
            return self.length * self.width
        else:
            return None

    def get_aspect(self):
        """
        Returns the aspect ratio
        """
        if self.aspect:
            # Trivial case
            return self.aspect
        if self.length and self.width:
            # If length and width both specified
            return self.length / self.width
        if self.length and self.area:
            # If length and area specified
            self.width = self.area / self.length
            return self.length / self.width
        if self.width and self.area:
            # If width and area specified
            self.length = self.area / self.width
            return self.length / self.width
        # out of options - returning None
        return None


class GCMTNodalPlanes(object):
    """
    Class to represent the nodal plane distribution of the tensor
    Each nodal plane is represented as a dictionary of the form:
    {'strike':, 'dip':, 'rake':}
    
    :param Union[dict, None] nodal_plane_1: First nodal plane
    :param Union[dict, None] nodal_plane_2: Second nodal plane
    """
    def __init__(self):
        self.nodal_plane_1 = None
        self.nodal_plane_2 = None


class GCMTPrincipalAxes(object):
    """
    Class to represent the eigensystem of the tensor in terms of
    T-, B- and P-plunge and azimuth
    #_axis = {'eigenvalue':, 'azimuth':, 'plunge':}

    :param dict | None t_axis: The eigensystem of the T-axis
    :param dict | None b_axis: The eigensystem of the B-axis
    :param dict | None p_axis: The eigensystem of the P-axis
    """
    def __init__(self):
        self.t_axis = None
        self.b_axis = None
        self.p_axis = None


class FocalMechanism(object):
    """
    Class to hold the full focal mechanism attribute set
    :param str eq_id:
        Identifier of the earthquake
    :param str name:
        Focal mechanism name
    :param nodal_planes:
        Nodal planes as instance of :class: GCMTNodalPlane
    :param eigenvalues:
        Eigenvalue decomposition as instance of :class: GCMTPrincipalAxes
    :param numpy.ndarray tensor:
        (3, 3) Moment Tensor
    :param str mechanism_type:
        Qualitative description of mechanism
    """
    def __init__(self,
                 eq_id,
                 name,
                 nodal_planes,
                 eigenvalues,
                 moment_tensor=None,
                 mechanism_type=None):
        self.id = eq_id
        self.name = name
        self.nodal_planes = nodal_planes
        self.eigenvalues = eigenvalues
        self.scalar_moment = None
        self.tensor = moment_tensor
        self.mechanism_type = mechanism_type

    def get_rake_from_mechanism_type(self):
        """
        Returns an idealised "rake" based on a qualitative description of the
        style of faulting
        """
        if self.mechanism_type in utils.MECHANISM_TYPE:
            return utils.MECHANISM_TYPE[self.mechanism_type]
        return 0.0

    def _moment_tensor_to_list(self):
        """
        Moment tensor to list
        """
        if self.tensor is None:
            return None
        else:
            return self.tensor.to_list() 


class Earthquake(object):
    """
    Class to hold earthquake event related information
    :param str id:
        Earthquake ID
    :param str name:
        Earthquake name
    :param datetime:
        Earthquake date and time as instance of :class: datetime.datetime
    :param float longitude:
        Earthquake hypocentre longitude
    :param float latitude:
        Earthquake hypocentre latitude
    :param float depth:
        Earthquake hypocentre depth (km)
    :param magnitude:
        Primary magnitude as instance of :class: Magnitude
    :param magnitude_list:
        Magnitude solutions for the earthquake as list of instances of the
        :class: Magntiude
    :param mechanism:
        Focal mechanism as instance of the :class: FocalMechanism
    :param rupture:
        Earthquake rupture as instance of the :class: Rupture
    """
    def __init__(self,
                 eq_id,
                 name,
                 date_time,
                 longitude,
                 latitude,
                 depth,
                 magnitude,
                 focal_mechanism=None,
                 eq_country=None,
                 tectonic_region=None):
        self.id = eq_id
        assert isinstance(date_time, datetime)
        self.datetime = date_time
        self.name = name
        self.country = eq_country
        self.longitude = longitude
        self.latitude = latitude
        self.depth = depth
        self.magnitude = magnitude
        self.magnitude_list = []
        self.mechanism = focal_mechanism
        self.rupture = None
        self.tectonic_region = tectonic_region


class RecordDistance(object):
    """
    Class to hold source to site distance information.
    :param float repi:
        Epicentral distance (km)
    :param float rhypo:
        Hypocentral distance (km)
    :param float rjb:
        Joyner-Boore distance (km)
    :param float rrup:
        Rupture distance (km)
    :param float r_x:
        Cross-track distance from site to up-dip projection of fault plane
        to surface
    :param float ry0:
        Along-track distance from site to surface projection of fault plane
    :param rvolc:
        Horizontal distance traversed through zone of volcanic activity.
    :param float rcdpp:
        Direct point parameter for directivity effect centered on the site-
        and earthquake-specific average DPP used
    """
    def __init__(self,
                 repi=None,
                 rhypo=None,
                 rjb=None,
                 rrup=None,
                 r_x=None,
                 ry0=None,
                 rvolc=None,
                 rcdpp=None):
        
        self.repi = repi
        self.rhypo = rhypo
        self.rjb = rjb
        self.rrup = rrup
        self.r_x = r_x
        self.ry0 = ry0
        self.rvolc = rvolc
        self.rcdpp = rcdpp


# Eurocode 8 Site Class Vs30 boundaries
EC8_VS30_BOUNDARIES = {
    "A": (800.0, np.inf),
    "B": (360.0, 800.0),
    "C": (180.0, 360.0),
    "D": (100.0, 180.0),
    "S1": (-np.inf, 100)
}


# Eurocode 8 Site Class NSPT boundaries
EC8_NSPT_BOUNDARIES = {
    "B": (50.0, np.inf),
    "C": (15.0, 50.0),
    "D": (-np.inf, 15.0)
}


# NEHRP Site Class Vs30 boundaries
NEHRP_VS30_BOUNDARIES = {
    "A": (1500.0, np.inf),
    "B": (760.0, 1500.0),
    "C": (360.0, 760.0),
    "D": (180.0, 360.0),
    "E": (-np.inf, 180.0)
}


# NEHRP Site Class NSPT boundaries
NEHRP_NSPT_BOUNDARIES = {
    "C": (50.0, np.inf),
    "D": (15.0, 50.0),
    "E": (-np.inf, 15.0)
}


class RecordSite(object):
    """
    Class to hold attributes belonging to the site
    :param str site_id:
        Site identifier
    :param str site_code:
        Network site code
    :param site_name:
        Network site name
    :param float longitude:
        Site longitude
    :param float latitude:
        Site latitude
    :param float altitude:
        Site elevation (m)
    :param site_class:
        Qualitative description of site class ("Rock", "Stiff Soil" etc.)
    :param float vs30:
        30-m average shear wave velocity (m/s)
    :param str vs30_measured:
        Vs30 is "measured" or "Inferred"
    :param str vs30_measured_type:
        Method for measuring Vs30
    :param float vs30_uncertainty:
        Standard error of Vs30
    :param float nspt:
        Number of blows of standard penetration test
    :param str nehrp:
        NEHRP Site Class
    :param str ec8:
        Eurocode 8 Site Class
    :param str building_structure:
        Description of structure hosting the instrument
    :param int number_floors:
        Number of floors of structure hosting the instrument
    :param int floor:
        Floor number for location of instrument
    :param str instrument_type:
        Description of instrument type
    :param str digitiser:
        Description of digitiser
    :param str network_code:
        Code of strong motion recording network
    :param str country:
        Country of site
    :param float z1pt0:
        Depth (m) to 1.0 km/s shear-wave velocity interface
    :param float z1pt5:
        Depth (m) to 1.5 km/s shear-wave velocity interface
    :param float z2pt5:
        Depth (km) to 2.5 km/s shear-wave velocity interface
    :param book backarc:
        True if site is in subduction backarc, False otherwise

    """
    def __init__(self,
                 site_id,
                 site_code,
                 site_name,
                 longitude,
                 latitude,
                 altitude,
                 vs30=None,
                 vs30_measured=None,
                 network_code=None,
                 country=None,
                 site_class=None,
                 backarc=False):
        self.id = site_id
        self.name = site_name
        self.code = site_code
        self.longitude = longitude
        self.latitude = latitude
        self.altitude = altitude
        self.site_class = site_class
        self.vs30 = vs30
        self.vs30_measured = vs30_measured
        self.vs30_measured_type = None
        self.vs30_uncertainty = None
        self.nspt = None
        self.nehrp = None
        self.ec8 = None
        self.building_structure = None
        self.number_floors = None
        self.floor = None
        self.instrument_type = None
        self.digitiser = None
        self.network_code = network_code
        self.sensor_depth = None
        self.country = country
        self.z1pt0 = None
        self.z1pt5 = None
        self.z2pt5 = None
        self.backarc = backarc
        self.morphology = None
        self.slope = None

    def to_openquake_site(self):
        """
        Returns the site as an instance of the :class:
        openquake.hazardlib.site.Site
        """
        if self.vs30:
            vs30 = self.vs30
            vs30_measured = self.vs30_measured

        if self.z1pt0:
            z1pt0 = self.z1pt0

        if self.z2pt5:
            z2pt5 = self.z2pt5
        
        location = Point(self.longitude,
                         self.latitude,
                        -self.altitude / 1000.)  # Elevation from m to km
        oq_site = Site(location,
                       vs30,
                       z1pt0,
                       z2pt5,
                       vs30measured=vs30_measured,
                       backarc=self.backarc)
        setattr(oq_site, "id", self.id)
        return oq_site

    def get_ec8_class(self):
        """
        Returns the EC8 class associated with a site given a Vs30
        """
        if self.ec8:
            return self.ec8
        if self.vs30:
            for key in EC8_VS30_BOUNDARIES:
                in_group = (self.vs30 >= EC8_VS30_BOUNDARIES[key][0]) and\
                    (self.vs30 < EC8_VS30_BOUNDARIES[key][1])
                if in_group:
                    self.ec8 = key
                    return self.ec8
        elif self.nspt:
            # Check to see if a site class can be determined from NSPT
            for key in EC8_NSPT_BOUNDARIES:
                in_group = (self.nspt >= EC8_NSPT_BOUNDARIES[key][0]) and\
                    (self.nspt < EC8_NSPT_BOUNDARIES[key][1])
                if in_group:
                    self.ec8 = key
                    return self.ec8
        else:
            print("Cannot determine EC8 site class - no Vs30 or NSPT measures!")
        return None
    
    def get_nehrp_class(self):
        """
        Returns the NEHRP class associated with a site given a Vs30 or NSPT
        """
        if self.nehrp:
            return self.nehrp
        if self.vs30:
            for key in NEHRP_VS30_BOUNDARIES:
                in_group = (self.vs30 >= NEHRP_VS30_BOUNDARIES[key][0]) and\
                    (self.vs30 < NEHRP_VS30_BOUNDARIES[key][1])
                if in_group:
                    self.nehrp = key
                    return self.nehrp
        elif self.nspt:
            # Check to see if a site class can be determined from NSPT
            for key in NEHRP_NSPT_BOUNDARIES:
                in_group = (self.nspt >= NEHRP_NSPT_BOUNDARIES[key][0]) and\
                    (self.nspt < NEHRP_NSPT_BOUNDARIES[key][1])
                if in_group:
                    self.nehrp = key
                    return self.nehrp
        else:
            print("Cannot determine NEHRP site class - no Vs30 or NSPT measures!")
        return None

    def vs30_from_ec8(self):
        """
        Returns an approximation of Vs30 given an EC8 site class (e.g. for the case
        when Vs30 is not measured but the site class is given).
        """
        if self.ec8 == 'A':
            return 900
        if self.ec8 == 'B':
            return 580
        if self.ec8 == 'C':
            return 220
        if self.ec8 == 'D':
            return 100
        if self.ec8 == 'E':
            return 100
        else:
            print("Cannot determine Vs30 from EC8 site class")


Filter = {
    'Type': None,
    'Order': None,
    'Passes': None,
    'Low-Cut': None,
    'High-Cut': None
}


Baseline = {
    'Type': None,
    'Start': None,
    'End': None
}


ims_dict = {
    'PGA': None,
    'PGV': None,
    'PGD': None,
    'CAV': None,
    'Ia': None,
    'CAV5': None,
    'arms': None,
    'd5_95': None,
    'd5_75': None
}


class Component(object):
    """
    Contains the metadata relating to waveform of the record
    :param str id:
        Waveform unique identifier
    :param orientation:
        Orientation of record as either azimuth (degrees, float) or string
    :param dict ims:
        Intensity Measures of component
    :param float longest_period:
        Longest usable period (s)
    :param dict waveform_filter:
        Waveform filter properties as dictionary
    :param dict baseline:
        Baseline correction metadata
    :param str units:
        Units of record
        
    """
    def __init__(self,
                 waveform_id,
                 orientation,
                 ims=None,
                 longest_period=None,
                 waveform_filter=None,
                 baseline=None,
                 units=None):
        self.id = waveform_id
        self.orientation = orientation
        self.lup = longest_period
        self.sup = None
        self.filter = waveform_filter
        self.baseline = baseline
        self.ims = ims
        self.units = units  # Equivalent to gain unit
        self.late_trigger = None



class GroundMotionRecord(object):
    """
    Class containing the full representation of the strong motion record
    :param str id:
        Ground motion record unique identifier
    :param str time_series_file:
        Path to time series file
    :param str spectra_file:
        Path to spectra file
    :param event:
        Earthquake event representation as :class: Earthquake
    :param distance:
        Distances representation as :class: RecordDistance
    :param site:
        Site representation as :class: RecordSite
    :param xrecord:
        x-component of record as instance of :class: Component
    :param yrecord:
        y-component of record as instance of :class: Component
    :param vertical:
         vertical component of record as instance of :class: Component
    :param float average_lup:
        Longest usable period of record-pair
    :param float average_sup:
        Shortest usable period of record-pair
    :param dict ims:
        Intensity measure of record
    :param directivity:
        ?
    :param str datafile:
        Data file for strong motion record
    """
    def __init__(self, gm_id, time_series_file, event, distance, record_site,
                 x_comp, y_comp, vertical=None, ims=None, longest_period=None,
                 shortest_period=None, spectra_file=None):
        self.id = gm_id
        self.time_series_file = time_series_file
        self.spectra_file = spectra_file
        assert isinstance(event, Earthquake)
        self.event = event
        assert isinstance(distance, RecordDistance)
        self.distance = distance
        assert isinstance(record_site, RecordSite)
        self.site = record_site
        assert isinstance(x_comp, Component) and isinstance(y_comp, Component)
        self.xrecord = x_comp
        self.yrecord = y_comp
        if vertical:
            assert isinstance(vertical, Component)
        self.vertical = vertical
        self.average_lup = longest_period
        self.average_sup = shortest_period
        self.ims = ims
        self.directivity = None
        self.datafile = None
        self.misc = None

    def get_azimuth(self):
        """
        If the azimuth is missing, returns the epicentre to station azimuth
        """
        if self.distance.azimuth:
            return self.distance.azimuth
        else:
            self.distance.azimuth = geodetic.azimuth(
                self.event.longitude,
                self.event.latitude,
                self.site.longitude,
                self.site.latitude)
        return self.distance.azimuth


class GroundMotionDatabase(ContextDB):
    """
    Class to represent a database of strong motions
    :param str db_id:
        Database identifier
    :param str db_name:
        Database name
    :param str db_directory:
        Path to database directory
    :param list records:
        Strong motion data as list of :class: GroundMotionRecord (defaults to
        None: empty list)
    :param list site_ids:
        List of site ids (defaults to None: empty list)
    """
    def __init__(self, db_id, db_name, db_directory=None, records=None,
                 site_ids=None):
        self.id = db_id
        self.name = db_name
        self.directory = db_directory
        self.records = list(records) if records is not None else []
        self.site_ids = list(site_ids) if site_ids is not None else []

    def __iter__(self):
        """
        Make this object iterable, i.e.
        `for rec in self` is equal to `for rec in self.records`
        """
        for record in self.records:
            yield record

    ############################################
    # Implementing ContextDB ABSTRACT METHODS: #
    ############################################

    def get_event_and_records(self):
        """
        Yield (event, records) tuples. See superclass docstring for details.
        """
        data = {}
        for record in self.records:
            evt_id = record.event.id
            if evt_id not in data:  # defaultdict might be an option
                data[evt_id] = []
            data[evt_id].append(record)

        for evt_id, records in data.items():
            yield evt_id, records

    SCALAR_IMTS = ["PGA", "PGV", "PGD", "Ia", "CAV"]

    def get_observations(self, imtx, records, component="Geometric"):
        """
        Return observed values for the given imt, as numpy array.
        See superclass docstring for details
        """
        values = []
        selection_string = "IMS/H/Spectra/Response/Acceleration/"
        for record in records:
            fle = h5py.File(record.datafile, "r")
            if imtx in self.SCALAR_IMTS:
                values.append(self.get_scalar(fle, imtx, component))
            elif "SA(" in imtx:
                spectrum = fle[selection_string + component + "/damping_05"][:]
                periods = fle["IMS/H/Spectra/Response/Periods"][:]
                target_period = imt.from_string(imtx).period
                values.append(
                    utils_imts.get_interpolated_period(target_period, periods, spectrum))
            else:
                raise ValueError("IMT %s is unsupported!" % imtx)
            fle.close()
            
        return values

    def get_rup(self, ctx):
        """
        Make a finite rupture for the given event inforamation.
        """ 
        # Get msr and aratio based on TRT if possible
        if hasattr(ctx, 'tectonic_region_type'):
            # NOTE: Admitted TRTs must be mapped to MBTK classifier TRTs
            eq_trt = ctx.tectonic_region_type
            if eq_trt in ['active_crustal', 'crustal']:    
                msr = scalerel.WC1994()
                aratio = 2
                trt = TRT.ACTIVE_SHALLOW_CRUST
            elif eq_trt == "stable":
                msr = scalerel.WC1994()
                aratio = 2
                trt = TRT.STABLE_CONTINENTAL
            elif eq_trt == 'slab':
                msr = scalerel.strasser2010.StrasserIntraslab()
                aratio = 5
                trt = TRT.SUBDUCTION_INTRASLAB
            elif eq_trt == 'int':
                msr = scalerel.strasser2010.StrasserInterface()
                aratio = 5
                trt = TRT.SUBDUCTION_INTERFACE
            else:
                # Has another TRT e.g. deep, induced, "unknown"
                # so make assumptions as for if no TRT provided
                msr = scalerel.WC1994()
                aratio = 3.0
                trt = None
        else:
            # No TRT so make some assumptions
            msr = scalerel.WC1994()
            aratio = 3.0
            trt = None

        # Avoid nodal plane issues
        if ctx.strike == 360.0:
            ctx.strike = 359.0
        if ctx.rake in {-180.0, 180.0}:
            ctx.rake = -179 if ctx.rake == -180.0 else 179

        # Make rupture from admitted event info
        rup = utils.make_rup(ctx.hypo_lon,
                             ctx.hypo_lat,
                             ctx.hypo_depth,
                             msr,
                             ctx.mag,
                             aratio,
                             ctx.strike,
                             ctx.dip,
                             ctx.rake,
                             trt,
                             ctx.ztor
                             )
        
        return rup

    def make_oq_ctx(self, ctx, rup, idx_site):
        """
        Make regular OQ context maker for computing missing distance metrics.

        NOTE: The user should be mindful that there will be inconsistencies
              between the distances obtained from reconstructing the ruptures
              and the distances provided in the flatfile. Therefore, it is
              advisable that the user either removes all provided distances
              and computes all of them from the reconstructed finite rupture,
              or they ensure the dataset they input already contains the
              distance metrics required for the GMMs they wish to consider.
         
        NOTE: This is tested within:
              `openquake.smt.tests.residuals.parsers.gem_flatfile_parser_test`
              which contains a row with completely empty distance metric cols.
        """  
        # Make site collection for given station
        pnt = Point(
            ctx.lons[idx_site], ctx.lats[idx_site], ctx.depths[idx_site])
        site = SiteCollection([
                    Site(
                        pnt,
                        ctx.vs30[idx_site],
                        ctx.z1pt0[idx_site],
                        ctx.z2pt5[idx_site]
                        )
                        ])
        
        # Make the ctx for given station which contains all distances
        mag_str = [f'{rup.mag:.2f}']
        oqp = {'imtls': {"PGA": []}, 'mags': mag_str}
        ctxm = ContextMaker(
            rup.tectonic_region_type, [utils.full_dtype_gmm()], oqp)
        ctxs = list(ctxm.get_ctx_iter([rup], site))

        return ctxs[0]

    def update_context(self, ctx, records, nodal_plane_index=1):
        """
        Updates the given RuptureContext with data from `records`.

        See superclass docstring for details
        """
        self._update_rupture_context(ctx, records, nodal_plane_index)
        self._update_sites_context(ctx, records)
        self._update_distances_context(ctx, records)

    def _update_rupture_context(self, ctx, records, nodal_plane_index=1):
        """
        Called by self.update_context
        """
        record = records[0]

        # Assign magnitude
        ctx.mag = record.event.magnitude.value

        # Assign nodal plane
        if nodal_plane_index == 2:
            ctx.strike = record.event.mechanism.nodal_planes.nodal_plane_2['strike']
            ctx.dip = record.event.mechanism.nodal_planes.nodal_plane_2['dip']
            ctx.rake = record.event.mechanism.nodal_planes.nodal_plane_2['rake']
        elif nodal_plane_index == 1:
            ctx.strike = record.event.mechanism.nodal_planes.nodal_plane_1['strike']
            ctx.dip = record.event.mechanism.nodal_planes.nodal_plane_1['dip']
            ctx.rake = record.event.mechanism.nodal_planes.nodal_plane_1['rake']
        else:
            ctx.strike = 0.0
            ctx.dip = 90.0
            ctx.rake = record.event.mechanism.get_rake_from_mechanism_type()

        # Assign a ztor if available
        if record.event.rupture.depth is not None:
            ctx.ztor = record.event.rupture.depth
        else:
            ctx.ztor = record.event.depth

        # Assign a rupture width if available
        if record.event.rupture.width is not None:
            ctx.width = record.event.rupture.width
        else:
            # Use WC1994 to define area and assume aratio of 1 to get width
            ctx.width = np.sqrt(scalerel.WC1994().get_median_area(ctx.mag, ctx.rake))

        # Default hypocentre location to the middle of the rupture
        ctx.hypo_loc = (0.5, 0.5)
        ctx.hypo_depth = record.event.depth
        ctx.hypo_lat = record.event.latitude
        ctx.hypo_lon = record.event.longitude

        # Add TRT if available
        if record.event.tectonic_region is not None:
            ctx.tectonic_region_type = record.event.tectonic_region

    def _update_sites_context(self, ctx, records):
        """
        Called by self.update_context.
        """
        for attname in self.sites_context_attrs:
            setattr(ctx, attname, [])

        for record in records:
            ctx.lons.append(record.site.longitude)
            ctx.lats.append(record.site.latitude)
            if record.site.altitude:
                depth = record.site.altitude * -1.0E-3
            else:
                depth = 0.0
            ctx.depths.append(depth)
            ctx.vs30.append(record.site.vs30)
            if record.site.vs30_measured is not None:
                vs30_measured = record.site.vs30_measured
            else:
                vs30_measured = 0
            ctx.vs30measured.append(vs30_measured)
            if record.site.z1pt0 is not None:
                z1pt0 = record.site.z1pt0
            else:
                z1pt0 = int(-999)
            ctx.z1pt0.append(z1pt0)
            if record.site.z2pt5 is not None:
                z2pt5 = record.site.z2pt5
            else:
                z2pt5 = int(-999)
            ctx.z2pt5.append(z2pt5)
            if getattr(record.site, "backarc", None) is not None:
                ctx.backarc.append(record.site.backarc)
        
        for attname in self.sites_context_attrs:
            attval = getattr(ctx, attname)
            # Remove attribute if its value is empty-like
            if attval is None or not len(attval):  
                delattr(ctx, attname)
            # Ensure some params are stored as bools
            elif attname in ('vs30measured', 'backarc'):
                setattr(ctx, attname, np.asarray(attval, dtype=bool))
            else:
                # dtype=float safely converts Nones to nans
                setattr(ctx, attname, np.asarray(attval, dtype=float))

    def _update_distances_context(self, ctx, records):
        """
        Called by self.update_context.

        NOTE: If a distance metric is missing from the record,
        then the SMT takes it from a finite rupture reconstructed
        within the engine.
        """
        # Set distance types in the "SMT" ctx
        for attname in self.distances_context_attrs:
            setattr(ctx, attname, [])
    
        # Get rupture for event
        rup = self.get_rup(ctx)

        # For each record manage the distances
        for idx_site, rec in enumerate(records):

            # Make ctx for given site
            site_ctx = self.make_oq_ctx(ctx, rup, idx_site)

            # Can take repi from regular ctx if missing
            if rec.distance.repi is not None:
                ctx.repi.append(rec.distance.repi)
            else:
                ctx.repi.append(getattr(site_ctx, 'repi')[0])

            # Can take rhypo from regular ctx if missing
            if rec.distance.rhypo is not None:
                ctx.rhypo.append(rec.distance.rhypo)
            else:
                ctx.rhypo.append(getattr(site_ctx, 'rhypo')[0])

            # Can take rjb from regular ctx if missing
            if rec.distance.rjb is not None:
                ctx.rjb.append(rec.distance.rjb)
            else:
                ctx.rjb.append(getattr(site_ctx, 'rjb')[0])

            # Can take rrup from regular ctx if missing
            if rec.distance.rrup is not None:
                ctx.rrup.append(rec.distance.rrup)
            else:
                ctx.rrup.append(getattr(site_ctx, 'rrup')[0])
                
            # Can take rx from regular ctx if missing
            if rec.distance.r_x is not None:
                ctx.rx.append(rec.distance.r_x) # r_x vs rx
            else:
                ctx.rx.append(getattr(site_ctx, 'rx')[0])

            # Can take ry0 from regular ctx if missing
            if rec.distance.ry0 is not None:
                ctx.ry0.append(rec.distance.ry0)
            else:
                ctx.ry0.append(getattr(site_ctx, 'ry0')[0])
            
            # Cannot compute rvolc from regular ctx
            if rec.distance.rvolc is None:
                # The regular ctx currently returns rvolc = 0
                # km by default (i.e. it cannot compute it),
                # but better to make this explicit here)
                ctx.rvolc.append(0.)
            else:
                ctx.rvolc.append(rec.distance.rvolc)
                                     
            # Cannot compute rcdpp from regular ctx
            if rec.distance.rcdpp is not None:
                ctx.rcdpp.append(rec.distance.rcdpp)
            else:
                # i.e. no directivity term (see CY14's
                # get_directivity function for example)
                ctx.rcdpp.append(0.) 

        for attname in self.distances_context_attrs:
            attval = getattr(ctx, attname)
            setattr(ctx, attname, np.asarray(attval, dtype=float))

    def get_scalar(self, fle, i_m, component="Geometric"):
        """
        Retrieves the scalar IM from the database
        :param fle:
            Instance of :class: h5py.File
        :param str i_m:
            Intensity measure
        :param str component:
            Horizontal component of IM
        """
        if not ("H" in fle["IMS"].keys()):
            x_im = fle[f"IMS/X/Scalar/{component}/{i_m}"][0]
            y_im = fle[f"IMS/Y/Scalar/{component}/{i_m}"][0]
            return utils_imts.SCALAR_XY[component](x_im, y_im)
        else:
            if i_m in fle[f"IMS/H/Scalar/{component}"].keys():
                return fle[f"IMS/H/Scalar/{component}/{i_m}"][0]
            else:
                raise ValueError("Scalar IM %s not in record database" % i_m)

    def number_records(self):
        """
        Returns number of records
        """
        return len(self.records)

    def __len__(self):
        """
        Returns the number of records
        """
        return len(self.records)

    def __repr__(self):
        """
        String with database ID and name
        """
        return "{:s} - ID({:s}) - Name ({:s})".format(
            self.__class__.__name__, self.id, self.name)

    def _get_event_id_list(self):
        """
        Returns the list of unique event keys from the database
        """
        event_list = []
        for record in self.records:
            if not record.event.id in event_list:
                event_list.append(record.event.id)
        return np.array(event_list)

    def _get_site_id(self, str_id):
        """
        Get site id 
        """
        if str_id not in self.site_ids:
            self.site_ids.append(str_id)
        _id = np.argwhere(str_id == np.array(self.site_ids))[0]
        return _id[0]

    def get_site_collection(self):
        """
        Returns the sites in the database as an instance of the :class:
        openquake.hazardlib.site.SiteCollection
        """
        return SiteCollection([
            rec.site.to_openquake_site() for rec in self.records])

    def rank_sites_by_record_count(self, threshold=0):
        """
        Function to determine count the number of records per site and return
        the list ranked in descending order
        """
        name_id_list = [(rec.site.id, rec.site.name) for rec in self.records]
        name_id = dict([])
        for name_id_pair in name_id_list:
            if name_id_pair[0] in name_id:
                name_id[name_id_pair[0]]["Count"] += 1
            else:
                name_id[name_id_pair[0]] = {"Count": 1, "Name": name_id_pair[1]}
        counts = np.array([name_id[key]["Count"] for key in name_id])
        sort_id = np.flipud(np.argsort(counts))

        key_vals = list(name_id)
        output_list = []
        for idx in sort_id:
            if name_id[key_vals[idx]]["Count"] >= threshold:
                output_list.append((key_vals[idx], name_id[key_vals[idx]]))

        return dict(output_list)


def load_database(directory):
    """
    Wrapper function to load the metadata of a :class:`GroundMotionDatabase`
    according to the filetype
    """
    metadata_file = None
    filetype = None
    fileset = os.listdir(directory)
    for ftype in ["pkl"]:
        if ("metadatafile.%s" % ftype) in fileset:
            metadata_file = "metadatafile.%s" % ftype
            filetype = ftype
            break
    if not metadata_file:
        raise IOError(
            "Expected metadata file of supported type not found in %s"
            % directory)
    metadata_path = os.path.join(directory, metadata_file)
    if filetype == "pkl":
        # pkl file type
        with open(metadata_path, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError("Metadata filetype %s not supported" % ftype)
