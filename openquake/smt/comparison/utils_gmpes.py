# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2023 GEM Foundation
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
Module with utility functions for gmpes
"""
import numpy as np
import pandas as pd

from openquake.hazardlib.geo import Point
from openquake.hazardlib.geo.surface import PlanarSurface
from openquake.hazardlib.source.rupture import BaseRupture
from openquake.hazardlib.geo.geodetic import npoints_towards
from openquake.hazardlib.site import Site, SiteCollection
from openquake.hazardlib.scalerel import WC1994
from openquake.hazardlib.const import TRT
from openquake.hazardlib.contexts import ContextMaker
from openquake.hazardlib.gsim.mgmpe import modifiable_gmpe as mgmpe
    

def get_sites_from_rupture(rup, from_point='TC', toward_azimuth=90,
                           direction='positive', hdist=100, step=5.,
                           site_props=''):
    """
    :param rup:
    :param from_point:
        A string. Options: 'TC', 'TL', 'TR'
    :return:
        A :class:`openquake.hazardlib.site.SiteCollection` instance
    """
    from_pnt = rup.surface._get_top_edge_centroid() # Get up-dip edge centroid
    r_lon = from_pnt.longitude
    r_lat = from_pnt.latitude
    r_dep = 0
    vdist = 0
    npoints = hdist / step
    strike = rup.surface.strike
    pointsp = []
    pointsn = []

    if direction=='positive':
        azi = (strike + toward_azimuth) % 360
        pointsp = npoints_towards(r_lon, r_lat, r_dep,
                                  azi, hdist, vdist, npoints)

    if direction=='negative':
        azi = (strike + toward_azimuth + 180) % 360
        pointsn = npoints_towards(r_lon, r_lat, r_dep,
                                  azi, hdist, vdist, npoints)

    sites = []
    keys = set(site_props.keys()) - set(['vs30', 'z1pt0', 'z2pt5'])

    if len(pointsn):
        lons = reversed(pointsn[0][0:])
        lats = reversed(pointsn[1][0:])
        for lon, lat in zip(lons, lats):
            site = Site(Point(lon, lat, 0.0), vs30=site_props['vs30'],
                        z1pt0=site_props['z1pt0'], z2pt5=site_props['z2pt5'])
            for key in list(keys):
                setattr(site, key, site_props[key])
            sites.append(site)

    if len(pointsp):
        for lon, lat in zip(pointsp[0], pointsp[1]):
            site = Site(Point(lon, lat, 0.0), vs30=site_props['vs30'],
                        z1pt0=site_props['z1pt0'], z2pt5=site_props['z2pt5'])
            for key in list(keys):
                setattr(site, key, site_props[key])
            sites.append(site)

    return SiteCollection(sites)


def get_rupture(lon, lat, dep, msr, mag, aratio, strike, dip, rake, trt,
                ztor=None):
    """
    Creates a rupture given the hypocenter position
    """
    hypoc = Point(lon, lat, dep)
    srf = PlanarSurface.from_hypocenter(hypoc, msr, mag, aratio, strike, dip,
                                        rake, ztor)
    rup = BaseRupture(mag, rake, trt, hypoc, srf)
    rup.hypocenter.depth = dep
    return rup


def att_curves(gmpe, orig_gmpe, depth, mag, aratio, strike, dip, rake, Vs30, 
               Z1, Z25, maxR, step, imt, ztor, eshm20_region, dist_type, trt,
               up_or_down_dip):    
    """
    Compute predicted ground-motion intensities
    """
    rup_trt = None
    if trt == 'ASCR':
        rup_trt = TRT.ACTIVE_SHALLOW_CRUST 
    if trt == 'InSlab':
        rup_trt = TRT.SUBDUCTION_INTRASLAB
    if trt == 'Interface':
        rup_trt = TRT.SUBDUCTION_INTERFACE
    if trt == 'Stable':
        rup_trt = TRT.STABLE_CONTINENTAL
    if trt == 'Upper_Mantle':
        rup_trt = TRT.UPPER_MANTLE
    if trt == 'Volcanic':
        rup_trt = TRT.VOLCANIC
    if trt == 'Induced':
        rup_trt = TRT.INDUCED
    if trt == 'Induced_Geothermal':
        rup_trt = TRT.GEOTHERMAL
    if trt == -999:
        rup_trt = gmpe.DEFINED_FOR_TECTONIC_REGION_TYPE
    if rup_trt is None:
     raise ValueError('Specify a TRT string within the toml file: ASCR, \
                       InSlab, Interface, Stable, Upper_Mantle, Volcanic, \
                       Induced, Induced_Geothermal')
                        
    # Get rup
    rup = get_rupture(0.0, 0.0, depth, WC1994(), mag=mag, aratio=aratio,
                      strike=strike, dip=dip, rake=rake, trt=rup_trt,
                      ztor=ztor)
    
    # Set site props
    if 'KothaEtAl2020ESHM20' in str(orig_gmpe):
        props = {'vs30': Vs30, 'z1pt0': Z1, 'z2pt5': Z25, 'backarc': False,
                 'vs30measured': True,'region': eshm20_region}  
    else:
        props = {'vs30': Vs30, 'z1pt0': Z1, 'z2pt5': Z25, 'backarc': False,
                 'vs30measured': True}
    
    # Check if site up-dip or down-dip of site
    if up_or_down_dip == float(1):
        direction = 'positive'
    elif up_or_down_dip == float(0):
        direction = 'negative'
    
    # Get sites
    sites = get_sites_from_rupture(rup, from_point='TC', toward_azimuth=90,
                                   direction=direction, hdist=maxR,
                                   step=step, site_props=props)
    
    # Add main R types to gmpe so can plot against repi, rrup, rjb and rhypo
    core_r_types = ['repi', 'rrup', 'rjb', 'rhypo']
    orig_r_types = list(gmpe.REQUIRES_DISTANCES)
    for core in core_r_types:
        if core not in orig_r_types:
            orig_r_types.append(core)
    gmpe.REQUIRES_DISTANCES = frozenset(orig_r_types)

    # Create context
    mag_str = [f'{mag:.2f}']
    oqp = {'imtls': {k: [] for k in [str(imt)]}, 'mags': mag_str}
    ctxm = ContextMaker(rup_trt, [gmpe], oqp)
    ctxs = list(ctxm.get_ctx_iter([rup], sites))
    ctxs = ctxs[0]
    ctxs.occurrence_rate = 0.0
    
    # Compute ground-motions
    mean, std, tau, phi = ctxm.get_mean_stds([ctxs])
    if dist_type == 'repi':
        distances = ctxs.repi
    if dist_type == 'rrup':
        distances = ctxs.rrup
    if dist_type == 'rjb':
        distances = ctxs.rjb
    if dist_type == 'rhypo':
        distances = ctxs.rhypo
    
    distances[len(distances)-1] = maxR

    return mean, std, distances


def _get_z1(Vs30, region):
    """
    :param region:
        Choose among: region= 0 for global; 1 for California; 2 for Japan;
        3 for China; 4 for Italy; 5 for Turkey (locally = 3); 6 for Taiwan (
        locally = 0)
    """
    if region == 2:   # in California and non-Japan region
        Z1 = (np.exp(-5.23 / 2 * np.log((Vs30**2 + 412.39**2) /
                                        (1360**2 + 412.39**2))))
    else:
        Z1 = (np.exp(-7.15 / 4 * np.log((Vs30**4 + 570.94**4) /
                                        (1360**4 + 570.94**4))))

    return Z1


def _get_z25(Vs30, region):
    """
    :param region:
        Choose among: region= 0 for global; 1 for California; 2 for Japan;
        3 for China; 4 for Italy; 5 for Turkey (locally = 3); 6 for Taiwan (
        locally = 0)
    """
    if region == 2:  # in Japan
        Z25 = np.exp(5.359 - 1.102 * np.log(Vs30))
    else:
        Z25 = np.exp(7.089 - 1.144 * np.log(Vs30))

    return Z25


def _param_gmpes(strike, dip, depth, aratio, rake, trt):
    """
    Get proxies for strike, dip, depth and aspect ratio if not provided
    """
    # Strike
    if strike == -999: 
        strike_s = 0
    else:
        strike_s = strike

    # Dip
    if dip == -999:
        if rake == 0:
            dip_s = 90 # strike slip
        else:
            dip_s = 45 # reverse or normal fault 
    else:
        dip_s = dip

    # Depth
    if depth == -999:
        if trt == 'Interface':
            depth_s = 30
        if trt == 'InSlab':
            depth_s = 50
        else:
            depth_s = 15
    else:
        depth_s = depth
        
    # a-ratio
    if aratio > -999.0 and np.isfinite(aratio):
        aratio_s = aratio
    else:
        if trt == 'InSlab' or trt == 'Interface':
            aratio_s = 5
        else:
            aratio_s = 2
            
    return strike_s, dip_s, depth_s, aratio_s


def mgmpe_check(gmpe):
    """
    Check if the GMPE should be modified using ModifiableGMPE
    :param gmpe:
        gmpe: GMPE to be modified if required (must be a gsim class)
    """

    # Preserve original GMPE prior and create base version of GMPE
    orig_gmpe = gmpe
    base_gsim = str(gmpe).splitlines()[0].replace('[', '').replace(']', '')

    # Get the additional params if specified
    inputs = pd.Series(str(gmpe).splitlines()[1:], dtype='object')
    add_inputs = {}
    add_as_int = ['eshm20_region']
    add_as_str = ['region', 'gmpe_table', 'volc_arc_file']

    if len(inputs) > 0:  # If greater than 0 must add required gsim inputs
        idx_to_drop = []
        for idx, par in enumerate(inputs):
            # Drop mgmpe params from the other gsim inputs
            if 'al_atik_2015_sigma' in str(par) or 'SiteTerm' in str(par):
                idx_to_drop.append(idx)
        inputs = inputs.drop(np.array(idx_to_drop))
        for idx, par in enumerate(inputs):
            key = str(par).split('=')[0].strip()
            if key in add_as_str:
                val = par.split('=')[1].replace('"', '').strip()
            elif key in add_as_int:
                val = int(par.split('=')[1])
            else:
                val = float(str(par).split('=')[1])
            add_inputs[key] = val

    # Sigma model implementation
    if 'al_atik_2015_sigma' in str(orig_gmpe):
        params = {"tau_model": "global", "ergodic": False}
        gmpe = mgmpe.ModifiableGMPE(gmpe={base_gsim: add_inputs},
                                    sigma_model_alatik2015=params)

    # Site term implementations
    msg_sigma_and_site_term = 'An alternative sigma model and an alternative \
        site term cannot be specified within a single GMPE implementation.'
    msg_multiple_site_terms = 'Two alternative site terms have been specified \
        within the toml for a single GMPE implementation'

    # Check only single site term specified
    if ('CY14SiteTerm' in str(orig_gmpe) and
            'NRCan15SiteTerm' in str(orig_gmpe)):
        raise ValueError(msg_multiple_site_terms)

    # Check if site term an dsigma model specified
    if ('CY14SiteTerm' in str(orig_gmpe) and
            'al_atik_2015_sigma' in str(orig_gmpe) or
            'CY14SiteTerm' in str(orig_gmpe) and
            'al_atik_2015_sigma' in str(orig_gmpe)):
        raise ValueError(msg_sigma_and_site_term)

    # CY14SiteTerm
    if 'CY14SiteTerm' in str(orig_gmpe):
        params = {}
        gmpe = mgmpe.ModifiableGMPE(gmpe={base_gsim: add_inputs},
                                    cy14_site_term=params)

    # NRCan15SiteTerm (kind = base)
    if ('NRCan15SiteTerm' in str(orig_gmpe) and
            'NRCan15SiteTermLinear' not in str(orig_gmpe)):
        params = {'kind': 'base'}
        gmpe = mgmpe.ModifiableGMPE(gmpe={base_gsim: add_inputs},
                                    nrcan15_site_term=params)
    # NRCan15SiteTerm (kind = linear)
    if 'NRCan15SiteTermLinear' in str(orig_gmpe):
        params = {'kind': 'linear'}
        gmpe = mgmpe.ModifiableGMPE(gmpe={base_gsim: add_inputs},
                                    nrcan15_site_term=params)

    return gmpe
