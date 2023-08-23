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
from openquake.hazardlib.geo import utils as geo_utils
from openquake.hazardlib.geo.geodetic import npoints_towards
from openquake.hazardlib.site import Site, SiteCollection
from openquake.hazardlib.scalerel import WC1994
from openquake.hazardlib.const import TRT
from openquake.hazardlib.contexts import ContextMaker
from openquake.hazardlib.gsim.mgmpe import modifiable_gmpe as mgmpe


def _get_first_point(rup, from_point):
    """
    :param rup:
    :param from_point:
    """
    sfc = rup.surface
    if from_point == 'TC': # Get the up-dip edge centre point
        return sfc._get_top_edge_centroid()
    elif from_point == 'BC': # Get the down-dip edge centre point
        lon, lat = geo_utils.get_middle_point(
            sfc.corner_lons[2], sfc.corner_lats[2],
            sfc.corner_lons[3], sfc.corner_lats[3]
        )
        return Point(lon, lat, sfc.corner_depths[2])
    elif from_point == 'TL':
        idx = 0
    elif from_point == 'TR':
        idx = 1
    elif from_point == 'BR':
        idx = 2
    elif from_point == 'BL':
        idx = 3
    else:
        raise ValueError('Unsupported option from first point')
    return Point(sfc.corner_lons[idx],
                 sfc.corner_lats[idx],
                 sfc.corner_depths[idx])


def get_sites_from_rupture(rup, from_point = 'TC', toward_azimuth = 90,
                           direction = 'positive', hdist = 100, step = 5.,
                           site_props = ''):
    """
    :param rup:
    :param from_point:
        A string. Options: 'TC', 'TL', 'TR'
    :return:
        A :class:`openquake.hazardlib.site.SiteCollection` instance
    """
    from_pnt = _get_first_point(rup, from_point)
    lon = from_pnt.longitude
    lat = from_pnt.latitude
    depth = 0
    vdist = 0
    npoints = hdist / step
    strike = rup.surface.strike
    pointsp = []
    pointsn = []

    if not len(site_props):
        raise ValueError()

    if direction in ['positive', 'both']:
        azi = (strike+toward_azimuth) % 360
        pointsp = npoints_towards(lon, lat, depth, azi, hdist, vdist, npoints)

    if direction in ['negative', 'both']:
        idx = 0 if direction == 'negative' else 1
        azi = (strike+toward_azimuth+180) % 360
        pointsn = npoints_towards(lon, lat, depth, azi, hdist, vdist, npoints)

    sites = []
    keys = set(site_props.keys()) - set(['vs30', 'z1pt0', 'z2pt5'])

    if len(pointsn):
        lons = reversed(pointsn[0][idx:])
        lats = reversed(pointsn[1][idx:])
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
    rup.hypo_depth = dep
    return rup


def att_curves(gmpe, orig_gmpe, depth, mag, aratio, strike, dip, rake, Vs30, 
               Z1, Z25, maxR, step, imt, ztor, eshm20_region, up_or_down_dip = None):    
    """
    Compute predicted ground-motion intensities w.r.t considered distance using
    the given GMPE
    """
    # Get trt
    trt = gmpe.DEFINED_FOR_TECTONIC_REGION_TYPE
    
    # Get rup
    rup = get_rupture(0.0, 0.0, depth, WC1994(), mag = mag, aratio = aratio,
                      strike = strike, dip = dip, rake = rake, trt = trt,
                      ztor = ztor)
    
    # Set site props
    if 'KothaEtAl2020ESHM20' in str(orig_gmpe):
        props = {'vs30': Vs30, 'z1pt0': Z1, 'z2pt5': Z25, 'backarc': False,
                 'vs30measured': True,'region': eshm20_region}  
    else:
        props = {'vs30': Vs30, 'z1pt0': Z1, 'z2pt5': Z25, 'backarc': False,
                 'vs30measured': True}
    
    # Check if site up-dip or down-dip of site
    if up_or_down_dip is None or up_or_down_dip == 1:
        direction = 'positive'
        from_point = 'TC'
    elif up_or_down_dip == 0:
        from_point = 'BC'
        direction = 'negative'
    
    # Get sites
    sites = get_sites_from_rupture(rup, from_point, toward_azimuth = 90,
                                   direction = direction, hdist = maxR,
                                   step = step, site_props = props)
    
    # Create context
    mag_str = [f'{mag:.2f}']
    oqp = {'imtls': {k: [] for k in [str(imt)]}, 'mags': mag_str}
    ctxm = ContextMaker(trt, [gmpe], oqp)
    
    ctxs = list(ctxm.get_ctx_iter([rup], sites)) 
    ctxs = ctxs[0]
    ctxs.occurrence_rate = 0.0
    
    # Compute ground-motions
    mean, std, tau, phi = ctxm.get_mean_stds([ctxs])
    distances = ctxs.rrup
    
    # Ensures can interpolate to max value in dist_list (within RS plotting)
    distances[len(distances)-1] = maxR
    
    return mean, std, distances


def _get_z1(Vs30,region):
    """
    :param region:
        Choose among: region= 0 for global; 1 for California; 2 for Japan;
        3 for China; 4 for Italy; 5 for Turkey (locally = 3); 6 for Taiwan (
        locally = 0)
    """      
    if region == 2: # in California and non-Japan region
        Z1 = np.exp(-5.23/2*np.log((Vs30**2+412.39**2)/(1360**2+412.39**2)))
    else:
        Z1 = np.exp(-7.15/4*np.log((Vs30**4+570.94**4)/(1360**4+570.94**4)))

    return Z1


def _get_z25(Vs30,region):
    """
    :param region:
        Choose among: region= 0 for global; 1 for California; 2 for Japan;
        3 for China; 4 for Italy; 5 for Turkey (locally = 3); 6 for Taiwan (
        locally = 0)
    """     
    if region == 2: # in Japan
        Z25 = np.exp(5.359 - 1.102 * np.log(Vs30))
        Z25A_default = np.exp(7.089 - 1.144 * np.log(1100))
    else:
        Z25 = np.exp(7.089 - 1.144 * np.log(Vs30))
        Z25A_default = np.exp(7.089 - 1.144 * np.log(1100))           

    return Z25


def _param_gmpes(gmpes, strike, dip, depth, aratio, rake):
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
        if gmpes.DEFINED_FOR_TECTONIC_REGION_TYPE == TRT.SUBDUCTION_INTERFACE:
            depth_s = 30
        elif gmpes.DEFINED_FOR_TECTONIC_REGION_TYPE == TRT.SUBDUCTION_INTRASLAB:
            depth_s = 50
        else:
            depth_s = 15
    else:
        depth_s = depth
        
    # a-ratio
    if aratio > -999.0 and np.isfinite(aratio):
        aratio_s = aratio
    else:
        if gmpes.DEFINED_FOR_TECTONIC_REGION_TYPE == TRT.SUBDUCTION_INTERFACE:
            aratio_s = 5
        elif gmpes.DEFINED_FOR_TECTONIC_REGION_TYPE == TRT.SUBDUCTION_INTRASLAB:
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
    base_gsim = str(gmpe).splitlines()[0].replace('[','').replace(']','')
    
    # Get the additional params if specified    
    inputs = pd.Series(str(gmpe).splitlines()[1:], dtype = 'object')
    add_inputs = {}
    add_as_int = ['eshm20_region']
    add_as_str = ['region', 'gmpe_table', 'volc_arc_file']
    
    if len(inputs) > 0: # If greater than 0 must add required gsim inputs 
        idx_to_drop = []
        for idx, par in enumerate(inputs):
            # Drop mgmpe params from the other gsim inputs
            if 'al_atik_2015_sigma' in str(par) or 'SiteTerm' in str(par):
                idx_to_drop.append(idx)
        inputs = inputs.drop(np.array(idx_to_drop))
        for idx, par in enumerate(inputs):
            key = str(par).split('=')[0].strip()
            if key in add_as_str:
                val = par.split('=')[1].replace('"','').strip()
            elif key in add_as_int:
                val = int(par.split('=')[1])
            else:
                val = float(str(par).split('=')[1])
            add_inputs[key] = val
    
    ### Sigma model implementation 
    if 'al_atik_2015_sigma' in str(orig_gmpe):
        params = {"tau_model": "global", "ergodic": False}
        gmpe = mgmpe.ModifiableGMPE(gmpe={base_gsim: add_inputs},
                             sigma_model_alatik2015=params)
    
    ### Site term implementations
    msg_sigma_and_site_term = 'An alternative sigma model and an alternative \
        site term cannot be specified within a single GMPE implementation.'
    msg_multiple_site_terms = 'Two alternative site terms have been specified \
        within the toml for a single GMPE implementation'
    
    # Check only single site term specified
    if 'CY14SiteTerm' in str(orig_gmpe) and 'NRCan15SiteTerm' in str(orig_gmpe):
        raise ValueError(msg_multiple_site_terms)
    
    # Check if site term an dsigma model specified
    if 'CY14SiteTerm' in str(orig_gmpe) and 'al_atik_2015_sigma' in str(
            orig_gmpe) or 'CY14SiteTerm' in str(orig_gmpe) and \
        'al_atik_2015_sigma' in str(orig_gmpe):
        raise ValueError(msg_sigma_and_site_term)
        
    # CY14SiteTerm
    if 'CY14SiteTerm' in str(orig_gmpe):
        params = {}
        gmpe = mgmpe.ModifiableGMPE(gmpe={base_gsim: add_inputs},
                                    cy14_site_term=params)
    
    # NRCan15SiteTerm (kind = base)
    if 'NRCan15SiteTerm' in str(orig_gmpe) and 'NRCan15SiteTermLinear' not \
        in str(orig_gmpe):
            params = {'kind': 'base'}
            gmpe = mgmpe.ModifiableGMPE(gmpe={base_gsim: add_inputs},
                                        nrcan15_site_term=params)
    # NRCan15SiteTerm (kind = linear)
    if 'NRCan15SiteTermLinear' in str(orig_gmpe):
            params = {'kind': 'linear'}
            gmpe = mgmpe.ModifiableGMPE(gmpe={base_gsim: add_inputs},
                                        nrcan15_site_term=params)

    return gmpe


