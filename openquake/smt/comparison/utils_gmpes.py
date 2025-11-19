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
Module with utility functions for gmpes
"""
import numpy as np
import pandas as pd
import ast
import re

from openquake.hazardlib import valid
from openquake.hazardlib import scalerel 
from openquake.hazardlib.geo import Point
from openquake.hazardlib.geo.geodetic import npoints_towards
from openquake.hazardlib.geo import utils as geo_utils
from openquake.hazardlib.site import Site, SiteCollection
from openquake.hazardlib.const import TRT
from openquake.hazardlib.contexts import ContextMaker
from openquake.hazardlib.gsim.mgmpe import modifiable_gmpe as mgmpe

from openquake.smt.utils import make_rup


def _get_first_point(rup, from_point):
    """
    Get the first point in the collection of sites from the rupture.
    
    Currently the SMT only computes ground-shaking for up or down-dip from the
    up-dip edge centre point (rrup, rjb), or from midpoint of the rupture (repi,
    rhypo). Will be expanded to include up-or-down dip from down-dip edge centre
    point or from a vertex of the rupture.
    """
    sfc = rup.surface

    if from_point == 'MP':
        return sfc.get_middle_point() # Get midpoint of rup surface

    elif from_point == 'TC':  # Get the up-dip edge centre point
        return sfc.get_top_edge_centroid()

    elif from_point == 'BC':  # Get the down-dip edge centre point
        lon, lat = geo_utils.get_middle_point(sfc.corner_lons[2],
                                              sfc.corner_lats[2],
                                              sfc.corner_lons[3],
                                              sfc.corner_lats[3])
        return Point(lon, lat, sfc.corner_depths[2])
        
    elif from_point == 'TL': # Get top left point
        idx = 0
    elif from_point == 'TR': # Get top right point
        idx = 1
    elif from_point == 'BR': # Get bottom right
        idx = 2
    elif from_point == 'BL': # Get bottom left
        idx = 3
    else:
        raise ValueError('Unsupported option from first point')

    return Point(
        sfc.corner_lons[idx], sfc.corner_lats[idx], sfc.corner_depths[idx])


def get_sites_from_rupture(rup,
                           from_point='TC',
                           toward_azimuth=90,
                           direction='positive',
                           hdist=100,
                           step=5.,
                           site_props=''):
    """
    Get the sites from the rupture to create the context with
    :param rup:
        Rupture object
    :param from_point:
        A string. Options: 'TC', 'TL', 'TR'
    :return:
        A :class:`openquake.hazardlib.site.SiteCollection` instance
    """
    from_pnt = _get_first_point(rup, from_point)
    r_lon = from_pnt.longitude
    r_lat = from_pnt.latitude
    r_dep = 0
    vdist = 0
    npoints = hdist / step
    strike = rup.surface.strike
    pointsp = []
    pointsn = []

    if direction == 'positive':
        azi = (strike + toward_azimuth) % 360
        pointsp = npoints_towards(r_lon,
                                  r_lat,
                                  r_dep,
                                  azi,
                                  hdist,
                                  vdist,
                                  npoints)
        
    if direction == 'negative':
        azi = (strike + toward_azimuth + 180) % 360
        pointsn = npoints_towards(r_lon,
                                  r_lat,
                                  r_dep,
                                  azi,
                                  hdist,
                                  vdist,
                                  npoints)

    sites = []
    keys = set(site_props.keys()) - set(['vs30', 'z1pt0', 'z2pt5'])

    if len(pointsn):
        lons = reversed(pointsn[0][0:])
        lats = reversed(pointsn[1][0:])
        for lon, lat in zip(lons, lats):
            site = Site(Point(lon, lat, 0.0),
                        vs30=site_props['vs30'],
                        z1pt0=site_props['z1pt0'],
                        z2pt5=site_props['z2pt5'])
            for key in list(keys):
                setattr(site, key, site_props[key])
            sites.append(site)

    if len(pointsp):
        for lon, lat in zip(pointsp[0], pointsp[1]):
            site = Site(Point(lon, lat, 0.0),
                        vs30=site_props['vs30'],
                        z1pt0=site_props['z1pt0'],
                        z2pt5=site_props['z2pt5'])
            for key in list(keys):
                setattr(site, key, site_props[key])
            sites.append(site)

    return SiteCollection(sites)


def get_rup(mag, lon, lat, depth, ztor, aratio, strike, dip, rake, trt):
    """
    Create an OQ rupture from the provided information.
    """
    # If TRT specified assign it and an MSR
    if trt == 'active_crustal':
        rup_trt = TRT.ACTIVE_SHALLOW_CRUST
        rup_msr = scalerel.WC1994()
    elif trt == 'stable':
        rup_trt = TRT.STABLE_CONTINENTAL
        rup_msr = scalerel.WC1994()
    elif trt == 'slab':
        rup_trt = TRT.SUBDUCTION_INTRASLAB
        rup_msr = scalerel.strasser2010.StrasserIntraslab()
    elif trt == 'interface':
        rup_trt = TRT.SUBDUCTION_INTERFACE
        rup_msr = scalerel.strasser2010.StrasserInterface()
    else:
        rup_trt = None
        rup_msr = scalerel.WC1994()

    if rup_trt == -999 and aratio == -999:
        msg = 'An aspect ratio must be provided by the user, or alternatively'
        msg += ' specify a TRT string within the toml file to assign a'
        msg += ' trt-dependent aratio proxy.'
        raise ValueError(msg)
    
    # Get rupture
    rup = make_rup(lon,
                   lat,
                   depth,
                   msr=rup_msr,
                   mag=mag,
                   aratio=aratio,
                   strike=strike,
                   dip=dip,
                   rake=rake,
                   trt=rup_trt,
                   ztor=ztor)
    
    return rup


def att_curves(gmpe,
               mag,
               lon,
               lat,
               depth,
               ztor,
               aratio,
               strike,
               dip,
               rake,
               trt,
               oq_rup,
               vs30,
               z1pt0,
               z2pt5,
               maxR,
               step,
               imt,
               dist_type,
               up_or_down_dip,
               volc_back_arc,
               eshm20_region):
    """
    Compute the ground-motion intensities for the given context created here
    """
    # Make rupture if not provided from XML or CSV
    if oq_rup is None:
        rup = get_rup(mag, lon, lat, depth, ztor, aratio, strike, dip, rake, trt)
    else:
        rup = oq_rup

    # Set site props
    props = {'vs30': vs30,
             'z1pt0': z1pt0,
             'z2pt5': z2pt5,
             'backarc': volc_back_arc,
             'vs30measured': False,
             'eshm20_region': eshm20_region}

    # Check if site up-dip or down-dip of site
    if up_or_down_dip == float(1):
        direction = 'positive'
    elif up_or_down_dip == float(0):
        direction = 'negative'
    else:
        raise ValueError('The site must be specified as up or down dip.')

    # Get sites
    if dist_type in ['repi', 'rhypo']:
        from_pnt = 'MP' # Sites from midpoint of rup surface
    else:
        from_pnt = 'TC' # Sites from center of top edge
    sites = get_sites_from_rupture(rup,
                                   from_point=from_pnt,
                                   toward_azimuth=90,
                                   direction=direction,
                                   hdist=maxR,
                                   step=step,
                                   site_props=props)

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
    ctxm = ContextMaker(rup.tectonic_region_type, [gmpe], oqp)
    ctxs = list(ctxm.get_ctx_iter([rup], sites))
    ctxs = ctxs[0]

    # Compute ground-motions
    mean, std, tau, phi = ctxm.get_mean_stds([ctxs])
    if dist_type == 'repi':
        distances = ctxs.repi
    elif dist_type == 'rrup':
        distances = ctxs.rrup
    elif dist_type == 'rjb':
        distances = ctxs.rjb
    elif dist_type == 'rhypo':
        distances = ctxs.rhypo
    else:
        raise ValueError('No valid distance type specified.')

    return mean, std, distances, tau, phi


def get_rup_pars(strike, dip, rake, aratio, trt):
    """
    Get (crude) proxies for strike, dip and aspect ratio if not
    provided by the user.
    """
    # Strike
    if strike == -999:
        strike_s = 0
    else:
        strike_s = strike

    # Dip
    if dip == -999:
        if rake == 0 or rake == 180:
            dip_s = 90  # Strike slip
        else:
            dip_s = 45  # Reverse or normal fault
    else:
        dip_s = dip

    # Aspect ratio
    if aratio != -999.0 and np.isfinite(aratio):
        aratio_s = aratio
    else:
        if trt in ['slab', 'interface']:
            aratio_s = 5
        else:
            aratio_s = 2 # Crustal

    return strike_s, dip_s, aratio_s


def mgmpe_check(gmpe):
    """
    Check if the GMPE should be modified using ModifiableGMPE. This function in
    effect parses the toml parameters for a GMPE into the equivalent parameters
    required for ModifiableGMPE. If a ModifiableGMPE is not required, a valid
    GSIM object with all specified kwargs is returned instead
    :param gmpe:
        gmpe: GMPE to be modified if required
    """
    if '[ModifiableGMPE]' in gmpe:

        # All of the inputs for this model
        params = pd.Series(gmpe.splitlines(), dtype=object)
        
        # Underlying GMM to modify
        base_gsim = re.search(r'gmpe\s*=\s*(.*)', params.iloc[1]).group(1).replace('"','')
        
        # Get the mgmpe params
        idx_params = []
        for idx, par in enumerate(params):
            if idx > 1:
                par = str(par)
                if ('sigma_model' in par or 'site_term' in par or 'basin_term' in par):
                    idx_params.append(idx)
                if 'fix_total_sigma' in par:
                    idx_params.append(idx)
                    base_vector = par.split('=')[1].replace('"', '')
                    fixed_sigma_vector = ast.literal_eval(base_vector)
                if 'with_betw_ratio' in par:
                    idx_params.append(idx)
                    with_betw_ratio = float(par.split('=')[1])
                if 'set_between_epsilon' in par:
                    idx_params.append(idx)
                    between_epsilon = float(par.split('=')[1])
                if 'add_delta_sigma_to_total_sigma' in par:
                    idx_params.append(idx)
                    delta_std = float(par.split('=')[1])
                if 'set_total_sigma_as_tau_plus_delta' in par:
                    idx_params.append(idx)
                    total_set_to_tau_and_delta = float(par.split('=')[1])
                if 'scaling' in par:
                    idx_params.append(idx)
                    if 'median_scaling_scalar' in par:
                        median_scalar = float(par.split('=')[1])
                    if 'median_scaling_vector' in par:
                        base_vector = par.split('=')[1].replace('"', '')
                        median_vector = ast.literal_eval(base_vector)
                    if 'sigma_scaling_scalar' in par:
                        sigma_scalar = float(par.split('=')[1])
                    if 'sigma_scaling_vector' in par:
                        base_vector = par.split('=')[1].replace('"', '')
                        sigma_vector = ast.literal_eval(base_vector)
                        
        # Now create kwargs
        kwargs = {}
        kwargs['gmpe'] = {base_gsim: {}}
        
        # Add the non-gmpe kwargs
        for idx_p, param in enumerate(params):
            if idx_p > 1 and idx_p not in idx_params:
                if 'lt_weight' not in param: # Skip if weight for logic tree
                    dic_key =  param.split('=')[0].strip().replace('"','')
                    dic_val =  param.split('=')[1].strip().replace('"','')
                    kwargs['gmpe'][base_gsim][dic_key] = dic_val
                
        # Al Atik 2015 sigma model
        if 'al_atik_2015_sigma' in gmpe:
            kwargs['sigma_model_alatik2015'] = {"tau_model": "global", "ergodic": False}
            
        # Fix total sigma per imt
        if 'fix_total_sigma' in gmpe:
            kwargs['set_fixed_total_sigma'] = {'total_sigma': fixed_sigma_vector}

        # Partition total sigma using a specified ratio of within:between
        if 'with_betw_ratio' in gmpe:
            kwargs['add_between_within_stds'] = {'with_betw_ratio': with_betw_ratio}

        # Set epsilon for tau and use instead of total sigma
        if 'set_between_epsilon' in gmpe:
            kwargs['set_between_epsilon'] = {'epsilon_tau': between_epsilon}
            
        # Add delta to total sigma
        if 'add_delta_sigma_to_total_sigma' in gmpe:
            kwargs['add_delta_std_to_total_std'] = {'delta': delta_std}
                
        # Set total sigma to sqrt(tau**2 + delta**2)
        if 'set_total_sigma_as_tau_plus_delta' in gmpe:
            kwargs['set_total_std_as_tau_plus_delta'] = {'delta': total_set_to_tau_and_delta}
        
        # Scale median by constant factor over all imts
        if 'median_scaling_scalar' in gmpe:
            kwargs['set_scale_median_scalar'] = {'scaling_factor': median_scalar}

        # Scale median by imt-dependent factor
        if 'median_scaling_vector' in gmpe:
            kwargs['set_scale_median_vector'] = {'scaling_factor': median_vector}

        # Scale sigma by constant factor over all imts
        if 'sigma_scaling_scalar' in gmpe:
            kwargs['set_scale_total_sigma_scalar'] = {'scaling_factor': sigma_scalar}

        # Scale sigma by imt-dependent factor
        if 'sigma_scaling_vector' in gmpe:
            kwargs['set_scale_total_sigma_vector'] = {'scaling_factor': sigma_vector}

        # CY14SiteTerm
        if 'CY14SiteTerm' in gmpe: kwargs['cy14_site_term'] = {}

        # BA08SiteTerm
        if 'BA08SiteTerm' in gmpe: kwargs['ba08_site_term'] = {}

        # BSSA14SiteTerm
        if "BSSA14SiteTerm" in gmpe: kwargs['bssa14_site_term'] = {}

        # NRCan15SiteTerm (Regular)
        if ('NRCan15SiteTerm' in gmpe and 'NRCan15SiteTermLinear' not in gmpe):
            kwargs['nrcan15_site_term'] = {'kind': 'base'}

        # NRCan15SiteTerm (linear)
        if 'NRCan15SiteTermLinear' in gmpe:
            kwargs['nrcan15_site_term'] = {'kind': 'linear'}

        # CB14 basin term
        if 'CB14BasinTerm' in gmpe: kwargs['cb14_basin_term'] = {}

        # M9 basin adjustment
        if 'M9BasinTerm' in gmpe: kwargs['m9_basin_term'] = {}
        
        gmm = mgmpe.ModifiableGMPE(**kwargs)

    # Not using ModifiableGMPE
    else:
        # Clean to ensure arguments can be passed (the logic tree weights
        # are retained in original GMM strings in utils_compare_gmpes.py)
        params = pd.Series(gmpe.splitlines())
        idx_to_drop = []
        for idx_p, par in enumerate(params):
            if 'lt_weight_gmc' in par:
                idx_to_drop.append(idx_p)
        params = params.drop(idx_to_drop)
        gmpe_clean = params.iloc[0].strip()
        if len(params) > 1:
            for idx_p, par in enumerate(params):
                if idx_p > 0:
                    gmpe_clean = gmpe_clean + '\n' + par
        else: # Ensures GSIM aliases work
            gmpe_clean = gmpe_clean.replace('[','').replace(']','')
        
        gmm = valid.gsim(gmpe_clean)

    return gmm


def get_gmm_str(gmm):
    """
    Return a clean GMM string (i.e. one without the GMM
    logic tree weight assigned to it, if present).
    """
    gmm_str = ''
    for idx_part, part in enumerate(gmm.split('\n')):
        if "lt_weight_gmc" not in part:
            if idx_part > 0:
                gmm_str += f", {part}"
            else:
                gmm_str += part.strip()

    return gmm_str


def get_imtl_unit(i):
    """
    Return a string of the intensity measure type's physical units of
    measurement
    """
    if str(i) in ['PGD', 'SDi']:
        unit = 'cm' # PGD, inelastic spectral displacement
    elif str(i) in ['PGV']:
        unit = 'cm/s' # PGV
    elif str(i) in ['IA']:
        unit = 'm/s' # Arias intensity
    elif str(i) in ['RSD', 'RSD595', 'RSD575', 'RSD2080', 'DRVT']:
        unit = 's' # Relative significant duration, DRVT
    elif str(i) in ['CAV']:
        unit = 'g-sec' # Cumulative absolute velocity
    elif str(i) in ['MMI']:
        unit = 'MMI' # Modified Mercalli Intensity
    elif str(i) in ['FAS', 'EAS']:
        unit = str(i) + ' (Hz)' # Fourier/Eff. Amp. Spectrum
    else:
        if str(i) not in ["PGA", "AvgSA"]:
            assert "SA" in str(i)
        unit = 'g' # PGA, SA, AvgSA

    return unit


def reformat_att_curves(att_curves, out=None):
    """
    Export the attenuation curves into a CSV for the given
    config (i.e. run parameters).
    """
    # Get the key describing the vs30 + truncation level
    params_key = pd.Series(att_curves.keys()).values[0]

    # Get Nstd and make an integer if appropriate
    nstd = float(params_key.split("GMM sigma epsilon = ")[1])
    if nstd.is_integer():
        nstd = int(nstd)

    # Then get the values per gmm (per imt-mag combination)
    vals = att_curves[params_key]['gmm att curves per imt-mag']

    # Now get the curves into a dictionary format
    store = {}
    for imt in vals.keys():
        unit = get_imtl_unit(imt)
        for scenario in vals[imt]:
            curves = vals[imt][scenario]
            for gmpe in curves: 
                
                # Get cleaned string of gmm
                gmm_str = get_gmm_str(gmpe)
                
                # Next per GMM get medians and sigmas
                if "(km)" not in gmpe:
                    key = f"{imt} ({unit}) | {gmm_str} | {scenario}"
                    
                    # Add median
                    store[f"Median | {key}"] = curves[gmpe][f'median ({unit})']
                    
                    # Will only be median plus/minus sigma if Nstd > 0
                    if f"median plus sigma ({unit})" in curves[gmpe]:
                        store[f"Median Plus Sigma (+ {nstd} epsilon) | {key}"
                              ] = curves[gmpe][f"median plus sigma ({unit})"]
                        store[f"Median Minus Sigma (- {nstd} epsilon) | {key}"
                              ] = curves[gmpe][f"median plus sigma ({unit})"]
                        
                # Then get the distance for given scenario
                else:
                    dkey = f"values of {gmpe} for {scenario}"
                    store.setdefault(dkey, curves[gmpe])
                    
    # Now into dataframe
    df = pd.DataFrame(store)

    # Reorder columns to get dist cols on left
    df = df[[
        col for col in df.columns if "(km)" in col] + [
            col for col in df.columns if "(km)" not in col]]

    # And export if required
    if out is not None:
        df.to_csv(out, index=False)

    return df


def reformat_spectra(spectra, out=None):
    """
    Export the response spectra into a CSV for the given
    config (i.e. run parameters).
    """
    store = {}
    eps = spectra['nstd']
    branches = ['median', 'median plus sigma', 'median minus sigma']
    for key in spectra.keys():
        
        # Don't need weighted GMMs (only used for computing aggregated LTs)
        if key in ["periods", "nstd"] or "_wei" in key:
            continue
        
        # Weighted gmm LTs
        if 'gmc' in key:
            for sc in spectra[key]:
                for idx_br, br in enumerate(spectra[key][sc]):

                    if br == {}:
                        continue # Empty dict if no epsilon applied
                    
                    bl = branches[idx_br]

                    if bl == "median plus sigma":
                        s_key = f"{bl.title()} (+ {eps} epsilon) (g) | {key} logic tree | {sc}"
                    elif bl == "median":
                        s_key = f"{bl.title()} (g) | {key} logic tree | {sc}"       
                    else:
                        assert bl == "median minus sigma"
                        s_key = f"{bl.title()} (- {eps} epsilon) (g) | {key} logic tree | {sc}"
                    
                    store[s_key] = np.array(list(br.values()))
        else:

            # Individual gmms
            for gmm in spectra[key]:
                gmm_str = get_gmm_str(gmm)
                for sc in spectra[key][gmm]:
                    assert 'lt_weight_gmc' in gmm
                    if key == "add":
                        s_key = f"{gmm_str}, Median Plus Sigma (+ {eps} epsilon) (g), {sc}"
                    elif key == "med":
                        s_key = f"{gmm_str}, Median (g), {sc}"
                    else:
                        assert key == "min"
                        s_key = f"{gmm_str}, Median Minus Sigma (- {eps} epsilon) (g), {sc}"
            
                    store[s_key] = spectra[key][gmm][sc]
                    
    # Make df
    df = pd.DataFrame(store, index=spectra['periods'])
    df.index.name = "Period (s)"

    # Export if required
    if out is not None:
        df.to_csv(out, index=True)

    return df