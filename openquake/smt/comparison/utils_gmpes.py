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
import os
import numpy 
from prettytable import PrettyTable

from openquake.hazardlib.geo import Point
from openquake.hazardlib import valid
from openquake.hazardlib.nrml import to_python
from openquake.hazardlib.geo.surface import PlanarSurface
from openquake.hazardlib.source.rupture import BaseRupture
from openquake.hazardlib.geo import utils as geo_utils
from openquake.hazardlib.geo.geodetic import npoints_towards
from openquake.hazardlib.site import Site, SiteCollection
from openquake.hazardlib.scalerel import WC1994
from openquake.hazardlib.const import TRT, StdDev
from openquake.hazardlib.contexts import ContextMaker


def _get_first_point(rup, from_point):
    """
    :param rup:
    :param from_point:
    """
    sfc = rup.surface
    if from_point == 'TC':
        return sfc._get_top_edge_centroid()
    elif from_point == 'BC':
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
        for lon, lat in reversed(pointsn[0][idx:], pointsn[1])[idx:]:
            site = Site(Point(lon, lat, 0.0), vs30=site_props['vs30'],
                        z1pt0=site_props['z1pt0'], z2pt5=site_props['z2pt5'])
            for key in list(keys):
                setattr(site, key, site_props[key])
            sites.append(site)

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

class GmcLt:
    """
    Class for managing the GMC logic tree
    """
    def __init__(self, gmclt, basedir=None):
        self.gmclt = gmclt
        self.basedir = basedir

    @classmethod
    def from_file(cls, fname):
        gmclt = to_python(fname)
        basedir = os.path.dirname(fname)
        self = cls(gmclt, basedir)
        return self

    def iter_gmms(self, trts=[]):
        """
        :param trts:
            A list of tectonic regions
        """
        for branching_level in self.gmclt:
            for branch_set in branching_level:
                for branch in branch_set:
                    gmm = valid.gsim(branch.uncertaintyModel, self.basedir)
                    trt = branch_set['applyToTectonicRegionType']
                    if trt in trts or len(trts) < 1:
                        yield gmm

    def iter_gmms_by_trt(self, trts=[]):
        """
        :param trts:
            A list of tectonic regions
        """
        for branching_level in self.gmclt:
            for branch_set in branching_level:
                trt = branch_set['applyToTectonicRegionType']
                gmms = []
                for branch in branch_set:
                    gmm = valid.gsim(branch.uncertaintyModel, self.basedir)
                    gmms.append(gmm)
                yield trt, gmms

def get_gm_info(ds, gm_vals):
    """
    :param ds:
        Datastore
    :param gm_vals:
        An iterable on a set of GM values
    :returns:
        Event ID and site ID
    """
    eids = []
    siteids = []
    df = ds.read_df('gmf_data', 'eid')
    for gm_val in gm_vals:
        print('Searching for ground motion value: {:.4f}'.format(gm_val))
        tmp = df[df.gmv_0 == gm_val]
        eids.append(tmp.index)
        siteids.append(tmp['sid'].iloc[0])
    return eids, siteids

def get_rupture_info(ds, eids, display=True):
    """
    :param ds:
    :param event_id:
    """
    x = PrettyTable()
    x.field_names = ["Event ID", "Rup ID", "Rlz ID", "Source ID"]
    x.align["Rup ID"] = "r"
    x.align["Rlz ID"] = "r"
    x.align["Source ID"] = "r"

    # Get rupture info
    ev = ds.read_df('events', 'id')
    rup = ds.read_df('ruptures', 'id')

    for eid in eids:
        rupid = ev.loc[eid]['rup_id']
        rlz_id = ev.loc[eid]['rlz_id'].iloc[0]
        source_id = rup.loc[rupid]['source_id'].iloc[0].decode("utf-8")
        ev.loc[eid]
        x.add_row([eid, rupid, rlz_id, source_id])

    return rupid, rlz_id, source_id

def get_rlz_info(ds, rlz_id, grp_id):
    full_lt = ds['full_lt']
    trt_by_grp = full_lt.trt_by_grp
    print(' ', trt_by_grp[grp_id])

    rlzs = full_lt.get_realizations()
    gmmrlz = full_lt.gsim_by_trt(rlzs[rlz_id])
    print(gmmrlz)
    print('\n ', gmmrlz[trt_by_grp[grp_id]])

def att_curves(gmpe,depth,mag,aratio,strike,dip,rake,Vs30,Z1,Z25,maxR,step,
              imt,ztor):    
    trt = gmpe.DEFINED_FOR_TECTONIC_REGION_TYPE
    rup = get_rupture(0.0, 0.0, depth, WC1994(), mag=mag, aratio=aratio,
                      strike=strike, dip=dip, rake=rake, trt=trt, ztor=ztor)
 
    props = {'vs30': Vs30, 'z1pt0': Z1, 'z2pt5': Z25, 'backarc': False,
             'vs30measured': True}
                
    sites = get_sites_from_rupture(rup, from_point='TC', toward_azimuth=90,
                                   direction='positive', hdist=maxR, step=step,
                                   site_props=props)
    
    param = dict(imtls={})
    cm = ContextMaker(trt, [gmpe], param)
    
    ctxs = list(cm.get_ctx_iter([rup], sites)) 
    ctxs = ctxs[0]
    ctxs.occurrence_rate = 0.0
    
    mean, std = gmpe.get_mean_and_stddevs(ctxs, ctxs, ctxs, imt, [StdDev.TOTAL])
    distances = ctxs.rrup
    
    return mean, std, distances

def att_curves_eshm20(gmpe,depth,mag,aratio,strike,dip,rake,Vs30,Z1,Z25,maxR,
                    step,imt,ztor,region):  
    """
    :param region:
        Choose among: region= 0 for global; 1 for California; 2 for Japan;
        3 for China; 4 for Italy; 5 for Turkey (locally = 3); 6 for Taiwan (
        locally = 0)
    """      
    trt = gmpe.DEFINED_FOR_TECTONIC_REGION_TYPE
    rup = get_rupture(0.0, 0.0, depth, WC1994(), mag=mag, aratio=aratio,
                      strike=strike, dip=dip, rake=rake, trt=trt, ztor=ztor)
    props = {'vs30': Vs30, 'z1pt0': Z1, 'z2pt5': Z25, 'backarc': False,
             'vs30measured': True,'region': numpy.array(region)}                
    sites = get_sites_from_rupture(rup, from_point='TC', toward_azimuth=90,
                                   direction='positive', hdist=maxR,
                                   step=step, site_props=props)
    param = dict(imtls={})
    cm = ContextMaker(trt, [gmpe], param)
    
    ctxs = list(cm.get_ctx_iter([rup], sites))          
    ctxs = ctxs[0]
    ctxs.occurrence_rate = 0.0
            
    mean, std = gmpe.get_mean_and_stddevs(ctxs, ctxs, ctxs, imt, [StdDev.TOTAL])
    distances = ctxs.rrup
    
    return mean, std, distances

def _get_z1(Vs30,region):
    """
    :param region:
        Choose among: region= 0 for global; 1 for California; 2 for Japan;
        3 for China; 4 for Italy; 5 for Turkey (locally = 3); 6 for Taiwan (
        locally = 0)
    """      
    if region == 2: # in California and non-Japan region
        Z1 = numpy.exp(-5.23/2*numpy.log((Vs30**2+412.39**2)/(1360**2+412.39**2)))
    else:
        Z1 = numpy.exp(-7.15/4*numpy.log((Vs30**4+570.94**4)/(1360**4+570.94**4)))
    return Z1

def _get_z25(Vs30,region):
    """
    :param region:
        Choose among: region= 0 for global; 1 for California; 2 for Japan;
        3 for China; 4 for Italy; 5 for Turkey (locally = 3); 6 for Taiwan (
        locally = 0)
    """     
    if region == 2: # in Japan
        Z25 = numpy.exp(5.359 - 1.102 * numpy.log(Vs30))
        Z25A_default = numpy.exp(7.089 - 1.144 * numpy.log(1100))
    else:
        Z25 = numpy.exp(7.089 - 1.144 * numpy.log(Vs30))
        Z25A_default = numpy.exp(7.089 - 1.144 * numpy.log(1100))           
    return Z25

def _param_gmpes(gmpes, strike, dip, depth, aratio, rake):
    # specific assumptions when the param are not available from the sources
    if strike == -999: 
        strike_s = 0
    else:
        strike_s = strike

    if dip == -999:
        if rake == 0:
            dip_s = 90 # strike slip
        else:
            dip_s = 45 # reverse or normal fault 
    else:
        dip_s = dip

    if depth == -999:
        print(gmpes)
        
        if str(valid.gsim(gmpes).DEFINED_FOR_TECTONIC_REGION_TYPE) == 'TRT.SUBDUCTION_INTERFACE':
            depth_s = 30
        elif str(valid.gsim(gmpes).DEFINED_FOR_TECTONIC_REGION_TYPE) == 'TRT.SUBDUCTION_INTRASLAB':
            depth_s = 50
        else:
            depth_s = 15
    else:
        depth_s = depth
        
    if aratio > -999.0 and numpy.isfinite(aratio):
        aratio_s = aratio
    else:
        if 'Inter' in gmpes:
            aratio_s = 5
        elif 'Slab' in gmpes:
            aratio_s = 5
        else:
            aratio_s = 2
            
    return strike_s, dip_s, depth_s, aratio_s