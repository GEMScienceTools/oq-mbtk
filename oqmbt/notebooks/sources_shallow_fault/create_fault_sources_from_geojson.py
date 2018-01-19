#!/usr/bin/env python3.5

import sys
import re
import numpy
import os
import json
import copy as cp

import logging

from ast import literal_eval
from shapely import wkt

from openquake.hazardlib.source import SimpleFaultSource
from openquake.hazardlib.geo.surface import SimpleFaultSurface

from openquake.hazardlib.scalerel.wc1994 import WC1994
from openquake.hazardlib.const import TRT
from openquake.hazardlib.tom import PoissonTOM
from openquake.hazardlib.geo import Point, Line
from openquake.hazardlib.mfd import EvenlyDiscretizedMFD
from openquake.hazardlib.sourcewriter import write_source_model
from openquake.hazardlib.tom import PoissonTOM

from oqmbt.oqt_project import OQtSource
from oqmbt.tools.faults import rates_for_double_truncated_mfd
from openquake.hmtk.faults.mfd.youngs_coppersmith import YoungsCoppersmithExponential
from openquake.hazardlib.geo.geodetic import azimuth

from oqmbt.notebooks.sources_shallow_fault.slip_utils import *

# Here I'm including "fake" categories, just for testing pourposes
SLIP_DIR_SET = set(['Dextral', 'Dextral-Normal',
                    'Normal', 'Normal-Dextral', 'Normal-Sinistral',
                    'Reverse', 'Sinistral',
                    'Sinistral-Normal', 'Sinistral-Reverse',
                    'Reverse-Dextral','Reverse-Sinistral',
                    'Thrust',
                    'Anticline',
                    'Blind-Thrust'])

# This classes are related with Leonard(2010) and were used  to compute
# geometries of the faults
WIDTH_CLASS = {'cl1': ['Normal', 'Reverse', 'Thrust', 'Normal-Dextral',
                       'Normal-Sinistral', 'Reverse-Sinistral',
                       'Reverse-Dextral',
                       'Blind-Thrust', 'Anticline'],
               'cl2': ['Dextral', 'Sinistral', 
                       'Dextral-Normal', 'Dextral-Reverse',
                       'Sinistral-Normal', 'Sinistral-Reverse']
               }

# Generic rake values used when this parameters is not available
RAKE_CLASS = {'Normal': -90,
              'Normal-Dextral': -135,
              'Normal-Sinistral': -45,
              'Reverse': 90,
              'Thrust': 90,
              'Blind-Thrust': 90,
              'Anticline': 90,           
              'Reverse-Dextral': 135,
              'Reverse-Sinistral': 45,
              'Sinistral': 0,
              'Sinistral-Normal': -45,
              'Sinistral-Reverse': 45,
              'Dextral': 180,
              'Dextral-Reverse': 135,
              'Dextral-Normal': -135
              }

# To transform literal values into numerical ones
DIRECTION_MAP = {'N': 0.,
                 'NNE': 22.5,
                 'NE': 45.,
                 'ENE': 67.5,
                 'E': 90.,
                 'ESE': 112.5,
                 'S': 180.,
                 'W': 270.,
                 'NW': 315.,
                 'SE': 135.,
                 'SW': 225.}


def get_net_slip(dip, rake, shor_rv=None, stk_rv=None, vert_rv=None):
    """
    to compute net_slip

    here I'm assuming:
    
    shortening_slip_rate [shor_rv] = heave
    vert_slip_rate [vert_rv] = trhow
    strike_slip_rate [stk_rv] = strike_slip_rate


    """
    option1 = [dip, rake, shor_rv, stk_rv]
    option2 = [dip, rake, shor_rv, vert_rv]
    option3 = [dip, rake, shor_rv]
    option4 = [dip, rake, stk_rv]
    d_slip = None
    s_slip = None
    if all(x is not None for x in option1):
        # computing the dip_slip using dip, rake and heave
        d_slip = dip_slip(dip, rake, shor_rv)
        if stk_rv is not None:
            s_slip = stk_rv
            print("s_slip = ", s_slip)
        print("OPTION-1")
    elif all(x is not None for x in option2):
        # computing the dip_slip using dip, rake and heave
        d_slip = dip_slip(dip, rake, heave=shor_rv)
        # computing the strike_slip using dip,rake,trhow
        s_slip = strike_slip(dip, rake, throw=vert_rv)
        # computing the net_slip when strike_slip is available
        slipr = net_slip(d_slip, s_slip)
        print("OPTION-2")
    elif all(x is not None for x in option3):
        # computing the dip_slip using dip, rake and heave
        d_slip = dip_slip(dip, rake, heave=shor_rv)
        # computing the strike_slip using dip_slip
        if d_slip > 1e-10:
            s_slip = strike_slip(dip, rake, d_slip=d_slip)
        else:
            s_slip = 0.0
        print("OPTION-3")
    elif all(x is not None for x in option4):
        # computing the dip_slip using dip, rake and strike_slip
        d_slip = dip_slip(dip, rake, sslip=stk_rv)
        s_slip = stk_rv
        print("OPTION-4")
    else:
        print('Not heave= %s or throw= %s or strike_slip =%s value provided '\
             %(shor_rv, vert_rv, stk_rv))
        msg = 'Not heave or throw or strike_slip value provided for this fault'
        logging.warning(msg)     
    
    # computing the net_slip when strike_slip is available
    if d_slip is not None and s_slip is not None:
        slipr = net_slip(d_slip, s_slip)
        print('net_slip =%s value computed in get_net_slip'%(slipr))
        return slipr
    else:
        return None


def _get_dip_dir_from_literal(dip_dir):
    """
    :paramater dir_str:
        A string defining a direction amongst the ones included in
        DIP_DIRECTION_MAP
    """
    if dip_dir in DIRECTION_MAP:
        dip_dir_angle = DIRECTION_MAP[dip_dir]
    else: 
        raise ValueError('Not supported dip_dir literal: %s' % (dip_dir))
    return dip_dir_angle


def _revert_fault_trace(fault_trace):
    """
   
    """
    fault_trace_orig = cp.deepcopy(fault_trace)
    fault_trace_new = fault_trace_orig
    fault_trace_new.points.reverse()  
    return fault_trace_new


def _need_to_revert(dip_dir_from_strike, dip_dir):
    """
    :parameter dip_dir_from_strike:
        Dip direction
    :parameter dip_dir:
        Dip direction
    :returns:
        Boolean
    """
    # print("dip_dir_database = ", dip_dir)
    # print("dip_dir_from_strike = ", dip_dir_from_strike)   
    msg = "dip direction from database = ", dip_dir
    logging.info(msg)  
    msg = "dip direction from strike = ", dip_dir_from_strike
    logging.info(msg)   
   
    dir1 = (dip_dir_from_strike+90.) % 360
    dir2 = (dip_dir) % 360
    # Computing difference
    diff = abs(dir1-dir2)
    if diff > 180:
        diff = 360 - diff
    if diff < 50:
        return False
    else:
        msg = '          ', dir1, dir2
        logging.info(msg)     
        msg = '     diff:', dir1-dir2
        logging.info(msg)     
        return True


def get_rake_from_rup_type(RAKE_CLASS, slipt):
    """
    Get rake using RAKE_CLASS and the style of slip [fault] 
    """
    if slipt in RAKE_CLASS:
        rake = RAKE_CLASS[slipt]
        # print("new rake value= %s"%(rake))
    else:
        raise ValueError('unsupported style of slip fault %s' % slipt)
    pass

    return rake 


def get_width_from_length(rld, slipt):
    """
    Get width from length 
    See Fig. 1 and 2 captions in Leonard 2010
    TODO: constrain the rld value to the limits
          proposed in Leonard 2010
    """     
    if slipt in WIDTH_CLASS['cl1'][:]:
        width = 1.75 * rld ** (2./3.)
        clas = 'cl1'
        # print("rld= %s, width = %s, class= %s"%(rld,width,slipt))
    elif slipt in WIDTH_CLASS['cl2'][:]:
        clas = 'cl2'
        if(rld<=45.):
            width = 1.50 * rld ** (2./3.)
            # print("rld= %s, width = %s, class= %s"%(rld,width,slipt))
            # constraining the width for values larger than 17. km
            # to be consistent with Leonard 2010 
            if width > 17.:
                width = 17.0
        else:
            width = 17.0 

            print("rld= %s, width = %s, class= %s"%(rld,width,slipt))
    else:
        raise ValueError('Not supported slip type value/class: %s' % (slipt))

    return width,clas


def _is_valid_dip_direction(tstr):
    """
    """
    if tstr in DIRECTION_MAP:
        return True
    raise ValueError('Unvalid dip direction string: %s' % tstr)


def _is_valid_slip_dir(tstr):
    """ 
    """
    if tstr in SLIP_DIR_SET:
        return True
    raise ValueError('Unvalid slip direction string: %s' % tstr)


def _is_valid_dip(dip):
    """
    """
    if (dip < 0.) or (dip > 90.):
        raise ValueError('Unvalid dip value: ', dip)
    else:
        return True


def _is_valid_strike(mean_azimuth):
    """
    """
    if (mean_azimuth < 0.) or (mean_azimuth > 360.):
        raise ValueError('Unvalid strike/mean azimuth value: %s' % mean_azimuth)
    else:
        return True


def _is_valid_rake(rake):
    """
    """
    if (rake) is None or ((rake <= -180.) or (rake > 360.)):
        raise ValueError('Unvalid or missing rake value: ', rake)
    else:
        return True


def get_dip_from_slip_type(slipt):
    """
    from sara_fault_tool.py
    fix a missing dip value using the slip type info

    """
    mtch = re.match('(\w*)-*\w*', slipt)

    if mtch:
        mech = mtch.group(1)
        print('mech:', mech)
        if (re.search('Reverse', mech) or 
           (re.search('Thrust', mech))):
            dip = 30.
        elif re.search('Anticline', mech):
            dip = 40.
        elif re.search('Blind', mech):
            dip = 40.            
        elif re.search('Normal', mech):
            dip = 60.
        elif (re.search('Dextral', mech) or
              re.search('Sinistral', mech)):
            dip = 90.
        else:
            raise ValueError('Unvalid dip angle from dip')
        return dip


def get_tples(tstr):
    """
    Extract information included in the tuples contained in the .geojson file

    :parameter string tstr:
        The string with the tuple
    :returns:
        A list containing the values of the tuple
    """
    if tstr is not None and len(tstr):
        tstra = re.sub('\(', '', re.sub('\)', '', tstr))
        flist = []
        for tmp in re.split('\,', tstra):
            if re.search('[0-9]', tmp):
                flist.append(float(tmp))
            else:
                flist.append(None)
    else:
        flist = (None, None, None)
    return flist


def get_line(dat):
    plist = []
    for tple in dat:
        plist.append(Point(tple[0], tple[1]))
    return Line(plist)


def _get_mean_az_from_trace(fault_trace):
    """
    :parameter:
        fault_trace
    :returns:
        A float defining the average azimuth
    """     
    mean_azimuth = fault_trace.average_azimuth()
    print("mean_azimuth= ", mean_azimuth)
    # valid_strike = False
    # if _is_valid_strike(mean_azimuth):
    #    valid_strike = True
    return mean_azimuth


def get_fault_sources(filename, slip_rate_class, bin_width=0.1, m_low=6.5, b_gr=1.0,
                      rupture_mesh_spacing=2.0, upper_seismogenic_depth=0.0, 
                      lower_seismogenic_depth=10.0, msr=WC1994(), 
                      rupture_aspect_ratio=2.0, temporal_occurrence_model=PoissonTOM(1.0),
                      aseismic_coeff=0.9, oqsource=False):
    """
    :parameter filename:
        The name of the .geojson file with fault data
    :parameter slip_rate_class:

    TODO: so far works only for slip_rate_class = "suggested/preferred"
    """

    logging.info('Reading %s and slip_type = %s' % (filename, slip_rate_class))
    with open(filename, 'r') as data_file:
        data = json.load(data_file)

    print('------------------------------------------------------------------------------')

    # Configuration parameters to create the sources
    # TODO:
    # use b_gr values from the area_sources and not a generic value
    # to test the whole processing

    # LOOP over the faults/traces
    srcl = []
    for idf, feature in enumerate(data['features']):

        source_id = '{0:d}'.format(idf)
        tectonic_region_type = TRT.ACTIVE_SHALLOW_CRUST

        fs_name = ''
        ns_name = ''

        # get fault name[s] - id
        if feature['properties']['fs_name'] is not None:
            fs_name = feature['properties']['fs_name']
        if feature['properties']['fs_name'] is not None:
            ns_name = feature['properties']['fs_name']
        name = '{0:s} | {1:s}'.format(fs_name, ns_name)
        if 'ogc_fid' in feature['properties']:
            id_fault = feature['properties']['ogc_fid']
        else:
            id_fault = '%d' % (idf)

        # get fault slip type
        if feature['properties']['slip_type'] is not None:
            slipt = feature['properties']['slip_type']
            msg = 'Slip type value is [%s] for fault with name'%(slipt)
            msg += '%s and id %s' % (name, id_fault)
            logging.info(msg)
        else:
            msg = 'Slip type value is missing for fault with name'
            msg += '%s and id %s' % (name, id_fault)
            logging.warning(msg)

        # get dip direction
        if feature['properties']['ns_dip_dir'] is not None:
            dip_dir = feature['properties']['ns_dip_dir']
            print("'dip_dir'= ", dip_dir)
        else:
            msg = 'Dip direction value is missing for fault with name'
            msg += '%s and id %s' % (name, id_fault)
            logging.warning(msg)
            # JUST FOR TESTING REASONS
            dip_dir = "N"
            print("dip_dir fazzula= ", dip_dir)
            # continue        

        # Get the tuples
        dipt = get_tples(feature['properties']['ns_average_dip'])
        print("dipt= ", dipt)
        raket = get_tples(feature['properties']['ns_average_rake'])
        print("raket= ", raket)
        sliprt = get_tples(feature['properties']['ns_net_slip_rate'])
        print("sliprt= ", sliprt)
        shor_rd = get_tples(feature['properties']['ns_shortening_rate'])
        print("shortening_rate= ", shor_rd)
        vert_rd = get_tples(feature['properties']['ns_vert_slip_rate'])
        print("vertical_slip_rate= ", vert_rd)
        stk_rd = get_tples(feature['properties']['ns_strike_slip_rate'])
        print("strike_slip_rate= ", stk_rd)
   
        # Set the value to be used [suggested, min, max]
        if slip_rate_class is 'suggested':                       
            valid_dip = False
            # Dip values
            dip = dipt[0]
            if dip is not None:
                valid_dip = _is_valid_dip(dip)
                if valid_dip:
                    print("Dip value = ", dip)
                    msg = 'Dip value is [%s] for fault with name'%(dip)
                    msg += '%s and id %s' % (name, id_fault)
                    logging.info(msg)           
            # if dip is None, but slip_type is available
            elif (dip) is None and (slipt) is not None:
                dip = get_dip_from_slip_type(slipt)
                print('Dip value for id= %s is missing and was computed using slipt= %s, new dip= %s ' %(id_fault, slipt, dip))
            # if dip is None, but slip_type is available
            elif (dip) is None and (slipt) is None:
                msg = 'Dip value is missing and could not be computed for fault with name '
                msg += '%s and id %s' % (name, id_fault)
                logging.warning(msg)
                continue

            valid_rake = False         
            # Rake values
            rake = raket[0]
            if rake is not None:
                valid_rake = _is_valid_rake(rake)
                if valid_rake:
                    print("Rake value = ", dip)
                    msg = 'Rake value is [%s] for fault with name'%(rake)
                    msg += '%s and id %s' % (name, id_fault)
                    logging.info(msg)           
            # if rake is None, but slip_type is available
            elif (rake) is None and (slipt) is not None:
                rake = get_rake_from_rup_type(RAKE_CLASS, slipt)
                print('Rake value for id= %s is missing and was computed using slipt= %s, new rake= %s '%(id_fault, slipt, rake))                     
                msg = 'Rake value is [%s] for slip_type= %s for fault with name'%(rake,slipt)
                msg += '%s and id %s' % (name, id_fault)
                logging.info(msg)
            # if rake and slip_type are not available
            elif (rake) is None and (slipt) is None:
                msg = 'Rake value is missing or could not be computed for fault with name '
                msg += '%s and id %s' % (name, id_fault)
                logging.warning(msg)
                continue
            
            # Slip rate values [shortening, vertical, strike_slip, net_slip]
            # If net_slip value is not available, a value is computed when 
            # other component are present in the database
            slipr = sliprt[0]
            shor_rv = shor_rd[0]
            stk_rv = stk_rd[0]
            vert_rv = vert_rd[0]

            msg = 'slipr= %s,  shor_rv= %s , stk_rv= %s, vert_rv= %s  for fault with name '%(slipr, shor_rv, stk_rv, vert_rv)
            msg += '%s and id %s' % (name, id_fault)
            logging.info(msg)

            if(slipr) is None and (shor_rv, stk_rv, vert_rv):
                print('slipr= %s'%(slipr))
                print('shor_rv= %s , stk_rv= %s, vert_rv= %s '%(shor_rv, stk_rv, vert_rv))        
                slipr = get_net_slip(dip, rake, shor_rv, stk_rv, vert_rv)

            if slipr is None:
                msg = 'net_slip  value is missing or can not be computed'
                msg += 'for fault with name %s and id %s' % (name, id_fault)
                logging.warning(msg)
                continue
            
            # Finally the net_slip is penalized using the aseismic_coeff
            net_slip = aseismic_coeff*float(slipr)
            print('net_slip value for id= %s is net_slip= %s [slipr = %s] ' %(id_fault, net_slip, slipt))
            
            msg = 'net_slip value [%.2f] computed for fault with name ' % (net_slip)
            msg += ' %s and id %s' % (name, id_fault)
            logging.info(msg)

        elif slip_rate_class is 'min':
            
            # Dip values
            dip = dipt[1]
            if (dip) is None and (slipt):
                dip = get_dip_from_slip_dir(slipt)
                print('Dip value for id= %s is missing and was computed using slipt= %s, new dip= %s ' %(id_fault, slipt, dip))
            else:
                msg = 'Dip value is missing for fault with name '
                msg += '%s and id %s' % (name, id_fault)
                logging.warning(msg)
                # continue

            # Rake values           
            rake = raket[1]
            if (rake) is None and (slipt):
                rake = get_rake_from_rup_type(RAKE_CLASS, slipt)
                print('Rake value for id= %s is missing and was computed using slipt= %s, new rake= %s '%(id_fault, slipt, rake))          
            else:
                msg = 'Rake value is missing for fault with name '
                msg += '%s and id %s' % (name, id_fault)
                logging.warning(msg)
                # continue
            # Slip rate values [shortening, vertical, strike_slip, net_slip]
            # If net_slip value is not available, a value is computed when 
            # other component are present in the database
            slipr = sliprt[1]
            shor_rd = shor_rv[1]
            stk_rd = stk_rv[1]
            vert_rv = vert_rd[1]

            if(slipr) is None and (shor_rv, stk_rv, vert_rv):
                slipr = get_net_slip(shor_rv, stk_rv, vert_rv)
            else:
                msg = 'net_slip  value is missing or not can be computed'
                msg += 'for fault with name %s and id %s' % (name, id_fault)
                logging.warning(msg)
                continue
            # Finally the net_slip is penalized using the aseismic_coeff
            net_slip = aseismic_coeff*float(slipr)

        elif slip_rate_class is 'max':
            
            # Dip values
            dip = dipt[2]
            if (dip) is None and (slipt):
                dip = get_dip_from_slip_dir(slipt)
                print('Dip value for id= %s is missing and was computed using slipt= %s, new dip= %s '\
                      % (id_fault, slipt, dip))
            else:
                msg = 'Dip value is missing for fault with name '
                msg += '%s and id %s' % (name, id_fault)
                logging.warning(msg)
                # continue

            # Rake values           
            rake = raket[2]
            if (rake) is None and (slipt):
                rake = get_rake_from_rup_type(RAKE_CLASS, slipt)
                print('Rake value for id= %s is missing and was computed using slipt= %s, new rake= %s '\
                      %(id_fault, slipt, rake))    
            else:
                msg = 'Rake value is missing for fault with name '
                msg += '%s and id %s' % (name, id_fault)
                logging.warning(msg)
                # continue
            
            # Slip rate values [shortening, vertical, strike_slip, net_slip]
            # If net_slip value is not available, a value is computed when 
            # other component are present in the database
            slipr = sliprt[2]
            shor_rd = shor_rv[2]
            stk_rd = stk_rv[2]
            vert_rv = vert_rd[0]

            if(slipr) is None and (shor_rv, stk_rv, vert_rv):
                slipr = get_net_slip(shor_rv, stk_rv, vert_rv)
            else:
                msg = 'net_slip  value is missing or not can be computed'
                msg += 'for fault with name %s and id %s' % (name, id_fault)
                logging.warning(msg)
                continue
            
            # Finally the net_slip is penalized using the aseismic_coeff
            net_slip = aseismic_coeff*float(slipr)

        else:
            raise ValueError('Invalid slip_rate_class')

        # Get fault trace geometry
        fault_trace = get_line(numpy.array(feature['geometry']['coordinates']))
        
        # Get dip direction angle from literal and strike from trace geometry
        mean_az_from_trace = _get_mean_az_from_trace(fault_trace)
        valid_az = False         
        valid_az = _is_valid_strike(mean_az_from_trace)
        if valid_az:
            # print("Mean azimuth value from trace = ", mean_az_from_trace)
            msg = 'Mean azimuth value is [%s] for fault with name'%(mean_az_from_trace)
            msg += '%s and id %s' % (name, id_fault)
            logging.info(msg)           

        dip_dir_angle = _get_dip_dir_from_literal(dip_dir)

        # Check if it's necessary to revert the fault trace
        if (dip_dir_angle is not None and
                _need_to_revert(mean_az_from_trace, dip_dir_angle)):
            new_fault_trace = _revert_fault_trace(fault_trace)
            logging.info('The fault trace for id= %s was reverted'%(id_fault))
        else:
            new_fault_trace = fault_trace

        
        if new_fault_trace:
            fault_trace = new_fault_trace

        # Get L from srl - See Table 5 of Leonard 2010
        # SRL: surface rupture length [km]
        # RLD: Subsurface horizontal rupture length [km]
        # IF SRL/RLD < 5. km the fault will be exclused
        srl = fault_trace.get_length()
        rld = 10**((numpy.log10(srl)+0.275)/1.1)
        
        if rld < 5.0:
            msg = 'SRL/RLD value is < 5.0 km for fault with name '
            msg += '%s and id %s' % (name, id_fault)
            logging.warning(msg)
            continue
        
        # Witdh
        width, cl = get_width_from_length(rld, slipt)
        # print("id=, %s, srl=, %.2f,  rld=, %.2f, width=, %.2f, slipt=, %s, cl=,%s "%(id_fault, srl, rld, width, slipt,cl))
        msg = "id=, %s, srl=, %.2f,  rld=, %.2f, width=, %.2f, slipt=, %s, cl=,%s "\
              % (id_fault, srl, rld, width, slipt, cl)
        logging.info(msg)

        # Get lower seismogenic depth from length
        lsd = width * numpy.sin(numpy.radians(float(dip)))
        lower_seismogenic_depth = lsd

        # create the surface from fault data 
        sfce = SimpleFaultSurface.from_fault_data(fault_trace,
                                                  upper_seismogenic_depth,
                                                  lower_seismogenic_depth,
                                                  dip,
                                                  rupture_mesh_spacing)
        # compute the area of the surface
        area = sfce.get_area()
        
        # compute the Mmax
        m_upp = msr.get_median_mag(sfce.get_area(), rake)
        if m_upp < m_low:
            msg = 'Mx [%.2f] is lesser than Mmin [%.2f] for fault with name '\
                  % (m_upp, m_low)         
            msg += '%s and id %s' % (name, id_fault)
            logging.warning(msg)

        tstr = '%3s - %-40s %5.2f' % (id_fault, name, m_upp)
        logging.info(tstr)
      
        if net_slip is not None:
            slip_rate = net_slip
            print("slip_rate= ", slip_rate)
        else:
            continue

        # constrainig the computation
        # Mx > Mmin=m_low
        # slip_rate >= 1e-10
        if slip_rate is not None and slip_rate >= 1e-10 and m_upp > m_low:

            # compute rates
            rates = rates_for_double_truncated_mfd(area,
                                                   slip_rate,
                                                   m_low,
                                                   m_upp,
                                                   b_gr,
                                                   bin_width)
            # MFD
            mfd = EvenlyDiscretizedMFD(m_low+bin_width/2, bin_width, rates)

            # Source
            if oqsource:
                src = SimpleFaultSource(source_id, name,
                                        tectonic_region_type,
                                        mfd,
                                        rupture_mesh_spacing,
                                        msr,
                                        rupture_aspect_ratio,
                                        temporal_occurrence_model,
                                        upper_seismogenic_depth,
                                        lower_seismogenic_depth,
                                        fault_trace,
                                        dip,
                                        rake)
            else:
                src = OQtSource(source_id, source_type='SimpleFaultSource')
                src.name = name   
                src.tectonic_region_type = tectonic_region_type   
                src.mfd = mfd  
                src.rupture_mesh_spacing = rupture_mesh_spacing
                src.slip_rate = slip_rate 
                src.msr = msr  
                src.rupture_aspect_ratio = rupture_aspect_ratio
                src.temporal_occurrence_model = temporal_occurrence_model
                src.upper_seismogenic_depth = upper_seismogenic_depth
                src.lower_seismogenic_depth = lower_seismogenic_depth
                src.trace = fault_trace
                src.dip = dip
                src.rake = rake
                print('right')

            srcl.append(src)

    return srcl
