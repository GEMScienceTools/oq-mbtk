#!/usr/bin/env python

import sys
import re
import numpy as np
import copy as cp

import os
import json


def get_apparent_dip(dip,rake):
    """
    """
    app_dip = np.degrees(np.arctan(np.tan(np.radians(rake)) * np.sin(np.radians(dip))))
    return app_dip

def offset_from_heave(heave, dip, rake):
    """
    Heave = shortering_rate

    """
    app_dip = get_apparent_dip(rake, dip)
    print("app_dip= ", app_dip)
    offset = heave / np.cos(np.radians(app_dip))
    print("offset in offset_from_heave= ", offset)
    
    return offset

def offset_from_dslip(d_slip, dip, rake):
    """
    """
    offset = d_slip / np.sin(np.radians(rake))
    
    return offset

def offset_from_sslip(strike_slip, dip, rake):
    """
    """
    offset = strike_slip / np.cos(np.radians(rake))

    return offset

def dip_slip(dip,rake,heave=None,throw=None, sslip=None):
    """
    Heave and Throw are the components of the dip slip
    Here I'm assuming:
    Heave = shortering_rate
    Throw = vertical_slip_rate
    rake = 90 ?
    
    """
    if heave is not None:
       # Firts I compute the offset using heave
       offset = offset_from_heave(heave, dip, rake)
       dip_slip = offset * np.sin(np.radians(rake))
       print("heave= %.2f, dip= %.2f,  rake= %.2f offset= %.2f dip_slip= %.2f " \
             %(heave, dip, rake, offset, dip_slip))
    elif throw is not None:
       # Firts I compute the offset using throw [usualmente no esta disponible]
       dip_slip = throw / np.sin(np.radians(dip))
       print("throw= %.2f, dip= %.2f,  rake= %.2f  dip_slip= %.2f " \
             %(throw, dip, rake, dip_slip))
    elif sslip is not None:
        offset = offset_from_sslip(sslip, dip, rake)
        dip_slip = offset * np.sin(np.radians(rake))
        print("strike_slip= %.2f, dip= %.2f,  rake= %.2f offset= %.2f dip_slip= %.2f " \
             %(sslip, dip, rake, offset, dip_slip))
    else:
       raise ValueError('Not heave= %s or throw= %s value provided' % (heave,throw))
    return dip_slip

def dip_slip_from_strike_slip(strike_slip, dip, rake):
    offset = offset_from_strike_slip(strike_slip, dip, rake)
    return dip_slip_from_offset(offset, dip, rake)

def strike_slip(dip, rake, heave=None, throw=None, d_slip=None):
    """
    
    """
    if heave is not None:
        offset = offset_from_heave(dip, rake, heave)
        # here rake=0?
        strike_slip = offset * np.cos(np.radians(rake))
        print("heave= %.2f, dip= %.2f,  rake= %.2f offset= %.2f strike_slip= %.2f " \
                 %(heave, dip, rake, offset, strike_slip))
    elif throw is not None:
        offset = offset_from_throw(dip, rake, throw)
        strike_slip = offset * np.cos(np.radians(rake))
        print("throw= %.2f, dip= %.2f,  rake= %.2f offset= %.2f strike_slip= %.2f " \
                 %(throw, dip, rake, offset, strike_slip))
    elif d_slip is not None:
        offset = offset_from_dslip(dip, rake, d_slip)
        strike_slip = offset * np.cos(np.radians(rake))
        print("d_slip= %.2f, dip= %.2f,  rake= %.2f offset= %.2f strike_slip= %.2f " \
                 %(d_slip, dip, rake, offset, strike_slip))    
    else:
       raise ValueError('Not heave= %s or throw= %s or d_slip= %s value provided' % (heave,throw,d_slip))
      
    return strike_slip

def net_slip(dip_slip,strike_slip):
    """
    """
    net_slip = np.sqrt(dip_slip**2 + strike_slip**2)
    print("dip_slip= %.2f,  strike_slip= %.2f net_slip= %.2f " \
             %(dip_slip, strike_slip, net_slip))

    return net_slip


