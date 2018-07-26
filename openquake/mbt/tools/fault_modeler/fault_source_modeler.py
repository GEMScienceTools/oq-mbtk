#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2015-2018 GEM Foundation
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
#
# Authors: Julio Garcia, Richard Styron, Valerio Poggi
# Last modify: 18/07/2018

# -----------------------------------------------------------------------------

import sys
import ast
import json
from copy import deepcopy

import configparser
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import openquake.mbt.tools.fault_modeler.fault_modeling_utils as fmu
from openquake.hazardlib.sourcewriter import write_source_model
from openquake.baselib import sap

# -----------------------------------------------------------------------------


# Parameters required from the fault modeler
"""
key_list = ['source_id',
            'name',
            'average_dip',
            'average_rake',
            'net_slip_rate',
            'vert_slip_rate',
            'strike_slip_rate',
            'shortening_rate',
            'dip_dir',
            'dip_slip_rate',
            'slip_type']

"""
option_types = {'b_value': float,
                'M_min': float,
                'M_max': float,
                'bin_width': float,
                'aseismic_coefficient': float,
                'rupture_aspect_ratio': float,
                'rupture_mesh_spacing': float,
                'minimum_fault_length': float,
                'tectonic_region_type': str,
                'upper_seismogenic_depth': float,
                'lower_seismogenic_depth': float,
                'magnitude_scaling_relationship': str}

# Conversion map for geojson input
param_map = deepcopy(fmu.param_map)

# Default arguments
defaults = deepcopy(fmu.defaults)


# -----------------------------------------------------------------------------

def build_fault_model(cfg_file=None,
                      geojson_file=None,
                      xml_output=None,
                      project_name=None,
                      black_list=None,
                      select_list=None,
                      param_map=param_map,
                      defaults=defaults,
                      **kwargs):
    """
    Note: we have to set the priority when using both ini file and
    optional arguments
    """

    # Import arguments from INI configuration file
    if cfg_file is not None:
        cfg_dict = read_config_file(cfg_file)

        if 'config' in cfg_dict:
            if 'geojson_file' in cfg_dict['config']:
                geojson_file = cfg_dict['config']['geojson_file']
            if 'xml_output' in cfg_dict['config']:
                xml_output = cfg_dict['config']['xml_output']
            if 'black_list' in cfg_dict['config']:
                black_list = ast.literal_eval(
                                cfg_dict['config']['black_list'])
            if 'select_list' in cfg_dict['config']:
                select_list = ast.literal_eval(
                                cfg_dict['config']['select_list'])

        if 'param_map' in cfg_dict:
            param_map.update(cfg_dict['param_map'])

        if 'defaults' in cfg_dict:
            defaults.update(cfg_dict['defaults'])

    for key in kwargs:
        defaults[key] = kwargs[key]

    # Import the fault database from geojson
    if geojson_file is not None:
        fault_db = fault_database()
        fault_db.import_from_geojson(geojson_file,
                                     black_list=black_list,
                                     select_list=select_list,
                                     param_map=param_map)
    else:
        print('Geojson file not specified')
        return

    # Create the fault source model in xml_format
    srcl = build_model_from_db(fault_db,
                               xml_output,
                               project_name,
                               param_map=param_map,
                               defaults=defaults)

    if xml_output is None:
        return srcl

# -----------------------------------------------------------------------------

def read_config_file(cfg_file):
    """
    """

    cfg_dict = {}
    cfg = configparser.ConfigParser(dict_type=dict)
    cfg.read(cfg_file)

    for key in ['config', 'param_map', 'defaults']:

        if cfg.has_section(key):
            cfg_dict[key] = {}

            for k, v in cfg.items(key):
                if k in option_types:
                    cast = option_types[k]
                else:
                    cast = str
                cfg_dict[key][k] = cast(v) 

    return cfg_dict

# -----------------------------------------------------------------------------

def build_model_from_db(fault_db,
                        xml_output=None,
                        project_name=None,
                        defaults=defaults,
                        param_map=param_map,
                        **kwargs):
    """
    """

    for key in kwargs:
        defaults[key] = kwargs[key]

    srcl = []

    for i, fl in enumerate(fault_db.db):
        try:
            sfs_dict = fmu.construct_sfs_dict(fl, defaults=defaults,
                                          param_map=param_map)
            sfs = fmu.make_fault_source(sfs_dict)
            srcl.append(sfs)
        except Exception as e:
            if 'source_id' in fl.keys():
                _id = fl['source_id']
            else:
                _id = '_unnamed_fault_number_{}'.format(i)
            print("Couldn't process Fault {}:  {}".format(_id, e))

    if xml_output is not None:
        # Write the final fault model
        write_source_model(xml_output, srcl, project_name)
    else:
        return srcl

# -----------------------------------------------------------------------------

class fault_database():
    """
    """

    def __init__(self, geojson_file=None):
        """
        """

        # Initialise an empty fault list
        self.db = []

        if geojson_file:
            self.import_from_geojson(geojson_file)

    def import_from_geojson(self, geojson_file,
                                  black_list=None,
                                  select_list=None,
                                  param_map=param_map,
                                  #defaults=defaults
                            ):
        """
        """

        # Import database
        with open(geojson_file, 'r') as f:
            data = json.load(f)

            # Loop over faults
            for feature in data['features']:
                fault = feature['properties']

                # Process only faults in the selection list
                if select_list is not None:
                    if fault[param_map['source_id']] not in select_list:
                        continue

                # Skip further processing for blacklisted faults
                if black_list is not None:
                    if fault['source_id'] in black_list:
                        continue

                # Get fault geometry
                fault['trace_coordinates'] = feature['geometry']['coordinates']

                self.db.append(fault)

            # Temporary adjustment to make the code running....!
            # self.add_property('name', 'XXX')


    def add_property(self, property, value=None, id=None):
        """
        """

        for fault in self.db:
            if fault['source_id'] is id or id is None:
                fault[property] = value

# -----------------------------------------------------------------------------

def main(argv):
    """
    """

    p = sap.Script(build_fault_model)
    p.opt(name='cfg_file',
          help='Parameter configuration file (.ini)',
          abbrev='-cfg',
          metavar='\'.ini\'')
    p.opt(name='geojson_file',
          help='Fault database in geojson format',
          abbrev='-geo',
          metavar='\'.geojson\'')
    p.opt(name='xml_output',
          help='Output xml containing the fault model',
          abbrev='-xml',
          metavar='\'.xml\'')
    p.opt(name='project_name',
          help='Name of the current project', abbrev='-h')
    p.opt(name='black_list',
          help='List of fault IDs NOT to be processed [id1,id2]', 
          type=ast.literal_eval, abbrev='-h')
    p.opt(name='select_list',
          help='List of selected fault IDs to be processed [id1,id2]', 
          type=ast.literal_eval, abbrev='-h')

    if len(argv) < 1:
        print(p.help())
    else:
        p.callfunc()

if __name__ == "__main__":
    main(sys.argv[1:])
