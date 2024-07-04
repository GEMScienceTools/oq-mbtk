# ------------------- The OpenQuake Model Building Toolkit --------------------
# Copyright (C) 2022 GEM Foundation
#           _______  _______        __   __  _______  _______  ___   _
#          |       ||       |      |  |_|  ||  _    ||       ||   | | |
#          |   _   ||   _   | ____ |       || |_|   ||_     _||   |_| |
#          |  | |  ||  | |  ||____||       ||       |  |   |  |      _|
#          |  |_|  ||  |_|  |      |       ||  _   |   |   |  |     |_
#          |       ||      |       | ||_|| || |_|   |  |   |  |    _  |
#          |_______||____||_|      |_|   |_||_______|  |___|  |___| |_|
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------
# vim: tabstop=4 shiftwidth=4 softtabstop=4
# coding: utf-8

import ast
import json
from copy import deepcopy
import configparser
import warnings

import openquake.mbt.tools.fault_modeler.fault_modeling_utils as fmu
from openquake.hazardlib.sourcewriter import write_source_model
from openquake.baselib import sap

# -----------------------------------------------------------------------------

warnings.simplefilter(action='ignore', category=FutureWarning)

# Parameters required from the fault modeler
option_types = {'b_value': float,
                'm_min': float,
                'm_max': float,
                'm_cli': float,
                'bin_width': float,
                'aseismic_coefficient': float,
                'rupture_aspect_ratio': float,
                'rupture_mesh_spacing': float,
                'minimum_fault_length': float,
                'tectonic_region_type': str,
                'upper_seismogenic_depth': float,
                'lower_seismogenic_depth': float,
                'magnitude_scaling_relation': str,
                'width_scaling_relation': str}


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def read_config_file(cfg_file):
    """
    Import various processing options from the (.ini) configuration file
    """
    cfg_dict = {}
    cfg = configparser.RawConfigParser(dict_type=dict)
    cfg.optionxform = str
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
                        width_method='length_scaling',
                        oqt_source=False,
                        project_name=None,
                        param_map=None,
                        defaults=None,
                        **kwargs):
    param_map_local = deepcopy(fmu.param_map)
    defaults_local = deepcopy(fmu.defaults)

    if param_map is not None:
        param_map_local.update(param_map)
    if defaults is not None:
        defaults_local.update(defaults)

    for key in kwargs:
        defaults_local[key] = kwargs[key]

    srcl = []

    for fl in fault_db.db:

        try:
            sfs_dict = fmu.construct_sfs_dict(fl,
                                              width_method=width_method,
                                              param_map=param_map_local,
                                              defaults=defaults_local)
            print("sfs_dict = ", sfs_dict)
            sfs = fmu.make_fault_source(sfs_dict, oqt_source=oqt_source)
            print("sfs en build_model_from_db = ", sfs)
            srcl.append(sfs)

        except Exception as e:
            if param_map:
                id = fl[param_map['source_id']]
                print("Couldn't process Fault {}: {}".format(id, e))
            else:
                raise

    if xml_output is not None:
        # Write the final fault model
        write_source_model(xml_output, srcl, project_name)
    else:
        return srcl


# -----------------------------------------------------------------------------

class FaultDatabase():
    """
    The fault_database object is used to retrieve, add, modify and export
    information from a database in geojson format
    """

    def __init__(self, geojson_file=None):
        # Initialise an empty fault list
        self.db = []

        if geojson_file:
            self.import_from_geojson(geojson_file)

    def import_from_geojson(self, geojson_file, black_list=None,
                            select_list=None, param_map=None,
                            update_keys=False):
        param_map_local = deepcopy(fmu.param_map)

        if param_map is not None:
            param_map_local.update(param_map)

        # Import database
        with open(geojson_file, 'r') as f:
            data = json.load(f)

            # Import geojson metadata
            self.meta = {k: data[k] for k in data if k != 'features'}

            # Loop over faults
            for feature in data['features']:

                fault = feature['properties']

                # Update parameter keys only if explicitly requested
                if update_keys:
                    pml = {}
                    for k in param_map_local:
                        k_map = param_map_local[k]
                        if k_map in fault:
                            fault[k] = fault.pop(k_map)
                            pml[k] = k
                else:
                    pml = param_map_local

                # Process only faults in the selection list
                if select_list is not None:
                    if not isinstance(select_list, (list, tuple)):
                        select_list = [select_list]
                    if fault[pml['source_id']] not in select_list:
                        continue

                # Skip further processing for blacklisted faults
                if black_list is not None:
                    if not isinstance(black_list, (list, tuple)):
                        black_list = [black_list]
                    if fault[pml['source_id']] in black_list:
                        continue

                # Get fault geometry
                fault['trace_coordinates'] = feature['geometry']['coordinates']

                self.db.append(fault)

    def export_to_geojson(self, geojson_file):
        with open(geojson_file, 'w') as f:

            data = {k: self.meta[k] for k in self.meta}
            data['features'] = []

            for fl in self.db:

                prop = {k: fl[k] for k in fl if k != 'trace_coordinates'}
                geom = {'coordinates': fl['trace_coordinates'],
                        'type': 'LineString'}

                feat = {'properties': prop,
                        'geometry': geom,
                        'type': 'Feature'}

                data['features'].append(feat)

            json.dump(data, f)

    def add_property(self, property, value=None, id=None, key='source_id'):
        for fault in self.db:
            if id is None:
                fault[property] = value
            else:
                if fault[key] is id:
                    fault[property] = value

    def remove_property(self, property, id=None, key='source_id'):
        for fault in self.db:
            if id is None:
                fault.pop(property)
            else:
                if fault[key] is id:
                    fault.pop(property)


# -----------------------------------------------------------------------------


def build_fault_model(*, cfg_file=None,
                      geojson_file=None,
                      xml_output=None,
                      black_list=None,
                      select_list=None,
                      project_name=None,
                      width_method='seismo_depth',
                      oqt_source=False,
                      param_map=None,
                      defaults=None,
                      **kwargs):
    """
    Main interface to create the fault source model from an active fault
    database in geojson format.
    Priority for the optional parameters is set to:
        1) single arguments
        2) dictionary
        3) .ini file
    """

    param_map_local = deepcopy(fmu.param_map)
    defaults_local = deepcopy(fmu.defaults)

    # Import arguments from INI configuration file
    if cfg_file is not None:
        cfg_dict = read_config_file(cfg_file)

        if 'config' in cfg_dict:
            config = cfg_dict['config']
            if 'geojson_file' in config:
                geojson_file = config['geojson_file']
            if 'xml_output' in config:
                xml_output = config['xml_output']
            if 'black_list' in config:
                black_list = ast.literal_eval(
                                config['black_list'])
            if 'select_list' in config:
                select_list = ast.literal_eval(
                                config['select_list'])

        if 'param_map' in cfg_dict:
            param_map_local.update(cfg_dict['param_map'])
        if 'defaults' in cfg_dict:
            defaults_local.update(cfg_dict['defaults'])

    if param_map is not None:
        param_map_local.update(param_map)
    if defaults is not None:
        defaults_local.update(defaults)

    for key in kwargs:
        defaults_local[key] = kwargs[key]

    # Import the fault database from geojson
    if geojson_file is not None:
        fault_db = FaultDatabase()
        fault_db.import_from_geojson(geojson_file,
                                     black_list=black_list,
                                     select_list=select_list,
                                     param_map=param_map_local)
    else:
        print('Geojson file not found')
        return

    # Create the fault source model in xml_format
    srcl = build_model_from_db(fault_db,
                               xml_output,
                               width_method=width_method,
                               oqt_source=oqt_source,
                               project_name=project_name,
                               param_map=param_map_local,
                               defaults=defaults_local)

    if xml_output is None:
        return srcl

    msg = 'Parameter configuration file (.ini)'
    build_fault_model.cfg_file = msg
    msg = 'Fault database in geojson format'
    build_fault_model.geojson_file = msg
    msg = 'Output xml containing the fault model'
    build_fault_model.xml_output = msg
    msg = 'List of fault IDs NOT to be processed [id1,id2]'
    build_fault_model.black_list = msg
    msg = 'List of selected fault IDs to be processed [id1,id2]'
    build_fault_model.select_list = msg
    msg = 'Name of the current project'
    build_fault_model.project_name = msg
    msg = 'Method to compute the fault width'
    build_fault_model.width_method = msg
    msg = 'Switch between hazardlib and oq-mbt source formats'
    build_fault_model.oqt_source = msg


if __name__ == "__main__":
    sap.run(build_fault_model)
