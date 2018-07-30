"""
Just a very preliminary testing script, to check if the code works in first
instance. More accurate testing procedures will follow....
"""

from openquake.mbt.tools.fault_modeler.fault_source_modeler import build_fault_model

param_map = {'source_id': 'ogc_fid',
             'name': 'ns_name',
             'average_dip': 'ns_average_dip',
             'average_rake': 'ns_average_rake',
             'net_slip_rate': 'ns_net_slip_rate',
             'vert_slip_rate': 'ns_vert_slip_rate',
             'strike_slip_rate': 'ns_strike_slip_rate',
             'shortening_rate': 'ns_shortening_rate',
             'dip_dir': 'ns_dip_dir',
             'dip_slip_rate': 'ns_dip_slip_rate'}

if 0:
    build_fault_model(
        geojson_file='Data/ne_asia_faults_rates_bvalue.geojson',
        xml_output='Data/test.xml',
        select_list=[1,2,3],
        param_map=param_map,
        lower_seismogenic_depth=30.)

if 1:
    build_fault_model(
        geojson_file='Data/ne_asia_faults_rates_bvalue.geojson',
        xml_output='Data/test.xml',
        param_map=param_map,
        defaults={'lower_seismogenic_depth':30.})

if 0:
    build_fault_model(cfg_file='config.ini')