from openquake.mbt.tools.fault_modeler.fault_source_modeler import fault_database

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

# Import the database
fault_db = fault_database()
fault_db.import_from_geojson('Data/ne_asia_faults_rates_bvalue.geojson',
                              param_map=param_map,
                              select_list=[1])

# Add a key/value to fault with id 1
# fault_db.add_property('upper_seismogenic_depth', value=20, id=1)

fault_db.export_to_geojson('Data/output.geojson')


"""
from openquake.mbt.tools.fault_modeler.fault_source_modeler import build_model_from_db

# Create and export the model
build_model_from_db(fault_db,xml_output='FaultModel.xml')
"""