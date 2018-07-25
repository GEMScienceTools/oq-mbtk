# Fault Model Building Tools

This module includes a set of functionalities to create an OpenQuake source model from an active fault database in geojson format.

## Getting Started

The fault model builder can be run as a Python module or directly from the command line.

### In Python

As a module, the main function *build_fault_model* can be called with:

```python
from openquake.mbt.tools.fault_modeler.fault_source_modeler import build_fault_model

build_fault_model(geojson_file='FaultDatabase.geojson', xml_output='FaultModel.xml')
```

or, alternatively, by passing a configuration file (.ini):

```python
build_fault_model(cfg_file='config.ini')
```

Note that if the *xml_output* is not specified (either as an argument of in the .ini file), the function will return directly the list of generated OpenQuake source objects as output.

### Using the Console

The above example can also be run a shell command as:

```console
fault_source_modeler.py -geo FaultDatabase.geojson -xml FaultModel.xml
```
or identically:

```console
fault_source_modeler.py -cfg config.ini
```

(Type ```fault_source_modeler.py -h``` to get a list of the available options)

### Optional Parameters

This is the list of optional parameters that can be passed as argument or in the configuration file (not yet available for the console):

| Key | Description | Default |
|:----|:------------|:--------|
| b_value | ... | ... |
| M_min | ... | ... |
| M_max | ... | ... |
| bin_width | ... | ... |
| aseismic_coefficient | ... | ... |
| rupture_aspect_ratio | ... | ... |
| rupture_mesh_spacing | ... | ... |
| minimum_fault_length | ... | ... |
| tectonic_region_type | ... | ... |
| upper_seismogenic_depth | ... | ... |
| lower_seismogenic_depth | ... | ... |
| magnitude_scaling_relationship | ... | ... |

Note that the provided settings will override the default (hardcoded) values for all the faults. To provide fault specific parametrisation, it is possible to:

1. Manually modify the geojson by adding the key-value pair to the specific fault item

2. Use the following programmatic approach:

```python
from openquake.mbt.tools.fault_modeler.fault_source_modeler import fault_database
from openquake.mbt.tools.fault_modeler.fault_source_modeler import build_model_from_db

# Import the database
fault_db = fault_database()
fault_db.import_from_geojson(geojson_file)

# Add a key/value to fault with id 1
fault_db.add_property('upper_seismogenic_depth', value=20, id=1)

# Create and export the model
build_model_from_db(fault_db,xml_output='FaultModel.xml')
```
The function *build_model_from_db* shares the same options of *build_fault_model*.

## Using non-standard Geojson file format

In case a geojson database with *non-standard* keyword (e.g. an old version),
it is possible to specify the translation map with a simple dictionary:

```
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

build_fault_model(geojson_file='FaultDatabase.geojson',
                  xml_output='FaultModel.xml')
                  param_map=param_map)
```

As well, the translation information can be specified in the configuration file (see following section)

## Configuration File Format

The configuration file (.ini) replaces the need of manually passing the arguments to the main function.
The file has the following standard format:

```
[config]
geojson_file = FaultDatabase.geojson
xml_output = FaultModel.xml
select_list = 1,2,3

[param_map]
source_id = ogc_fid
name = ns_name
average_dip = ns_average_dip
average_rake = ns_average_rake
net_slip_rate = ns_net_slip_rate
vert_slip_rate = ns_vert_slip_rate
strike_slip_rate = ns_strike_slip_rate
shortening_rate = ns_shortening_rate
dip_dir = ns_dip_dir
dip_slip_rate = ns_dip_slip_rate

[defaults]
upper_seismogenic_depth = 0
lower_seismogenic_depth = 15
tectonic_region_type = Active Shallow Crust
rupture_mesh_spacing = 5
```

The first block is referred to the input/output. The second block contains the keyword translation map when using non-standard a geojson fault database. The third block includes a list of settings to be used as default parameters.

The use of any of the three blocks is optional, as well as any of the directives. The combined used of the configuration file and the direct passing of arguments is allowed. Note, however, that in case an option is specified twice, the directive in the configuration file will have priority.