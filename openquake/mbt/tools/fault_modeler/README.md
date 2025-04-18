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

Note that if the *xml_output* argument is not specified (either as an argument of in the .ini file), the code returns a list of the generated OpenQuake fault source objects as output.

### From the Command Line

The above examples can also be run as a shell command:

```console
fault_source_modeler.py -geo FaultDatabase.geojson -xml FaultModel.xml
```
or identically:

```console
fault_source_modeler.py -cfg config.ini
```

(Type `fault_source_modeler.py -h` to get a list of the available options)

### Optional Parameters

Several optional parameters can be passed, either as arguments or within the configuration file. If these parameters are not passed, then the default values will be used, or default algorithms will compute the values. **Thus, it is best practice for the user to assign each parameter value using one of the following approaches.**

Modifiable parameters are:

| Key | Description | Default Value |
|:----|:------------|:--------|
| b_value | Gutenberg-Richter b-value of the derived MFD for each fault | 1.0 |
| m_min | Minimum magnitude of the MFD for each fault; seimsic moment is distributed between m_min and m_max  | 4.0 |
| m_max | Maximum magnitude of the MFD for each fault; seimsic moment is distributed between m_min and m_max | None |
| m_cli | Minimum magnitude of rupture to be admitted by the fault source; does not affect distribution of seismic moment | 6.0 |
| m_upp | maximum possible magnitude, which can not be exceeded regardless of fault dimensions | 10.0 |
| bin_width | Bin width of the MFD for each fault | 0.1 |
| aseismic_coefficient | Coefficient used to scale slip rate according to the portion released aseismically; 0.0 means all slip is seismic | 0.0 |
| rupture_aspect_ratio | Ratio used by OQ to deterimine length and width dimensions of ruptures  | 2.0 |
| rupture_mesh_spacing | OQ parameter, dimension of mesh cells on a rupture | 5.0 |
| minimum_fault_length | threshold below which fault sources won't be generated | 5.0 |
| tectonic_region_type | OQ parameter, tectonic region type of the fault source | 'Active Shallow Crust' |
| upper_seismogenic_depth | upper (shallow) limit of fault rupture | 0.0 |
| lower_seismogenic_depth | lower (deep) limit of fault rupture | 35.0 or computed from rupture length |
| magnitude_scaling_relationship | empirical relationship used to convert between rupture area and magnitude | Leonard2014_Interplate |
| mfd_type | shape of the MFD produced for each fault | DoubleTruncatedGR | 

A single parameter can be passed as optional argument like:

```python
build_fault_model(geojson_file='FaultDatabase.geojson', 
                  aseismic_coefficient=0.9,
                  lower_seismogenic_depth=25.)
```

(note that passing single parameters as flags is presently not yet implemented in the command line interface)

For multiple parameters, it is also possible to use a dictionary:

```python
defaults = {'b_value': 1.,
            'aseismic_coefficient': 0.1,
            'upper_seismogenic_depth': 0.,
            'lower_seismogenic_depth': 20.}

build_fault_model(geojson_file='FaultDatabase.geojson', 
                  defaults=defaults)
```

As final options, optional parameters can be specified in an external configuration file (see below).

However, providing an optional setting will always override the default 
(hardcoded) value for all the faults in the database at the same time. To 
provide fault specific parameterisation, it is possible to:

1. Manually modify the geojson by adding the key-value pair to the specific fault item

2. Use the following programmatic approach:

```python
from openquake.mbt.tools.fault_modeler.fault_source_modeler import FaultDatabase
from openquake.mbt.tools.fault_modeler.fault_source_modeler import build_model_from_db

# Import the database
fault_db = FaultDatabase()
fault_db.import_from_geojson(geojson_file)

# Add a key/value to fault with id 1
fault_db.add_property('upper_seismogenic_depth', value=20, id=1)

# Create and export the model
build_model_from_db(fault_db, xml_output='FaultModel.xml')
```
(note that the function *build_model_from_db* shares the same options of *build_fault_model*)

The *fault_database* class can be used also to export the modified database to geojson format:

```python
fault_db.export_geojson(geojson_file)
```

## Using non-standard Geojson file formats

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

The use of any of the three blocks is optional, as well as any of the directives. The combined used of the configuration file and the direct passing of arguments is allowed. Note, however, that in case an option is specified twice, the directive passed as optional argument will have priority.
